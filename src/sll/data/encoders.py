"""Encodeurs fixes (non appris), partagés par tous les modèles.

- E0 `pixels` : l'image d'encre aplatie (1024).
- E1 `random` : projection aléatoire fixe + ReLU (contrôle « features sans a priori »).
- E2 `edges`  : 8 filtres d'orientation fixes (dérivées de gaussienne orientées, type V1),
                redressés, puis pooling moyen 8×8 -> 4×4×8 = 128 dimensions.

Aucun encodeur ne contient de détecteur de « coin » ou de « cercle » : ce serait donner la réponse
(décision D26).
"""
from __future__ import annotations

import numpy as np

from ..core.rng import make_rng


def _conv2d_same(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Convolution 2D 'same' (images (N,H,W), noyau (h,w)) sans dépendance externe."""
    n, H, W = img.shape
    h, w = k.shape
    ph, pw = h // 2, w // 2
    pad = np.pad(img, ((0, 0), (ph, ph), (pw, pw)))
    win = np.lib.stride_tricks.sliding_window_view(pad, (h, w), axis=(1, 2))  # (n,H,W,h,w)
    return np.einsum("nijkl,kl->nij", win, k[::-1, ::-1])


def orientation_kernels(n_orient: int = 8, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    ax = np.arange(size) - size // 2
    X, Y = np.meshgrid(ax, ax)
    ks = []
    for i in range(n_orient):
        th = np.pi * i / n_orient
        u = X * np.cos(th) + Y * np.sin(th)
        g = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        k = -u * g  # dérivée première orientée
        k -= k.mean()
        k /= np.abs(k).sum()
        ks.append(k)
    return np.stack(ks)


class Encoder:
    name = "base"
    dim = 0

    def __call__(self, X: np.ndarray) -> np.ndarray:  # X : (N,32,32) encre
        raise NotImplementedError


class PixelEncoder(Encoder):
    name, dim = "pixels", 1024

    def __call__(self, X):
        X = np.asarray(X, float)
        return X.reshape(len(X), -1)


class RandomEncoder(Encoder):
    name = "random"

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        rng = make_rng(seed, "encoder", "random")
        self.W = rng.normal(0, 1 / np.sqrt(1024), (1024, dim))
        self.b = rng.normal(0, 0.1, dim)

    def __call__(self, X):
        X = np.asarray(X, float).reshape(len(X), -1)
        return np.maximum(X @ self.W + self.b, 0.0)


class EdgeEncoder(Encoder):
    name = "edges"

    def __init__(self, n_orient: int = 8, pool: int = 8):
        self.kernels = orientation_kernels(n_orient)
        self.pool = pool
        self.dim = n_orient * (32 // pool) ** 2

    def __call__(self, X):
        X = np.asarray(X, float)
        maps = []
        for k in self.kernels:
            r = np.abs(_conv2d_same(X, k))  # énergie orientée (signe du contraste ignoré)
            n, H, W = r.shape
            p = self.pool
            maps.append(r.reshape(n, H // p, p, W // p, p).mean(axis=(2, 4)))
        return np.stack(maps, axis=1).reshape(len(X), -1)


def get_encoder(name: str, seed: int = 0) -> Encoder:
    return {"pixels": PixelEncoder, "random": lambda: RandomEncoder(seed=seed),
            "edges": EdgeEncoder}[name]()
