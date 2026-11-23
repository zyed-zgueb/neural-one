"""Moteur de rendu partagé.

Toutes les images du laboratoire — formes du maître ET dessins de l'élève — passent par cette
unique fonction. Il n'existe donc aucun « style » propre à une classe ou à un auteur : seule la
géométrie du tracé diffère.

Méthode : un tracé est une liste de polylignes en coordonnées continues (pixels, origine en haut à
gauche, y vers le bas). Un sous-pixel est encré s'il est à une distance <= épaisseur/2 du tracé.
On sur-échantillonne (ss × ss sous-pixels par pixel) puis on moyenne : anti-crénelage identique
pour tous.

Convention : on renvoie l'**encre** (1 = noir, 0 = fond blanc). L'image affichée est 1 - encre.
"""
from __future__ import annotations

import numpy as np

IMG_SIZE = 32


def _segments(polylines: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    for pl in polylines:
        pl = np.asarray(pl, dtype=float)
        if len(pl) < 2:
            continue
        a.append(pl[:-1])
        b.append(pl[1:])
    if not a:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(a), np.concatenate(b)


def render(polylines: list[np.ndarray], thickness: float, size: int = IMG_SIZE, ss: int = 4) -> np.ndarray:
    """Rend des polylignes en image d'encre (size × size, valeurs dans [0, 1])."""
    A, B = _segments(polylines)
    N = size * ss
    hi = np.zeros((N, N), dtype=bool)
    r = thickness / 2.0
    r2 = r * r
    for a, b in zip(A, B):
        lo = np.floor((np.minimum(a, b) - r) * ss - 0.5).astype(int)
        up = np.ceil((np.maximum(a, b) + r) * ss + 0.5).astype(int)
        x0, y0 = max(lo[0], 0), max(lo[1], 0)
        x1, y1 = min(up[0], N), min(up[1], N)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = (np.arange(x0, x1) + 0.5) / ss
        ys = (np.arange(y0, y1) + 0.5) / ss
        ab = b - a
        l2 = max(float(ab @ ab), 1e-12)
        px = xs[None, :] - a[0]
        py = ys[:, None] - a[1]
        t = np.clip((px * ab[0] + py * ab[1]) / l2, 0.0, 1.0)
        dx = px - t * ab[0]
        dy = py - t * ab[1]
        hi[y0:y1, x0:x1] |= (dx * dx + dy * dy) <= r2
    return hi.reshape(size, ss, size, ss).mean(axis=(1, 3))


def to_display(ink: np.ndarray) -> np.ndarray:
    """Encre -> image affichable (noir sur blanc)."""
    return 1.0 - np.clip(ink, 0.0, 1.0)
