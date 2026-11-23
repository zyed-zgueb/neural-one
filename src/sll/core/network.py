"""Réseau commun à tous les élèves.

    x ∈ R^D  ->  u = W1 x + b1 (+ ξ)  ->  h = relu(u)  ->  z = W2 h + b2  ->  p = softmax(z / T)

Un seul réseau, une seule initialisation pour tous les modèles : seule la **règle de mise à jour**
change d'un élève à l'autre (même nombre de paramètres, mêmes conditions initiales).

- `forward` renvoie un `Activity` : tout l'état interne d'un passage (x, u, ξ, h, z, p), ce qui
  sert à l'instrumentation et, plus tard, à la Shadow Memory.
- `backward` : rétropropagation exacte (réservée aux références M0/M1/G0/G1), vérifiée par
  différences finies dans les tests.
- `local_traces` : produits locaux pré × post utilisés par les règles à trois facteurs (M2/G2) :
    sortie  : c2 = (onehot(a) − p) ⊗ h            (local : activité de l'unité de sortie et de son entrée)
    cachée  : c1 = (ξ ⊙ 1[u>0]) ⊗ x / σ²          (perturbation de nœud : bruit propre au neurone)
  Aucune information d'erreur ne circule de la sortie vers la couche cachée.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Activity:
    x: np.ndarray
    u: np.ndarray
    xi: np.ndarray
    h: np.ndarray
    z: np.ndarray
    p: np.ndarray


class MLP:
    def __init__(self, d_in: int, d_hidden: int, d_out: int, rng: np.random.Generator,
                 temperature: float = 1.0, init_scale: float = 1.0):
        self.d_in, self.d_hidden, self.d_out = d_in, d_hidden, d_out
        self.W1 = rng.normal(0, init_scale * np.sqrt(2.0 / d_in), (d_hidden, d_in))
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(0, init_scale * np.sqrt(1.0 / d_hidden), (d_out, d_hidden))
        self.b2 = np.zeros(d_out)
        self.T = temperature

    # -------------------------------------------------------------- état
    def params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def n_params(self) -> int:
        return sum(v.size for v in self.params().values())

    def copy(self) -> "MLP":
        m = MLP.__new__(MLP)
        m.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return m

    # -------------------------------------------------------------- calcul
    def forward(self, x: np.ndarray, xi: np.ndarray | None = None) -> Activity:
        u = self.W1 @ x + self.b1
        if xi is None:
            xi = np.zeros_like(u)
        h = np.maximum(u + xi, 0.0)
        z = self.W2 @ h + self.b2
        zz = z / self.T
        e = np.exp(zz - zz.max())
        return Activity(x=x, u=u, xi=xi, h=h, z=z, p=e / e.sum())

    def forward_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H = np.maximum(X @ self.W1.T + self.b1, 0.0)
        Z = H @ self.W2.T + self.b2
        return H, Z

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward_batch(X)[1].argmax(1)

    def backward(self, act: Activity, dz: np.ndarray) -> dict[str, np.ndarray]:
        """Gradient de L par rapport aux paramètres, étant donné dL/dz (rétropropagation)."""
        dW2 = np.outer(dz, act.h)
        db2 = dz
        dh = self.W2.T @ dz
        du = dh * ((act.u + act.xi) > 0)
        return {"W1": np.outer(du, act.x), "b1": du, "W2": dW2, "b2": db2}

    def local_traces(self, act: Activity, a: int, sigma: float) -> dict[str, np.ndarray]:
        """Produits locaux pour une règle à trois facteurs (pas de rétropropagation)."""
        post = -act.p.copy()
        post[a] += 1.0
        c2 = np.outer(post, act.h)
        if sigma > 0:
            g = act.xi * ((act.u + act.xi) > 0) / (sigma ** 2)
        else:
            g = np.zeros_like(act.u)
        return {"W1": np.outer(g, act.x), "b1": g, "W2": c2, "b2": post}

    def apply(self, delta: dict[str, np.ndarray], lr: float, layers: tuple[str, ...] = ("W1", "b1", "W2", "b2"),
              max_norm: float | None = None) -> float:
        """Ajoute lr × delta aux paramètres choisis ; renvoie la norme de la modification."""
        total = 0.0
        for k in layers:
            d = lr * delta[k]
            if max_norm is not None:
                n = np.linalg.norm(d)
                if n > max_norm:
                    d = d * (max_norm / n)
            getattr(self, k)[...] += d
            total += float((d ** 2).sum())
        return float(np.sqrt(total))


def softmax_xent_grad(p: np.ndarray, y: int) -> np.ndarray:
    """dL/dz pour l'entropie croisée (température 1)."""
    g = p.copy()
    g[y] -= 1.0
    return g
