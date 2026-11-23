"""Outils statistiques de base (sans dépendance externe)."""
from __future__ import annotations

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def rule_of_three_upper(n: int) -> float:
    """Borne supérieure (95 %) d'une probabilité quand 0 succès sur n essais."""
    return 3.0 / n if n else 1.0


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
