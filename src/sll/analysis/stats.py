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


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> dict:
    """Test de Wilcoxon apparié (approximation normale, correction des ex-æquo). Bilatéral."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return {"n": 0, "W": 0.0, "z": 0.0, "p": 1.0}
    a = np.abs(d)
    order = np.argsort(a)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    # rangs moyens pour les ex-æquo
    vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    W = ranks[d > 0].sum()
    mu = n * (n + 1) / 4
    tie = (counts ** 3 - counts).sum() / 48
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie)
    z = (W - mu) / sd if sd > 0 else 0.0
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))
    return {"n": int(n), "W": float(W), "z": float(z), "p": float(p)}


def holm(pvalues: dict) -> dict:
    """Correction de Holm-Bonferroni : renvoie les p-valeurs ajustées."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
