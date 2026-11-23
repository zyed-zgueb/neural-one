"""Sondes anti-raccourcis.

Question : peut-on distinguer cercles et carrés à partir de statistiques **qui ne sont pas la forme**
(quantité d'encre, position, épaisseur) ? Si oui, le générateur offre un raccourci et un modèle
pourrait « réussir » sans rien comprendre.

Méthode : régression logistique (NumPy, standardisée, L2) en validation croisée 5 plis.
On sépare :
- sondes de **nuisance** (doivent rester au hasard) : encre totale, centroïde, épaisseur ;
- sondes de **compacité** (propriétés géométriques légitimes, rapportées à titre informatif) :
  étendue (boîte englobante), dispersion spatiale de l'encre.
"""
from __future__ import annotations

import numpy as np

from .stats import wilson_ci

NUISANCE = ["encre_totale", "centroide_x", "centroide_y", "epaisseur_moyenne"]
GEOMETRIC = ["largeur_boite", "hauteur_boite", "aire_boite", "dispersion"]


def image_statistics(X: np.ndarray) -> dict[str, np.ndarray]:
    X = np.asarray(X, float)
    n = len(X)
    ys, xs = np.mgrid[0:32, 0:32]
    tot = X.sum((1, 2))
    w = X / np.maximum(tot[:, None, None], 1e-9)
    cx = (w * xs).sum((1, 2))
    cy = (w * ys).sum((1, 2))
    spread = np.sqrt((w * ((xs - cx[:, None, None]) ** 2 + (ys - cy[:, None, None]) ** 2)).sum((1, 2)))
    mask = X > 0.5
    bw, bh = np.zeros(n), np.zeros(n)
    for i in range(n):
        yy, xx = np.where(mask[i])
        if len(xx):
            bw[i] = xx.max() - xx.min() + 1
            bh[i] = yy.max() - yy.min() + 1
    # proxy d'épaisseur : encre / longueur du squelette approchée par le nombre de pixels > 0.5
    thick = tot / np.maximum(mask.sum((1, 2)), 1)
    return {"encre_totale": tot, "centroide_x": cx, "centroide_y": cy, "epaisseur_moyenne": thick,
            "largeur_boite": bw, "hauteur_boite": bh, "aire_boite": bw * bh, "dispersion": spread}


def _fit_logreg(X, y, l2=1e-2, iters=500, lr=0.5):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def probe_accuracy(F: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0, quadratic: bool = True) -> dict:
    """Précision en validation croisée d'une sonde logistique sur les features F."""
    F = np.asarray(F, float)
    if F.ndim == 1:
        F = F[:, None]
    if quadratic:  # ajoute carrés et produits : capture des seuils et des rapports simples
        cols = [F] + [F[:, i:i + 1] * F[:, j:j + 1] for i in range(F.shape[1]) for j in range(i, F.shape[1])]
        F = np.concatenate(cols, 1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    correct = 0
    for k in range(folds):
        te = parts[k]
        tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        w, b = _fit_logreg((F[tr] - mu) / sd, y[tr])
        pred = ((F[te] - mu) / sd @ w + b) > 0
        correct += int((pred == y[te]).sum())
    acc = correct / len(y)
    lo, hi = wilson_ci(correct, len(y))
    return {"accuracy": acc, "ci95": (lo, hi), "n": len(y)}


def run_probes(X: np.ndarray, y: np.ndarray) -> dict:
    stats = image_statistics(X)
    out = {}
    for k, v in stats.items():
        out[k] = probe_accuracy(v, y)
    out["NUISANCE_jointe"] = probe_accuracy(np.stack([stats[k] for k in NUISANCE], 1), y)
    out["GEOMETRIE_jointe"] = probe_accuracy(np.stack([stats[k] for k in GEOMETRIC + ["encre_totale"]], 1), y)
    return out
