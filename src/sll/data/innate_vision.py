"""SV0 — système visuel inné de l'élève (fixe, non appris). Voir PARADIGME_IMITATION.md §13.

Abstractions fonctionnelles de mécanismes présents tôt dans le développement :

| Étage | Inspiration | Sortie |
|-------|-------------|--------|
| orientation locale | cellules simples V1 | énergie par orientation (8) |
| tolérance de position | cellules complexes V1, grands champs récepteurs | histogramme global (8) [+ carte grossière 2×2 (32), option] |
| contiguïté | connexité du trait | nombre de morceaux |
| terminaisons | cellules « end-stopped » | nombre de bouts de ligne |
| jonctions | — | nombre d'embranchements |
| fermeture | perception topologique (Chen 1982) + complétion par bonne continuation | niveau de fermeture ∈ [0,1] ; nombre de trous |

Aucun détecteur d'angle, de coin, ni de courbure (frontière inné / appris, D34).

Mise en œuvre : le trait est binarisé après un léger flou (élimine le bruit isolé), les
morceaux de moins de 4 pixels sont ignorés (bruit). Squelette par amincissement de Zhang–Suen ;
terminaisons / jonctions par le « nombre de passages » autour de chaque point du squelette (1 / ≥ 3). Fermeture = existence
d'une région de fond enclose (non reliée au bord) ; la complétion consiste à « épaissir »
virtuellement le trait de k pixels (k = 0..3) : une forme fermée à k = 0 est franchement fermée,
une forme qui ne se ferme qu'à k = 3 a de grandes lacunes.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .encoders import Encoder, _conv2d_same, orientation_kernels

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _label(mask: np.ndarray, neigh) -> tuple[np.ndarray, int]:
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    n = 0
    for i, j in zip(*np.nonzero(mask)):
        if lab[i, j]:
            continue
        n += 1
        lab[i, j] = n
        q = deque([(i, j)])
        while q:
            a, b = q.popleft()
            for da, db in neigh:
                x, y = a + da, b + db
                if 0 <= x < H and 0 <= y < W and mask[x, y] and not lab[x, y]:
                    lab[x, y] = n
                    q.append((x, y))
    return lab, n


def binarize(ink: np.ndarray, thr: float = 0.3, min_size: int = 4) -> np.ndarray:
    p = np.pad(ink, 1, mode="edge")
    blur = sum(p[i:i + ink.shape[0], j:j + ink.shape[1]] for i in range(3) for j in range(3)) / 9
    m = blur > thr
    lab, n = _label(m, N8)
    if n:
        sizes = np.bincount(lab.ravel())
        small = np.where(sizes < min_size)[0]
        small = small[small > 0]
        m[np.isin(lab, small)] = False
    return m


def _dilate(m: np.ndarray, k: int) -> np.ndarray:
    out = m.copy()
    for _ in range(k):
        p = np.pad(out, 1)
        out = np.zeros_like(out)
        for i in range(3):
            for j in range(3):
                out |= p[i:i + m.shape[0], j:j + m.shape[1]]
    return out


def thin(m: np.ndarray) -> np.ndarray:
    """Amincissement de Zhang–Suen (squelette d'un pixel d'épaisseur)."""
    img = np.pad(m.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = img
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            c = P[1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(x.astype(int) for x in nb)
            seq = nb + [p2]
            A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(int) for k in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def _crossing_number(s: np.ndarray) -> np.ndarray:
    """Nombre de passages fond -> trait en faisant le tour des 8 voisins (robuste aux amas de pixels) :
    1 = bout de ligne, 2 = trait continu, >= 3 = embranchement."""
    p = np.pad(s.astype(int), 1)
    H, W = s.shape
    ring = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    v = [p[1 + a:1 + a + H, 1 + b:1 + b + W] for a, b in ring]
    return sum(((v[k] == 0) & (v[(k + 1) % 8] == 1)).astype(int) for k in range(8))


def _holes(m: np.ndarray) -> tuple[int, int]:
    """(nombre de régions de fond encloses, aire enclose totale)."""
    lab, n = _label(~m, N4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    areas = [int((lab == k).sum()) for k in range(1, n + 1) if k not in border]
    return len(areas), int(sum(areas))


def topology(ink: np.ndarray, max_bridge: int = 3) -> dict:
    """Primitives topologiques d'une image de trait."""
    m = binarize(ink)
    if not m.any():
        return {"morceaux": 0, "terminaisons": 0, "jonctions": 0, "fermeture": 0.0, "trous": 0,
                "complétion": max_bridge + 1}
    _, n_comp = _label(m, N8)
    sk = thin(m)
    cn = _crossing_number(sk)
    ends = int((sk & (cn == 1)).sum())
    junc_mask = sk & (cn >= 3)
    _, n_junc = _label(junc_mask, N8)  # jonctions voisines comptées une fois
    # Purement topologique : on compte les régions encloses, sans mesurer leur aire (une aire
    # relative serait une information de forme — compacité — et non de topologie).
    k_closed, n_holes = max_bridge + 1, 0
    for k in range(max_bridge + 1):
        nh, _ = _holes(_dilate(m, k))
        if nh > 0:
            k_closed, n_holes = k, nh
            break
    return {"morceaux": int(n_comp), "terminaisons": ends, "jonctions": int(n_junc),
            "fermeture": float(1 - k_closed / (max_bridge + 1)), "trous": int(n_holes),
            "complétion": int(k_closed)}


class InnateVision(Encoder):
    """SV0 : orientation (globale + carte 2×2) + primitives topologiques."""
    name = "SV0"

    def __init__(self, use_orientation: bool = True, use_topology: bool = True, n_orient: int = 8,
                 use_map: bool = False):
        """use_map : ajoute une carte grossière 2×2 des orientations (rétinotopique). Désactivée par
        défaut : CR-07 montre qu'elle dégrade fortement l'apprentissage à peu d'exemples (elle ajoute
        32 dimensions dépendantes de la position, sans rapport avec la forme)."""
        self.use_orientation, self.use_topology, self.use_map = use_orientation, use_topology, use_map
        self.kernels = orientation_kernels(n_orient)
        self.dim = ((n_orient + (4 * n_orient if use_map else 0)) if use_orientation else 0) + (5 if use_topology else 0)
        if not use_topology:
            self.name = "SV0-orientation"
        elif not use_orientation:
            self.name = "SV0-topologie"

    def orientation(self, X):
        maps = np.stack([np.abs(_conv2d_same(X, k)) for k in self.kernels], 1)  # (n,8,32,32)
        tot = maps.sum((1, 2, 3))[:, None] + 1e-9
        glob = maps.sum((2, 3)) / tot
        if not self.use_map:
            return glob
        coarse = maps.reshape(len(X), maps.shape[1], 2, 16, 2, 16).sum((3, 5)).reshape(len(X), -1) / tot
        return np.concatenate([glob, coarse], 1)

    def topology_features(self, X):
        rows = []
        for x in X:
            t = topology(x)
            rows.append([min(t["morceaux"], 5), min(t["terminaisons"], 8), min(t["jonctions"], 8),
                         t["fermeture"], min(t["trous"], 4)])
        return np.array(rows, float)

    def __call__(self, X):
        X = np.asarray(X, float)
        parts = []
        if self.use_orientation:
            parts.append(self.orientation(X))
        if self.use_topology:
            parts.append(self.topology_features(X))
        return np.concatenate(parts, 1)


def _enclosed(m: np.ndarray) -> bool:
    """Existe-t-il du fond non relié au bord ? (propagation vectorisée, rapide)"""
    free = ~m
    reach = np.zeros_like(m)
    reach[0, :], reach[-1, :], reach[:, 0], reach[:, -1] = free[0, :], free[-1, :], free[:, 0], free[:, -1]
    while True:
        p = np.pad(reach, 1)
        grown = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | reach) & free
        if (grown == reach).all():
            break
        reach = grown
    return bool((free & ~reach).any())


def closure_level(ink: np.ndarray, max_bridge: int = 3) -> float:
    """Sensation innée de fermeture ∈ [0, 1] (même définition que `topology`, version rapide) :
    1 = franchement fermé ; décroît avec la taille des lacunes à combler ; 0 = ouvert."""
    m = binarize(ink)
    if not m.any():
        return 0.0
    for k in range(max_bridge + 1):
        if _enclosed(_dilate(m, k)):
            return float(1 - k / (max_bridge + 1))
    return 0.0
