"""Le maître.

Deux maîtres (décision D11) :

- `CategoricalTeacher` (principal) juge **la forme** d'un tracé : « est-ce un carré ? » /
  « est-ce un cercle ? », indépendamment de sa taille, sa position, son orientation. Il rend :
    * `accepted` : le pouce (vrai/faux) ;
    * `score` ∈ [0, 1] : l'encouragement gradué, « c'est presque ça » (D22) ;
    * `components` : chaque critère et sa note — base du futur « remontrer en insistant » (D21),
      qui mettra l'accent sur le critère le plus raté (`worst`).
- `CopyTeacher` (contrôle) juge la **ressemblance pixel** avec le modèle montré. Il récompense la
  copie, donc la mémorisation : il sert à montrer la différence.

Le maître catégoriel analyse le **tracé** (suite de points), pas les pixels. C'est un choix
d'environnement : le maître est un juge géométrique fiable. Il juge de la même façon les formes du
générateur (leur contour) et les gestes de l'élève.

Les tolérances (`Tolerance`) définissent ce qu'est « un carré acceptable ». Elles sont figées avant
les expériences et sont les mêmes pour tous les élèves. Elles restent ouvertes à l'intuition humaine
(CR-05, question 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.render import render
from ..data.shapes_generator import CIRCLE, SQUARE


@dataclass(frozen=True)
class Tolerance:
    closure_gap: float = 0.10          # écart fin-début / périmètre
    total_turn_deg: float = 45.0       # |tour total| doit être 360° ± ceci
    corner_threshold_deg: float = 55.0 # virage local au-delà duquel on parle de « coin »
    corner_angle_deg: tuple[float, float] = (60.0, 120.0)
    side_cv: float = 0.20              # inégalité des côtés (écart-type / moyenne)
    straightness: float = 0.12         # flèche max d'un côté / longueur du côté
    radial_cv: float = 0.09            # irrégularité du rayon d'un cercle


@dataclass
class Analysis:
    perimeter: float
    closure_gap: float
    total_turn_deg: float
    corners: np.ndarray                # indices des coins dans le tracé ré-échantillonné
    corner_angles_deg: np.ndarray
    side_lengths: np.ndarray
    straightness: np.ndarray
    radial_cv: float
    resampled: np.ndarray

    @property
    def n_corners(self) -> int:
        return len(self.corners)


@dataclass
class Judgment:
    target: int
    accepted: bool
    score: float
    components: dict = field(default_factory=dict)   # critère -> (valeur, note ∈ [0,1], ok)
    worst: str = ""
    analysis: Analysis | None = None


def _resample(pts: np.ndarray, n: int, closed: bool) -> np.ndarray:
    P = np.vstack([pts, pts[:1]]) if closed else pts
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 1e-9:
        return np.repeat(pts[:1], n, 0)
    q = np.linspace(0, s[-1], n, endpoint=not closed)
    return np.stack([np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])], 1)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def analyze(points: np.ndarray, tol: Tolerance = Tolerance(), n: int = 128) -> Analysis:
    pts = np.asarray(points, float)
    # retire les points dupliqués consécutifs
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
    pts = pts[keep]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    perim = float(seg.sum())
    gap = float(np.linalg.norm(pts[-1] - pts[0]) / max(perim, 1e-9))
    closed = gap <= tol.closure_gap
    if closed and np.linalg.norm(pts[-1] - pts[0]) < 1e-6:
        pts = pts[:-1]
    R = _resample(pts, n, closed)
    d = np.diff(np.vstack([R, R[:1]]) if closed else R, axis=0)
    head = np.arctan2(d[:, 1], d[:, 0])
    if closed:
        turn = _wrap(np.roll(head, -1) - head)
    else:
        turn = np.concatenate([_wrap(np.diff(head)), [0.0]])
    total = float(np.rad2deg(turn.sum()))

    # coins : virage cumulé sur une fenêtre d'environ 1/18 du périmètre
    w = max(3, n // 18)
    kern = np.ones(w)
    if closed:
        ext = np.concatenate([turn[-w:], turn, turn[:w]])
        W = np.convolve(ext, kern, mode="same")[w:-w]
    else:
        W = np.convolve(turn, kern, mode="same")
    A = np.abs(np.rad2deg(W))
    thr = tol.corner_threshold_deg
    corners = []
    cand = np.where(A > thr)[0]
    # suppression des non-maxima : un coin par plage contiguë au-dessus du seuil
    if len(cand):
        groups, cur = [], [cand[0]]
        for i in cand[1:]:
            if i - cur[-1] <= 1:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        if closed and len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == n - 1:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
        # position du coin = centre de masse du virage dans la plage (robuste aux coins arrondis)
        absturn = np.abs(turn)
        for g in groups:
            g = np.array(g)
            lo_i, hi_i = g[0] - w // 2, g[-1] + w // 2
            idx = np.arange(lo_i, hi_i + 1)
            idx_m = idx % n if closed else np.clip(idx, 0, n - 1)
            wts = absturn[idx_m]
            pos = float((idx * wts).sum() / max(wts.sum(), 1e-12))
            corners.append(int(round(pos)) % n if closed else int(np.clip(round(pos), 0, n - 1)))
    corners = np.array(sorted(corners), dtype=int)
    # angle de chaque coin = virage cumulé sur une fenêtre plus large (coins arrondis)
    wide = 2 * w
    angles = []
    for c in corners:
        idx = (c + np.arange(-wide // 2, wide // 2 + 1))
        idx = idx % n if closed else np.clip(idx, 0, n - 1)
        angles.append(abs(np.rad2deg(turn[np.unique(idx)].sum())))
    angles = np.array(angles)

    # côtés entre coins consécutifs
    sides, straight = [], []
    if len(corners) >= 2:
        cs = list(corners) + ([corners[0] + n] if closed else [])
        for a, b in zip(cs[:-1], cs[1:]):
            idx = np.arange(a, b + 1) % n
            seg_pts = R[idx]
            length = float(np.linalg.norm(np.diff(seg_pts, axis=0), axis=1).sum())
            # flèche mesurée sur la partie centrale du côté (on tolère les coins arrondis)
            m = len(seg_pts)
            core = seg_pts[int(0.15 * m): max(int(0.85 * m), int(0.15 * m) + 2)]
            seg_pts = core if len(core) >= 2 else seg_pts
            chord = seg_pts[-1] - seg_pts[0]
            cl = np.linalg.norm(chord)
            if cl > 1e-9:
                nrm = np.array([-chord[1], chord[0]]) / cl
                dev = np.abs((seg_pts - seg_pts[0]) @ nrm).max() / cl
            else:
                dev = 1.0
            sides.append(length)
            straight.append(dev)
    rad = np.linalg.norm(R - R.mean(0), axis=1)
    radial_cv = float(rad.std() / max(rad.mean(), 1e-9))
    return Analysis(perimeter=perim, closure_gap=gap, total_turn_deg=total, corners=corners,
                    corner_angles_deg=angles, side_lengths=np.array(sides), straightness=np.array(straight),
                    radial_cv=radial_cv, resampled=R)


def _note(value: float, limit: float) -> float:
    """Note continue ∈ ]0, 1] : 1 si parfait, ≈ 0,61 à la limite de tolérance, puis chute rapide.

    C'est l'encouragement « c'est presque ça » : il varie même à l'intérieur de la tolérance, pour
    qu'un dessin meilleur reçoive toujours une meilleure note.
    """
    return float(np.exp(-0.5 * (value / limit) ** 2))


class CategoricalTeacher:
    name = "categoriel"

    def __init__(self, tol: Tolerance = Tolerance()):
        self.tol = tol

    def judge(self, points: np.ndarray, target: int) -> Judgment:
        t = self.tol
        an = analyze(points, t)
        comp = {}
        comp["fermeture"] = (an.closure_gap, _note(an.closure_gap, t.closure_gap),
                             an.closure_gap <= t.closure_gap)
        turn_err = abs(abs(an.total_turn_deg) - 360.0)
        comp["tour_complet"] = (turn_err, _note(turn_err, t.total_turn_deg), turn_err <= t.total_turn_deg)
        nc = an.n_corners
        if target == SQUARE:
            comp["quatre_coins"] = (nc, float(np.exp(-abs(nc - 4))), nc == 4)
            half = (t.corner_angle_deg[1] - t.corner_angle_deg[0]) / 2
            dev = float(np.abs(an.corner_angles_deg - 90.0).max()) if nc else 90.0
            comp["angles_droits"] = (dev, _note(dev, half), nc > 0 and dev <= half)
            if len(an.side_lengths) >= 2:
                cv = float(an.side_lengths.std() / an.side_lengths.mean())
                st = float(an.straightness.max())
            else:
                cv, st = 1.0, 1.0
            comp["cotes_egaux"] = (cv, _note(cv, t.side_cv), cv <= t.side_cv)
            comp["cotes_droits"] = (st, _note(st, t.straightness), st <= t.straightness)
        elif target == CIRCLE:
            comp["sans_coin"] = (nc, float(np.exp(-nc)), nc == 0)
            comp["rond_regulier"] = (an.radial_cv, _note(an.radial_cv, t.radial_cv), an.radial_cv <= t.radial_cv)
        else:
            raise ValueError(target)
        notes = np.array([c[1] for c in comp.values()])
        score = float(np.exp(np.log(np.maximum(notes, 1e-12)).mean()))  # moyenne géométrique
        accepted = all(c[2] for c in comp.values())
        worst = min(comp, key=lambda k: comp[k][1])
        return Judgment(target=target, accepted=accepted, score=score, components=comp, worst=worst, analysis=an)


class CopyTeacher:
    """Contrôle : récompense la ressemblance pixel avec le modèle montré (donc la copie)."""
    name = "copiste"

    def __init__(self, threshold: float = 0.5, thickness: float = 2.0):
        self.threshold = threshold
        self.thickness = thickness

    @staticmethod
    def _blur(x):
        p = np.pad(x, 1)
        return sum(p[i:i + 32, j:j + 32] for i in range(3) for j in range(3)) / 9

    def judge(self, points: np.ndarray, model_ink: np.ndarray, target: int | None = None) -> Judgment:
        a = self._blur(render([np.asarray(points, float)], self.thickness))
        b = self._blur(np.asarray(model_ink, float))
        dice = float(2 * (a * b).sum() / max((a * a).sum() + (b * b).sum(), 1e-9))
        return Judgment(target=-1 if target is None else target, accepted=dice >= self.threshold, score=dice,
                        components={"ressemblance": (dice, dice, dice >= self.threshold)}, worst="ressemblance")
