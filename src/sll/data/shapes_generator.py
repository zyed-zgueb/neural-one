"""Générateur de cercles et de carrés 32×32.

Chaque image est entièrement décrite par un `ShapeParams` (traçabilité : on garde les paramètres,
pas seulement les pixels). Le tracé vectoriel (polylignes) est aussi conservé : le maître juge
les formes sur leur tracé, exactement comme il jugera les dessins de l'élève.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .levels import Level
from .render import IMG_SIZE, render

CIRCLE, SQUARE = 0, 1
CLASS_NAMES = {CIRCLE: "cercle", SQUARE: "carré"}


@dataclass
class ShapeParams:
    kind: int
    cx: float
    cy: float
    perimeter: float
    thickness: float
    rotation: float            # radians (carré) ; orientation de l'ellipse (cercle)
    ellipticity: float
    radial_jitter: float
    vertex_jitter: float
    corner_rounding: float
    noise_sigma: float
    fragment: float
    shape_seed: int            # graine des petites irrégularités (jitter, fragments, bruit)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Shape:
    params: ShapeParams
    polylines: list[np.ndarray]   # tracé effectivement rendu (après fragmentation)
    outline: np.ndarray           # contour complet fermé (avant fragmentation), pour le maître
    ink: np.ndarray               # image 32×32, 1 = encre


# ---------------------------------------------------------------- contours

def _circle_outline(p: ShapeParams, rng: np.random.Generator, n: int = 96) -> np.ndarray:
    r = p.perimeter / (2 * np.pi)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = np.ones(n)
    if p.radial_jitter > 0:
        for k in (2, 3, 4, 5):
            rad += p.radial_jitter / 2 * rng.normal() * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    x = r * rad * (1 + p.ellipticity) * np.cos(th)
    y = r * rad * (1 - p.ellipticity) * np.sin(th)
    c, s = np.cos(p.rotation), np.sin(p.rotation)
    pts = np.stack([c * x - s * y, s * x + c * y], 1)
    return pts


def _square_outline(p: ShapeParams, rng: np.random.Generator) -> np.ndarray:
    side = p.perimeter / 4
    h = side / 2
    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=float)
    if p.vertex_jitter > 0:
        corners += rng.normal(0, p.vertex_jitter * side, corners.shape)
    pts = []
    rho = p.corner_rounding * side
    for i in range(4):
        prev, cur, nxt = corners[i - 1], corners[i], corners[(i + 1) % 4]
        if rho <= 1e-6:
            pts.append(cur)
            continue
        u1 = (prev - cur) / np.linalg.norm(prev - cur)
        u2 = (nxt - cur) / np.linalg.norm(nxt - cur)
        a, b = cur + rho * u1, cur + rho * u2
        # courbe de Bézier quadratique a -> cur -> b (coin arrondi)
        for t in np.linspace(0, 1, 7):
            pts.append((1 - t) ** 2 * a + 2 * (1 - t) * t * cur + t ** 2 * b)
    pts = np.array(pts)
    c, s = np.cos(p.rotation), np.sin(p.rotation)
    return np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], 1)


def _resample_closed(pts: np.ndarray, step: float) -> np.ndarray:
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(s[-1] / step), 8)
    q = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(q, s, closed[:, 0]), np.interp(q, s, closed[:, 1])], 1)


def _fragment(outline: np.ndarray, frac: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Efface 1 à 3 arcs du contour, totalisant `frac` du périmètre."""
    dense = _resample_closed(outline, 0.25)
    n = len(dense)
    keep = np.ones(n, bool)
    n_gaps = int(rng.integers(1, 4))
    for _ in range(n_gaps):
        L = int(round(frac / n_gaps * n))
        start = int(rng.integers(0, n))
        keep[(start + np.arange(L)) % n] = False
    # découpe en polylignes contiguës (circulaire)
    if keep.all():
        return [np.vstack([dense, dense[:1]])]
    first_gap = int(np.argmin(keep))
    order = (first_gap + np.arange(n)) % n
    pls, cur = [], []
    for i in order:
        if keep[i]:
            cur.append(dense[i])
        elif cur:
            pls.append(np.array(cur))
            cur = []
    if cur:
        pls.append(np.array(cur))
    return [pl for pl in pls if len(pl) >= 2]


def outline_extent(kind: int, perimeter: float, rotation: float, ellipticity: float = 0.0) -> float:
    """Demi-étendue maximale (x ou y) d'une forme centrée en 0, sans jitter."""
    if kind == CIRCLE:
        return perimeter / (2 * np.pi) * (1 + ellipticity)
    h = perimeter / 8
    return h * (abs(np.cos(rotation)) + abs(np.sin(rotation)))


# ---------------------------------------------------------------- tirage

def sample_params(level: Level, kind: int, rng: np.random.Generator) -> ShapeParams:
    P = rng.uniform(*level.perimeter)
    t = rng.uniform(*level.thickness)
    # angle tiré pour les deux classes (même consommation du flux aléatoire)
    angle = rng.uniform(0, np.pi / 2)
    ell = rng.uniform(*level.ellipticity)
    if kind == SQUARE:
        rotation = angle if level.rotate_square else 0.0
        ell_eff = 0.0
    else:
        rotation = angle * 4  # orientation de l'ellipse dans [0, 2π)
        ell_eff = ell
    rounding = rng.uniform(*level.corner_rounding) if kind == SQUARE else 0.0
    noise = rng.uniform(*level.noise_sigma)
    frag = rng.uniform(*level.fragment)
    # marge de sécurité pour les irrégularités
    slack = 1.0 + 2 * max(level.vertex_jitter, level.radial_jitter)
    # Zone de position indépendante de la classe : on prend l'étendue maximale possible parmi les
    # deux formes pour ce périmètre. Sinon les petites formes (carrés droits) pourraient aller plus
    # près des bords, et la position deviendrait un indice de la classe (fuite détectée par la sonde).
    ext_c = outline_extent(CIRCLE, P, 0.0, level.ellipticity[1])
    ext_s = outline_extent(SQUARE, P, np.pi / 4 if level.rotate_square else 0.0)
    ext = max(ext_c, ext_s) * slack + t / 2 + level.margin
    if level.centered:
        cx = cy = IMG_SIZE / 2
    else:
        lo, hi = ext, IMG_SIZE - ext
        if hi <= lo:
            raise ValueError(f"Forme trop grande pour l'image (P={P:.1f})")
        cx, cy = rng.uniform(lo, hi, size=2)
    return ShapeParams(kind=kind, cx=float(cx), cy=float(cy), perimeter=float(P), thickness=float(t),
                       rotation=float(rotation), ellipticity=float(ell_eff),
                       radial_jitter=level.radial_jitter if kind == CIRCLE else 0.0,
                       vertex_jitter=level.vertex_jitter if kind == SQUARE else 0.0,
                       corner_rounding=float(rounding), noise_sigma=float(noise), fragment=float(frag),
                       shape_seed=int(rng.integers(0, 2**31 - 1)))


def build_shape(p: ShapeParams) -> Shape:
    rng = np.random.default_rng(p.shape_seed)
    outline = _circle_outline(p, rng) if p.kind == CIRCLE else _square_outline(p, rng)
    outline = outline + np.array([p.cx, p.cy])
    if p.fragment > 0:
        polylines = _fragment(outline, p.fragment, rng)
    else:
        polylines = [np.vstack([outline, outline[:1]])]
    ink = render(polylines, p.thickness)
    if p.noise_sigma > 0:
        ink = np.clip(ink + rng.normal(0, p.noise_sigma, ink.shape), 0.0, 1.0)
    return Shape(params=p, polylines=polylines, outline=outline, ink=ink)


def generate(level: Level, kind: int, rng: np.random.Generator) -> Shape:
    return build_shape(sample_params(level, kind, rng))
