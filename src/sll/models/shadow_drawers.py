"""Élèves dessinateurs à Shadow (étape 4b).

Deux Shadows (D12) :
- **Shadow motrice** : les états internes de chacun des 16 pas du geste (entrée, activité cachée,
  bruit propre, action, probabilités) sont gardés jusqu'au jugement, puis la plasticité est appliquée
  à ces états. Le crédit d'un dessin va exactement à ses 16 pas — contrairement à la trace
  d'éligibilité de G2, qui mélange les pas (décroissance λ) et déborde sur l'essai précédent.
- **Shadow perceptive** (gabarit intérieur, D18, D20) : la perception du modèle montré (histogramme de
  16 orientations, système inné) est gardée pendant le dessin. Après le dessin, l'élève **regarde son
  propre dessin** avec le même système visuel et le compare au gabarit (comparaison insensible à la
  rotation : décalage circulaire des orientations). Il ressent aussi, de façon innée, si son trait se
  referme (D35).

Signal interne (sans maître) : s_int = w_g · ressemblance_au_gabarit + w_f · fermeture ∈ [0, 1].
Signal modulateur : δ = (R_maître + β · s_int) − b, b moyenne glissante par forme demandée.

Contrôles (pour ne pas confondre les causes) :
- gabarit **brouillé** : le gabarit vient d'une autre image (même quantité de signal, sans rapport) ;
- le même signal interne donné à l'élève à rétropropagation (G1+interne) : le gain vient-il du
  signal ou de la Shadow ?
- G1 + bonus d'exploration (entropie) : l'effondrement sur le cercle se corrige-t-il simplement en
  forçant l'exploration ?
"""
from __future__ import annotations

import numpy as np

from ..data.encoders import _conv2d_same, orientation_kernels
from ..data.innate_vision import closure_level
from .drawers import Drawer, G1Reinforce
from ..world.student import draw

_K16 = orientation_kernels(16, size=7, sigma=1.2)


def orientation_template(ink: np.ndarray) -> np.ndarray:
    """Histogramme de 16 orientations (système inné, cellules complexes à grand champ)."""
    m = np.stack([np.abs(_conv2d_same(ink[None], k))[0] for k in _K16]).sum((1, 2))
    return m / max(m.sum(), 1e-9)


def template_similarity(a: np.ndarray, b: np.ndarray, k: float = 8.0) -> float:
    d = min(np.abs(np.roll(a, s) - b).sum() for s in range(len(a)))
    return float(np.exp(-k * d))


class InternalSense:
    def __init__(self, w_template: float, w_closure: float):
        self.w_template, self.w_closure = w_template, w_closure

    def __call__(self, gesture, template) -> float:
        if self.w_template == 0 and self.w_closure == 0:
            return 0.0
        ink = gesture.draw(1.5)
        s = 0.0
        if self.w_template > 0:
            s += self.w_template * template_similarity(orientation_template(ink), template)
        if self.w_closure > 0:
            s += self.w_closure * closure_level(ink)
        return s / (self.w_template + self.w_closure)


class G3ShadowMotor(Drawer):
    """Plasticité locale ; crédit appliqué aux états du geste gardés en Shadow motrice."""
    name = "G3-shadow-motrice"

    def __init__(self, d_percept, rng, lr, lr_hidden, sigma, beta: float = 0.0, w_template: float = 0.0,
                 w_closure: float = 0.0, baseline_rate: float = 0.05):
        super().__init__(d_percept, rng)
        self.lr, self.lr_hidden, self.sigma = lr, lr_hidden, sigma
        self.beta = beta
        self.sense = InternalSense(w_template, w_closure)
        self.baseline_rate = baseline_rate
        self.last_internal = 0.0

    def attempt(self, percept, kind, start, step, rng, judge, template=None):
        ep = draw(self.net, percept, start, step, rng, sigma=self.sigma)
        shadow = [(act, a) for act, a in zip(ep.activities, ep.actions)]   # Shadow motrice
        R = judge(ep.gesture.points(), kind)
        s_int = self.sense(ep.gesture, template) if self.beta > 0 else 0.0
        self.last_internal = s_int
        total = R + self.beta * s_int
        self.attempts += 1
        delta = total - self.baseline[kind]
        self.baseline[kind] += self.baseline_rate * (total - self.baseline[kind])
        acc = None
        for act, a in shadow:
            c = self.net.local_traces(act, a, self.sigma)
            acc = c if acc is None else {k: acc[k] + c[k] for k in c}
        self.net.apply(acc, self.lr * delta, layers=("W2", "b2"), max_norm=1.0)
        if self.lr_hidden > 0:
            self.net.apply(acc, self.lr_hidden * delta, layers=("W1", "b1"), max_norm=1.0)
        self.updates += 1
        return R


class G1Plus(G1Reinforce):
    """REINFORCE (rétropropagation) + signal interne et/ou bonus d'exploration."""

    def __init__(self, d_percept, rng, lr, beta: float = 0.0, w_template: float = 0.0, w_closure: float = 0.0,
                 entropy: float = 0.0):
        super().__init__(d_percept, rng, lr)
        self.beta, self.entropy = beta, entropy
        self.sense = InternalSense(w_template, w_closure)

    def attempt(self, percept, kind, start, step, rng, judge, template=None):
        ep = draw(self.net, percept, start, step, rng)
        R = judge(ep.gesture.points(), kind)
        s_int = self.sense(ep.gesture, template) if self.beta > 0 else 0.0
        total = R + self.beta * s_int
        self.attempts += 1
        delta = total - self.baseline[kind]
        self.baseline[kind] += self.baseline_rate * (total - self.baseline[kind])
        grads = None
        for act, a in zip(ep.activities, ep.actions):
            dz = -act.p.copy()
            dz[a] += 1.0
            dz = delta * dz
            if self.entropy > 0:   # gradient de l'entropie de la politique (favorise l'exploration)
                lp = np.log(act.p + 1e-12)
                H = -(act.p * lp).sum()
                dz = dz + self.entropy * (-act.p * (lp + H))
            g = self.net.backward(act, dz)
            grads = g if grads is None else {k: grads[k] + g[k] for k in g}
        self.net.apply(grads, self.lr, max_norm=1.0)
        self.updates += 1
        return R


# Élèves comparés : (classe, paramètres fixes)
STUDENTS = {
    "G2-trace": None,  # référence de l'étape 3 (drawers.G2LocalTrace)
    "G3-shadow-motrice": dict(cls="G3"),
    "G5-gabarit+fermeture": dict(cls="G3", w_template=1.0, w_closure=1.0),
    "G5-gabarit": dict(cls="G3", w_template=1.0, w_closure=0.0),
    "G5-fermeture": dict(cls="G3", w_template=0.0, w_closure=1.0),
    "G5-gabarit-brouillé": dict(cls="G3", w_template=1.0, w_closure=1.0, scramble=True),
    "G1-reinforce": dict(cls="G1"),
    "G1+interne": dict(cls="G1", w_template=1.0, w_closure=1.0),
    "G1+exploration": dict(cls="G1", entropy_search=True),
}

SPACES = {
    "G3": {"lr": ("log", 1e-3, 0.5), "lr_hidden": ("log", 1e-5, 1e-2), "sigma": ("log", 0.02, 0.5)},
    "G1": {"lr": ("log", 1e-3, 0.5)},
    "beta": ("log", 0.1, 3.0),
    "entropy": ("log", 1e-3, 0.3),
}


def _draw_hp(space, rng):
    t, lo, hi = space
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if t == "log" else float(rng.uniform(lo, hi))


def sample_student_hp(name, rng):
    spec = STUDENTS[name]
    hp = {k: _draw_hp(v, rng) for k, v in SPACES[spec["cls"]].items()}
    if spec.get("w_template", 0) > 0 or spec.get("w_closure", 0) > 0:
        hp["beta"] = _draw_hp(SPACES["beta"], rng)
    if spec.get("entropy_search"):
        hp["entropy"] = _draw_hp(SPACES["entropy"], rng)
    return hp


def make_student(name, d_percept, hp, rng):
    spec = STUDENTS[name]
    wt, wc = spec.get("w_template", 0.0), spec.get("w_closure", 0.0)
    if spec["cls"] == "G3":
        return G3ShadowMotor(d_percept, rng, hp["lr"], hp["lr_hidden"], hp["sigma"], beta=hp.get("beta", 0.0),
                             w_template=wt, w_closure=wc)
    return G1Plus(d_percept, rng, hp["lr"], beta=hp.get("beta", 0.0), w_template=wt, w_closure=wc,
                  entropy=hp.get("entropy", 0.0))
