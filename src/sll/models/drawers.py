"""Élèves dessinateurs de référence (sans Shadow) — paradigme maître–élève.

| Élève | Ce qu'il reçoit | Apprentissage |
|-------|-----------------|---------------|
| G0 imitation | le **geste** du maître (information que les autres n'ont pas) | rétropropagation, entropie croisée pas à pas |
| G1 REINFORCE | le jugement du maître sur son dessin | gradient de politique sur les 16 pas, rétropropagation |
| G2 local + trace | le jugement du maître | règle locale à trois facteurs ; la trace d'éligibilité accumule les 16 pas ; couche cachée par perturbation de nœud |

Jugement : « pouce » (+1 accepté / −1 refusé) ou « presque » (note graduée ∈ [0, 1]).
Le signal modulateur est l'erreur de prédiction δ = R − b, b moyenne glissante par forme demandée.
"""
from __future__ import annotations

import numpy as np

from ..core.network import MLP
from ..world.pen import perfect_circle, perfect_square
from ..world.student import CONTEXT_DIM, N_ACTIONS, draw


class Drawer:
    name = "base"

    def __init__(self, d_percept: int, rng: np.random.Generator, hidden: int = 64):
        self.net = MLP(d_percept + CONTEXT_DIM, hidden, N_ACTIONS, rng)
        self.attempts = 0
        self.updates = 0
        self.baseline = {0: 0.0, 1: 0.0}

    def greedy(self, percept, start, step):
        return draw(self.net, percept, start, step, None, greedy=True)


def teacher_gesture(kind: int, rng: np.random.Generator):
    """Le geste « canonique » du maître pour une forme (sens de rotation au hasard)."""
    d = int(rng.choice([-1, 1]))
    return (perfect_square((16, 16), 12, direction=d) if kind == 1 else perfect_circle((16, 16), 14, direction=d))


class G0Imitation(Drawer):
    name = "G0-imitation"

    def __init__(self, d_percept, rng, lr: float):
        super().__init__(d_percept, rng)
        self.lr = lr

    def learn_from_demo(self, percept, kind, start, step, rng):
        g = teacher_gesture(kind, rng)
        ep = draw(self.net, percept, start, step, rng, forced_actions=g.actions)
        for act, a in zip(ep.activities, ep.actions):
            dz = act.p.copy()
            dz[a] -= 1.0
            self.net.apply(self.net.backward(act, dz), -self.lr)
            self.updates += 1


class G1Reinforce(Drawer):
    name = "G1-reinforce"

    def __init__(self, d_percept, rng, lr: float, baseline_rate: float = 0.05):
        super().__init__(d_percept, rng)
        self.lr, self.baseline_rate = lr, baseline_rate

    def attempt(self, percept, kind, start, step, rng, judge):
        ep = draw(self.net, percept, start, step, rng)
        R = judge(ep.gesture.points(), kind)
        self.attempts += 1
        delta = R - self.baseline[kind]
        self.baseline[kind] += self.baseline_rate * (R - self.baseline[kind])
        grads = None
        for act, a in zip(ep.activities, ep.actions):
            dz = -act.p.copy()
            dz[a] += 1.0
            g = self.net.backward(act, dz)
            grads = g if grads is None else {k: grads[k] + g[k] for k in g}
        self.net.apply(grads, self.lr * delta, max_norm=1.0)
        self.updates += 1
        return R


class G2LocalTrace(Drawer):
    name = "G2-trace"

    def __init__(self, d_percept, rng, lr: float, lr_hidden: float, sigma: float, lam: float,
                 baseline_rate: float = 0.05):
        super().__init__(d_percept, rng)
        self.lr, self.lr_hidden, self.sigma, self.lam = lr, lr_hidden, sigma, lam
        self.baseline_rate = baseline_rate
        self.trace = {k: np.zeros_like(v) for k, v in self.net.params().items()}

    def attempt(self, percept, kind, start, step, rng, judge):
        ep = draw(self.net, percept, start, step, rng, sigma=self.sigma)
        # la trace s'accumule pendant le geste, pas à pas, avec décroissance λ (elle n'est pas remise à zéro)
        for act, a in zip(ep.activities, ep.actions):
            c = self.net.local_traces(act, a, self.sigma)
            for k in self.trace:
                self.trace[k] = self.lam * self.trace[k] + c[k]
        R = judge(ep.gesture.points(), kind)
        self.attempts += 1
        delta = R - self.baseline[kind]
        self.baseline[kind] += self.baseline_rate * (R - self.baseline[kind])
        self.net.apply(self.trace, self.lr * delta, layers=("W2", "b2"), max_norm=1.0)
        if self.lr_hidden > 0:
            self.net.apply(self.trace, self.lr_hidden * delta, layers=("W1", "b1"), max_norm=1.0)
        self.updates += 1
        return R


DRAWER_SPACES = {
    "G0-imitation": {"lr": ("log", 1e-3, 0.5)},
    "G1-reinforce": {"lr": ("log", 1e-3, 0.5)},
    "G2-trace": {"lr": ("log", 1e-3, 0.5), "lr_hidden": ("log", 1e-5, 1e-2), "sigma": ("log", 0.02, 0.5),
                 "lam": ("uni", 0.5, 1.0)},
}


def sample_drawer_hp(kind, rng):
    hp = {}
    for k, (t, lo, hi) in DRAWER_SPACES[kind].items():
        hp[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if t == "log" else float(rng.uniform(lo, hi))
    return hp


def make_drawer(kind, d_percept, hp, rng):
    if kind == "G0-imitation":
        return G0Imitation(d_percept, rng, hp["lr"])
    if kind == "G1-reinforce":
        return G1Reinforce(d_percept, rng, hp["lr"])
    if kind == "G2-trace":
        return G2LocalTrace(d_percept, rng, hp["lr"], hp["lr_hidden"], hp["sigma"], hp["lam"])
    raise ValueError(kind)
