"""Élèves de référence pour la tâche de classification (secondaire, D14).

Tous partagent le réseau `MLP` et la même initialisation ; seule la règle d'apprentissage change.

| Modèle | Signal reçu | Mise à jour |
|--------|-------------|-------------|
| M0  | étiquette y | rétropropagation, entropie croisée |
| M1  | récompense r = ±1 | REINFORCE (gradient de politique) avec rétropropagation |
| M2  | récompense r = ±1 | règle locale à trois facteurs + trace d'éligibilité (λ) ; couche cachée par perturbation de nœud |
| M2-lecture | idem | idem, couche cachée figée (seule la lecture apprend) |
| 1-PPV, prototype | étiquette | mémoire des exemples (référence « mémorisation pure ») |

Protocole d'interaction (identique pour tous) : `a, info = act(x)` puis `feedback(info, r, y)`.
Les modèles à récompense n'utilisent jamais `y`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.network import MLP, softmax_xent_grad


@dataclass
class Counters:
    presentations: int = 0
    updates: int = 0
    weight_change: float = 0.0


class Learner:
    name = "base"
    uses_label = False

    def __init__(self):
        self.counters = Counters()

    def act(self, x: np.ndarray, rng: np.random.Generator):
        raise NotImplementedError

    def feedback(self, info, r: float, y: int):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NetLearner(Learner):
    def __init__(self, net: MLP, lr: float):
        super().__init__()
        self.net, self.lr = net, lr

    def predict(self, X):
        return self.net.predict(X)


class M0Backprop(NetLearner):
    """Référence haute : apprentissage supervisé par rétropropagation."""
    name, uses_label = "M0-backprop", True

    def act(self, x, rng):
        act = self.net.forward(x)
        self.counters.presentations += 1
        return int(act.p.argmax()), act

    def feedback(self, act, r, y):
        g = self.net.backward(act, softmax_xent_grad(act.p, y))
        self.counters.weight_change += self.net.apply(g, -self.lr)
        self.counters.updates += 1


class M1Reinforce(NetLearner):
    """Récompense globale, gradient de politique exact (rétropropagation), sans trace."""
    name = "M1-reinforce"

    def __init__(self, net, lr, baseline_rate: float = 0.1):
        super().__init__(net, lr)
        self.baseline, self.baseline_rate = 0.0, baseline_rate

    def act(self, x, rng):
        act = self.net.forward(x)
        a = int(rng.choice(len(act.p), p=act.p))
        self.counters.presentations += 1
        return a, (act, a)

    def feedback(self, info, r, y):
        act, a = info
        delta = r - self.baseline          # erreur de prédiction de récompense
        self.baseline += self.baseline_rate * (r - self.baseline)
        dz = -act.p.copy()
        dz[a] += 1.0
        g = self.net.backward(act, dz)
        self.counters.weight_change += self.net.apply(g, self.lr * delta)
        self.counters.updates += 1


class M2Local(NetLearner):
    """Plasticité locale modulée par la récompense, avec trace d'éligibilité classique.

    e ← λ·e + c(t)  (c : produits locaux, voir MLP.local_traces)
    Δw = η · δ · e  avec δ = r − b (b : moyenne glissante de r)
    """
    name = "M2-local"

    def __init__(self, net, lr, lr_hidden: float = 0.0, sigma: float = 0.0, lam: float = 0.0,
                 baseline_rate: float = 0.1, readout_only: bool = False):
        super().__init__(net, lr)
        self.lr_hidden, self.sigma, self.lam = lr_hidden, sigma, lam
        self.readout_only = readout_only
        if readout_only:
            self.name = "M2-lecture"
        self.baseline, self.baseline_rate = 0.0, baseline_rate
        self.trace = {k: np.zeros_like(v) for k, v in net.params().items()}

    def act(self, x, rng):
        xi = None
        if not self.readout_only and self.sigma > 0:
            xi = rng.normal(0, self.sigma, self.net.d_hidden)
        act = self.net.forward(x, xi)
        a = int(rng.choice(len(act.p), p=act.p))
        c = self.net.local_traces(act, a, 0.0 if self.readout_only else self.sigma)
        for k in self.trace:
            self.trace[k] = self.lam * self.trace[k] + c[k]
        self.counters.presentations += 1
        return a, None

    def feedback(self, info, r, y):
        delta = r - self.baseline
        self.baseline += self.baseline_rate * (r - self.baseline)
        dw = self.net.apply(self.trace, self.lr * delta, layers=("W2", "b2"))
        if not self.readout_only:
            dw += self.net.apply(self.trace, self.lr_hidden * delta, layers=("W1", "b1"), max_norm=1.0)
        self.counters.weight_change += dw
        self.counters.updates += 1


class NearestNeighbor(Learner):
    """1 plus proche voisin : mémorise les exemples, ne généralise que par ressemblance."""
    name, uses_label = "1-PPV", True

    def __init__(self):
        super().__init__()
        self.X, self.y = [], []

    def act(self, x, rng):
        self.counters.presentations += 1
        return (int(self.predict(x[None])[0]) if self.X else 0), x

    def feedback(self, x, r, y):
        self.X.append(x)
        self.y.append(y)
        self.counters.updates += 1

    def predict(self, X):
        if not self.X:
            return np.zeros(len(X), int)
        M = np.array(self.X)
        d = (X ** 2).sum(1)[:, None] + (M ** 2).sum(1)[None] - 2 * X @ M.T
        return np.array(self.y)[d.argmin(1)]


class Prototype(NearestNeighbor):
    """Prototype : moyenne des exemples de chaque classe (abstraction minimale)."""
    name = "prototype"

    def predict(self, X):
        if not self.X:
            return np.zeros(len(X), int)
        M, y = np.array(self.X), np.array(self.y)
        classes = np.unique(y)
        P = np.stack([M[y == c].mean(0) for c in classes])
        d = (X ** 2).sum(1)[:, None] + (P ** 2).sum(1)[None] - 2 * X @ P.T
        return classes[d.argmin(1)]


def make_learner(kind: str, d_in: int, hp: dict, init_rng: np.random.Generator, hidden: int = 64) -> Learner:
    if kind == "1-PPV":
        return NearestNeighbor()
    if kind == "prototype":
        return Prototype()
    net = MLP(d_in, hidden, 2, init_rng)
    if kind == "M0-backprop":
        return M0Backprop(net, hp["lr"])
    if kind == "M1-reinforce":
        return M1Reinforce(net, hp["lr"])
    if kind == "M2-local":
        return M2Local(net, hp["lr"], lr_hidden=hp["lr_hidden"], sigma=hp["sigma"], lam=hp["lam"])
    if kind == "M2-lecture":
        return M2Local(net, hp["lr"], lam=hp["lam"], readout_only=True)
    if kind == "non-entraîné":
        return M0Backprop(net, 0.0)
    raise ValueError(kind)


# Espaces de recherche d'hyperparamètres : même budget de tirages pour chaque modèle.
SEARCH_SPACES = {
    "M0-backprop": {"lr": ("log", 1e-3, 1.0)},
    "M1-reinforce": {"lr": ("log", 1e-3, 1.0)},
    "M2-local": {"lr": ("log", 1e-3, 1.0), "lr_hidden": ("log", 1e-5, 1e-1), "sigma": ("log", 0.01, 1.0),
                 "lam": ("uni", 0.0, 0.9)},
    "M2-lecture": {"lr": ("log", 1e-3, 1.0), "lam": ("uni", 0.0, 0.9)},
}


def sample_hp(kind: str, rng: np.random.Generator) -> dict:
    hp = {}
    for k, (t, lo, hi) in SEARCH_SPACES.get(kind, {}).items():
        hp[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if t == "log" else float(rng.uniform(lo, hi))
    return hp
