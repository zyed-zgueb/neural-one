"""Élèves à mémoire (reconnaissance) : replay et variantes Shadow, à nombre de mises à jour égal.

Tous partent de la règle locale de lecture (M2-lecture, la meilleure règle locale en 1 passage,
CR-07) : Δw2 = η · δ · (onehot(a) − p) ⊗ h. Après chaque expérience réelle, l'élève fait R
**réactivations** de souvenirs ; chaque réactivation est une mise à jour. Les modèles ne diffèrent
que par ce qu'ils gardent et la façon dont ils se rappellent :

| Élève | Souvenir rappelé | Activité utilisée pour la plasticité |
|-------|------------------|--------------------------------------|
| Replay | entrée x + décision + probabilités + récompense, choix uniforme | x rejouée telle quelle avec les poids actuels, pondérée par min(1, p_actuel(a) / p_ancien(a)) |
| Shadow A | activités (x, h) + récompense, **sans la décision** | décision reconstruite â = argmax actuel (pas de pondération possible : la décision n'a pas été gardée) |
| Shadow B | activités + décision + récompense ; force qui décroît (γ), choix ∝ force, intensité ∝ force | x rejouée telle quelle |
| Shadow C | comme B + **rappel reconstructif** : indice partiel et bruité, complété par le réseau ; le rappel **renforce** le souvenir | activité recalculée à partir de l'indice, mélangée à l'activité stockée (β) |

Vérifications (tests) : B avec R = 0 ≡ M2-lecture ; C avec indice complet, sans bruit, β = 0,
sans renforcement, choix uniforme et γ = 1 ≡ Replay.

Une variante `M0+replay` donne exactement le même mécanisme de replay à la rétropropagation :
sans elle, un avantage de la mémoire ne pourrait pas être attribué à la plasticité locale.
"""
from __future__ import annotations

import numpy as np

from ..core.network import MLP, softmax_xent_grad
from ..learning.shadow_memory import ShadowEntry, ShadowMemory
from .classifiers import NetLearner


class MemoryLearner(NetLearner):
    name = "memoire"

    def __init__(self, net: MLP, lr: float, R: int, variant: str = "replay", decay: float = 1.0,
                 cue_keep: float = 1.0, cue_noise: float = 0.0, blend: float = 0.0, recall_boost: float = 0.0,
                 capacity: int = 1000, baseline_rate: float = 0.1, rule: str = "local",
                 importance: bool = True):
        super().__init__(net, lr)
        self.importance = importance
        self.R, self.variant = R, variant
        self.cue_keep, self.cue_noise, self.blend = cue_keep, cue_noise, blend
        self.rule = rule
        selection = "strength" if variant in ("B", "C") else "uniform"
        self.memory = ShadowMemory(capacity=capacity, decay=decay if variant in ("B", "C") else 1.0,
                                   selection=selection, recall_boost=recall_boost if variant == "C" else 0.0,
                                   store_action=(variant != "A"))
        self.baseline, self.baseline_rate = 0.0, baseline_rate
        self.name = {"replay": "Replay", "A": "Shadow-A", "B": "Shadow-B", "C": "Shadow-C"}[variant]
        if rule == "backprop":
            self.name = "M0+replay"
        self.uses_label = rule == "backprop"
        self._last = None

    # ----------------------------------------------------------- expérience réelle
    def act(self, x, rng):
        act = self.net.forward(x)
        a = int(rng.choice(2, p=act.p)) if self.rule == "local" else int(act.p.argmax())
        self.counters.presentations += 1
        self._rng = rng
        return a, (act, a)

    def _update(self, act, a, r, intensity=1.0, h=None, y=None):
        if self.rule == "backprop":
            g = self.net.backward(act, softmax_xent_grad(act.p, y))
            self.counters.weight_change += self.net.apply(g, -self.lr * intensity)
        else:
            delta = r - self.baseline
            post = -act.p.copy()
            post[a] += 1.0
            hh = act.h if h is None else h
            g = {"W2": np.outer(post, hh), "b2": post}
            self.counters.weight_change += self.net.apply(g, self.lr * intensity * delta, layers=("W2", "b2"))
        self.counters.updates += 1

    def feedback(self, info, r, y):
        act, a = info
        self._update(act, a, r, y=y)
        if self.rule == "local":
            self.baseline += self.baseline_rate * (r - self.baseline)
        self.memory.store(ShadowEntry(x=act.x.copy(), h=act.h.copy(), a=a, p=act.p.copy(), r=r,
                                      extra={"y": y}))
        for _ in range(self.R):
            self.reactivate(self._rng)

    # ----------------------------------------------------------- rappel
    def reactivate(self, rng):
        e = self.memory.sample(rng)
        if e is None:
            return
        x = e.x
        if self.variant == "C" and (self.cue_keep < 1.0 or self.cue_noise > 0):
            mask = rng.random(x.shape) < self.cue_keep
            x = np.where(mask, x, 0.0) + rng.normal(0, self.cue_noise, x.shape)
        act = self.net.forward(x)
        h = act.h
        if self.variant == "C" and self.blend > 0:
            h = self.blend * e.h + (1 - self.blend) * act.h
        a = e.a if e.a is not None else int(act.p.argmax())   # Shadow A : décision reconstruite
        r = e.r
        intensity = e.strength if self.variant in ("B", "C") else 1.0
        if self.rule == "local" and e.p is not None and self.importance:
            # Correction hors-politique standard (pondération d'importance tronquée) : un ancien choix
            # que l'élève ne referait plus aujourd'hui compte moins. Sans elle, rejouer sans fin une
            # vieille erreur pousse les poids sans limite (constaté : le replay dégradait l'élève).
            intensity *= min(1.0, act.p[a] / max(e.p[a], 1e-6))
        self._update(act, a, r, intensity=intensity, h=h, y=e.extra.get("y"))


MEMORY_SPACES = {
    "Replay": {"lr": ("log", 1e-3, 1.0)},
    "M0+replay": {"lr": ("log", 1e-3, 1.0)},
    "Shadow-A": {"lr": ("log", 1e-3, 1.0)},
    "Shadow-B": {"lr": ("log", 1e-3, 1.0), "decay": ("uni", 0.8, 1.0)},
    "Shadow-C": {"lr": ("log", 1e-3, 1.0), "decay": ("uni", 0.8, 1.0), "cue_keep": ("uni", 0.5, 1.0),
                 "cue_noise": ("log", 0.01, 1.0), "blend": ("uni", 0.0, 0.8), "recall_boost": ("uni", 0.0, 0.5)},
}


def make_memory_learner(kind: str, d_in: int, hp: dict, init_rng, R: int, hidden: int = 64):
    net = MLP(d_in, hidden, 2, init_rng)
    if kind == "Replay":
        return MemoryLearner(net, hp["lr"], R, "replay")
    if kind == "M0+replay":
        return MemoryLearner(net, hp["lr"], R, "replay", rule="backprop")
    if kind == "Shadow-A":
        return MemoryLearner(net, hp["lr"], R, "A")
    if kind == "Shadow-B":
        return MemoryLearner(net, hp["lr"], R, "B", decay=hp["decay"])
    if kind == "Shadow-C":
        return MemoryLearner(net, hp["lr"], R, "C", decay=hp["decay"], cue_keep=hp["cue_keep"],
                             cue_noise=hp["cue_noise"], blend=hp["blend"], recall_boost=hp["recall_boost"])
    raise ValueError(kind)


def sample_memory_hp(kind, rng):
    hp = {}
    for k, (t, lo, hi) in MEMORY_SPACES[kind].items():
        hp[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if t == "log" else float(rng.uniform(lo, hi))
    return hp
