"""Shadow Memory : mémoire temporaire des états internes, indépendante de l'activité courante.

Chaque souvenir (`ShadowEntry`) garde un instantané de l'activité au moment d'une expérience :
entrée x, activité cachée h, décision a, probabilités p, récompense r (quand elle arrive), et une
**force** s ∈ [0, 1].

Paramètres (CONCEPTION §3) :
- capacité K (nombre de souvenirs conservés ; le plus faible est oublié en premier) ;
- décroissance γ : à chaque nouvelle expérience, s ← γ·s (oubli passif) ;
- sélection des souvenirs à réactiver : uniforme, ou proportionnelle à la force ;
- renforcement par le rappel (intuition humaine, CR-03) : à chaque rappel, s ← s + α(1 − s).

La mémoire ne fait que garder et restituer. Ce qu'on fait d'un souvenir rappelé (le rejouer à
l'identique, le reconstruire à partir d'un indice partiel, l'utiliser comme gabarit) est décidé par
l'élève (voir `models/shadow_classifiers.py`, `models/shadow_drawers.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ShadowEntry:
    x: np.ndarray
    h: np.ndarray
    a: int | None
    p: np.ndarray | None
    r: float | None = None
    strength: float = 1.0
    recalls: int = 0
    age: int = 0
    extra: dict = field(default_factory=dict)


class ShadowMemory:
    def __init__(self, capacity: int = 64, decay: float = 1.0, selection: str = "uniform",
                 recall_boost: float = 0.0, store_action: bool = True):
        self.capacity, self.decay = capacity, decay
        self.selection, self.recall_boost = selection, recall_boost
        self.store_action = store_action
        self.entries: list[ShadowEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def store(self, entry: ShadowEntry) -> ShadowEntry:
        for e in self.entries:          # oubli passif des souvenirs plus anciens
            e.strength *= self.decay
            e.age += 1
        if not self.store_action:       # Shadow A : la décision n'est pas conservée
            entry.a, entry.p = None, None
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            weakest = min(range(len(self.entries) - 1), key=lambda i: self.entries[i].strength)
            self.entries.pop(weakest)
        return entry

    def sample(self, rng: np.random.Generator) -> ShadowEntry | None:
        pool = [e for e in self.entries if e.r is not None]
        if not pool:
            return None
        if self.selection == "uniform":
            e = pool[int(rng.integers(len(pool)))]
        elif self.selection == "strength":
            w = np.array([e.strength for e in pool]) + 1e-12
            e = pool[int(rng.choice(len(pool), p=w / w.sum()))]
        else:
            raise ValueError(self.selection)
        e.recalls += 1
        if self.recall_boost > 0:       # se rappeler ancre le souvenir
            e.strength += self.recall_boost * (1.0 - e.strength)
        return e

    def strengths(self) -> np.ndarray:
        return np.array([e.strength for e in self.entries])
