"""Jeux de données : tirages disjoints, épisodes few-shot, contrôle de fuite.

Garanties :
- chaque rôle (« train », « val », « test ») a son propre flux aléatoire (sll.core.rng) ;
- classes équilibrées, ordre mélangé de façon déterministe ;
- `leakage_report` mesure la distance de chaque image test à l'image d'apprentissage la plus proche.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.rng import make_rng
from .levels import Level, get_level
from .shapes_generator import CIRCLE, SQUARE, Shape, generate


@dataclass
class ShapeSet:
    shapes: list[Shape]
    level: str
    role: str

    @property
    def X(self) -> np.ndarray:
        return np.stack([s.ink for s in self.shapes])

    @property
    def y(self) -> np.ndarray:
        return np.array([s.params.kind for s in self.shapes])

    def __len__(self) -> int:
        return len(self.shapes)


def make_set(level: str | Level, n_per_class: int, master_seed: int, role: str, *extra_roles) -> ShapeSet:
    lv = get_level(level) if isinstance(level, str) else level
    rng = make_rng(master_seed, "shapes", lv.name, role, *extra_roles)
    shapes = []
    for _ in range(n_per_class):
        shapes.append(generate(lv, CIRCLE, rng))
        shapes.append(generate(lv, SQUARE, rng))
    order = rng.permutation(len(shapes))
    return ShapeSet([shapes[i] for i in order], lv.name, role)


def few_shot_episode(level: str, n_per_class: int, master_seed: int, repetition: int,
                     test_level: str | None = None, n_test_per_class: int = 1000) -> tuple[ShapeSet, ShapeSet]:
    """Un épisode : n exemples d'apprentissage par classe + un jeu de test indépendant.

    Le jeu de test dépend seulement de (graine, niveau de test), pas de la répétition : tous les
    modèles et toutes les répétitions sont évalués sur les mêmes images, ce qui permet des
    comparaisons appariées.
    """
    train = make_set(level, n_per_class, master_seed, "train", f"rep{repetition}")
    test = make_set(test_level or level, n_test_per_class, master_seed, "test")
    return train, test


def leakage_report(train: ShapeSet, test: ShapeSet, near_dup_threshold: float = 1.0) -> dict:
    """Distance L2 (en pixels d'encre) de chaque image test à la plus proche image d'apprentissage."""
    Xtr = train.X.reshape(len(train), -1)
    Xte = test.X.reshape(len(test), -1)
    d2 = (Xte ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None] - 2 * Xte @ Xtr.T
    dmin = np.sqrt(np.maximum(d2.min(1), 0))
    # référence : distance typique entre deux images de test différentes
    k = min(len(test), 500)
    Xs = Xte[:k]
    d2s = (Xs ** 2).sum(1)[:, None] + (Xs ** 2).sum(1)[None] - 2 * Xs @ Xs.T
    np.fill_diagonal(d2s, np.inf)
    ref = np.sqrt(np.maximum(d2s.min(1), 0))
    return {
        "min_distance": float(dmin.min()),
        "median_distance": float(np.median(dmin)),
        "near_duplicates": int((dmin < near_dup_threshold).sum()),
        "reference_median_nn_distance_within_test": float(np.median(ref)),
        "distances": dmin,
    }
