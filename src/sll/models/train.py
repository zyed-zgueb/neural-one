"""Boucle d'interaction élève–environnement pour la classification (identique pour tous les modèles)."""
from __future__ import annotations

import time

import numpy as np

from .classifiers import make_learner


def run_classification(kind: str, hp: dict, Xtr, ytr, Xte, yte, passes: int,
                       order_rng: np.random.Generator, init_rng: np.random.Generator,
                       reward: str = "normal") -> dict:
    """Entraîne un élève en ligne (`passes` passages sur les exemples) puis l'évalue.

    reward = "normal" : r = +1 si correct, −1 sinon ;
             "aleatoire" : r tiré au hasard (±1), étiquettes aléatoires pour les modèles supervisés
                           (contrôle négatif : on ne doit rien apprendre).
    """
    t0 = time.perf_counter()
    learner = make_learner(kind, Xtr.shape[1], hp, init_rng)
    n = len(ytr)
    for _ in range(passes):
        for i in order_rng.permutation(n):
            a, info = learner.act(Xtr[i], order_rng)
            if reward == "normal":
                r, y = (1.0 if a == ytr[i] else -1.0), int(ytr[i])
            else:
                r, y = float(order_rng.choice([-1.0, 1.0])), int(order_rng.integers(0, 2))
            learner.feedback(info, r, y)
    pred = learner.predict(Xte)
    tp = int(((pred == 1) & (yte == 1)).sum())
    tn = int(((pred == 0) & (yte == 0)).sum())
    fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    return {
        "acc": (tp + tn) / len(yte),
        "train_acc": float((learner.predict(Xtr) == ytr).mean()),
        "confusion": [[tn, fp], [fn, tp]],   # lignes : vrai cercle, vrai carré
        "recall_square": tp / max(tp + fn, 1),
        "recall_circle": tn / max(tn + fp, 1),
        "presentations": learner.counters.presentations,
        "updates": learner.counters.updates,
        "seconds": time.perf_counter() - t0,
    }


def run_memory_classification(kind: str, hp: dict, R: int, Xtr, ytr, Xte, yte,
                              order_rng: np.random.Generator, init_rng: np.random.Generator) -> dict:
    """Un seul passage sur les exemples ; R réactivations de souvenirs après chaque expérience."""
    from .shadow_classifiers import make_memory_learner
    t0 = time.perf_counter()
    learner = make_memory_learner(kind, Xtr.shape[1], hp, init_rng, R)
    for i in order_rng.permutation(len(ytr)):
        a, info = learner.act(Xtr[i], order_rng)
        learner.feedback(info, 1.0 if a == ytr[i] else -1.0, int(ytr[i]))
    pred = learner.predict(Xte)
    return {"acc": float((pred == yte).mean()), "train_acc": float((learner.predict(Xtr) == ytr).mean()),
            "presentations": learner.counters.presentations, "updates": learner.counters.updates,
            "seconds": time.perf_counter() - t0}
