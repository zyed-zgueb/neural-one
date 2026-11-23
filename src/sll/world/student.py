"""L'élève dessinateur : perception du modèle -> geste au stylo, pas à pas.

À chaque pas t, l'entrée du réseau est :
    [ perception du modèle montré (fixe pendant le dessin)  |  contexte moteur du pas t ]
Contexte moteur (« proprioception », D20) : pas courant (one-hot 16), virage précédent (one-hot 7),
cap cumulé depuis le départ (cos, sin).

Réflexe postural fourni (documenté, non appris) : le stylo part du coin haut-gauche de la boîte
englobante du modèle, cap horizontal, avec un pas égal à l'étendue / 4. Le maître catégoriel ne
jugeant que la forme, ce réflexe ne donne aucune information sur la catégorie ; il évite seulement
de devoir apprendre « où poser le stylo ». Il sera retiré quand on testera la taille (test de
l'élève qui sait).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.network import MLP, Activity
from .pen import T_STEPS, TURNS, Gesture

N_ACTIONS = len(TURNS)
CONTEXT_DIM = T_STEPS + N_ACTIONS + 2


def posture(ink: np.ndarray) -> tuple[np.ndarray, float]:
    """Point de départ et longueur de pas déduits de l'étendue de l'encre du modèle."""
    ys, xs = np.where(ink > 0.3)
    if len(xs) == 0:
        return np.array([10.0, 10.0]), 3.0
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    extent = max(x1 - x0, y1 - y0)
    return np.array([x0 + 0.5, y0 + 0.5], float), max(extent / 4.0, 1.5)


def context(t: int, prev_action: int | None, heading: float) -> np.ndarray:
    c = np.zeros(CONTEXT_DIM)
    c[t] = 1.0
    if prev_action is not None:
        c[T_STEPS + prev_action] = 1.0
    c[T_STEPS + N_ACTIONS] = np.cos(heading)
    c[T_STEPS + N_ACTIONS + 1] = np.sin(heading)
    return c


@dataclass
class Episode:
    gesture: Gesture
    activities: list[Activity]
    actions: np.ndarray
    inputs: np.ndarray


def draw(net: MLP, percept: np.ndarray, start: np.ndarray, step: float, rng: np.random.Generator | None,
         greedy: bool = False, sigma: float = 0.0, forced_actions: np.ndarray | None = None) -> Episode:
    """Fait dessiner l'élève. `forced_actions` : imposer les actions (imitation du geste du maître)."""
    acts, actions, inputs = [], [], []
    heading, prev = 0.0, None
    for t in range(T_STEPS):
        x = np.concatenate([percept, context(t, prev, heading)])
        xi = rng.normal(0, sigma, net.d_hidden) if (sigma > 0 and rng is not None) else None
        act = net.forward(x, xi)
        if forced_actions is not None:
            a = int(forced_actions[t])
        elif greedy:
            a = int(act.p.argmax())
        else:
            a = int(rng.choice(N_ACTIONS, p=act.p))
        acts.append(act)
        actions.append(a)
        inputs.append(x)
        heading += TURNS[a]
        prev = a
    g = Gesture(start=np.asarray(start, float), heading=0.0, step=step, actions=np.array(actions))
    return Episode(gesture=g, activities=acts, actions=np.array(actions), inputs=np.array(inputs))
