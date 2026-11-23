"""Le stylo de l'élève.

L'élève ne produit pas des pixels : il pilote un stylo (décision D10).
- départ (x0, y0), cap initial, longueur de pas L : choisis par l'élève (à partir de sa perception) ;
- puis T pas : à chaque pas le stylo avance de L, puis tourne d'un angle pris dans `TURNS`.

Le tracé est rendu par **le même moteur** que les formes du maître (sll.data.render).

Proprioception : à chaque pas l'élève peut « sentir » son virage, son cap et sa position
(`Gesture.proprioception`). C'est la matière première des sous-concepts moteurs
(« je tourne fort », « je vais tout droit », rythme entre deux virages) — décisions D20, D26.

Biais inductif assumé : l'ensemble des virages contient ±90°. Une variante sans 90° exact
(`TURNS_NO90`) sert de contrôle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.render import render

TURNS = np.deg2rad(np.array([-90.0, -45.0, -22.5, 0.0, 22.5, 45.0, 90.0]))
TURNS_NO90 = np.deg2rad(np.array([-80.0, -40.0, -20.0, 0.0, 20.0, 40.0, 80.0]))
T_STEPS = 16


@dataclass
class Gesture:
    start: np.ndarray          # (2,)
    heading: float             # radians
    step: float                # longueur d'un pas (pixels)
    actions: np.ndarray        # (T,) indices dans `turns`
    turns: np.ndarray = field(default_factory=lambda: TURNS.copy())

    def points(self) -> np.ndarray:
        """Positions successives du stylo (T+1, 2)."""
        pts = [np.asarray(self.start, float)]
        h = self.heading
        for a in self.actions:
            p = pts[-1] + self.step * np.array([np.cos(h), np.sin(h)])
            pts.append(p)
            h += self.turns[a]
        return np.array(pts)

    def proprioception(self) -> np.ndarray:
        """Par pas : (virage signé, |virage|, cos cap, sin cap) — ce que l'élève « ressent »."""
        h = self.heading
        out = []
        for a in self.actions:
            out.append([self.turns[a], abs(self.turns[a]), np.cos(h), np.sin(h)])
            h += self.turns[a]
        return np.array(out)

    def draw(self, thickness: float = 1.5) -> np.ndarray:
        return render([self.points()], thickness)


def turn_index(deg: float, turns: np.ndarray = TURNS) -> int:
    return int(np.argmin(np.abs(np.rad2deg(turns) - deg)))


def _centered(g: Gesture, center) -> Gesture:
    """Translate le geste pour que le centre de ses sommets soit `center`."""
    pts = g.points()[:-1]
    g.start = g.start + (np.asarray(center, float) - pts.mean(0))
    return g


def perfect_square(center, side: float, rotation: float = 0.0, direction: int = 1) -> Gesture:
    """Geste « parfait » : 4 côtés de 4 pas, virage de 90° après chaque côté."""
    a90, a0 = turn_index(90 * direction), turn_index(0)
    actions = np.array(([a0] * 3 + [a90]) * 4)
    return _centered(Gesture(start=np.zeros(2), heading=rotation, step=side / 4, actions=actions), center)


def perfect_circle(center, diameter: float, direction: int = 1, heading: float = 0.0) -> Gesture:
    """Geste « parfait » : 16 pas, virage constant de 22,5° (hexadécagone régulier)."""
    n = T_STEPS
    step = diameter * np.sin(np.pi / n)
    actions = np.full(n, turn_index(22.5 * direction))
    return _centered(Gesture(start=np.zeros(2), heading=heading, step=step, actions=actions), center)


def random_gesture(rng: np.random.Generator, center=(16.0, 16.0), step: float = 3.0,
                   p_straight: float | None = None, turns: np.ndarray = TURNS) -> Gesture:
    """Élève qui « gribouille » au hasard (référence de l'exploration aveugle)."""
    k = len(turns)
    if p_straight is None:
        probs = np.full(k, 1 / k)
    else:
        probs = np.full(k, (1 - p_straight) / (k - 1))
        probs[int(np.argmin(np.abs(turns)))] = p_straight
    actions = rng.choice(k, size=T_STEPS, p=probs)
    heading = rng.uniform(0, 2 * np.pi)
    return Gesture(start=np.asarray(center, float), heading=heading, step=step, actions=actions, turns=turns)
