"""Niveaux de difficulté du générateur.

Chaque niveau est une distribution de paramètres. Les tailles sont tirées en **périmètre** P,
commun aux deux classes : cercle de rayon P/2π, carré de côté P/4. Ainsi la quantité d'encre
(≈ P × épaisseur) a la même distribution pour les deux formes (voir CONCEPTION.md §2.3).

Remarque géométrique : en moyenne sur une rotation uniforme, l'étendue d'un carré de périmètre P
vaut P/π, exactement le diamètre du cercle de même périmètre. L'étendue est donc équilibrée en
moyenne dès que la rotation varie (L2+), mais pas en L0/L1 (carré non tourné : étendue P/4 < P/π).
Ce n'est pas une fuite « non géométrique » : c'est une propriété de forme (compacité). Elle est
mesurée et rapportée par les sondes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Level:
    name: str
    description: str
    perimeter: tuple[float, float] = (60.0, 60.0)
    centered: bool = True
    thickness: tuple[float, float] = (1.5, 1.5)
    rotate_square: bool = False
    ellipticity: tuple[float, float] = (0.0, 0.0)
    radial_jitter: float = 0.0          # amplitude relative des ondulations du cercle
    vertex_jitter: float = 0.0          # écart-type relatif (au côté) du déplacement des sommets
    corner_rounding: tuple[float, float] = (0.0, 0.0)  # rayon relatif au côté
    noise_sigma: tuple[float, float] = (0.0, 0.0)
    fragment: tuple[float, float] = (0.0, 0.0)          # fraction du contour effacée
    margin: float = 1.0
    extra: dict = field(default_factory=dict)


LEVELS: dict[str, Level] = {
    "L0": Level("L0", "Contrôle : centré, taille et épaisseur fixes, carré droit"),
    "L1": Level("L1", "Position et taille variables", perimeter=(32.0, 68.0), centered=False),
    "L2": Level("L2", "+ épaisseur variable, rotation du carré", perimeter=(32.0, 68.0), centered=False,
                thickness=(1.0, 3.0), rotate_square=True),
    "L3": Level("L3", "+ déformations géométriques et bruit", perimeter=(32.0, 68.0), centered=False,
                thickness=(1.0, 3.0), rotate_square=True, ellipticity=(0.0, 0.12), radial_jitter=0.04,
                vertex_jitter=0.04, corner_rounding=(0.0, 0.15), noise_sigma=(0.0, 0.15)),
    "L4": Level("L4", "+ contour fragmenté", perimeter=(32.0, 68.0), centered=False,
                thickness=(1.0, 3.0), rotate_square=True, ellipticity=(0.0, 0.12), radial_jitter=0.04,
                vertex_jitter=0.04, corner_rounding=(0.0, 0.15), noise_sigma=(0.0, 0.15),
                fragment=(0.10, 0.25)),
}

# Hors distribution : apprentissage sur petites formes, test sur grandes (niveau L2 sinon).
LEVELS["OOD-train"] = replace(LEVELS["L2"], name="OOD-train", description="L2, petites formes",
                              perimeter=(32.0, 48.0))
LEVELS["OOD-test"] = replace(LEVELS["L2"], name="OOD-test", description="L2, grandes formes jamais vues",
                             perimeter=(56.0, 68.0))


def get_level(name: str) -> Level:
    return LEVELS[name]
