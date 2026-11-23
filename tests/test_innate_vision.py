"""Tests du système visuel inné SV0."""
import numpy as np

from sll.data.datasets import make_set
from sll.data.encoders import get_encoder
from sll.data.innate_vision import topology
from sll.data.render import render
from sll.world.pen import perfect_circle, perfect_square


def test_closed_shapes_are_closed_single_piece_without_ends():
    for g in (perfect_square((16, 16), 14, rotation=0.3), perfect_circle((16, 16), 18)):
        t = topology(g.draw(1.5))
        assert t["morceaux"] == 1 and t["terminaisons"] == 0 and t["fermeture"] == 1.0 and t["trous"] == 1


def test_open_line_has_two_ends_and_is_not_closed():
    t = topology(render([np.array([[5.0, 16.0], [27.0, 16.0]])], 1.5))
    assert t["terminaisons"] == 2 and t["fermeture"] == 0.0 and t["trous"] == 0


def test_small_gap_is_completed_but_less_closed():
    pts = perfect_square((16, 16), 16).points()
    full = topology(render([pts], 1.5))
    gapped = topology(render([pts[:-1], ], 1.5))  # dernier pas (4 px) manquant
    assert full["fermeture"] == 1.0
    assert 0.0 < gapped["fermeture"] < 1.0 and gapped["terminaisons"] == 2


def test_two_pieces():
    t = topology(render([np.array([[3.0, 5.0], [12.0, 5.0]]), np.array([[3.0, 25.0], [28.0, 25.0]])], 1.5))
    assert t["morceaux"] == 2


def test_topology_does_not_reveal_circle_vs_square():
    """La fermeture etc. ne donnent pas la réponse : cercle et carré ont la même topologie."""
    s = make_set("L2", 40, 0, "topo")
    T = [topology(x) for x in s.X]
    for key in ("morceaux", "terminaisons", "trous", "fermeture"):
        a = [t[key] for t, y in zip(T, s.y) if y == 0]
        b = [t[key] for t, y in zip(T, s.y) if y == 1]
        assert np.mean(a) == np.mean(b), key


def test_sv0_dimensions():
    X = make_set("L2", 3, 0, "dim").X
    for name, d in (("SV0", 13), ("SV0-carte", 45), ("SV0-orientation", 8), ("SV0-topologie", 5)):
        e = get_encoder(name)
        assert e(X).shape == (6, d) and e.dim == d
