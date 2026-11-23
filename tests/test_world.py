"""Tests de l'étape 2 : générateur, encodeurs, sondes, stylo, maître."""
import numpy as np
import pytest

from sll.analysis.probes import image_statistics, probe_accuracy, run_probes
from sll.core.rng import make_rng
from sll.data.datasets import few_shot_episode, leakage_report, make_set
from sll.data.encoders import get_encoder
from sll.data.levels import LEVELS
from sll.data.render import render
from sll.data.shapes_generator import CIRCLE, SQUARE, ShapeParams, build_shape, generate
from sll.world.pen import TURNS, perfect_circle, perfect_square, random_gesture
from sll.world.teacher import CategoricalTeacher, CopyTeacher

T = CategoricalTeacher()


# ------------------------------------------------------------------ générateur

def test_determinism():
    a = make_set("L3", 5, 42, "train")
    b = make_set("L3", 5, 42, "train")
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)


def test_roles_are_independent():
    a = make_set("L2", 5, 42, "train")
    b = make_set("L2", 5, 42, "test")
    assert not np.array_equal(a.X, b.X)


@pytest.mark.parametrize("level", list(LEVELS))
def test_shapes_inside_canvas_and_balanced(level):
    s = make_set(level, 40, 1, "check")
    assert (s.y == CIRCLE).sum() == (s.y == SQUARE).sum() == 40
    for sh in s.shapes:
        pts = sh.outline
        assert pts.min() >= 0 and pts.max() <= 32, f"forme hors image ({level})"
    assert s.X.min() >= 0 and s.X.max() <= 1


def test_ink_is_matched_between_classes():
    s = make_set("L2", 400, 3, "ink")
    ink = s.X.sum((1, 2))
    c, q = ink[s.y == CIRCLE].mean(), ink[s.y == SQUARE].mean()
    assert abs(c - q) / c < 0.05


def test_render_thickness_scales_ink():
    line = [np.array([[4.0, 16.0], [28.0, 16.0]])]
    a, b = render(line, 1.0).sum(), render(line, 2.0).sum()
    assert 1.8 < b / a < 2.2


# ------------------------------------------------------------------ sondes et fuites

@pytest.mark.parametrize("level", ["L2", "L3"])
def test_no_nuisance_shortcut(level):
    """Encre, position, épaisseur ne doivent pas révéler la classe (niveaux de référence)."""
    s = make_set(level, 800, 7, "probe-test")
    r = run_probes(s.X, s.y)
    for key in ["encre_totale", "centroide_x", "centroide_y", "epaisseur_moyenne", "NUISANCE_jointe"]:
        assert r[key]["ci95"][1] < 0.58, (level, key, r[key])


def test_probe_detects_a_planted_shortcut():
    """Contrôle de la sonde elle-même : un raccourci volontaire doit être détecté."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    f = y + rng.normal(0, 0.5, 1000)
    assert probe_accuracy(f, y)["accuracy"] > 0.8


def test_train_test_disjoint():
    tr, te = few_shot_episode("L2", 16, 5, repetition=0, n_test_per_class=200)
    rep = leakage_report(tr, te)
    assert rep["near_duplicates"] == 0
    tr2, te2 = few_shot_episode("L2", 16, 5, repetition=1, n_test_per_class=200)
    assert np.array_equal(te.X, te2.X), "le test doit être commun à toutes les répétitions (appariement)"
    assert not np.array_equal(tr.X, tr2.X)


# ------------------------------------------------------------------ encodeurs

@pytest.mark.parametrize("name", ["pixels", "random", "edges"])
def test_encoders_shape_and_determinism(name):
    X = make_set("L2", 4, 0, "enc").X
    e1, e2 = get_encoder(name, seed=3), get_encoder(name, seed=3)
    F = e1(X)
    assert F.shape == (8, e1.dim)
    assert np.allclose(F, e2(X))
    assert np.isfinite(F).all()


# ------------------------------------------------------------------ stylo

def test_pen_and_generator_share_the_renderer():
    """Un carré tracé au stylo et le même carré du générateur donnent la même image (aucun biais de style)."""
    g = perfect_square((15.3, 16.8), 12.0, rotation=0.37)
    p = ShapeParams(kind=SQUARE, cx=15.3, cy=16.8, perimeter=48.0, thickness=1.5, rotation=0.37,
                    ellipticity=0, radial_jitter=0, vertex_jitter=0, corner_rounding=0, noise_sigma=0,
                    fragment=0, shape_seed=0)
    a, b = g.draw(1.5), build_shape(p).ink
    assert np.minimum(a, b).sum() / np.maximum(a, b).sum() > 0.97


def test_perfect_gestures_are_closed():
    for g in (perfect_square((16, 16), 14), perfect_circle((16, 16), 18)):
        pts = g.points()
        assert np.linalg.norm(pts[-1] - pts[0]) < 1e-6
        assert np.allclose(pts[:-1].mean(0), [16, 16])


def test_proprioception_shape():
    g = random_gesture(make_rng(0, "p"))
    pr = g.proprioception()
    assert pr.shape == (16, 4)
    assert np.allclose(pr[:, 0], TURNS[g.actions])


# ------------------------------------------------------------------ maître

@pytest.mark.parametrize("seed", range(5))
def test_teacher_accepts_perfect_gestures_any_size_rotation(seed):
    rng = np.random.default_rng(seed)
    side, rot = rng.uniform(7, 20), rng.uniform(0, np.pi / 2)
    sq = perfect_square((16, 16), side, rotation=rot, direction=int(rng.choice([-1, 1])))
    ci = perfect_circle((16, 16), rng.uniform(9, 24), direction=int(rng.choice([-1, 1])))
    assert T.judge(sq.points(), SQUARE).accepted
    assert T.judge(ci.points(), CIRCLE).accepted
    assert not T.judge(sq.points(), CIRCLE).accepted
    assert not T.judge(ci.points(), SQUARE).accepted


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "OOD-test"])
def test_teacher_agrees_with_generator(level):
    s = make_set(level, 60, 11, "teacher")
    for sh in s.shapes:
        pts = np.vstack([sh.outline, sh.outline[:1]])
        assert T.judge(pts, sh.params.kind).accepted
        assert not T.judge(pts, 1 - sh.params.kind).accepted


def test_teacher_rejects_open_shape_and_three_sides():
    g = perfect_square((16, 16), 14)
    pts = g.points()[:13]  # trois côtés seulement
    j = T.judge(pts, SQUARE)
    assert not j.accepted and j.score < 0.5


def test_encouragement_is_graded():
    """Un carré légèrement déformé est « presque ça » : note entre le gribouillis et le parfait."""
    perfect = perfect_square((16, 16), 14).points()
    rng = np.random.default_rng(0)
    slightly = perfect + rng.normal(0, 0.4, perfect.shape)
    slightly[-1] = slightly[0]
    scrib = random_gesture(make_rng(1, "s"), step=3.5).points()
    s_p, s_s, s_r = (T.judge(x, SQUARE).score for x in (perfect, slightly, scrib))
    assert s_p > s_s > s_r


def test_random_scribbles_are_rarely_rewarded():
    rng = make_rng(0, "rs")
    acc = sum(T.judge(random_gesture(rng, step=3.5).points(), SQUARE).accepted for _ in range(2000))
    assert acc <= 2


def test_copy_teacher_prefers_the_shown_model():
    p = ShapeParams(kind=SQUARE, cx=16, cy=16, perimeter=56.0, thickness=2.0, rotation=0.0, ellipticity=0,
                    radial_jitter=0, vertex_jitter=0, corner_rounding=0, noise_sigma=0, fragment=0, shape_seed=0)
    model = build_shape(p).ink
    same = perfect_square((16, 16), 14).points()
    other = perfect_square((8, 8), 6).points()
    C = CopyTeacher()
    assert C.judge(same, model).score > C.judge(other, model).score
