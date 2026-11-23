"""Tests de l'étape 3 : réseau commun, règles d'apprentissage, contrôles."""
import numpy as np
import pytest

from sll.core.network import MLP, softmax_xent_grad
from sll.core.rng import make_rng
from sll.data.datasets import make_set
from sll.data.encoders import Normalizer, get_encoder
from sll.models.classifiers import make_learner
from sll.models.drawers import G0Imitation
from sll.models.train import run_classification
from sll.world.student import posture
from sll.world.teacher import CategoricalTeacher


def _net(seed=0, d=6, h=5, k=3):
    return MLP(d, h, k, np.random.default_rng(seed))


def test_backprop_matches_finite_differences():
    net = _net()
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=6), 2

    def loss():
        return -np.log(net.forward(x).p[y])

    act = net.forward(x)
    g = net.backward(act, softmax_xent_grad(act.p, y))
    eps = 1e-6
    for name in ("W1", "b1", "W2", "b2"):
        P = getattr(net, name)
        num = np.zeros_like(P)
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + eps
            lp = loss()
            P[idx] = old - eps
            lm = loss()
            P[idx] = old
            num[idx] = (lp - lm) / (2 * eps)
        assert np.allclose(num, g[name], atol=1e-5), name


def test_local_output_trace_equals_policy_gradient():
    """Couche de sortie : la trace locale (post − p) ⊗ h EST le gradient de log π(a).
    Donc M2 et M1 ne diffèrent que par la couche cachée (perturbation de nœud vs rétropropagation)."""
    net = _net()
    x = np.random.default_rng(2).normal(size=6)
    act = net.forward(x)
    a = 1
    local = net.local_traces(act, a, sigma=0.0)
    dz = -act.p.copy()
    dz[a] += 1
    bp = net.backward(act, dz)
    assert np.allclose(local["W2"], bp["W2"]) and np.allclose(local["b2"], bp["b2"])
    assert np.allclose(local["W1"], 0.0), "sans bruit, la règle locale ne touche pas la couche cachée"


def test_node_perturbation_estimates_the_gradient():
    """En moyenne, (R(ξ) − R(0)) · ξ/σ² ≈ dR/du : la perturbation de nœud est un estimateur sans biais
    (au premier ordre) du gradient, sans rétropropagation."""
    net = _net(seed=3)
    rng = np.random.default_rng(4)
    x = rng.normal(size=6)
    c = rng.normal(size=net.d_hidden)
    R = lambda act: float(c @ act.h)  # noqa: E731
    R0 = R(net.forward(x))
    sigma, n = 0.01, 40000
    est = np.zeros(net.d_hidden)
    for _ in range(n):
        xi = rng.normal(0, sigma, net.d_hidden)
        act = net.forward(x, xi)
        est += (R(act) - R0) * net.local_traces(act, 0, sigma)["b1"]
    est /= n
    true = c * (net.forward(x).u > 0)
    assert np.corrcoef(est, true)[0, 1] > 0.95


@pytest.fixture(scope="module")
def orient_data():
    enc = get_encoder("orientations")
    nz = Normalizer(enc(make_set("L2", 200, 0, "norm").X))
    te = make_set("L2", 200, 0, "test")
    pools = [make_set("L2", 8, 0, "train", f"rep{r}") for r in range(8)]
    return [(nz(enc(p.X)), p.y) for p in pools], (nz(enc(te.X)), te.y)


def test_random_reward_teaches_nothing(orient_data):
    pools, (Xte, yte) = orient_data
    accs = []
    for kind, hp in [("M2-local", {"lr": 0.1, "lr_hidden": 1e-3, "sigma": 0.1, "lam": 0.0}),
                     ("M1-reinforce", {"lr": 0.1})]:
        for r, (X, y) in enumerate(pools):
            res = run_classification(kind, hp, X, y, Xte, yte, 20, make_rng(0, "o", r), make_rng(0, "i", r),
                                     reward="aleatoire")
            accs.append(res["acc"])
    assert abs(np.mean(accs) - 0.5) < 0.08


def test_true_reward_beats_random_reward(orient_data):
    pools, (Xte, yte) = orient_data
    hp = {"lr": 0.1, "lam": 0.0}
    good = [run_classification("M2-lecture", hp, X, y, Xte, yte, 20, make_rng(0, "o", r), make_rng(0, "i", r))["acc"]
            for r, (X, y) in enumerate(pools)]
    assert np.mean(good) > 0.75


def test_same_initialisation_for_all_models():
    a = make_learner("M0-backprop", 8, {"lr": 0.1}, make_rng(0, "init", 3))
    b = make_learner("M2-local", 8, {"lr": 0.1, "lr_hidden": 0, "sigma": 0, "lam": 0}, make_rng(0, "init", 3))
    assert np.array_equal(a.net.W1, b.net.W1) and np.array_equal(a.net.W2, b.net.W2)


def test_imitation_student_learns_to_draw_a_square():
    """G0 qui imite le geste du maître sur un seul carré finit par tracer un carré accepté."""
    ink = make_set("L2", 1, 0, "train", "g0").shapes
    sq = next(s for s in ink if s.params.kind == 1)
    percept = np.zeros(8)
    start, step = posture(sq.ink)
    rng = make_rng(0, "g0")
    d = G0Imitation(8, make_rng(0, "init"), lr=0.1)
    for _ in range(100):
        d.learn_from_demo(percept, 1, start, step, rng)
    ep = d.greedy(percept, start, step)
    assert CategoricalTeacher().judge(ep.gesture.points(), 1).accepted
