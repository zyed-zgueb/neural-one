"""Tests de l'étape 4 : Shadow Memory et équivalences avec les mécanismes connus."""
import numpy as np

from sll.core.rng import make_rng
from sll.learning.shadow_memory import ShadowEntry, ShadowMemory
from sll.models.classifiers import make_learner
from sll.models.shadow_classifiers import MemoryLearner, make_memory_learner
from sll.core.network import MLP


def _data(n=12, d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    X = rng.normal(size=(n, d)) + 1.5 * y[:, None]
    return X, y


def _train(learner, X, y, seed=1):
    rng = make_rng(seed, "o")
    for i in rng.permutation(len(y)):
        a, info = learner.act(X[i], rng)
        learner.feedback(info, 1.0 if a == y[i] else -1.0, int(y[i]))
    return learner


def test_shadow_B_without_reactivation_equals_eligibility_rule():
    """Sans réactivation (R = 0) et feedback immédiat, Shadow B ≡ règle locale M2-lecture (λ = 0)."""
    X, y = _data()
    m2 = _train(make_learner("M2-lecture", 8, {"lr": 0.1, "lam": 0.0}, make_rng(0, "i")), X, y)
    sb = _train(make_memory_learner("Shadow-B", 8, {"lr": 0.1, "decay": 0.9}, make_rng(0, "i"), R=0), X, y)
    assert np.allclose(m2.net.W2, sb.net.W2) and np.allclose(m2.net.b2, sb.net.b2)


def test_shadow_C_with_neutral_settings_equals_replay():
    """Indice complet, sans bruit, sans mélange, sans renforcement, γ = 1, choix uniforme : C ≡ Replay."""
    X, y = _data()
    rp = _train(make_memory_learner("Replay", 8, {"lr": 0.1}, make_rng(0, "i"), R=5), X, y)
    net = MLP(8, 64, 2, make_rng(0, "i"))
    c = MemoryLearner(net, 0.1, 5, "C", decay=1.0, cue_keep=1.0, cue_noise=0.0, blend=0.0, recall_boost=0.0)
    c.memory.selection = "uniform"
    c = _train(c, X, y)
    assert np.allclose(rp.net.W2, c.net.W2)


def test_update_count_is_matched():
    X, y = _data()
    for kind, hp in [("Replay", {"lr": 0.1}), ("Shadow-B", {"lr": 0.1, "decay": 0.9}),
                     ("M0+replay", {"lr": 0.1})]:
        L = _train(make_memory_learner(kind, 8, hp, make_rng(0, "i"), R=4), X, y)
        assert L.counters.updates == len(y) * 5, kind


def test_memory_decay_capacity_and_recall_boost():
    m = ShadowMemory(capacity=3, decay=0.5, selection="strength", recall_boost=0.5)
    for k in range(4):
        m.store(ShadowEntry(x=np.array([k]), h=np.zeros(1), a=0, p=None, r=1.0))
    assert len(m) == 3
    assert np.allclose(sorted(m.strengths()), [0.25, 0.5, 1.0])
    rng = np.random.default_rng(0)
    e = m.sample(rng)
    before = e.strength
    assert e.recalls == 1 and before > 0
    m2 = ShadowMemory(recall_boost=0.5)
    m2.store(ShadowEntry(x=np.zeros(1), h=np.zeros(1), a=0, p=None, r=1.0, strength=0.2))
    e2 = m2.sample(rng)
    assert np.isclose(e2.strength, 0.6)


def test_shadow_A_forgets_the_decision():
    m = ShadowMemory(store_action=False)
    e = m.store(ShadowEntry(x=np.zeros(2), h=np.zeros(2), a=1, p=np.array([0.3, 0.7]), r=1.0))
    assert e.a is None and e.p is None
