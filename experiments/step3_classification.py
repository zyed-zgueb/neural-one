"""Étape 3a — références de classification (M0, M1, M2, M2-lecture, 1-PPV, prototype, contrôles).

Protocole (CONCEPTION.md §7) :
1. Réglage des hyperparamètres : même budget (N_CONFIGS tirages) pour chaque modèle, sur des tirages
   d'apprentissage ET un jeu de validation distincts de ceux de l'évaluation finale.
2. Évaluation finale appariée : pour chaque répétition, tous les modèles reçoivent les mêmes
   exemples, dans le même ordre, avec la même initialisation, et sont testés sur le même jeu.
3. Budgets : N ∈ {1,2,4,8,16,32} exemples par classe ; régimes : 1 passage (en ligne) et 20 passages.

Usage : PYTHONPATH=src python experiments/step3_classification.py [--reps 100] [--quick]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sll.analysis.stats import bootstrap_ci, holm, wilcoxon_signed_rank  # noqa: E402
from sll.core.rng import make_rng  # noqa: E402
from sll.data.datasets import make_set  # noqa: E402
from sll.data.encoders import Normalizer, get_encoder  # noqa: E402
from sll.models.classifiers import sample_hp  # noqa: E402
from sll.models.train import run_classification  # noqa: E402

SEED = 0
MODELS = ["M0-backprop", "M1-reinforce", "M2-local", "M2-lecture", "1-PPV", "prototype", "non-entraîné"]
TUNED = ["M0-backprop", "M1-reinforce", "M2-local", "M2-lecture"]
BUDGETS = [1, 2, 4, 8, 16, 32]
PASSES = [1, 20]
ENCODERS = ["orientations", "edges", "pixels"]
N_CONFIGS = 16
TUNE_REPS = 16
TUNE_BUDGETS = [2, 8]

DATA: dict = {}


def prefix(pool_X, pool_y, n):
    """Les n premiers exemples de chaque classe, dans l'ordre du tirage (budgets emboîtés)."""
    idx = np.sort(np.concatenate([np.where(pool_y == c)[0][:n] for c in (0, 1)]))
    return pool_X[idx], pool_y[idx]


def build_data(level_train: str, level_test: str, reps: int, n_test: int):
    """Encode une seule fois toutes les images (partagées entre processus par fork)."""
    out = {}
    norm_set = make_set(level_train, 300, SEED, "norm")
    test = make_set(level_test, n_test, SEED, "test")
    val = make_set(level_train, 300, SEED, "val")
    pools = [make_set(level_train, max(BUDGETS), SEED, "train", f"rep{r}") for r in range(reps)]
    tune_pools = [make_set(level_train, max(TUNE_BUDGETS), SEED, "tune", f"rep{r}") for r in range(TUNE_REPS)]
    for e in ENCODERS:
        enc = get_encoder(e, seed=SEED)
        nz = Normalizer(enc(norm_set.X))
        f = lambda X: nz(enc(X))  # noqa: E731
        out[e] = {
            "test": (f(test.X), test.y), "val": (f(val.X), val.y),
            "pools": [(f(p.X), p.y) for p in pools], "tune": [(f(p.X), p.y) for p in tune_pools],
        }
    return out


def _tune_job(args):
    enc, kind, passes, cfg = args
    hp = sample_hp(kind, make_rng(SEED, "hp", kind, cfg))
    Xv, yv = DATA[enc]["val"]
    accs = []
    for r, (PX, Py) in enumerate(DATA[enc]["tune"]):
        for n in TUNE_BUDGETS:
            Xtr, ytr = prefix(PX, Py, n)
            res = run_classification(kind, hp, Xtr, ytr, Xv, yv, passes,
                                     make_rng(SEED, "tune-order", r, n), make_rng(SEED, "tune-init", r))
            accs.append(res["acc"])
    return (enc, kind, passes), hp, float(np.mean(accs))


def _final_job(args):
    enc, kind, passes, hp, rep, reward = args
    PX, Py = DATA[enc]["pools"][rep]
    Xte, yte = DATA[enc]["test"]
    rows = []
    for n in BUDGETS:
        Xtr, ytr = prefix(PX, Py, n)
        res = run_classification(kind, hp, Xtr, ytr, Xte, yte, passes,
                                 make_rng(SEED, "order", rep, n), make_rng(SEED, "init", rep), reward=reward)
        res.update(encoder=enc, model=kind, passes=passes, rep=rep, n_per_class=n, reward=reward)
        rows.append(res)
    return rows


def summarize(rows):
    """Moyennes, IC, proportion de répétitions atteignant 90/95 %, N90/N95 sur la courbe moyenne."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["encoder"], r["passes"], r["reward"], r["model"], r["n_per_class"])].append(r)
    table = {}
    for key, rs in groups.items():
        acc = np.array([r["acc"] for r in rs])
        tr = np.array([r["train_acc"] for r in rs])
        table[key] = {"mean": float(acc.mean()), "std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                      "ci95": bootstrap_ci(acc, n_boot=2000), "p_reach_90": float((acc >= 0.9).mean()),
                      "p_reach_95": float((acc >= 0.95).mean()), "train_acc": float(tr.mean()),
                      "updates": float(np.mean([r["updates"] for r in rs])),
                      "seconds": float(np.mean([r["seconds"] for r in rs])), "n": len(rs)}
    curves = {}
    for (e, p, rw, m, n), v in table.items():
        curves.setdefault((e, p, rw, m), {})[n] = v
    thresholds = {}
    for key, c in curves.items():
        ns = sorted(c)
        def first(th):
            for n in ns:
                if c[n]["mean"] >= th:
                    return n
            return None  # jamais atteint dans les budgets testés (donnée censurée)
        thresholds[key] = {"N90": first(0.9), "N95": first(0.95)}
    return table, curves, thresholds


def paired_tests(rows, encoder, passes, n, reference="M2-local"):
    """Comparaisons appariées (même répétition) de chaque modèle contre la référence."""
    by = {}
    for r in rows:
        if r["encoder"] == encoder and r["passes"] == passes and r["n_per_class"] == n and r["reward"] == "normal":
            by.setdefault(r["model"], {})[r["rep"]] = r["acc"]
    ref = by[reference]
    pv, out = {}, {}
    for m, d in by.items():
        if m == reference:
            continue
        reps = sorted(set(d) & set(ref))
        x, y = np.array([d[i] for i in reps]), np.array([ref[i] for i in reps])
        w = wilcoxon_signed_rank(x, y)
        out[m] = {"diff_mean": float((x - y).mean()), "diff_ci95": bootstrap_ci(x - y, n_boot=2000), **w}
        pv[m] = w["p"]
    for m, p in holm(pv).items():
        out[m]["p_holm"] = p
    return out


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=1000, help="images de test par classe")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    global N_CONFIGS, TUNE_REPS
    if args.quick:
        args.reps, args.n_test, N_CONFIGS, TUNE_REPS = 8, 300, 4, 4
    t0 = time.time()
    DATA = build_data("L2", "L2", args.reps, args.n_test)
    print(f"données encodées en {time.time() - t0:.0f}s", flush=True)

    # 1. Réglage
    jobs = [(e, k, p, c) for e in ENCODERS for k in TUNED for p in PASSES for c in range(N_CONFIGS)]
    with Pool(args.workers) as pool:
        tune = pool.map(_tune_job, jobs, chunksize=4)
    best = {}
    for key, hp, acc in tune:
        if key not in best or acc > best[key][1]:
            best[key] = (hp, acc)
    print(f"réglage terminé en {time.time() - t0:.0f}s", flush=True)

    # 2. Évaluation finale appariée (+ contrôle récompense aléatoire)
    jobs = []
    for e, k, p in product(ENCODERS, MODELS, PASSES):
        hp = best.get((e, k, p), ({}, None))[0]
        for rep in range(args.reps):
            jobs.append((e, k, p, hp, rep, "normal"))
            if k in TUNED and e == "orientations":
                jobs.append((e, k, p, hp, rep, "aleatoire"))
    with Pool(args.workers) as pool:
        rows = [r for rs in pool.map(_final_job, jobs, chunksize=4) for r in rs]
    print(f"évaluation terminée en {time.time() - t0:.0f}s", flush=True)

    table, curves, thresholds = summarize(rows)
    tests = {f"{e}|P{p}|N{n}": paired_tests(rows, e, p, n)
             for e in ENCODERS for p in PASSES for n in (2, 8, 32)}
    out_dir = ROOT / "results" / "step3"
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    k2s = lambda k: "|".join(map(str, k))  # noqa: E731
    summary = {
        "seed": SEED, "git_commit": commit, "reps": args.reps, "n_test_per_class": args.n_test,
        "level": "L2", "budgets": BUDGETS, "passes": PASSES, "encoders": ENCODERS,
        "best_hyperparameters": {k2s(k): {"hp": v[0], "val_acc": v[1]} for k, v in best.items()},
        "table": {k2s(k): v for k, v in table.items()},
        "thresholds": {k2s(k): v for k, v in thresholds.items()},
        "paired_tests_vs_M2-local": tests,
        "duration_s": time.time() - t0,
    }
    name = "classification_quick.json" if args.quick else "classification.json"
    with open(out_dir / name, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    np.save(out_dir / name.replace(".json", "_rows.npy"), np.array(rows, dtype=object), allow_pickle=True)
    for key, th in sorted(thresholds.items()):
        if key[2] == "normal":
            c = curves[key]
            print(key, th, " ".join(f"{n}:{c[n]['mean']:.2f}" for n in BUDGETS))


if __name__ == "__main__":
    main()
