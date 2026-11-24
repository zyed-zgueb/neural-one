"""Étape 4a — la Shadow comble-t-elle le retard de l'élève local quand chaque exemple n'est vu qu'une fois ?

Cible (D37, P26, P27) : en 1 passage, M2-lecture plafonne à 80 % (8 ex./forme) contre 96 % pour la
rétropropagation. On donne à chaque élève R réactivations par expérience (mêmes nombres de mises à
jour) et on compare : Replay, Shadow A, B, C, et la rétropropagation avec le même replay (M0+replay).

Réglage équitable (même budget de configurations par élève et par R), puis 100 répétitions
appariées (mêmes exemples, même ordre, même initialisation, même test). Encodeur E3, niveau L2.

Usage : OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/step4_shadow_classification.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from sll.analysis.stats import bootstrap_ci, holm, wilcoxon_signed_rank  # noqa: E402
from sll.core.rng import make_rng  # noqa: E402
from sll.data.datasets import make_set  # noqa: E402
from sll.data.encoders import Normalizer, get_encoder  # noqa: E402
from sll.models.classifiers import sample_hp  # noqa: E402
from sll.models.shadow_classifiers import sample_memory_hp  # noqa: E402
from sll.models.train import run_classification, run_memory_classification  # noqa: E402

SEED = 0
BUDGETS = [1, 2, 4, 8, 16, 32]
RS = [4, 19]
MEMORY_MODELS = ["Replay", "Shadow-A", "Shadow-B", "Shadow-C", "M0+replay"]
REFERENCES = [("M2-lecture", 1), ("M0-backprop", 1), ("M2-lecture", 20)]
N_CONFIGS, TUNE_REPS, TUNE_BUDGETS = 12, 16, [2, 8]
DATA = {}


def prefix(PX, Py, n):
    idx = np.sort(np.concatenate([np.where(Py == c)[0][:n] for c in (0, 1)]))
    return PX[idx], Py[idx]


def build(reps, n_test):
    enc = get_encoder("orientations")
    nz = Normalizer(enc(make_set("L2", 300, SEED, "norm").X))
    f = lambda s: (nz(enc(s.X)), s.y)  # noqa: E731
    return {"test": f(make_set("L2", n_test, SEED, "test")), "val": f(make_set("L2", 300, SEED, "val")),
            "pools": [f(make_set("L2", 32, SEED, "train", f"rep{r}")) for r in range(reps)],
            "tune": [f(make_set("L2", 8, SEED, "tune", f"rep{r}")) for r in range(TUNE_REPS)]}


def condition_name(kind, R=None, passes=None):
    return f"{kind}|R{R}" if R is not None else f"{kind}|P{passes}"


def run_one(kind, hp, R, passes, X, y, Xte, yte, rep, n, tag):
    o, i = make_rng(SEED, tag, "order", rep, n), make_rng(SEED, tag, "init", rep)
    if R is not None:
        return run_memory_classification(kind, hp, R, X, y, Xte, yte, o, i)
    return run_classification(kind, hp, X, y, Xte, yte, passes, o, i)


def _tune(args):
    kind, R, passes, cfg = args
    hp_rng = make_rng(SEED, "hp4", kind, R, passes, cfg)
    hp = sample_memory_hp(kind, hp_rng) if R is not None else sample_hp(kind, hp_rng)
    Xv, yv = DATA["val"]
    accs = []
    for r, (PX, Py) in enumerate(DATA["tune"]):
        for n in TUNE_BUDGETS:
            X, y = prefix(PX, Py, n)
            accs.append(run_one(kind, hp, R, passes, X, y, Xv, yv, r, n, "tune")["acc"])
    return condition_name(kind, R, passes), hp, float(np.mean(accs))


def _final(args):
    name, kind, R, passes, hp, rep = args
    PX, Py = DATA["pools"][rep]
    Xte, yte = DATA["test"]
    out = []
    for n in BUDGETS:
        X, y = prefix(PX, Py, n)
        res = run_one(kind, hp, R, passes, X, y, Xte, yte, rep, n, "final")
        out.append({"cond": name, "rep": rep, "n": n, "acc": res["acc"], "train_acc": res["train_acc"],
                    "updates": res["updates"], "seconds": res["seconds"]})
    return out


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    global N_CONFIGS, TUNE_REPS
    if args.quick:
        args.reps, N_CONFIGS, TUNE_REPS = 6, 3, 4
    t0 = time.time()
    DATA = build(args.reps, 1000 if not args.quick else 300)
    conds = [(k, R, None) for k in MEMORY_MODELS for R in RS] + [(k, None, p) for k, p in REFERENCES]
    jobs = [(k, R, p, c) for (k, R, p) in conds for c in range(N_CONFIGS)]
    with Pool(args.workers) as pool:
        tuned = pool.map(_tune, jobs, chunksize=2)
    best = {}
    for name, hp, acc in tuned:
        if name not in best or acc > best[name][1]:
            best[name] = (hp, acc)
    print(f"réglage : {time.time() - t0:.0f}s", flush=True)
    jobs = [(condition_name(k, R, p), k, R, p, best[condition_name(k, R, p)][0], rep)
            for (k, R, p) in conds for rep in range(args.reps)]
    with Pool(args.workers) as pool:
        rows = [r for rs in pool.map(_final, jobs, chunksize=2) for r in rs]
    print(f"évaluation : {time.time() - t0:.0f}s", flush=True)

    names = sorted({r["cond"] for r in rows})
    table = {}
    for c in names:
        for n in BUDGETS:
            a = np.array([r["acc"] for r in rows if r["cond"] == c and r["n"] == n])
            table[f"{c}|{n}"] = {"mean": float(a.mean()), "ci95": bootstrap_ci(a, n_boot=2000),
                                 "p_reach_90": float((a >= 0.9).mean()), "n": len(a),
                                 "updates": float(np.mean([r["updates"] for r in rows if r["cond"] == c and r["n"] == n])),
                                 "train_acc": float(np.mean([r["train_acc"] for r in rows if r["cond"] == c and r["n"] == n]))}

    def paired(ref, n):
        d = {}
        for r in rows:
            if r["n"] == n:
                d.setdefault(r["cond"], {})[r["rep"]] = r["acc"]
        out, pv = {}, {}
        for c in names:
            if c == ref:
                continue
            reps = sorted(set(d[c]) & set(d[ref]))
            x, y = np.array([d[c][i] for i in reps]), np.array([d[ref][i] for i in reps])
            w = wilcoxon_signed_rank(x, y)
            out[c] = {"diff": float((x - y).mean()), "diff_ci95": bootstrap_ci(x - y, n_boot=2000), **w}
            pv[c] = w["p"]
        for c, p in holm(pv).items():
            out[c]["p_holm"] = p
        return out

    tests = {}
    for n in (2, 8, 32):
        for R in RS:
            tests[f"vs_Replay|R{R}|N{n}"] = paired(f"Replay|R{R}", n)
        tests[f"vs_M2-lecture|P1|N{n}"] = paired("M2-lecture|P1", n)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"seed": SEED, "git_commit": commit, "reps": args.reps, "budgets": BUDGETS, "R": RS,
           "best_hyperparameters": {k: {"hp": v[0], "val_acc": v[1]} for k, v in best.items()},
           "table": table, "paired_tests": tests, "duration_s": time.time() - t0}
    name = "shadow_classification_quick.json" if args.quick else "shadow_classification.json"
    (ROOT / "results" / "step4").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / "step4" / name, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for c in names:
        print(c.ljust(18), " ".join(f"{table[f'{c}|{n}']['mean']:.3f}" for n in BUDGETS),
              " maj(N=8)=", int(table[f"{c}|8"]["updates"]))


if __name__ == "__main__":
    main()
