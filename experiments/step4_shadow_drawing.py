"""Étape 4b — la Shadow (motrice + gabarit intérieur) et la sensation de fermeture débloquent-elles le carré ?

Cible (D37, P24, P28) : à l'étape 3, tous les élèves finissent par tracer des cercles quoi qu'on leur
montre (carrés : 0 %). On compare, à budget d'essais égal et avec le même réglage pour tous :
G2 (trace classique), G3 (Shadow motrice), G5 (G3 + gabarit intérieur et/ou fermeture), le
contrôle « gabarit brouillé », et côté rétropropagation G1, G1 + même signal interne,
G1 + bonus d'exploration. Jugement du maître : « presque ça » ; plus deux conditions « pouce seul ».

Usage : OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/step4_shadow_drawing.py [--reps 10] [--quick]
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

from sll.analysis.stats import bootstrap_ci  # noqa: E402
from sll.core.rng import make_rng  # noqa: E402
from sll.data.datasets import make_set  # noqa: E402
from sll.data.encoders import Normalizer, get_encoder  # noqa: E402
from sll.models.drawers import make_drawer, sample_drawer_hp  # noqa: E402
from sll.models.shadow_drawers import STUDENTS, make_student, orientation_template, sample_student_hp  # noqa: E402
from sll.world.student import posture  # noqa: E402
from sll.world.teacher import CategoricalTeacher  # noqa: E402

SEED = 0
TEACHER = CategoricalTeacher()
JUDGES = {"presque": lambda pts, k: TEACHER.judge(pts, k).score,
          "pouce": lambda pts, k: 1.0 if TEACHER.judge(pts, k).accepted else -1.0}
CONDITIONS = [(name, "presque") for name in STUDENTS] + [("G3-shadow-motrice", "pouce"),
                                                          ("G5-gabarit+fermeture", "pouce")]
DATA: dict = {}


def build(reps, n_test):
    enc = get_encoder("orientations")
    nz = Normalizer(enc(make_set("L2", 300, SEED, "norm").X))

    def pack(s):
        return {"P": nz(enc(s.X)), "y": s.y, "post": [posture(x) for x in s.X],
                "tmpl": np.array([orientation_template(x) for x in s.X])}
    scr = make_set("L2", 50, SEED, "scramble")
    return {"test": pack(make_set("L2", n_test, SEED, "test-draw")),
            "val": pack(make_set("L2", 25, SEED, "val-draw")),
            "pools": [pack(make_set("L2", 4, SEED, "demo", f"rep{r}")) for r in range(reps)],
            "tune": [pack(make_set("L2", 4, SEED, "demo-tune", f"rep{r}")) for r in range(2)],
            "scramble": np.array([orientation_template(x) for x in scr.X])}


def take(pool, n):
    return np.sort(np.concatenate([np.where(pool["y"] == c)[0][:n] for c in (0, 1)]))


def evaluate(d, data, idx=None):
    idx = range(len(data["y"])) if idx is None else idx
    acc = {0: [], 1: []}
    for i in idx:
        ep = d.greedy(data["P"][i], *data["post"][i])
        acc[int(data["y"][i])].append(TEACHER.judge(ep.gesture.points(), int(data["y"][i])).accepted)
    return {"acc": float(np.mean(acc[0] + acc[1])), "acc_cercle": float(np.mean(acc[0])),
            "acc_carre": float(np.mean(acc[1]))}


def make(name, d, hp, rng):
    if name == "G2-trace":
        return make_drawer("G2-trace", d, hp, rng)
    return make_student(name, d, hp, rng)


def run(name, reward, hp, pool, n_demo, attempts, rep, cps, test, tag):
    rng = make_rng(SEED, tag, "draw4", name, reward, n_demo, rep)
    d = make(name, pool["P"].shape[1], hp, make_rng(SEED, "draw-init", rep))
    idx = take(pool, n_demo)
    scramble = STUDENTS.get(name) and STUDENTS[name].get("scramble")
    judge = JUDGES[reward]
    curve = [{"attempts": 0, **evaluate(d, test)}]
    internal = []
    for a in range(1, attempts + 1):
        i = idx[rng.integers(len(idx))]
        tmpl = DATA["scramble"][rng.integers(len(DATA["scramble"]))] if scramble else pool["tmpl"][i]
        if name == "G2-trace":
            d.attempt(pool["P"][i], int(pool["y"][i]), *pool["post"][i], rng, judge)
        else:
            d.attempt(pool["P"][i], int(pool["y"][i]), *pool["post"][i], rng, judge, template=tmpl)
            internal.append(getattr(d, "last_internal", 0.0))
        if a in cps:
            curve.append({"attempts": a, **evaluate(d, test)})
    train = evaluate(d, pool, idx)
    return {"name": name, "reward": reward, "n_demo": n_demo, "rep": rep, "hp": hp, "curve": curve,
            "train": train}


def _tune(args):
    name, reward, cfg, attempts = args
    hrng = make_rng(SEED, "hp-draw4", name, reward, cfg)
    hp = sample_drawer_hp("G2-trace", hrng) if name == "G2-trace" else sample_student_hp(name, hrng)
    scores = []
    for r, pool in enumerate(DATA["tune"]):
        res = run(name, reward, hp, pool, 1, attempts, r, {attempts // 2, attempts}, DATA["val"], "tune")
        scores.append(np.mean([c["acc"] for c in res["curve"][1:]]))
    return (name, reward), hp, float(np.mean(scores))


def _final(args):
    name, reward, hp, n_demo, rep, attempts, cps = args
    return run(name, reward, hp, DATA["pools"][rep], n_demo, attempts, rep, cps, DATA["test"], "final")


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--attempts", type=int, default=8000)
    ap.add_argument("--n-configs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.attempts, args.n_configs = 1, 400, 1
    t0 = time.time()
    DATA = build(args.reps, 50)
    jobs = [(n, r, c, args.attempts // 2) for n, r in CONDITIONS for c in range(args.n_configs)]
    with Pool(args.workers) as pool:
        tuned = pool.map(_tune, jobs)
    best = {}
    for key, hp, s in tuned:
        if key not in best or s > best[key][1]:
            best[key] = (hp, s)
    print(f"réglage : {time.time() - t0:.0f}s", flush=True)
    cps = sorted({int(args.attempts * f) for f in (0.125, 0.25, 0.5, 0.75, 1.0)})
    jobs = [(n, r, best[(n, r)][0], nd, rep, args.attempts, cps) for n, r in CONDITIONS for nd in (1, 4)
            for rep in range(args.reps)]
    with Pool(args.workers) as pool:
        results = pool.map(_final, jobs)
    print(f"évaluation : {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for n, r in CONDITIONS:
        for nd in (1, 4):
            rs = [x for x in results if x["name"] == n and x["reward"] == r and x["n_demo"] == nd]
            pts = {}
            for x in rs:
                for c in x["curve"]:
                    pts.setdefault(c["attempts"], []).append(c)
            summary[f"{n}|{r}|{nd}"] = {
                "curve": {str(a): {k: float(np.mean([c[k] for c in cs])) for k in ("acc", "acc_carre", "acc_cercle")}
                          for a, cs in sorted(pts.items())},
                "final_acc_ci95": bootstrap_ci([x["curve"][-1]["acc"] for x in rs], n_boot=2000),
                "final_carre_ci95": bootstrap_ci([x["curve"][-1]["acc_carre"] for x in rs], n_boot=2000),
                "reps_with_any_square": float(np.mean([x["curve"][-1]["acc_carre"] > 0 for x in rs])),
                "train_acc": float(np.mean([x["train"]["acc"] for x in rs])),
                "train_carre": float(np.mean([x["train"]["acc_carre"] for x in rs])),
            }
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"seed": SEED, "git_commit": commit, "reps": args.reps, "attempts": args.attempts,
           "best_hyperparameters": {"|".join(k): {"hp": v[0], "tune": v[1]} for k, v in best.items()},
           "summary": summary, "runs": results, "duration_s": time.time() - t0}
    (ROOT / "results" / "step4").mkdir(parents=True, exist_ok=True)
    name = "shadow_drawing_quick.json" if args.quick else "shadow_drawing.json"
    with open(ROOT / "results" / "step4" / name, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    for k, v in summary.items():
        last = list(v["curve"].values())[-1]
        print(k.ljust(36), f"acc={last['acc']:.2f} carré={last['acc_carre']:.2f} cercle={last['acc_cercle']:.2f}",
              f"| entraînement carré={v['train_carre']:.2f}")


if __name__ == "__main__":
    main()
