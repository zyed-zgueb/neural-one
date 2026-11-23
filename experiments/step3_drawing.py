"""Étape 3b — premiers élèves dessinateurs (sans Shadow) : G0 imitation, G1 REINFORCE, G2 trace locale.

Questions :
1. Un élève sans Shadow parvient-il à remonter la pente de « presque ça » ?
2. « Pouce » (±1) contre « presque ça » (note graduée) : lequel permet d'apprendre ? (H8, P13)
3. Combien d'essais, pour combien de démonstrations ?

Mesure : les élèves dessinent (sans exploration) des formes **jamais vues** (jeu de test L2) ;
le maître catégoriel juge chaque dessin pour la forme montrée.

Usage : PYTHONPATH=src python experiments/step3_drawing.py [--reps 10] [--attempts 8000] [--quick]
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

from sll.analysis.stats import bootstrap_ci  # noqa: E402
from sll.core.rng import make_rng  # noqa: E402
from sll.data.datasets import make_set  # noqa: E402
from sll.data.encoders import Normalizer, get_encoder  # noqa: E402
from sll.models.drawers import make_drawer, sample_drawer_hp  # noqa: E402
from sll.world.student import posture  # noqa: E402
from sll.world.teacher import CategoricalTeacher  # noqa: E402

SEED = 0
TEACHER = CategoricalTeacher()
JUDGES = {
    "presque": lambda pts, k: TEACHER.judge(pts, k).score,
    "pouce": lambda pts, k: 1.0 if TEACHER.judge(pts, k).accepted else -1.0,
}
DATA: dict = {}


def build_data(reps, n_demo_max, n_test):
    enc = get_encoder("orientations")
    nz = Normalizer(enc(make_set("L2", 300, SEED, "norm").X))

    def pack(s):
        return {"P": nz(enc(s.X)), "y": s.y, "post": [posture(x) for x in s.X]}
    return {
        "test": pack(make_set("L2", n_test, SEED, "test-draw")),
        "pools": [pack(make_set("L2", n_demo_max, SEED, "demo", f"rep{r}")) for r in range(reps)],
        "tune": [pack(make_set("L2", n_demo_max, SEED, "demo-tune", f"rep{r}")) for r in range(3)],
    }


def take(pool, n):
    idx = np.sort(np.concatenate([np.where(pool["y"] == c)[0][:n] for c in (0, 1)]))
    return idx


def evaluate(drawer, data):
    acc, sc = {0: [], 1: []}, {0: [], 1: []}
    for i in range(len(data["y"])):
        ep = drawer.greedy(data["P"][i], *data["post"][i])
        j = TEACHER.judge(ep.gesture.points(), int(data["y"][i]))
        acc[int(data["y"][i])].append(j.accepted)
        sc[int(data["y"][i])].append(j.score)
    return {"acc": float(np.mean(acc[0] + acc[1])), "acc_cercle": float(np.mean(acc[0])),
            "acc_carre": float(np.mean(acc[1])), "score": float(np.mean(sc[0] + sc[1]))}


def run(kind, hp, pool, n_demo, attempts, reward, rep, checkpoints, test, tag="final"):
    rng = make_rng(SEED, tag, "draw", kind, reward, n_demo, rep)
    d = make_drawer(kind, pool["P"].shape[1], hp, make_rng(SEED, "draw-init", rep))
    idx = take(pool, n_demo)
    curve, t0 = [], time.perf_counter()
    curve.append({"attempts": 0, **evaluate(d, test)})
    if kind == "G0-imitation":
        # l'imitation ne fait pas d'essais : elle apprend du geste montré ; on compte les présentations
        presentations = 0
        for e in range(attempts):
            for i in rng.permutation(idx):
                d.learn_from_demo(pool["P"][i], int(pool["y"][i]), *pool["post"][i], rng)
                presentations += 1
            if (e + 1) in checkpoints:
                curve.append({"attempts": e + 1, "presentations": presentations, **evaluate(d, test)})
    else:
        judge = JUDGES[reward]
        Rs = []
        for a in range(1, attempts + 1):
            i = idx[rng.integers(len(idx))]
            Rs.append(d.attempt(pool["P"][i], int(pool["y"][i]), *pool["post"][i], rng, judge))
            if a in checkpoints:
                curve.append({"attempts": a, "mean_reward_recent": float(np.mean(Rs[-500:])), **evaluate(d, test)})
    return {"model": kind, "reward": reward, "n_demo": n_demo, "rep": rep, "hp": hp, "curve": curve,
            "seconds": time.perf_counter() - t0}


def _tune_job(args):
    kind, reward, cfg, attempts, checkpoints = args
    hp = sample_drawer_hp(kind, make_rng(SEED, "draw-hp", kind, cfg))
    scores = []
    for r, pool in enumerate(DATA["tune"]):
        res = run(kind, hp, pool, 1, attempts, reward, r, checkpoints, DATA["val"], tag="tune")
        scores.append(np.mean([c["score"] for c in res["curve"][1:]]))  # aire sous la courbe
    return (kind, reward), hp, float(np.mean(scores))


def _final_job(args):
    kind, reward, n_demo, hp, rep, attempts, checkpoints = args
    return run(kind, hp, DATA["pools"][rep], n_demo, attempts, reward, rep, checkpoints, DATA["test"])


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
        args.reps, args.attempts, args.n_configs = 2, 1000, 2
    t0 = time.time()
    DATA = build_data(args.reps, 4, 50)
    enc = get_encoder("orientations")
    nz = Normalizer(enc(make_set("L2", 300, SEED, "norm").X))
    v = make_set("L2", 25, SEED, "val-draw")
    DATA["val"] = {"P": nz(enc(v.X)), "y": v.y, "post": [posture(x) for x in v.X]}
    cps = sorted({int(args.attempts * f) for f in (0.125, 0.25, 0.5, 0.75, 1.0)})
    tune_attempts = args.attempts // 2
    tune_cps = sorted({int(tune_attempts * f) for f in (0.25, 0.5, 0.75, 1.0)})

    # Réglage (même budget de configurations pour chaque élève et chaque mode de jugement)
    jobs = [(k, r, c, tune_attempts, tune_cps) for k, r in [("G1-reinforce", "presque"), ("G1-reinforce", "pouce"),
                                                             ("G2-trace", "presque"), ("G2-trace", "pouce")]
            for c in range(args.n_configs)]
    jobs += [("G0-imitation", "geste", c, 40, [10, 20, 30, 40]) for c in range(args.n_configs)]
    with Pool(args.workers) as pool:
        tuned = pool.map(_tune_job, jobs)
    best = {}
    for key, hp, s in tuned:
        if key not in best or s > best[key][1]:
            best[key] = (hp, s)
    print(f"réglage : {time.time() - t0:.0f}s", {k: (v[0], round(v[1], 3)) for k, v in best.items()}, flush=True)

    # Évaluation finale
    jobs = []
    for (kind, reward), (hp, _) in best.items():
        for n_demo, rep in product([1, 4], range(args.reps)):
            if kind == "G0-imitation":
                jobs.append((kind, reward, n_demo, hp, rep, 40, [1, 2, 5, 10, 20, 40]))
            else:
                jobs.append((kind, reward, n_demo, hp, rep, args.attempts, cps))
    with Pool(args.workers) as pool:
        results = pool.map(_final_job, jobs)
    print(f"évaluation : {time.time() - t0:.0f}s", flush=True)

    # Résumé
    summary = {}
    for (kind, reward) in best:
        for n_demo in (1, 4):
            rs = [r for r in results if r["model"] == kind and r["reward"] == reward and r["n_demo"] == n_demo]
            pts = {}
            for r in rs:
                for c in r["curve"]:
                    pts.setdefault(c["attempts"], []).append(c)
            summary[f"{kind}|{reward}|{n_demo}"] = {
                str(a): {"acc": float(np.mean([c["acc"] for c in cs])),
                         "acc_ci95": bootstrap_ci([c["acc"] for c in cs], n_boot=2000),
                         "acc_carre": float(np.mean([c["acc_carre"] for c in cs])),
                         "acc_cercle": float(np.mean([c["acc_cercle"] for c in cs])),
                         "score": float(np.mean([c["score"] for c in cs])),
                         "frac_reps_acc>=0.9": float(np.mean([c["acc"] >= 0.9 for c in cs])), "n": len(cs)}
                for a, cs in sorted(pts.items())}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"seed": SEED, "git_commit": commit, "reps": args.reps, "attempts": args.attempts,
           "best_hyperparameters": {"|".join(k): {"hp": v[0], "tune_score": v[1]} for k, v in best.items()},
           "summary": summary, "runs": results, "duration_s": time.time() - t0}
    out_dir = ROOT / "results" / "step3"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "drawing_quick.json" if args.quick else "drawing.json"
    with open(out_dir / name, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    for k, v in summary.items():
        print(k, " ".join(f"{a}:{x['acc']:.2f}({x['score']:.2f})" for a, x in v.items()))


if __name__ == "__main__":
    main()
