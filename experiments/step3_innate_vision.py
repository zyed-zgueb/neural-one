"""Étape 3c (exploratoire) — système visuel inné SV0 contre l'histogramme d'orientations E3.

Hyperparamètres fixés (non réglés) : comparaison **exploratoire**, à confirmer par une campagne
réglée et pré-enregistrée. Niveaux L2 (contours complets) et L4 (contours fragmentés).

Usage : PYTHONPATH=src python experiments/step3_innate_vision.py [--reps 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sll.analysis.probes import probe_accuracy  # noqa: E402
from sll.analysis.stats import bootstrap_ci  # noqa: E402
from sll.core.rng import make_rng  # noqa: E402
from sll.data.datasets import make_set  # noqa: E402
from sll.data.encoders import Normalizer, get_encoder  # noqa: E402
from sll.models.train import run_classification  # noqa: E402

SEED = 0
ENCODERS = ["orientations", "SV0", "SV0-carte", "SV0-orientation", "SV0-topologie"]
MODELS = {"M0-backprop": {"lr": 0.05}, "M2-lecture": {"lr": 0.1, "lam": 0.0}, "1-PPV": {}}
BUDGETS = [1, 2, 4, 8, 16, 32]
DATA = {}


def build(level, reps):
    out = {}
    norm = make_set(level, 200, SEED, "norm")
    test = make_set(level, 500, SEED, "test")
    pools = [make_set(level, 32, SEED, "train", f"rep{r}") for r in range(reps)]
    for e in ENCODERS:
        enc = get_encoder(e)
        nz = Normalizer(enc(norm.X))
        out[e] = {"test": (nz(enc(test.X)), test.y), "pools": [(nz(enc(p.X)), p.y) for p in pools],
                  "raw_test": enc(test.X)}
    return out


def job(args):
    level, e, m, rep = args
    PX, Py = DATA[level][e]["pools"][rep]
    Xte, yte = DATA[level][e]["test"]
    res = []
    for n in BUDGETS:
        idx = np.sort(np.concatenate([np.where(Py == c)[0][:n] for c in (0, 1)]))
        r = run_classification(m, MODELS[m], PX[idx], Py[idx], Xte, yte, 20, make_rng(SEED, "o", rep, n),
                               make_rng(SEED, "i", rep))
        res.append((level, e, m, n, rep, r["acc"]))
    return res


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    DATA = {lv: build(lv, args.reps) for lv in ("L2", "L4")}
    # Sonde : la topologie seule ne doit pas séparer cercle et carré
    probes = {}
    for lv in ("L2", "L4"):
        F = DATA[lv]["SV0-topologie"]["raw_test"]
        y = DATA[lv]["SV0-topologie"]["test"][1]
        probes[lv] = probe_accuracy(F, y)
    jobs = [(lv, e, m, r) for lv in ("L2", "L4") for e in ENCODERS for m in MODELS for r in range(args.reps)]
    with Pool(args.workers) as pool:
        rows = [x for rs in pool.map(job, jobs) for x in rs]
    table = {}
    for lv in ("L2", "L4"):
        for e in ENCODERS:
            for m in MODELS:
                for n in BUDGETS:
                    a = np.array([r[5] for r in rows if r[:4] == (lv, e, m, n)])
                    table[f"{lv}|{e}|{m}|{n}"] = {"mean": float(a.mean()), "ci95": bootstrap_ci(a, n_boot=2000)}
    out = {"reps": args.reps, "probes_topologie_seule": probes, "table": table,
           "note": "exploratoire : hyperparamètres fixés, non réglés"}
    (ROOT / "results" / "step3").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / "step3" / "innate_vision.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("sondes topologie seule :", {k: round(v["accuracy"], 3) for k, v in probes.items()})
    for lv in ("L2", "L4"):
        for m in MODELS:
            for e in ENCODERS:
                print(lv, m.ljust(12), e.ljust(16), " ".join(f"{table[f'{lv}|{e}|{m}|{n}']['mean']:.2f}" for n in BUDGETS))


if __name__ == "__main__":
    main()
