"""Étape 4a bis — d'où vient l'avantage de Shadow C sur le replay ? Ablations, hyperparamètres réglés figés.

On retire un ingrédient à la fois de Shadow C (réglée) : indice partiel/bruité, mélange avec l'état
stocké, renforcement par le rappel, oubli (décroissance + choix selon la force). Et on ajoute
l'ingrédient seul au replay. 100 répétitions appariées, mêmes graines que l'étape 4a.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import step4_shadow_classification as S  # noqa: E402

from sll.analysis.stats import bootstrap_ci, holm, wilcoxon_signed_rank  # noqa: E402
from sll.core.network import MLP  # noqa: E402
from sll.core.rng import make_rng  # noqa: E402
from sll.models.shadow_classifiers import MemoryLearner  # noqa: E402

BEST = json.loads((ROOT / "results" / "step4" / "shadow_classification.json").read_text())["best_hyperparameters"]


def variants(R):
    hp = BEST[f"Shadow-C|R{R}"]["hp"]
    rp = BEST[f"Replay|R{R}"]["hp"]
    full = dict(lr=hp["lr"], decay=hp["decay"], cue_keep=hp["cue_keep"], cue_noise=hp["cue_noise"],
                blend=hp["blend"], recall_boost=hp["recall_boost"], selection="strength")
    neutral = dict(decay=1.0, cue_keep=1.0, cue_noise=0.0, blend=0.0, recall_boost=0.0, selection="uniform")
    return {
        "C complet": full,
        "C sans indice partiel": {**full, "cue_keep": 1.0, "cue_noise": 0.0},
        "C sans mélange": {**full, "blend": 0.0},
        "C sans renforcement": {**full, "recall_boost": 0.0},
        "C sans oubli": {**full, "decay": 1.0, "selection": "uniform"},
        "C neutre (= replay, lr de C)": {**full, **neutral},
        "Replay (réglé)": {**neutral, "lr": rp["lr"]},
        "Replay + indice partiel": {**neutral, "lr": rp["lr"], "cue_keep": hp["cue_keep"], "cue_noise": hp["cue_noise"]},
    }


def run(args):
    R, name, hp, rep = args
    PX, Py = S.DATA["pools"][rep]
    Xte, yte = S.DATA["test"]
    out = []
    for n in S.BUDGETS:
        X, y = S.prefix(PX, Py, n)
        net = MLP(X.shape[1], 64, 2, make_rng(S.SEED, "final", "init", rep))
        L = MemoryLearner(net, hp["lr"], R, "C", decay=hp["decay"], cue_keep=hp["cue_keep"],
                          cue_noise=hp["cue_noise"], blend=hp["blend"], recall_boost=hp["recall_boost"])
        L.memory.selection = hp["selection"]
        rng = make_rng(S.SEED, "final", "order", rep, n)
        for i in rng.permutation(len(y)):
            a, info = L.act(X[i], rng)
            L.feedback(info, 1.0 if a == y[i] else -1.0, int(y[i]))
        out.append((R, name, rep, n, float((L.predict(Xte) == yte).mean())))
    return out


if __name__ == "__main__":
    reps = 100
    S.DATA = S.build(reps, 1000)
    jobs = [(R, name, hp, rep) for R in S.RS for name, hp in variants(R).items() for rep in range(reps)]
    with Pool(4) as pool:
        rows = [x for rs in pool.map(run, jobs, chunksize=4) for x in rs]
    res = {}
    for R in S.RS:
        for n in (2, 8, 32):
            d = {}
            for r in rows:
                if r[0] == R and r[3] == n:
                    d.setdefault(r[1], {})[r[2]] = r[4]
            ref = d["Replay (réglé)"]
            pv, entry = {}, {}
            for name, v in d.items():
                x = np.array([v[i] for i in range(reps)])
                y = np.array([ref[i] for i in range(reps)])
                w = wilcoxon_signed_rank(x, y)
                entry[name] = {"mean": float(x.mean()), "diff_vs_replay": float((x - y).mean()),
                               "diff_ci95": bootstrap_ci(x - y, n_boot=2000), "p": w["p"]}
                if name != "Replay (réglé)":
                    pv[name] = w["p"]
            for k, p in holm(pv).items():
                entry[k]["p_holm"] = p
            res[f"R{R}|N{n}"] = entry
    with open(ROOT / "results" / "step4" / "ablation_shadowC.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    for k, e in res.items():
        print(k)
        for name, v in e.items():
            print("   ", name.ljust(30), f"{v['mean']:.3f}", f"{v['diff_vs_replay']:+.3f}", f"p_holm={v.get('p_holm', 1):.2g}")
