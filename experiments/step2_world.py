"""Étape 2 — validation du monde : formes, sondes, maître, stylo, élèves aléatoires.

Usage : PYTHONPATH=src python experiments/step2_world.py [--seed 0] [--n-random 20000]
Sorties : results/step2/summary.json, docs/figures/step2_*.png
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

from sll.analysis import gallery  # noqa: E402
from sll.analysis.probes import run_probes  # noqa: E402
from sll.analysis.stats import rule_of_three_upper, wilson_ci  # noqa: E402
from sll.core.rng import make_rng  # noqa: E402
from sll.data.datasets import few_shot_episode, leakage_report, make_set  # noqa: E402
from sll.data.render import render  # noqa: E402
from sll.data.shapes_generator import CIRCLE, SQUARE, ShapeParams, build_shape  # noqa: E402
from sll.world.pen import TURNS, TURNS_NO90, perfect_circle, perfect_square, random_gesture  # noqa: E402
from sll.world.teacher import CategoricalTeacher  # noqa: E402

LEVELS = ["L0", "L1", "L2", "L3", "L4", "OOD-test"]
TEACHER = CategoricalTeacher()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "inconnu"


# ------------------------------------------------------------------ élèves aléatoires

def _random_batch(args):
    seed, cond, n = args
    rng = make_rng(seed, "random-student", cond)
    turns = TURNS_NO90 if cond == "uniforme_sans90" else TURNS
    p_straight = 0.5 if cond == "souvent_tout_droit" else None
    out = {SQUARE: [], CIRCLE: []}
    acc = {SQUARE: 0, CIRCLE: 0}
    best = {SQUARE: (-1, None), CIRCLE: (-1, None)}
    for _ in range(n):
        g = random_gesture(rng, step=3.5, p_straight=p_straight, turns=turns)
        pts = g.points()
        for tgt in (SQUARE, CIRCLE):
            j = TEACHER.judge(pts, tgt)
            out[tgt].append(j.score)
            acc[tgt] += j.accepted
            if j.score > best[tgt][0]:
                best[tgt] = (j.score, (g.start.tolist(), g.heading, g.step, g.actions.tolist(), cond))
    return cond, {k: np.array(v) for k, v in out.items()}, acc, best


def random_students(seed: int, n: int, workers: int = 4):
    conds = ["uniforme", "souvent_tout_droit", "uniforme_sans90"]
    chunks = 8
    jobs = [(seed * 1000 + c, cond, n // chunks) for cond in conds for c in range(chunks)]
    with Pool(workers) as pool:
        res = pool.map(_random_batch, jobs)
    agg = {}
    for cond, scores, acc, best in res:
        a = agg.setdefault(cond, {"scores": {SQUARE: [], CIRCLE: []}, "acc": {SQUARE: 0, CIRCLE: 0},
                                  "best": {SQUARE: (-1, None), CIRCLE: (-1, None)}})
        for t in (SQUARE, CIRCLE):
            a["scores"][t].append(scores[t])
            a["acc"][t] += acc[t]
            if best[t][0] > a["best"][t][0]:
                a["best"][t] = best[t]
    for cond in agg:
        for t in (SQUARE, CIRCLE):
            agg[cond]["scores"][t] = np.concatenate(agg[cond]["scores"][t])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=20_000)
    ap.add_argument("--n-probe", type=int, default=1500, help="images par classe pour les sondes")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = ROOT / "results" / "step2"
    fig_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "git_commit": git_commit(), "n_random": args.n_random}

    # 1. Galerie des niveaux
    rows = {lv: make_set(lv, 6, args.seed, "gallery").shapes for lv in LEVELS}
    gallery.level_gallery(rows, str(fig_dir / "step2_niveaux.png"))

    # 2. Sondes anti-raccourcis
    probes = {}
    for lv in ["L1", "L2", "L3", "L4"]:
        s = make_set(lv, args.n_probe, args.seed, "probe")
        probes[lv] = run_probes(s.X, s.y)
    summary["probes"] = probes
    gallery.probe_chart(probes, str(fig_dir / "step2_sondes.png"),
                        ["encre_totale", "centroide_x", "epaisseur_moyenne", "NUISANCE_jointe", "GEOMETRIE_jointe"],
                        ["encre totale", "position (x)", "texture du trait", "nuisances jointes",
                         "compacité (géométrie)"])

    # 3. Fuite train/test
    leak = {}
    for lv in ["L1", "L2", "L3"]:
        tr, te = few_shot_episode(lv, 32, args.seed, repetition=0, n_test_per_class=500)
        r = leakage_report(tr, te)
        r.pop("distances")
        leak[lv] = r
    summary["leakage"] = leak

    # 4. Le maître juge les formes du générateur
    teacher_val = {}
    for lv in LEVELS:
        s = make_set(lv, 300, args.seed, "teacher-validation")
        res = {}
        for kind, name in ((CIRCLE, "cercle"), (SQUARE, "carré")):
            shapes = [x for x in s.shapes if x.params.kind == kind]
            ok = [TEACHER.judge(np.vstack([x.outline, x.outline[:1]]), kind).accepted for x in shapes]
            cross = [TEACHER.judge(np.vstack([x.outline, x.outline[:1]]), 1 - kind).accepted for x in shapes]
            res[name] = {"accepté_comme_sa_classe": float(np.mean(ok)),
                         "accepté_comme_autre_classe": float(np.mean(cross)), "n": len(shapes)}
        teacher_val[lv] = res
    summary["teacher_validation"] = teacher_val

    # 5. Élève « parfait » : gestes idéaux à toutes tailles et rotations
    rng = make_rng(args.seed, "perfect")
    perfect = {"carré": [], "cercle": []}
    items = []
    for i in range(200):
        side = rng.uniform(7, 20)
        g = perfect_square(rng.uniform(12, 20, 2), side, rotation=rng.uniform(0, np.pi / 2),
                           direction=int(rng.choice([-1, 1])))
        j = TEACHER.judge(g.points(), SQUARE)
        perfect["carré"].append(j.accepted)
        d = rng.uniform(9, 24)
        gc = perfect_circle(rng.uniform(12, 20, 2), d, direction=int(rng.choice([-1, 1])),
                            heading=rng.uniform(0, 2 * np.pi))
        jc = TEACHER.judge(gc.points(), CIRCLE)
        perfect["cercle"].append(jc.accepted)
        if i < 8:
            items.append((g.draw(rng.uniform(1, 2.5)), f"carré {side:.0f}px", j.accepted))
            items.append((gc.draw(rng.uniform(1, 2.5)), f"cercle {d:.0f}px", jc.accepted))
    summary["perfect_student"] = {k: float(np.mean(v)) for k, v in perfect.items()}

    # 6. Même moteur de rendu : geste parfait vs forme du générateur aux mêmes paramètres
    ious = []
    for i in range(50):
        side = rng.uniform(9, 16)
        rot = rng.uniform(0, np.pi / 2)
        c = rng.uniform(13, 19, 2)
        g = perfect_square(c, side, rotation=rot)
        p = ShapeParams(kind=SQUARE, cx=c[0], cy=c[1], perimeter=4 * side, thickness=1.5, rotation=rot,
                        ellipticity=0, radial_jitter=0, vertex_jitter=0, corner_rounding=0, noise_sigma=0,
                        fragment=0, shape_seed=0)
        a, b = g.draw(1.5), build_shape(p).ink
        ious.append(float(np.minimum(a, b).sum() / np.maximum(a, b).sum()))
    summary["pen_vs_generator_iou"] = {"mean": float(np.mean(ious)), "min": float(np.min(ious))}

    # 7. Élèves aléatoires
    agg = random_students(args.seed, args.n_random)
    rnd = {}
    for cond, a in agg.items():
        n = len(a["scores"][SQUARE])
        entry = {"n": n}
        for t, name in ((SQUARE, "carré"), (CIRCLE, "cercle")):
            k = a["acc"][t]
            sc = a["scores"][t]
            entry[name] = {
                "acceptés": int(k),
                "taux": k / n,
                "ic95": wilson_ci(k, n) if k else (0.0, rule_of_three_upper(n)),
                "note_moyenne": float(sc.mean()),
                "note_p99": float(np.quantile(sc, 0.99)),
                "note_max": float(sc.max()),
                "frac_note_sup_0.5": float((sc > 0.5).mean()),
            }
        rnd[cond] = entry
    summary["random_students"] = rnd
    gallery.score_histograms({"cible : carré (au hasard)": agg["uniforme"]["scores"][SQUARE],
                              "cible : cercle (au hasard)": agg["uniforme"]["scores"][CIRCLE]},
                             str(fig_dir / "step2_notes_hasard.png"))

    # galerie des gestes : parfaits + gribouillis + meilleurs gribouillis
    rng2 = make_rng(args.seed, "gallery-random")
    for _ in range(8):
        g = random_gesture(rng2, step=3.5)
        jq, jc = TEACHER.judge(g.points(), SQUARE), TEACHER.judge(g.points(), CIRCLE)
        items.append((g.draw(1.5), f"□{jq.score:.2f} ○{jc.score:.2f}", jq.accepted or jc.accepted))
    from sll.world.pen import Gesture
    for t, name in ((SQUARE, "□"), (CIRCLE, "○")):
        for cond in ("uniforme", "souvent_tout_droit"):
            sc, spec = agg[cond]["best"][t]
            if spec is None:
                continue
            start, heading, step, actions, c = spec
            turns = TURNS_NO90 if c == "uniforme_sans90" else TURNS
            g = Gesture(np.array(start), heading, step, np.array(actions), turns)
            items.append((g.draw(1.5), f"meilleur {name} {sc:.2f}", TEACHER.judge(g.points(), t).accepted))
    gallery.gesture_gallery(items, str(fig_dir / "step2_gestes.png"),
                            "Gestes de l'élève : parfaits (2 premières lignes), gribouillis, meilleurs gribouillis")

    summary["duration_s"] = time.time() - t0
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({k: v for k, v in summary.items() if k != "probes"}, indent=1, ensure_ascii=False, default=str))
    print({lv: {k: round(v["accuracy"], 3) for k, v in r.items()} for lv, r in probes.items()})


if __name__ == "__main__":
    main()
