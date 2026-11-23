"""Figures de l'étape 3 à partir des résultats JSON (results/step3/)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sll.analysis.gallery import INK, INK2  # noqa: E402,F401  (applique le style commun)

COLORS = {"M0-backprop": "#2a78d6", "M1-reinforce": "#eb6834", "M2-local": "#1baf7a", "M2-lecture": "#eda100",
          "1-PPV": "#e87ba4", "prototype": "#008300", "non-entraîné": "#8a8a85"}
LABELS = {"M0-backprop": "M0 rétropropagation (étiquette)", "M1-reinforce": "M1 REINFORCE (récompense)",
          "M2-local": "M2 local + trace (récompense)", "M2-lecture": "M2 lecture seule (récompense)",
          "1-PPV": "1 plus proche voisin (mémoire)", "prototype": "prototype (moyenne)",
          "non-entraîné": "non entraîné"}
ENC_TITLES = {"orientations": "histogramme d'orientations (E3)", "edges": "orientations localisées (E2)",
              "pixels": "pixels bruts (E0)"}


def classification_figure(path_json: Path, passes: int, out: Path):
    S = json.loads(path_json.read_text())
    table = S["table"]
    budgets = S["budgets"]
    encs = S["encoders"]
    fig, axes = plt.subplots(1, len(encs), figsize=(4.1 * len(encs), 4.2), sharey=True)
    for ax, e in zip(axes, encs):
        ax.axhline(0.9, color="#c9c8c1", lw=1, ls=(0, (3, 3)))
        ax.axhline(0.5, color="#c9c8c1", lw=1)
        for m, col in COLORS.items():
            key = lambda n: f"{e}|{passes}|normal|{m}|{n}"  # noqa: E731
            if key(1) not in table:
                continue
            mu = np.array([table[key(n)]["mean"] for n in budgets])
            lo = np.array([table[key(n)]["ci95"][0] for n in budgets])
            hi = np.array([table[key(n)]["ci95"][1] for n in budgets])
            ax.fill_between(budgets, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(budgets, mu, "-o", color=col, lw=2, ms=4, mec="white", mew=1, label=LABELS[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_xlabel("exemples par forme")
        ax.set_title(ENC_TITLES[e], fontsize=9.5, color=INK)
        ax.set_ylim(0.45, 1.01)
        ax.grid(axis="y", color="#ecebe6", lw=0.8)
    axes[0].set_ylabel("précision sur formes jamais vues")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    regime = "1 passage (en ligne)" if passes == 1 else f"{passes} passages"
    fig.suptitle(f"Reconnaître cercle / carré avec peu d'exemples — niveau L2, {regime}, "
                 f"{S['reps']} répétitions (bandes : IC 95 %)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def drawing_figure(path_json: Path, out: Path):
    S = json.loads(path_json.read_text())
    summ = S["summary"]
    series = [("G1-reinforce|presque", "#eb6834", "-", "G1 REINFORCE, « presque ça »"),
              ("G1-reinforce|pouce", "#eb6834", (0, (4, 3)), "G1 REINFORCE, pouce seul"),
              ("G2-trace|presque", "#1baf7a", "-", "G2 local + trace, « presque ça »"),
              ("G2-trace|pouce", "#1baf7a", (0, (4, 3)), "G2 local + trace, pouce seul")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, n_demo in zip(axes[:2], (1, 4)):
        for key, col, ls, lab in series:
            k = f"{key}|{n_demo}"
            xs = [int(a) for a in summ[k]]
            ys = [summ[k][a]["acc"] for a in summ[k]]
            ax.plot(xs, ys, ls=ls, marker="o", color=col, lw=2, ms=4, mec="white", mew=1, label=lab)
        ax.set_title(f"{n_demo} démonstration{'s' if n_demo > 1 else ''} par forme", fontsize=9.5, color=INK)
        ax.set_xlabel("essais de l'élève (dessins jugés)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color="#ecebe6", lw=0.8)
    axes[0].set_ylabel("dessins acceptés (formes jamais vues)")
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    # ce qui a été appris, par forme (fin d'apprentissage, 4 démonstrations)
    ax = axes[2]
    rows = [("G0-imitation|geste|4", "G0 imitation du geste"), ("G1-reinforce|presque|4", "G1 « presque ça »"),
            ("G2-trace|presque|4", "G2 « presque ça »"), ("G1-reinforce|pouce|4", "G1 pouce seul"),
            ("G2-trace|pouce|4", "G2 pouce seul")]
    for i, (k, lab) in enumerate(rows):
        last = list(summ[k].values())[-1]
        ax.plot([last["acc_carre"], last["acc_cercle"]], [i, i], color="#d8d7d0", lw=2, zorder=1)
        ax.plot(last["acc_cercle"], i, "o", color="#2a78d6", ms=8, mec="white", mew=1.5, zorder=2,
                label="cercles" if i == 0 else None)
        ax.plot(last["acc_carre"], i, "s", color="#eb6834", ms=8, mec="white", mew=1.5, zorder=3,
                label="carrés" if i == 0 else None)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("dessins acceptés en fin d'apprentissage")
    ax.set_title("Par forme (4 démonstrations par forme)", fontsize=9.5, color=INK)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.grid(axis="x", color="#ecebe6", lw=0.8)
    fig.suptitle(f"Apprendre à dessiner sans Shadow — {S['reps']} répétitions, formes de test jamais vues",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    R = ROOT / "results" / "step3"
    F = ROOT / "docs" / "figures"
    if (R / "classification.json").exists():
        classification_figure(R / "classification.json", 1, F / "step3_classification_1passage.png")
        classification_figure(R / "classification.json", 20, F / "step3_classification_20passages.png")
    if (R / "drawing.json").exists():
        drawing_figure(R / "drawing.json", F / "step3_dessin.png")
