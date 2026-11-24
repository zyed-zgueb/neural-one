"""Figures de l'étape 4."""
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
from sll.analysis.gallery import INK, INK2  # noqa: E402,F401

R = ROOT / "results" / "step4"
F = ROOT / "docs" / "figures"


def classification():
    S = json.loads((R / "shadow_classification.json").read_text())
    A = json.loads((R / "ablation_shadowC.json").read_text())
    t, budgets = S["table"], S["budgets"]
    series = [("M2-lecture|P1", "#8a8a85", "-", "local, 1 passage, sans mémoire"),
              ("Replay|R19", "#eda100", "-", "local + replay"),
              ("Shadow-C|R19", "#1baf7a", "-", "local + Shadow C"),
              ("M0-backprop|P1", "#2a78d6", (0, (4, 3)), "rétropropagation, 1 passage"),
              ("M0+replay|R19", "#2a78d6", "-", "rétropropagation + replay")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.axhline(0.9, color="#c9c8c1", lw=1, ls=(0, (3, 3)))
    for key, col, ls, lab in series:
        mu = [t[f"{key}|{n}"]["mean"] for n in budgets]
        lo = [t[f"{key}|{n}"]["ci95"][0] for n in budgets]
        hi = [t[f"{key}|{n}"]["ci95"][1] for n in budgets]
        ax.fill_between(budgets, lo, hi, color=col, alpha=0.13, lw=0)
        ax.plot(budgets, mu, ls=ls, marker="o", color=col, lw=2, ms=4, mec="white", mew=1, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets, [str(b) for b in budgets])
    ax.set_xlabel("exemples par forme (chacun vécu une seule fois)")
    ax.set_ylabel("précision sur formes jamais vues")
    ax.set_ylim(0.55, 1.01)
    ax.set_title("20 mises à jour par expérience (19 réactivations) — 100 répétitions", fontsize=9.5, color=INK)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.grid(axis="y", color="#ecebe6", lw=0.8)

    ax = axes[1]
    e = A["R19|N8"]
    order = ["C complet", "C sans indice partiel", "C sans mélange", "C sans renforcement", "C sans oubli",
             "Replay + indice partiel"]
    labels = ["Shadow C complète", "  … sans rappel partiel", "  … sans mélange", "  … sans renforcement",
              "  … sans oubli", "replay + rappel partiel seul"]
    for i, k in enumerate(order):
        d = e[k]["diff_vs_replay"] * 100
        lo, hi = (np.array(e[k]["diff_ci95"]) * 100)
        col = "#1baf7a" if lo > 0 else ("#e34948" if hi < 0 else "#8a8a85")
        ax.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
        ax.plot(d, i, "o", color=col, ms=7, mec="white", mew=1.5)
        ax.text(hi + 0.4, i, f"{d:+.1f}", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color="#8a8a85", lw=1)
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("différence avec le replay (points de précision, IC 95 %)")
    ax.set_title("D'où vient le gain ? (8 exemples par forme)", fontsize=9.5, color=INK)
    ax.grid(axis="x", color="#ecebe6", lw=0.8)
    fig.suptitle("Reconnaître avec des exemples vécus une seule fois : la mémoire réactivée", fontsize=10.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(F / "step4_reconnaissance.png", dpi=160)
    plt.close(fig)


def drawing():
    S = json.loads((R / "shadow_drawing.json").read_text())
    summ = S["summary"]
    rows = [("G2-trace|presque", "G2 trace classique"), ("G3-shadow-motrice|presque", "G3 Shadow motrice"),
            ("G5-fermeture|presque", "G3 + sensation de fermeture"), ("G5-gabarit|presque", "G3 + gabarit intérieur"),
            ("G5-gabarit+fermeture|presque", "G3 + gabarit + fermeture"),
            ("G5-gabarit-brouillé|presque", "  contrôle : gabarit brouillé"),
            ("G1-reinforce|presque", "G1 rétropropagation"), ("G1+interne|presque", "G1 + gabarit + fermeture"),
            ("G1+exploration|presque", "G1 + bonus d'exploration"),
            ("G3-shadow-motrice|pouce", "G3, pouce seul"), ("G5-gabarit+fermeture|pouce", "G3 + gabarit + fermeture, pouce seul")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, nd in zip(axes, (1, 4)):
        for i, (k, lab) in enumerate(rows):
            s = summ[f"{k}|{nd}"]
            last = list(s["curve"].values())[-1]
            ax.plot([last["acc_carre"], last["acc_cercle"]], [i, i], color="#d8d7d0", lw=2, zorder=1)
            ax.plot(last["acc_cercle"], i, "o", color="#2a78d6", ms=7, mec="white", mew=1.5, zorder=2,
                    label="cercles" if i == 0 else None)
            ax.plot(last["acc_carre"], i, "s", color="#eb6834", ms=7, mec="white", mew=1.5, zorder=3,
                    label="carrés" if i == 0 else None)
        ax.set_yticks(range(len(rows)), [r[1] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("dessins acceptés (formes jamais vues), après 8 000 essais")
        ax.set_title(f"{nd} démonstration{'s' if nd > 1 else ''} par forme", fontsize=9.5, color=INK)
        ax.grid(axis="x", color="#ecebe6", lw=0.8)
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(f"Apprendre à dessiner avec la Shadow — {S['reps']} répétitions", fontsize=10.5, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(F / "step4_dessin.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    if (R / "ablation_shadowC.json").exists():
        classification()
    if (R / "shadow_drawing.json").exists():
        drawing()
