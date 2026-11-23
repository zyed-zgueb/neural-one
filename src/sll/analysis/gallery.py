"""Figures de l'étape 2 (galeries et résumés), en PNG statiques pour les comptes rendus."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..data.render import to_display  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"
INK, INK2 = "#1f1f1e", "#5f5e58"
CLASS_COLOR = {0: BLUE, 1: ORANGE}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c9c8c1", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
})


def _img(ax, ink, title=None, color=INK):
    ax.imshow(to_display(ink), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#d8d7d0")
    if title:
        ax.set_title(title, fontsize=7, color=color, pad=2)


def level_gallery(rows: dict[str, list], path: str, per_class: int = 6):
    """rows : nom de niveau -> liste de Shape (cercles et carrés mélangés)."""
    names = list(rows)
    fig, axes = plt.subplots(len(names), 2 * per_class, figsize=(2 * per_class * 0.75, len(names) * 0.95))
    for r, name in enumerate(names):
        circ = [s for s in rows[name] if s.params.kind == 0][:per_class]
        sq = [s for s in rows[name] if s.params.kind == 1][:per_class]
        for c, s in enumerate(circ + sq):
            _img(axes[r, c], s.ink)
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9, color=INK)
    axes[0, per_class // 2].set_title("cercles", color=BLUE, fontsize=10, loc="left")
    axes[0, per_class + per_class // 2].set_title("carrés", color=ORANGE, fontsize=10, loc="left")
    fig.suptitle("Formes du maître par niveau de difficulté", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gesture_gallery(items: list[tuple[np.ndarray, str, bool]], path: str, title: str, ncols: int = 8):
    """items : (image d'encre, légende, accepté ?)."""
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.0, nrows * 1.15))
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        ink, label, ok = items[k]
        _img(ax, ink, ("✓ " if ok else "✗ ") + label, color=INK if ok else INK2)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def probe_chart(results: dict[str, dict], path: str, keys: list[str], labels: list[str]):
    """Précision des sondes anti-raccourcis par niveau (petits multiples, une sonde par panneau)."""
    levels = list(results)
    fig, axes = plt.subplots(1, len(keys), figsize=(2.3 * len(keys), 2.4), sharey=True)
    for ax, k, lab in zip(axes, keys, labels):
        acc = [results[lv][k]["accuracy"] for lv in levels]
        lo = [results[lv][k]["ci95"][0] for lv in levels]
        hi = [results[lv][k]["ci95"][1] for lv in levels]
        x = np.arange(len(levels))
        ax.axhline(0.5, color=GRAY, lw=1, ls=(0, (3, 3)))
        ax.vlines(x, lo, hi, color=INK2, lw=1.5)
        ax.plot(x, acc, "o", ms=6, color=BLUE, mec="white", mew=1.5)
        ax.set_xticks(x, levels)
        ax.set_ylim(0.4, 1.02)
        ax.set_title(lab, fontsize=9, color=INK)
        ax.grid(axis="y", color="#ecebe6", lw=0.8)
    axes[0].set_ylabel("précision de la sonde")
    fig.suptitle("Peut-on deviner la forme sans regarder la forme ? (0,5 = non)", fontsize=10.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def score_histograms(scores: dict[str, np.ndarray], path: str):
    """Distribution de l'encouragement gradué reçu par des gribouillis aléatoires."""
    names = list(scores)
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 2.4), sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, name in zip(np.atleast_1d(axes), names):
        s = scores[name]
        h, _ = np.histogram(s, bins=bins)
        ax.bar(bins[:-1], h / len(s), width=bins[1] - bins[0] - 0.006, align="edge",
               color=ORANGE if "carré" in name else BLUE)
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_xlabel("note du maître (0 = raté, 1 = parfait)")
        ax.set_yscale("log")
        ax.grid(axis="y", color="#ecebe6", lw=0.8)
    np.atleast_1d(axes)[0].set_ylabel("fraction des essais")
    fig.suptitle("« C'est presque ça » : notes reçues par 20 000 gribouillis au hasard", fontsize=10.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160)
    plt.close(fig)
