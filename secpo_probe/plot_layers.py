"""Plot probe accuracy by layer for both models, and the gap between them.

    python -m secpo_probe.plot_layers --results runs/placeholder/probes/results.json

The final hidden-state index is taken after the model's final norm, so it is drawn
with a hollow marker and left out when the plot names the largest gap.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .acts import MODEL_STYLE

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def layer_axis_label(last: int) -> str:
    return f"hidden-state index (0 = embeddings, {last} = last block after the final norm)"


def hollow_final_point(ax, x, y, color):
    """Mark the final index, which includes the final norm, as different from the block outputs."""
    ax.plot(x[-1], y[-1], marker="o", markersize=6, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.6, linestyle="none", zorder=3)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--metric", default="test_acc", choices=["test_acc", "test_auroc"])
    ap.add_argument("--out", default=None, help="default: next to results.json")
    ap.add_argument("--title-note", default="", help="extra text for the title, e.g. PLACEHOLDER DATA")
    args = ap.parse_args()
    res = json.load(open(args.results))
    models = res["config"]["models"]
    name = {"test_acc": "accuracy", "test_auroc": "AUROC"}[args.metric]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    series = {}
    for m in models:
        rows = sorted((r for r in res["rows"] if r["model"] == m), key=lambda r: r["layer"])
        x = np.array([r["layer"] for r in rows])
        y = np.array([r[args.metric] for r in rows])
        st = MODEL_STYLE.get(m, {"color": MUTED, "label": m, "marker": "o"})
        if args.metric == "test_acc":
            ax.fill_between(x, [r["test_acc_lo"] for r in rows], [r["test_acc_hi"] for r in rows],
                            color=st["color"], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=st["color"], marker=st["marker"], markersize=4, linewidth=2, label=st["label"])
        hollow_final_point(ax, x, y, st["color"])
        series[m] = y
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax.annotate("chance", (x[-1], 0.5), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8, color=MUTED)
    ax.set_ylabel(f"probe test {name}", color=INK)
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    title = f"Linear probe: benign vs injected, test {name} by layer"
    ax.set_title(title + (f"\n{args.title_note}" if args.title_note else ""), color=INK, fontsize=11, loc="left")
    style_axis(ax)

    diff = series[models[1]] - series[models[0]]
    ax2.axhline(0, color=MUTED, linewidth=1)
    if args.metric == "test_acc":
        s = res["summary"]
        ax2.fill_between(x, s["diff_lo"], s["diff_hi"], color=MUTED, alpha=0.18, linewidth=0,
                         label="95% interval (bootstrap over test pairs)")
        ax2.legend(frameon=False, loc="best", fontsize=8)
    ax2.plot(x, diff, color=INK, marker="o", markersize=3.5, linewidth=1.8)
    hollow_final_point(ax2, x, diff, INK)
    k = int(np.abs(diff[:-1]).argmax())  # the final index includes the final norm, so it is not a candidate
    note = f"largest gap before the final norm: index {k} ({diff[k]:+.3f})"
    onset = res["summary"].get("onset_layer") if args.metric == "test_acc" else None
    if onset is not None:
        for a in (ax, ax2):
            a.axvline(onset, color=MUTED, linewidth=0.8, alpha=0.6, zorder=0)
        note = f"gap opens at index {onset} ({diff[onset]:+.3f}); " + note
    ax2.text(0.01, 0.04, note, transform=ax2.transAxes, ha="left", va="bottom", fontsize=8, color=INK)
    ax2.set_ylabel(f"{name} gap\n({models[1]} − {models[0]})", color=INK)
    ax2.set_xlabel(layer_axis_label(int(x[-1])) + "; hollow point = that final index", color=INK)
    ax2.set_xticks(x[::2])
    ax2.margins(y=0.35)
    style_axis(ax2)

    out = args.out or os.path.join(os.path.dirname(args.results), f"{args.metric}_by_layer.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
