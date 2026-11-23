"""2D sanity plot: do benign and injected prompts separate at a given layer?

    python -m secpo_probe.plot_2d --acts runs/placeholder --split train --layers 4 16 28

One row per model, one column per layer. Run this before training any probe.

--method pca       top two principal components (unsupervised; uses no labels).
--method meandiff  x = projection on the difference of the class means, fitted on a
                   random half of the points and drawn on the other half;
                   y = first principal component orthogonal to that direction.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .acts import LABEL_STYLE, MODEL_STYLE, Split

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def top_pcs(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k principal directions of centred X and their explained-variance ratios."""
    if X.shape[0] >= X.shape[1]:
        evals, evecs = np.linalg.eigh(X.T @ X)
        comps = evecs[:, ::-1][:, :k].T
    else:
        evals, evecs = np.linalg.eigh(X @ X.T)
        comps = (X.T @ evecs[:, ::-1][:, :k]).T
        comps /= np.linalg.norm(comps, axis=1, keepdims=True) + 1e-12
    evals = np.clip(evals[::-1], 0, None)
    return comps, evals[:k] / max(evals.sum(), 1e-12)


def project(X: np.ndarray, y: np.ndarray, method: str, rng) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = X - X.mean(0)
    if method == "pca":
        comps, ratio = top_pcs(X, 2)
        return X @ comps.T, y, [f"PC1 ({ratio[0]:.0%} var.)", f"PC2 ({ratio[1]:.0%} var.)"]
    perm = rng.permutation(len(X))
    fit, show = perm[: len(X) // 2], perm[len(X) // 2:]
    d = X[fit][y[fit] == 1].mean(0) - X[fit][y[fit] == 0].mean(0)
    d /= np.linalg.norm(d) + 1e-12
    resid = X[fit] - np.outer(X[fit] @ d, d)
    comps, _ = top_pcs(resid, 1)
    Z = np.stack([X[show] @ d, X[show] @ comps[0]], axis=1)
    return Z, y[show], ["class-mean difference direction (held-out half)", "orthogonal PC1"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--acts", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 16, 28])
    ap.add_argument("--models", nargs="+", default=["base", "secpo"])
    ap.add_argument("--method", choices=["pca", "meandiff"], default="pca")
    ap.add_argument("--max-points", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title-note", default="")
    args = ap.parse_args()

    fig, axes = plt.subplots(len(args.models), len(args.layers), squeeze=False,
                             figsize=(3.6 * len(args.layers), 3.4 * len(args.models)))
    for i, m in enumerate(args.models):
        sp = Split(args.acts, m, args.split)
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.permutation(len(sp.labels))[: args.max_points * 2])
        for j, l in enumerate(args.layers):
            ax = axes[i, j]
            Z, y, names = project(sp.layer(l)[keep], sp.labels[keep], args.method, rng)
            for lab, st in LABEL_STYLE.items():
                pts = Z[y == lab][: args.max_points]
                ax.scatter(pts[:, 0], pts[:, 1], s=14, c=st["color"], marker=st["marker"],
                           alpha=0.6, linewidths=0.4, edgecolors="white", label=st["label"])
            ax.set_title(f"{MODEL_STYLE.get(m, {'label': m})['label']}\nlayer {l}",
                         fontsize=9, color=INK, loc="left")
            ax.set_xlabel(names[0], fontsize=8, color=MUTED)
            ax.set_ylabel(names[1], fontsize=8, color=MUTED)
            ax.tick_params(colors=MUTED, length=0, labelsize=7)
            for s in ax.spines.values():
                s.set_color(GRID)
    axes[0, -1].legend(frameon=False, fontsize=8, loc="best")
    head = f"Last-token hidden states, {args.method} projection, split '{args.split}'"
    fig.suptitle(head + (f" — {args.title_note}" if args.title_note else ""), fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.out or os.path.join(args.acts, f"sanity2d_{args.method}_{args.split}.png")
    fig.savefig(out, dpi=180, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
