"""Hewitt & Liang (2019) shuffled-label control task for probing selectivity.

    python -m secpo_probe.control_task --acts runs/real

Trains the exact same L2-regularized logistic regression probe (`fit_probe`) on
activations whose benign/injected labels are randomly flipped per twin pair,
and evaluates on the test split. Comparing real test accuracy against control
test accuracy measures probe selectivity (Real - Control) and rules out
high-dimensional memorization.

Writes <acts>/probes/control_results.json, control_results.csv, and
selectivity_by_layer.{png,pdf}.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .acts import MODEL_STYLE, Split
from .plot_layers import INK, MUTED, style_axis
from .train_probes import fit_probe


def flip_pair_labels(labels: np.ndarray, pair_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly swap labels (0 <-> 1) for half of the twin pairs, keeping 50/50 balance."""
    uniq, inv = np.unique(pair_ids, return_inverse=True)
    flip_pair = rng.random(len(uniq)) < 0.5
    return np.where(flip_pair[inv], 1 - labels, labels).astype(np.int64)


def run_control_model(model: str, args, device) -> list[dict]:
    train = Split(args.acts, model, args.train)
    test = Split(args.acts, model, args.test)
    rng = np.random.default_rng(args.seed)
    shuffled_trains = [
        torch.tensor(flip_pair_labels(train.labels, train.pair_ids, rng),
                     dtype=torch.float32, device=device)
        for _ in range(args.n_shuffles)
    ]
    y_te = torch.tensor(test.labels, dtype=torch.float32, device=device)

    rows = []
    for l in range(train.n_layers):
        X_tr = torch.from_numpy(train.layer(l)).to(device)
        X_te = torch.from_numpy(test.layer(l)).to(device)
        mu, sd = X_tr.mean(0), X_tr.std(0)
        sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
        X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd

        tr_accs, te_accs, te_aucs = [], [], []
        for s_idx, y_tr_ctrl in enumerate(shuffled_trains):
            w, b = fit_probe(X_tr, y_tr_ctrl, args.seed + s_idx + 1, args.l2, args.max_iter)
            logit_tr = X_tr @ w + b
            logit_te = X_te @ w + b
            tr_accs.append(((logit_tr > 0).float() == y_tr_ctrl).float().mean().item())
            te_accs.append(((logit_te > 0).float() == y_te).float().mean().item())
            te_aucs.append(float(roc_auc_score(test.labels, logit_te.cpu().numpy())))

        rows.append({
            "model": model,
            "layer": l,
            "ctrl_train_acc": float(np.mean(tr_accs)),
            "ctrl_test_acc": float(np.mean(te_accs)),
            "ctrl_test_acc_std": float(np.std(te_accs)),
            "ctrl_test_auroc": float(np.mean(te_aucs)),
        })
        print(
            f"  [control] {model} layer {l:2d}  "
            f"ctrl_train {rows[-1]['ctrl_train_acc']:.3f}  "
            f"ctrl_test {rows[-1]['ctrl_test_acc']:.3f}  "
            f"ctrl_auroc {rows[-1]['ctrl_test_auroc']:.3f}",
            flush=True,
        )
    return rows


def plot_selectivity(merged_rows: list[dict], models: list[str], out_path: str, title_note: str) -> None:
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
    )
    for m in models:
        rows = sorted((r for r in merged_rows if r["model"] == m), key=lambda r: r["layer"])
        x = np.array([r["layer"] for r in rows])
        real_acc = np.array([r["real_test_acc"] for r in rows])
        ctrl_acc = np.array([r["ctrl_test_acc"] for r in rows])
        sel = np.array([r["selectivity"] for r in rows])
        st = MODEL_STYLE.get(m, {"color": MUTED, "label": m, "marker": "o"})

        ax.plot(x, real_acc, color=st["color"], marker=st["marker"], markersize=4,
                linewidth=2, label=f"{st['label']} (real)")
        ax.plot(x, ctrl_acc, color=st["color"], linestyle="--", linewidth=1.6, alpha=0.75,
                label=f"{st['label']} (shuffled control)")
        ax2.plot(x, sel, color=st["color"], marker=st["marker"], markersize=3.5,
                 linewidth=1.8, label=st["label"])

    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax.set_ylabel("probe test accuracy", color=INK)
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, loc="best", fontsize=8)
    title = "Hewitt & Liang control task: real vs. shuffled-label probe accuracy"
    ax.set_title(title + (f"\n{title_note}" if title_note else ""), color=INK, fontsize=11, loc="left")
    style_axis(ax)

    ax2.axhline(0.0, color=MUTED, linewidth=1)
    ax2.set_ylabel("selectivity\n(real − control)", color=INK)
    ax2.set_xlabel("hidden-state index (0 = embeddings, 32 = last block)", color=INK)
    ax2.set_xticks(x[::2])
    ax2.set_ylim(-0.08, 0.55)
    ax2.legend(frameon=False, loc="lower right", fontsize=8)
    style_axis(ax2)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--acts", required=True, help="root directory written by secpo_probe.extract")
    ap.add_argument("--results", default=None, help="real probe results.json (default: <acts>/probes/results.json)")
    ap.add_argument("--train", default="train")
    ap.add_argument("--test", default="test")
    ap.add_argument("--models", nargs=2, default=["base", "secpo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--n-shuffles", type=int, default=3, help="number of random label shuffles to average")
    ap.add_argument("--title-note", default="")
    args = ap.parse_args()

    probes_dir = os.path.join(args.acts, "probes")
    os.makedirs(probes_dir, exist_ok=True)
    results_path = args.results or os.path.join(probes_dir, "results.json")
    real_res = json.load(open(results_path))
    real_map = {(r["model"], r["layer"]): r["test_acc"] for r in real_res["rows"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctrl_rows = []
    for m in args.models:
        ctrl_rows += run_control_model(m, args, device)

    for r in ctrl_rows:
        r["real_test_acc"] = real_map[(r["model"], r["layer"])]
        r["selectivity"] = r["real_test_acc"] - r["ctrl_test_acc"]

    json_out = os.path.join(probes_dir, "control_results.json")
    csv_out = os.path.join(probes_dir, "control_results.csv")
    plot_out = os.path.join(probes_dir, "selectivity_by_layer.png")
    json.dump({"n_shuffles": args.n_shuffles, "rows": ctrl_rows}, open(json_out, "w"), indent=2)
    with open(csv_out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(ctrl_rows[0]))
        wr.writeheader()
        wr.writerows(ctrl_rows)

    plot_selectivity(ctrl_rows, args.models, plot_out, args.title_note)
    print(f"wrote {json_out} and {csv_out}")


if __name__ == "__main__":
    main()
