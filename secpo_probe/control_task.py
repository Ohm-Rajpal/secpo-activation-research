"""Sanity check: can the probe memorize?

    python -m secpo_probe.control_task --acts runs/real_crossed

Fits the same probe as train_probes on the train activations, after swapping the
benign/injected labels inside a random half of the twin pairs. Those labels carry no
information about injections, so any train accuracy above 0.5 is memorization. If the
probe fits them well, its train accuracy on the real labels proves nothing, and only
accuracy on data it was not trained on counts.

This checks the method; it is not a finding about the models, and it is not Hewitt &
Liang's selectivity, which needs control labels that can be scored on held-out data.

Writes <acts>/probes/control_results.json, control_results.csv and
control_capacity_by_layer.{png,pdf}.
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

from .acts import MODEL_STYLE, Split
from .plot_layers import INK, MUTED, hollow_final_point, layer_axis_label, style_axis
from .train_probes import fit_probe, standardize


def flip_pair_labels(labels: np.ndarray, pair_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly swap labels (0 <-> 1) for half of the twin pairs, keeping 50/50 balance."""
    uniq, inv = np.unique(pair_ids, return_inverse=True)
    flip_pair = rng.random(len(uniq)) < 0.5
    return np.where(flip_pair[inv], 1 - labels, labels).astype(np.int64)


def run_control_model(model: str, args, device) -> list[dict]:
    train = Split(args.acts, model, args.train)
    rng = np.random.default_rng(args.seed)  # recreated per model, so both models get the same shuffles
    shuffled = [
        torch.tensor(flip_pair_labels(train.labels, train.pair_ids, rng), dtype=torch.float32, device=device)
        for _ in range(args.n_shuffles)
    ]
    rows = []
    for l in range(train.n_layers):
        _, _, (X,) = standardize(torch.from_numpy(train.layer(l)).to(device))
        accs = []
        for s_idx, y_ctrl in enumerate(shuffled):
            w, b = fit_probe(X, y_ctrl, args.seed + s_idx + 1, args.l2, args.max_iter)
            accs.append((((X @ w + b) > 0).float() == y_ctrl).float().mean().item())
        rows.append({
            "model": model,
            "layer": l,
            "ctrl_train_acc": float(np.mean(accs)),
            "ctrl_train_acc_min": float(np.min(accs)),
            "ctrl_train_acc_max": float(np.max(accs)),
        })
        print(f"  [control] {model} index {l:2d}  shuffled-label train accuracy {rows[-1]['ctrl_train_acc']:.3f}",
              flush=True)
    return rows


def plot_capacity(rows: list[dict], models: list[str], out_path: str, title_note: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m in models:
        mine = sorted((r for r in rows if r["model"] == m), key=lambda r: r["layer"])
        x = np.array([r["layer"] for r in mine])
        ctrl = np.array([r["ctrl_train_acc"] for r in mine])
        st = MODEL_STYLE.get(m, {"color": MUTED, "label": m, "marker": "o"})
        ax.fill_between(x, [r["ctrl_train_acc_min"] for r in mine], [r["ctrl_train_acc_max"] for r in mine],
                        color=st["color"], alpha=0.15, linewidth=0)
        ax.plot(x, ctrl, color=st["color"], marker=st["marker"], markersize=4, linewidth=2, label=st["label"])
        hollow_final_point(ax, x, ctrl, st["color"])
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax.annotate("no memorization", (x[-1], 0.5), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8, color=MUTED)
    real = [r["real_train_acc"] for r in rows if r["layer"] > 0]
    span = f"{min(real):.3f}" if max(real) - min(real) < 5e-4 else f"{min(real):.3f} to {max(real):.3f}"
    ax.text(0.99, 0.5, f"for comparison, train accuracy on the real labels\nis {span} from index 1 up, in both models",
            transform=ax.transAxes, ha="right", va="center", fontsize=8, color=INK, linespacing=1.4)
    ax.set_ylabel("probe TRAIN accuracy on shuffled labels", color=INK)
    ax.set_xlabel(layer_axis_label(int(x[-1])) + "; hollow point = that final index", color=INK)
    ax.set_xticks(x[::2])
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.08, 0.5), fontsize=8)
    title = "Capacity check: the probe fits shuffled labels, so train accuracy is uninformative"
    ax.set_title(title + (f"\n{title_note}" if title_note else ""), color=INK, fontsize=11, loc="left")
    style_axis(ax)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--acts", required=True, help="root directory written by secpo_probe.extract")
    ap.add_argument("--results", default=None, help="real probe results.json (default: <out>/results.json)")
    ap.add_argument("--train", default="train")
    ap.add_argument("--models", nargs=2, default=["base", "secpo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--n-shuffles", type=int, default=3, help="number of random label shuffles to average")
    ap.add_argument("--out", default=None, help="default: <acts>/probes")
    ap.add_argument("--title-note", default="")
    args = ap.parse_args()

    args.out = args.out or os.path.join(args.acts, "probes")
    os.makedirs(args.out, exist_ok=True)
    real_res = json.load(open(args.results or os.path.join(args.out, "results.json")))
    real_train = {(r["model"], r["layer"]): r["train_acc"] for r in real_res["rows"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for m in args.models:
        rows += run_control_model(m, args, device)
    for r in rows:
        r["real_train_acc"] = real_train[(r["model"], r["layer"])]

    json_out = os.path.join(args.out, "control_results.json")
    csv_out = os.path.join(args.out, "control_results.csv")
    config = {k: getattr(args, k) for k in ("train", "models", "seed", "l2", "max_iter", "n_shuffles")}
    json.dump({"config": config, "rows": rows}, open(json_out, "w"), indent=2)
    with open(csv_out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    plot_capacity(rows, args.models, os.path.join(args.out, "control_capacity_by_layer.png"), args.title_note)
    for m in args.models:
        accs = [r["ctrl_train_acc"] for r in rows if r["model"] == m and r["layer"] > 0]
        print(f"{m}: shuffled-label train accuracy {min(accs):.3f} to {max(accs):.3f} over indices 1 and up")
    print(f"wrote {json_out} and {csv_out}")


if __name__ == "__main__":
    main()
