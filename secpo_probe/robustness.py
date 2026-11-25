"""Two follow-up checks on the per-layer probes, computed from the saved activations.

    python -m secpo_probe.robustness --acts runs/real_crossed --data data/real_crossed

1. Inside the test file
    train_probes trains a probe on one dataset and tests it on another. A lower score
    for one model can then mean two things: the injection is harder to read in that
    model, or the probe just carries over less well between datasets. This check
    removes the second reading. The classifier is trained and tested inside the test
    file (5-fold cross-validation, folds grouped by injection text, so it is always
    scored on prompts and injection questions it was not trained on), with two
    classifiers:
        probe      the logistic regression of train_probes
        meandiff   the class-mean rule: is a prompt closer to the average injected
                   prompt or to the average benign one? Nothing else is fitted.

2. Clean-injected distance (no classifier)
    How far an injection moves the model's internal state away from the clean prompt's:
    the distance between an injected prompt and its benign ("clean") twin, divided by
    the typical distance between two unrelated benign prompts of the same file. Raw
    last-token states, one value per model, file and hidden-state index. At 1.0 the
    injected prompt is as far from its twin as an unrelated prompt would be. It is the
    size of a change at one token position, not its meaning: plain text added to the
    data moves the state too, so an instruction-specific claim needs that control.

Writes <acts>/probes/robustness.json, robustness.csv, within_test_by_layer.{png,pdf}
and clean_injected_distance_by_layer.{png,pdf}. --replot redraws the figures from the
saved json.
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
from .data import INJECTED, load_jsonl
from .plot_layers import INK, MUTED, hollow_final_point, layer_axis_label, style_axis
from .train_probes import fit_probe, pair_bootstrap, standardize

# name: title of its panel
READOUTS = {"probe": "Logistic probe", "meandiff": "Class-mean rule"}


def injection_groups(path: str, split: Split) -> np.ndarray:
    """Per activation row: the injection text of its twin pair, used to group the folds."""
    rows = load_jsonl(path)
    if [r["id"] for r in rows] != split.ids:
        raise SystemExit(f"{path} does not match the rows the activations were extracted from")
    injection = {r["pair_id"]: " ".join(r["injection"].split()) for r in rows if r["label"] == INJECTED}
    return np.array([injection[pid] for pid in split.pair_ids])


def group_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold number of every row; rows that share a group share a fold."""
    uniq, inv = np.unique(groups, return_inverse=True)
    return (np.random.default_rng(seed).permutation(len(uniq)) % k)[inv]


def predict(readout: str, X_fit, y_fit, X_eval, args) -> torch.Tensor:
    """True where the readout fitted on (X_fit, y_fit) calls a row of X_eval injected."""
    _, _, (A, B) = standardize(X_fit, X_eval)  # statistics of the fit rows only
    if readout == "probe":
        w, b = fit_probe(A, y_fit, args.seed, args.l2, args.max_iter)
    else:
        m1, m0 = A[y_fit == 1].mean(0), A[y_fit == 0].mean(0)
        w = m1 - m0
        b = -0.5 * ((m1 + m0) @ w)  # threshold halfway between the two class means
    return (B @ w + b) > 0


def twin_index(labels: np.ndarray, pair_ids: np.ndarray, seed: int):
    """Row numbers of each pair's benign and injected twin, and an unrelated benign row for each pair."""
    uniq, inv = np.unique(pair_ids, return_inverse=True)
    ben, inj = np.empty(len(uniq), dtype=np.int64), np.empty(len(uniq), dtype=np.int64)
    ben[inv[labels == 0]] = np.where(labels == 0)[0]
    inj[inv[labels == 1]] = np.where(labels == 1)[0]
    order = np.random.default_rng(seed).permutation(len(uniq))
    other = np.empty(len(uniq), dtype=np.int64)
    other[order] = ben[np.roll(order, 1)]  # a cycle, so no pair is matched with itself
    return ben, inj, other


def clean_injected_distance(X: torch.Tensor, ben, inj, other) -> float | None:
    """Mean distance from an injected prompt to its benign twin, over the mean distance between unrelated benign prompts."""
    between = (X[ben] - X[other]).norm(dim=1).mean()
    if between < 1e-6:  # index 0: the last token's embedding is the same in every prompt
        return None
    return ((X[inj] - X[ben]).norm(dim=1).mean() / between).item()


def run_model(model: str, args, device, info: dict) -> tuple[dict, list[dict]]:
    """Per-row correctness [n_layers, N_test] for each readout, and the clean-injected distance rows."""
    splits = {"train": Split(args.acts, model, args.train), "test": Split(args.acts, model, args.test)}
    for s, sp in splits.items():
        if sp.ids != info[s]["ids"]:
            raise SystemExit(f"{model}/{s}: activations were extracted from a different file than the other model's")
    y = torch.tensor(splits["test"].labels, dtype=torch.float32, device=device)
    n_layers = splits["test"].n_layers
    correct = {r: np.zeros((n_layers, len(y)), dtype=bool) for r in READOUTS}
    distances = []
    for l in range(n_layers):
        X = {s: torch.from_numpy(sp.layer(l)).to(device) for s, sp in splits.items()}
        for r in READOUTS:
            # cross-validation: every row is scored once, by a readout that never saw its injection text
            pred = torch.zeros(len(y), dtype=torch.bool, device=device)
            for k in range(args.folds):
                held = torch.from_numpy(info["test"]["fold"] == k).to(device)
                pred[held] = predict(r, X["test"][~held], y[~held], X["test"][held], args)
            correct[r][l] = (pred == (y > 0.5)).cpu().numpy()
        for s in splits:
            distances.append({"model": model, "split": s, "layer": l,
                              "clean_injected_distance": clean_injected_distance(X[s], *info[s]["twins"])})
        print(f"  {model} index {l:2d}  inside the test file: "
              + "  ".join(f"{r} {correct[r][l].mean():.3f}" for r in READOUTS), flush=True)
    return correct, distances


def figure_header(fig, title: str, note: str) -> float:
    """Title at the top left with the legend on its own row beneath it.

    Returns the top of the area left for the axes, as a fraction of the figure height.
    """
    h = fig.get_figheight()
    used = 0.08  # inches from the top edge
    fig.text(0.01, 1 - used / h, title + (f"\n{note}" if note else ""), ha="left", va="top", fontsize=11, color=INK)
    used += 0.21 * (2 if note else 1) + 0.04
    fig.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.003, 1 - used / h), fontsize=9, ncol=4)
    return 1 - (used + 0.34) / h


def plot_within_test(res: dict, out_path: str, title_note: str) -> None:
    """One panel per classifier, both models in each: accuracy on held-out rows of the test file."""
    models = res["config"]["models"]
    fig, axes = plt.subplots(1, len(READOUTS), figsize=(9, 4.3), sharex=True, sharey=True)
    for ax, (r, title) in zip(axes, READOUTS.items()):
        for m in models:
            rows = sorted((x for x in res["rows"] if (x["model"], x["readout"]) == (m, r)), key=lambda x: x["layer"])
            x = np.array([row["layer"] for row in rows])
            acc = np.array([row["acc"] for row in rows])
            st = MODEL_STYLE.get(m, {"color": MUTED, "label": m, "marker": "o"})
            ax.fill_between(x, [row["acc_lo"] for row in rows], [row["acc_hi"] for row in rows],
                            color=st["color"], alpha=0.15, linewidth=0)
            ax.plot(x, acc, color=st["color"], linewidth=2, label=st["label"] if r == "probe" else None)
            hollow_final_point(ax, x, acc, st["color"])  # that index is taken after the final norm
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style_axis(ax)
        ax.tick_params(labelsize=8)
    axes[0].set_ylim(0.4, 1.02)
    axes[0].set_xticks(x[::4])
    axes[0].set_ylabel("accuracy on held-out rows", color=INK, fontsize=9)
    fig.supxlabel(layer_axis_label(int(x[-1])) + "; hollow point = that final index", fontsize=9, color=INK)
    top = figure_header(fig, "Telling benign from injected prompts when the classifier is trained and tested "
                             "inside the test file", title_note)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path, dpi=200, facecolor="white")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", facecolor="white")
    print("wrote", out_path)


def plot_clean_injected_distance(res: dict, out_path: str, title_note: str) -> None:
    """Top row: the distance in each model. Bottom row: the second model's distance divided by the first's."""
    models = res["config"]["models"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [3, 2]})
    for j, s in enumerate(("train", "test")):
        ax, ax2 = axes[0, j], axes[1, j]
        series = {}
        for m in models:
            rows = sorted((x for x in res["clean_injected_distance"]
                           if (x["model"], x["split"]) == (m, s) and x["clean_injected_distance"] is not None),
                          key=lambda x: x["layer"])
            x = np.array([row["layer"] for row in rows])
            v = np.array([row["clean_injected_distance"] for row in rows])
            st = MODEL_STYLE.get(m, {"color": MUTED, "label": m, "marker": "o"})
            ax.plot(x, v, color=st["color"], linewidth=2, label=st["label"] if s == "train" else None)
            hollow_final_point(ax, x, v, st["color"])
            series[m] = v
        ratio = series[models[1]] / series[models[0]]
        ax2.axhline(1.0, color=MUTED, linewidth=1, linestyle=":")
        ax2.plot(x, ratio, color=INK, linewidth=1.8)
        hollow_final_point(ax2, x, ratio, INK)
        ax2.text(0.03, 0.06, f"{ratio[:-1].min():.2f} at its lowest before the final norm (index {x[ratio[:-1].argmin()]})",
                 transform=ax2.transAxes, ha="left", va="bottom", fontsize=8, color=INK)
        ax.set_title(f"{s} file", fontsize=9, color=INK, loc="left")
        for a in (ax, ax2):
            style_axis(a)
            a.tick_params(labelsize=8)
    for a in axes[:, 0]:
        a.set_ylim(bottom=0)  # after both panels are drawn, so the shared top covers both
    axes[0, 0].set_ylabel("clean-injected distance\n(1.0 = injected prompt as far from its clean\n"
                          "twin as an unrelated prompt)", color=INK, fontsize=9)
    axes[1, 0].set_ylabel(f"{models[1]} distance divided\nby {models[0]} distance\n(1.0 = no difference)", color=INK, fontsize=9)
    axes[1, 0].set_xticks(np.arange(0, x[-1] + 1, 4))
    fig.supxlabel(layer_axis_label(int(x[-1])) + "; hollow point = that final index\n"
                  "index 0 is not drawn: the last token's embedding is the same in every prompt", fontsize=9, color=INK)
    top = figure_header(fig, "Clean-injected distance: how far does the injection move the model's internal state, "
                             "at the last prompt token?", title_note)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path, dpi=200, facecolor="white")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", facecolor="white")
    print("wrote", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--acts", required=True, help="root directory written by secpo_probe.extract")
    ap.add_argument("--data", help="directory holding the test file the activations came from")
    ap.add_argument("--train", default="train", help="name of the train file (without .jsonl)")
    ap.add_argument("--test", default="test", help="name of the test file (without .jsonl)")
    ap.add_argument("--models", nargs=2, default=["base", "secpo"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", type=float, default=1e-3, help="L2 penalty of the logistic probe")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="default: <acts>/probes")
    ap.add_argument("--title-note", default="")
    ap.add_argument("--replot", action="store_true", help="only redraw the figures from an existing robustness.json")
    args = ap.parse_args()
    args.out = args.out or os.path.join(args.acts, "probes")
    os.makedirs(args.out, exist_ok=True)
    figures = (os.path.join(args.out, "within_test_by_layer.png"),
               os.path.join(args.out, "clean_injected_distance_by_layer.png"))
    if args.replot:
        res = json.load(open(os.path.join(args.out, "robustness.json")))
        plot_within_test(res, figures[0], args.title_note)
        plot_clean_injected_distance(res, figures[1], args.title_note)
        return
    if not args.data:
        ap.error("--data is required unless --replot is given")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Everything that depends on the rows and not on the model: twin rows, and the folds of the test file.
    info = {}
    for s, name in (("train", args.train), ("test", args.test)):
        sp = Split(args.acts, args.models[0], name)
        info[s] = {"ids": sp.ids, "labels": sp.labels, "pair_ids": sp.pair_ids,
                   "twins": twin_index(sp.labels, sp.pair_ids, args.seed)}
        if s == "test":
            groups = injection_groups(os.path.join(args.data, name + ".jsonl"), sp)
            info[s]["fold"] = group_folds(groups, args.folds, args.seed)

    correct, distances = {}, []
    for m in args.models:
        c, d = run_model(m, args, device, info)
        distances += d
        for r, arr in c.items():
            correct.setdefault(r, {})[m] = arr

    rows, summary = [], {}
    for r, by_model in correct.items():
        ci, diff_ci = pair_bootstrap(by_model, info["test"]["pair_ids"], args.n_boot, args.seed)
        for m in args.models:
            for l, ok in enumerate(by_model[m]):
                rows.append({"model": m, "layer": l, "readout": r, "acc": float(ok.mean()),
                             "acc_lo": float(ci[m][0, l]), "acc_hi": float(ci[m][1, l])})
        diff = by_model[args.models[1]].mean(1) - by_model[args.models[0]].mean(1)
        summary[r] = {"diff_by_layer": diff.tolist(), "diff_lo": diff_ci[0].tolist(), "diff_hi": diff_ci[1].tolist()}

    config = {k: getattr(args, k) for k in ("train", "test", "models", "folds", "seed", "l2", "max_iter", "n_boot")}
    config.update(rows="accuracy inside the test file, cross-validated",
                  difference=f"{args.models[1]} minus {args.models[0]} accuracy",
                  fold_grouping="injection text",
                  n_rows={s: len(info[s]["ids"]) for s in info},
                  n_distinct_injections=len(set(groups)))
    res = {"config": config, "summary": summary, "rows": rows, "clean_injected_distance": distances}
    json.dump(res, open(os.path.join(args.out, "robustness.json"), "w"), indent=2)
    with open(os.path.join(args.out, "robustness.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\ninside the test file, {args.models[1]} minus {args.models[0]} accuracy (indices before the final norm)")
    for r, s in summary.items():
        diff = np.array(s["diff_by_layer"][:-1])
        k = int(np.abs(diff).argmax())
        print(f"  {READOUTS[r]:15s} largest gap {diff[k]:+.3f} at index {k}")
    print(f"wrote {args.out}/robustness.json")
    plot_within_test(res, figures[0], args.title_note)
    plot_clean_injected_distance(res, figures[1], args.title_note)


if __name__ == "__main__":
    main()
