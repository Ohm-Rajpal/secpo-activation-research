"""Train one linear probe per layer per model: benign (0) vs injected (1).

    python -m secpo_probe.train_probes --acts runs/placeholder --train train --test test

For each model and each of the 33 hidden-state outputs, fits logistic regression
on the train file's activations and scores it on the test file's activations.
Every probe starts from the same seeded initial weights and uses the same
optimiser settings, so the two models differ only in their features.

Writes <acts>/probes/results.json, results.csv and probe_weights_<model>.npz.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .acts import Split


def fit_probe(X: torch.Tensor, y: torch.Tensor, seed: int, l2: float, max_iter: int):
    """L2-regularised logistic regression, full batch, L-BFGS. Returns (w, b)."""
    torch.manual_seed(seed)  # identical initial weights for every layer and model
    lin = torch.nn.Linear(X.shape[1], 1)
    w = lin.weight.detach().clone().squeeze(0).to(X.device).requires_grad_(True)
    b = lin.bias.detach().clone().to(X.device).requires_grad_(True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-6, tolerance_change=1e-10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def run_model(model: str, args, device) -> tuple[list[dict], np.ndarray, Split]:
    train, test = Split(args.acts, model, args.train), Split(args.acts, model, args.test)
    y_tr = torch.tensor(train.labels, dtype=torch.float32, device=device)
    y_te = torch.tensor(test.labels, dtype=torch.float32, device=device)
    rows, correct_all, weights = [], [], []
    for l in range(train.n_layers):
        X_tr = torch.from_numpy(train.layer(l)).to(device)
        X_te = torch.from_numpy(test.layer(l)).to(device)
        # Standardise with train statistics only. Constant features (all of layer 0,
        # where the last token's embedding is the same in every prompt) map to zero.
        mu, sd = X_tr.mean(0), X_tr.std(0)
        sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
        X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd

        w, b = fit_probe(X_tr, y_tr, args.seed, args.l2, args.max_iter)
        logit_tr, logit_te = X_tr @ w + b, X_te @ w + b
        correct = ((logit_te > 0).float() == y_te)
        rows.append({
            "model": model,
            "layer": l,
            "train_acc": ((logit_tr > 0).float() == y_tr).float().mean().item(),
            "test_acc": correct.float().mean().item(),
            "test_auroc": float(roc_auc_score(test.labels, logit_te.cpu().numpy())),
            "test_loss": torch.nn.functional.binary_cross_entropy_with_logits(logit_te, y_te).item(),
        })
        correct_all.append(correct.cpu().numpy())
        weights.append(np.concatenate([(w / sd).cpu().numpy(), [(b - (w * mu / sd).sum()).item()]]))
        print(f"  {model} layer {l:2d}  train {rows[-1]['train_acc']:.3f}  "
              f"test {rows[-1]['test_acc']:.3f}  auroc {rows[-1]['test_auroc']:.3f}", flush=True)
    # weights are stored in raw activation space: logit = x @ w[:-1] + w[-1]
    np.savez_compressed(os.path.join(args.out, f"probe_weights_{model}.npz"),
                        weights=np.stack(weights).astype(np.float32))
    return rows, np.stack(correct_all), test


def pair_bootstrap(correct: dict[str, np.ndarray], pair_ids: np.ndarray, n_boot: int, seed: int):
    """95% intervals from resampling test pairs (twins stay together).

    Returns per-model accuracy intervals and the interval of (last model - first model).
    """
    uniq, inv = np.unique(pair_ids, return_inverse=True)
    per_pair = {m: np.stack([np.bincount(inv, weights=c[l], minlength=len(uniq))
                             for l in range(c.shape[0])]) for m, c in correct.items()}
    counts = np.bincount(inv, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    n = counts[draws].sum(1)                                      # [n_boot]
    acc = {m: pp[:, draws].sum(2) / n for m, pp in per_pair.items()}  # [layers, n_boot]
    ci = {m: np.percentile(a, [2.5, 97.5], axis=1) for m, a in acc.items()}
    names = list(correct)
    diff_ci = np.percentile(acc[names[-1]] - acc[names[0]], [2.5, 97.5], axis=1)
    return ci, diff_ci


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--acts", required=True, help="root directory written by secpo_probe.extract")
    ap.add_argument("--train", default="train", help="name of the train file (without .jsonl)")
    ap.add_argument("--test", default="test", help="name of the test file (without .jsonl)")
    ap.add_argument("--models", nargs=2, default=["base", "secpo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", type=float, default=1e-3, help="L2 penalty on the probe weights")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="default: <acts>/probes")
    args = ap.parse_args()
    args.out = args.out or os.path.join(args.acts, "probes")
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows, correct, tests = [], {}, {}
    for m in args.models:
        r, correct[m], tests[m] = run_model(m, args, device)
        rows += r
    a, b = (tests[m] for m in args.models)
    if a.ids != b.ids:
        raise SystemExit("the two models were extracted from different test files")

    ci, diff_ci = pair_bootstrap(correct, a.pair_ids, args.n_boot, args.seed)
    for r in rows:
        r["test_acc_lo"], r["test_acc_hi"] = (float(v) for v in ci[r["model"]][:, r["layer"]])

    acc = {m: np.array([r["test_acc"] for r in rows if r["model"] == m]) for m in args.models}
    diff = acc[args.models[1]] - acc[args.models[0]]
    step = np.diff(diff)
    summary = {
        "difference": f"{args.models[1]} minus {args.models[0]} test accuracy",
        "diff_by_layer": diff.tolist(),
        "diff_lo": diff_ci[0].tolist(),
        "diff_hi": diff_ci[1].tolist(),
        "layer_of_largest_abs_diff": int(np.abs(diff).argmax()),
        "layer_of_largest_change_in_diff": int(np.abs(step).argmax()) + 1,
    }
    config = {k: getattr(args, k) for k in ("train", "test", "models", "seed", "l2", "max_iter", "n_boot")}
    config.update(n_train=int(Split(args.acts, args.models[0], args.train).acts.shape[0]),
                  n_test=len(a.ids),
                  extraction={m: tests[m].info for m in args.models})
    json.dump({"config": config, "summary": summary, "rows": rows},
              open(os.path.join(args.out, "results.json"), "w"), indent=2)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    l1, l2 = summary["layer_of_largest_abs_diff"], summary["layer_of_largest_change_in_diff"]
    print(f"\nlargest |accuracy gap|: layer {l1} ({diff[l1]:+.3f}, "
          f"95% CI [{diff_ci[0][l1]:+.3f}, {diff_ci[1][l1]:+.3f}])")
    print(f"largest layer-to-layer change in the gap: into layer {l2} ({step[l2 - 1]:+.3f})")
    print(f"wrote {args.out}/results.json")


if __name__ == "__main__":
    main()
