"""Build data/real/train.jsonl (SecPO Alpaca) and test.jsonl (SEP) in DATA_FORMAT.md schema.

Downloads the two raw JSON files into <out>/_raw/ on first run (using stdlib only)
and writes paired benign/injected JSONL files ready for `python -m secpo_probe.data`.

    python -m secpo_probe.build_real --out data/real --n-train 2000 --n-test 2000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request

from .data import validate

SECPO_TRAIN_URL = (
    "https://huggingface.co/datasets/Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset"
    "/resolve/main/preference_Llama-3.1-8B-Instruct_dpo_NaiveCompletion_randpos_synthetic_alpaca_optimized.json"
)
SEP_TEST_URL = (
    "https://raw.githubusercontent.com/egozverev/Should-It-Be-Executed-Or-Processed"
    "/main/SEP_dataset/SEP_dataset.json"
)


def ensure_file(url: str, dest: str) -> str:
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"downloading {url} -> {dest} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _norm(s: str) -> str:
    return " ".join(s.split())


def build_secpo_train(raw_path: str, n_pairs: int, max_chars: int, rng: random.Random) -> list[dict]:
    raw = json.load(open(raw_path, encoding="utf-8"))
    candidates = []
    for idx, r in enumerate(raw):
        chosen_input = r.get("chosen_input", "")
        chosen_data = r.get("chosen_data", "")
        injection = r.get("rejected_input_whole", "").strip()
        if not chosen_data.strip() or not injection or not chosen_input.endswith(chosen_data):
            continue
        instruction = chosen_input[: -len(chosen_data)].strip()
        benign_data = chosen_data.strip()
        if not instruction or not benign_data:
            continue
        at_end = bool(r.get("inject_at_the_end", False))
        injected_data = f"{benign_data} {injection}" if at_end else f"{injection} {benign_data}"
        if len(instruction) + len(injected_data) > max_chars:
            continue
        if injection not in injected_data:
            continue
        if _norm(injected_data.replace(injection, "", 1)) != _norm(benign_data):
            continue
        candidates.append((idx, instruction, benign_data, injected_data, injection, at_end))

    if n_pairs and n_pairs < len(candidates):
        candidates = rng.sample(candidates, n_pairs)

    rows = []
    for k, (idx, instruction, benign_data, injected_data, injection, at_end) in enumerate(candidates):
        pair_id = f"secpo-train-{k:05d}"
        rows.append({
            "id": f"{pair_id}-benign",
            "pair_id": pair_id,
            "label": 0,
            "instruction": instruction,
            "data": benign_data,
            "injection": None,
            "source": "secpo-trainset",
            "meta": {"orig_index": idx, "inject_at_the_end": at_end},
        })
        rows.append({
            "id": f"{pair_id}-injected",
            "pair_id": pair_id,
            "label": 1,
            "instruction": instruction,
            "data": injected_data,
            "injection": injection,
            "source": "secpo-trainset",
            "meta": {"orig_index": idx, "inject_at_the_end": at_end},
        })
    return rows


def build_sep_test(raw_path: str, n_pairs: int, max_chars: int, rng: random.Random) -> list[dict]:
    raw = json.load(open(raw_path, encoding="utf-8"))
    candidates = []
    for idx, r in enumerate(raw):
        instruction = r.get("system_prompt_clean", "").strip()
        clean_raw = r.get("prompt_clean", "")
        inst_raw = r.get("prompt_instructed", "")
        benign_data = clean_raw.strip()
        injected_data = inst_raw.strip()
        if not instruction or not benign_data or clean_raw not in inst_raw:
            continue
        injection = inst_raw.replace(clean_raw, "", 1).strip()
        if not injection or injection not in injected_data:
            continue
        if len(instruction) + len(injected_data) > max_chars:
            continue
        if _norm(injected_data.replace(injection, "", 1)) != _norm(benign_data):
            continue
        candidates.append((idx, instruction, benign_data, injected_data, injection, r.get("info", {})))

    if n_pairs and n_pairs < len(candidates):
        candidates = rng.sample(candidates, n_pairs)

    rows = []
    for k, (idx, instruction, benign_data, injected_data, injection, info) in enumerate(candidates):
        pair_id = f"sep-{k:05d}"
        rows.append({
            "id": f"{pair_id}-benign",
            "pair_id": pair_id,
            "label": 0,
            "instruction": instruction,
            "data": benign_data,
            "injection": None,
            "source": "sep",
            "meta": {"orig_index": idx, **info},
        })
        rows.append({
            "id": f"{pair_id}-injected",
            "pair_id": pair_id,
            "label": 1,
            "instruction": instruction,
            "data": injected_data,
            "injection": injection,
            "source": "sep",
            "meta": {"orig_index": idx, **info},
        })
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/real", help="output directory (default: data/real)")
    ap.add_argument("--n-train", type=int, default=2000, help="number of train pairs (0 = all)")
    ap.add_argument("--n-test", type=int, default=2000, help="number of test pairs (0 = all)")
    ap.add_argument("--max-chars", type=int, default=6000, help="skip unusually long prompts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw_dir = os.path.join(args.out, "_raw")
    secpo_raw = ensure_file(SECPO_TRAIN_URL, os.path.join(raw_dir, "secpo_llama_train.json"))
    sep_raw = ensure_file(SEP_TEST_URL, os.path.join(raw_dir, "SEP_dataset.json"))

    train_rows = build_secpo_train(secpo_raw, args.n_train, args.max_chars, rng)
    test_rows = build_sep_test(sep_raw, args.n_test, args.max_chars, rng)

    train_path = os.path.join(args.out, "train.jsonl")
    test_path = os.path.join(args.out, "test.jsonl")
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    for label, path, rows in [("train", train_path, train_rows), ("test", test_path, test_rows)]:
        errs, warns = validate(rows)
        print(f"{label} ({path}): {len(rows)} rows ({len(rows)//2} pairs) | errors={len(errs)} warnings={len(warns)}")
        if errs:
            raise SystemExit(f"Validation failed on {path}: {errs[:5]}")


if __name__ == "__main__":
    main()
