"""Build train.jsonl (SecPO Alpaca) and test.jsonl (SEP) in DATA_FORMAT.md schema.

Downloads the two raw JSON files into --raw-dir on first run (using stdlib only)
and writes paired benign/injected JSONL files ready for `python -m secpo_probe.data`.

The sampled examples, the benign train rows and the whole test file are the same in both
modes of --train-injection. Only the injected train rows differ:

    crossed  the sampled examples are put into random couples and each example's injected
             row is its own data followed by its partner's instruction, in both directions:
             (P1, D1 + " " + P2) and (P2, D2 + " " + P1). Written to data/real_crossed/.
    attack   the attack string SecPO ships (`rejected_input_whole`), at the start or the
             end of the data as SecPO placed it. Written to data/real/; with the default
             sizes and seed these are the files of the first real run, byte for byte.

    python -m secpo_probe.build_real                             # crossed -> data/real_crossed/
    python -m secpo_probe.build_real --train-injection attack    # attack  -> data/real/

The couples are drawn by their own generator, random.Random(f"crossing-{seed}"), so the
generator that samples the train and then the test examples is in the same state in both
modes. No example is crossed with a partner whose instruction is identical to its own.
With an odd number of train examples the last sampled one is left out of the crossed file.
An existing train.jsonl or test.jsonl is never replaced by different content unless
--overwrite is given.
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

# Defaults of the command line.
TRAIN_INJECTION = "crossed"
OUT_DIR = {"crossed": "data/real_crossed", "attack": "data/real"}
RAW_DIR = "data/real/_raw"  # one copy of the downloads, shared by both modes


def ensure_file(url: str, dest: str) -> str:
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"downloading {url} -> {dest} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _norm(s: str) -> str:
    return " ".join(s.split())


def cross_partners(candidates: list[tuple], rng: random.Random, max_tries: int = 1000) -> list[int]:
    """Split an even number of candidates into random couples; returns each one's partner.

    A shuffle is drawn again if it holds a couple with identical instructions, or one
    where removing the appended instruction would not give back the benign data (rule 3).
    """
    def clash(a: tuple, b: tuple) -> bool:  # b's instruction appended to a's data
        instruction, benign_data, injection = a[1], a[2], b[1]
        if _norm(instruction) == _norm(injection):
            return True
        return _norm(f"{benign_data} {injection}".replace(injection, "", 1)) != _norm(benign_data)

    order = list(range(len(candidates)))
    for _ in range(max_tries):
        rng.shuffle(order)
        couples = list(zip(order[::2], order[1::2]))
        if not any(clash(candidates[a], candidates[b]) or clash(candidates[b], candidates[a]) for a, b in couples):
            partner = [0] * len(candidates)
            for a, b in couples:
                partner[a], partner[b] = b, a
            return partner
    raise SystemExit(f"no valid crossing in {max_tries} shuffles: too many identical instructions")


def build_secpo_train(raw_path: str, n_pairs: int, max_chars: int, rng: random.Random,
                      cross_rng: random.Random | None = None) -> list[dict]:
    """Twin pairs with SecPO's attack string, or crossed with each other if cross_rng is given."""
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

    partner = None
    if cross_rng is not None:
        if len(candidates) % 2:
            print(f"crossing needs an even number of examples: left out the last of the {len(candidates)} sampled")
            candidates = candidates[:-1]
        partner = cross_partners(candidates, cross_rng)

    rows = []
    for k, (idx, instruction, benign_data, injected_data, injection, at_end) in enumerate(candidates):
        pair_id = f"secpo-train-{k:05d}"
        meta = {"orig_index": idx, "inject_at_the_end": at_end}
        if partner is not None:  # the partner's instruction alone, always at the end
            j = partner[k]
            injection = candidates[j][1]
            injected_data = f"{benign_data} {injection}"
            meta = {"orig_index": idx, "inject_at_the_end": True,
                    "partner_pair_id": f"secpo-train-{j:05d}", "partner_orig_index": candidates[j][0]}
        rows.append({
            "id": f"{pair_id}-benign",
            "pair_id": pair_id,
            "label": 0,
            "instruction": instruction,
            "data": benign_data,
            "injection": None,
            "source": "secpo-trainset",
            "meta": meta,
        })
        rows.append({
            "id": f"{pair_id}-injected",
            "pair_id": pair_id,
            "label": 1,
            "instruction": instruction,
            "data": injected_data,
            "injection": injection,
            "source": "secpo-trainset",
            "meta": dict(meta),
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


def jsonl_text(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def would_change(path: str, rows: list[dict]) -> bool:
    """True if `path` already exists and holds something other than `rows`."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        return f.read() != jsonl_text(rows).encode("utf-8")


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(jsonl_text(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-injection", choices=list(OUT_DIR), default=TRAIN_INJECTION,
                    help="injected train rows: 'crossed' appends the instruction of another sampled example, "
                         "'attack' inserts the attack string SecPO ships (default: %(default)s)")
    ap.add_argument("--out", default=None, help="output directory (default: "
                    f"{OUT_DIR['crossed']} for crossed, {OUT_DIR['attack']} for attack)")
    ap.add_argument("--raw-dir", default=RAW_DIR,
                    help="where the two raw downloads are kept, for both modes (default: %(default)s)")
    ap.add_argument("--n-train", type=int, default=2000,
                    help="number of train pairs (0 = all); crossed leaves out the last one if the number is odd")
    ap.add_argument("--n-test", type=int, default=2000, help="number of test pairs (0 = all)")
    ap.add_argument("--max-chars", type=int, default=6000,
                    help="skip unusually long prompts (in both modes measured with SecPO's attack string in the "
                         "data, so that both modes sample the same examples)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed of the train and test samples; the crossed couples are drawn by a separate "
                         "random.Random(f'crossing-{seed}')")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing train.jsonl or test.jsonl even if its content would change")
    args = ap.parse_args()
    args.out = args.out or OUT_DIR[args.train_injection]

    rng = random.Random(args.seed)
    # The couples get their own generator: drawing them from `rng` would shift the test sample.
    cross_rng = random.Random(f"crossing-{args.seed}") if args.train_injection == "crossed" else None
    secpo_raw = ensure_file(SECPO_TRAIN_URL, os.path.join(args.raw_dir, "secpo_llama_train.json"))
    sep_raw = ensure_file(SEP_TEST_URL, os.path.join(args.raw_dir, "SEP_dataset.json"))

    train_rows = build_secpo_train(secpo_raw, args.n_train, args.max_chars, rng, cross_rng)
    test_rows = build_sep_test(sep_raw, args.n_test, args.max_chars, rng)

    train_path = os.path.join(args.out, "train.jsonl")
    test_path = os.path.join(args.out, "test.jsonl")
    changed = [p for p, rows in ((train_path, train_rows), (test_path, test_rows)) if would_change(p, rows)]
    if changed and not args.overwrite:
        raise SystemExit(f"not replacing {' and '.join(changed)}: this run (--train-injection "
                         f"{args.train_injection}) would change the content; use another --out or --overwrite")
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    train_label = f"train/{args.train_injection}"
    for label, path, rows in [(train_label, train_path, train_rows), ("test", test_path, test_rows)]:
        errs, warns = validate(rows)
        print(f"{label} ({path}): {len(rows)} rows ({len(rows)//2} pairs) | errors={len(errs)} warnings={len(warns)}")
        if errs:
            raise SystemExit(f"Validation failed on {path}: {errs[:5]}")


if __name__ == "__main__":
    main()
