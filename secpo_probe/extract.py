"""Extract last-token hidden states for every layer of one model.

    python -m secpo_probe.extract --model base  --data data/placeholder/train.jsonl data/placeholder/test.jsonl --out runs/placeholder
    python -m secpo_probe.extract --model secpo --data data/placeholder/train.jsonl data/placeholder/test.jsonl --out runs/placeholder

Writes, under <out>/<model>/<data file stem>/ :
    acts.npy    float16 array of shape [N, 33, 4096], rows in dataset order
    meta.jsonl  one line per row: id, pair_id, label, n_tokens
    info.json   model, prompt format, dataset hash, shapes

Index 0 of the layer axis is the embedding output. Indices 1 to 31 are the raw
outputs of transformer blocks 1 to 31. Index 32 is the output of block 32 AFTER
the model's final norm (transformers returns the last hidden state normed), so
the step from 31 to 32 is one block plus the norm, unlike every other step.
An interrupted run resumes from its last checkpoint, saved every 20 batches.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from .data import MODELS, build_messages, file_sha256, load_jsonl, template_source, validate


def render_prompts(rows: list[dict], prompt_format: str) -> tuple[list[list[int]], AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(template_source(prompt_format))
    ids = []
    for r in rows:
        text = tok.apply_chat_template(
            build_messages(r, prompt_format), tokenize=False, add_generation_prompt=True
        )
        # The template already emits <|begin_of_text|>; do not add a second one.
        ids.append(tok(text, add_special_tokens=False).input_ids)
    return ids, tok


def make_batches(lengths: list[int], max_tokens: int, max_size: int) -> list[list[int]]:
    """Group similar-length prompts into batches of row indices.

    Each batch has at most max_size prompts and batch_size * longest <= max_tokens.
    A single prompt longer than max_tokens still gets a batch of its own.
    """
    # largest to smallest
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches, cur = [], []
    for i in order:
        longest = lengths[cur[0]] if cur else lengths[i]
        if cur and (len(cur) >= max_size or (len(cur) + 1) * longest > max_tokens):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


@torch.inference_mode() # no gradients are being computed
def last_token_states(model, batch_ids: list[list[int]], pad_id: int) -> torch.Tensor:
    """[B, n_layers + 1, hidden] hidden states at each prompt's final token."""
    width = max(len(x) for x in batch_ids)
    input_ids = torch.full((len(batch_ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), width), dtype=torch.long)
    for j, x in enumerate(batch_ids):  # left-pad, so the final token sits at position -1
        input_ids[j, width - len(x):] = torch.tensor(x)
        mask[j, width - len(x):] = 1
    # base_model is the transformer without the vocabulary head. The hidden states are
    # the same tensors; the unused [B, T, 128256] logits (1.5 GB at 6000 tokens) are skipped.
    out = model.base_model(
        input_ids=input_ids.to(model.device),
        attention_mask=mask.to(model.device),
        output_hidden_states=True,
    )
    return torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1).float().cpu()


def extract_file(model, model_name: str, tag: str, data_path: str, args) -> None:
    split = os.path.splitext(os.path.basename(data_path))[0]
    out_dir = os.path.join(args.out, tag, split)
    os.makedirs(out_dir, exist_ok=True)

    rows = load_jsonl(data_path)
    errors, _ = validate(rows)
    if errors:
        raise SystemExit(f"{data_path} fails validation; run: python -m secpo_probe.data {data_path}")
    if args.limit:
        rows = rows[: args.limit]

    ids, tok = render_prompts(rows, args.prompt_format)
    lengths = [len(x) for x in ids]
    if max(lengths) > args.max_prompt_tokens:
        raise SystemExit(f"longest prompt has {max(lengths)} tokens > --max-prompt-tokens")
    if len({x[-1] for x in ids}) != 1:
        raise SystemExit("prompts do not all end in the same token; the template changed?")
    print(f"[{tag}/{split}] {len(rows)} prompts, tokens min/median/max = "
          f"{min(lengths)}/{int(np.median(lengths))}/{max(lengths)}")
    print("example prompt:\n" + tok.decode(ids[0]) + "\n---")

    info = {
        "model": model_name,
        "prompt_format": args.prompt_format,
        "template_from": template_source(args.prompt_format),
        "data_file": data_path,
        "data_sha256": file_sha256(data_path),
        "n_rows": len(rows),
        "position": "last prompt token (end of the assistant header)",
        "last_token": tok.convert_ids_to_tokens(ids[0][-1]),
    }
    info_path = os.path.join(out_dir, "info.json")
    acts_path = os.path.join(out_dir, "acts.npy")
    done_path = os.path.join(out_dir, "done.npy")
    shape = (len(rows), model.config.num_hidden_layers + 1, model.config.hidden_size)

    if os.path.exists(info_path) and os.path.exists(acts_path):
        old = json.load(open(info_path))
        same = all(old.get(k) == v for k, v in info.items()) and old.get("shape") == list(shape)
        if same and old.get("complete"):
            print(f"[{tag}/{split}] already complete, skipping")
            return
        resume = same and os.path.exists(done_path)
        if not resume:
            print("existing output was made with different settings; starting over")
    else:
        resume = False

    if resume:
        acts = np.load(acts_path, mmap_mode="r+")
        done = np.load(done_path)
        print(f"resuming: {int(done.sum())}/{len(rows)} rows already done")
    else:
        acts = np.lib.format.open_memmap(acts_path, mode="w+", dtype=np.float16, shape=shape)
        done = np.zeros(len(rows), dtype=bool)
        np.save(done_path, done)
        json.dump({**info, "shape": list(shape), "complete": False}, open(info_path, "w"), indent=2)

    pad_id = tok.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    batches = [b for b in make_batches(lengths, args.max_batch_tokens, args.max_batch_size)
               if not done[b].all()]
    t0, n_done = time.time(), 0
    for k, b in enumerate(batches):
        states = last_token_states(model, [ids[i] for i in b], pad_id).numpy()
        if not np.isfinite(states).all() or np.abs(states).max() > np.finfo(np.float16).max:
            raise SystemExit("non-finite or float16-overflowing activation; stopping")
        acts[b] = states.astype(np.float16)
        done[b] = True
        n_done += len(b)
        if k % 20 == 0 or k == len(batches) - 1:
            acts.flush()
            np.save(done_path, done)
            print(f"  batch {k + 1}/{len(batches)}  {n_done} rows  {time.time() - t0:.0f}s", flush=True)

    acts.flush()
    assert done.all()
    with open(os.path.join(out_dir, "meta.jsonl"), "w") as f:
        for r, n in zip(rows, lengths):
            f.write(json.dumps({"id": r["id"], "pair_id": r["pair_id"],
                                "label": r["label"], "n_tokens": n}) + "\n")
    json.dump({**info, "shape": list(shape), "complete": True}, open(info_path, "w"), indent=2)
    os.remove(done_path)
    print(f"saved {acts_path}  shape={shape}")


def already_complete(args, tag: str, data_path: str, model_name: str) -> bool:
    """True if a finished extraction of this exact file, model and format exists."""
    split = os.path.splitext(os.path.basename(data_path))[0]
    info_path = os.path.join(args.out, tag, split, "info.json")
    if not os.path.exists(info_path):
        return False
    old = json.load(open(info_path))
    return bool(old.get("complete")) and old.get("model") == model_name \
        and old.get("prompt_format") == args.prompt_format \
        and old.get("data_sha256") == file_sha256(data_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="'base', 'secpo', or a Hugging Face model name")
    ap.add_argument("--data", required=True, nargs="+", help="one or more dataset .jsonl files")
    ap.add_argument("--out", required=True, help="output root directory")
    ap.add_argument("--prompt-format", choices=["secpo", "native"], default="secpo")
    ap.add_argument("--max-batch-tokens", type=int, default=6000,
                    help="padded tokens per batch. A memory limit, not a speed knob: on the 23 GB L4 "
                         "speed is flat from about 1000 up, and above about 12000 it runs out of memory")
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--max-prompt-tokens", type=int, default=4096,
                    help="fail if any prompt is longer than this (never truncates)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (debugging)")
    args = ap.parse_args()

    model_name = MODELS.get(args.model, args.model)
    tag = args.model if args.model in MODELS else args.model.replace("/", "__")
    hf_logging.disable_progress_bar()
    if not args.limit and all(already_complete(args, tag, d, model_name) for d in args.data):
        print(f"[{tag}] all files already extracted, nothing to do")
        return
    print(f"loading {model_name} (about 1 minute if not cached in memory) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    for data_path in args.data:
        extract_file(model, model_name, tag, data_path, args)


if __name__ == "__main__":
    main()
