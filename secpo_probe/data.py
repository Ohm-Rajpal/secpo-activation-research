"""Dataset format: loading, validation, and prompt rendering.

The format is documented for humans in DATA_FORMAT.md. Run the validator with

    python -m secpo_probe.data path/to/file.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict

BASE = "meta-llama/Llama-3.1-8B-Instruct"
SECPO = "Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO"
MODELS = {"base": BASE, "secpo": SECPO}

BENIGN, INJECTED = 0, 1


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: not valid JSON ({e})") from None
    return rows


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(s: str) -> str:
    return " ".join(s.split())


def validate(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors mean the file must be fixed."""
    errors, warnings = [], []
    seen_ids = set()
    pairs = defaultdict(list)
    for i, r in enumerate(rows):
        where = f"row {i + 1} (id={r.get('id')!r})"
        for key in ("id", "pair_id", "instruction", "data"):
            if not isinstance(r.get(key), str) or not r[key].strip():
                errors.append(f"{where}: '{key}' must be a non-empty string")
        if r.get("label") not in (BENIGN, INJECTED) or isinstance(r.get("label"), bool):
            errors.append(f"{where}: 'label' must be the integer 0 (benign) or 1 (injected)")
            continue
        if r.get("id") in seen_ids:
            errors.append(f"{where}: duplicate id")
        seen_ids.add(r.get("id"))
        if r.get("system") is not None and not isinstance(r["system"], str):
            errors.append(f"{where}: 'system' must be a string if present")
        inj = r.get("injection")
        if r["label"] == INJECTED:
            if not isinstance(inj, str) or not inj.strip():
                errors.append(f"{where}: injected rows need a non-empty 'injection'")
            elif isinstance(r.get("data"), str) and inj not in r["data"]:
                errors.append(f"{where}: 'injection' is not a substring of 'data'")
        elif inj:
            errors.append(f"{where}: benign rows must not have an 'injection'")
        pairs[r.get("pair_id")].append(r)

    for pid, members in pairs.items():
        labels = sorted(m["label"] for m in members if m.get("label") in (0, 1))
        if labels != [BENIGN, INJECTED]:
            errors.append(
                f"pair_id={pid!r}: needs exactly one benign and one injected row, has labels {labels}"
            )
            continue
        ben, inj = sorted(members, key=lambda m: m["label"])
        if ben.get("instruction") != inj.get("instruction") or ben.get("system") != inj.get("system"):
            errors.append(f"pair_id={pid!r}: the twins must share the same instruction and system")
        if isinstance(inj.get("injection"), str) and isinstance(inj.get("data"), str):
            stripped = inj["data"].replace(inj["injection"], "", 1)
            if _norm(stripped) != _norm(ben.get("data") or ""):
                warnings.append(
                    f"pair_id={pid!r}: injected data minus the injection differs from the benign data"
                )
    return errors, warnings


def build_messages(row: dict, prompt_format: str) -> list[dict]:
    """Chat messages for one row.

    secpo  : instruction in `user`, untrusted data in the `input` role (SecPO's layout).
    native : instruction and data joined in one `user` turn (how base Llama is normally used).
    """
    msgs = []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
    if prompt_format == "secpo":
        msgs.append({"role": "user", "content": row["instruction"]})
        msgs.append({"role": "input", "content": row["data"]})
    elif prompt_format == "native":
        msgs.append({"role": "user", "content": row["instruction"] + "\n\n" + row["data"]})
    else:
        raise ValueError(f"unknown prompt format {prompt_format!r}")
    return msgs


def template_source(prompt_format: str) -> str:
    """Which tokenizer's chat template renders the prompt.

    This depends only on the format, never on the model being run, so both models
    receive byte-identical token ids (their tokenizer.json files are identical).
    """
    return SECPO if prompt_format == "secpo" else BASE


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m secpo_probe.data FILE.jsonl [FILE.jsonl ...]")
    failed = False
    for path in sys.argv[1:]:
        rows = load_jsonl(path)
        errors, warnings = validate(rows)
        n_inj = sum(r.get("label") == INJECTED for r in rows)
        print(f"{path}: {len(rows)} rows, {n_inj} injected, {len(rows) - n_inj} benign")
        for w in warnings[:10]:
            print("  warning:", w)
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
        for e in errors[:20]:
            print("  ERROR:", e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        print("  " + ("FAILED" if errors else "OK"))
        failed |= bool(errors)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
