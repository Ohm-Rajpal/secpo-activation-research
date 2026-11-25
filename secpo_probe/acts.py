"""Loading saved activations."""
from __future__ import annotations

import json
import os

import numpy as np

# Two-series palette (color-blind-safe blue / orange) used by every figure.
BLUE, ORANGE = "#2a78d6", "#eb6834"
MODEL_STYLE = {
    "base": {"color": BLUE, "label": "Llama-3.1-8B-Instruct (undefended)", "marker": "o"},
    "secpo": {"color": ORANGE, "label": "Llama-3.1-8B-Instruct-SecPO", "marker": "s"},
}
LABEL_STYLE = {
    0: {"color": BLUE, "label": "benign", "marker": "o"},
    1: {"color": ORANGE, "label": "injected", "marker": "^"},
}


class Split:
    """Activations of one model on one dataset file."""

    def __init__(self, root: str, model: str, split: str):
        d = os.path.join(root, model, split)
        self.info = json.load(open(os.path.join(d, "info.json")))
        if not self.info.get("complete"):
            raise SystemExit(f"{d}: extraction is not complete")
        self.acts = np.load(os.path.join(d, "acts.npy"), mmap_mode="r")
        meta = [json.loads(line) for line in open(os.path.join(d, "meta.jsonl"))]
        self.ids = [m["id"] for m in meta]
        self.pair_ids = np.array([m["pair_id"] for m in meta])
        self.labels = np.array([m["label"] for m in meta], dtype=np.int64)
        self.n_layers = self.acts.shape[1]
        assert len(meta) == self.acts.shape[0]

    def layer(self, l: int) -> np.ndarray:
        return np.asarray(self.acts[:, l, :], dtype=np.float32)
