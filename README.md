# SecPO activation probing

CS182 project (Michaux, Rajpal, Nguyen, Cheng). We ask at which layer SecPO's defensive
training changes how linearly readable "this prompt contains an injection" is, compared
with the undefended base model.

- Undefended: `meta-llama/Llama-3.1-8B-Instruct`
- Defended: `Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO`

## Method

1. Each prompt is a trusted instruction plus untrusted data. Half the prompts are benign;
   the other half are the same prompts with a harmless injection inserted in the data.
2. For each model, save the hidden state at the last prompt token for all 33 hidden-state
   outputs (embeddings + 32 blocks, width 4096).
3. For each model and each layer, train a logistic-regression probe (benign vs injected)
   on the SecPO-training-distribution file and test it on the SEP file. All probes share
   the same initial weights, seed, and optimiser settings.
4. Compare the two models' test accuracy layer by layer.

## Run it

```bash
./run_pipeline.sh data/placeholder runs/placeholder "PLACEHOLDER DATA"   # smoke test, ~6 min
./run_pipeline.sh data/real runs/real                                    # real data
```

`data/real/` must contain `train.jsonl` and `test.jsonl` in the format of
[DATA_FORMAT.md](DATA_FORMAT.md). Outputs go to the run directory and are synced to
`s3://dl-team-activations-ohmrajpal/<run name>/`.

| Step | Command | Output |
|---|---|---|
| Validate | `python -m secpo_probe.data FILE.jsonl` | `OK` or a list of problems |
| Extract | `python -m secpo_probe.extract --model base --data train.jsonl test.jsonl --out RUN` | `RUN/base/train/acts.npy` `[N, 33, 4096]` float16, `meta.jsonl`, `info.json` |
| 2D check | `python -m secpo_probe.plot_2d --acts RUN --layers 4 16 28` | `RUN/sanity2d_pca_train.png` |
| Probes | `python -m secpo_probe.train_probes --acts RUN` | `RUN/probes/results.json`, `results.csv`, probe weights |
| Layer plot | `python -m secpo_probe.plot_layers --results RUN/probes/results.json` | `RUN/probes/test_acc_by_layer.png` and `.pdf` |

Extraction resumes if interrupted and skips files that are already complete.

## Design decisions worth knowing

**Both models get identical token ids.** The two tokenizers are the same file; only the
chat templates differ (base Llama's adds a default system turn with a date). By default
(`--prompt-format secpo`) both models are fed SecPO's layout: instruction in `user`, data
in the `input` role, no system turn. The weights are then the only difference between
the two runs. The cost is that base Llama never saw an `input` role in training.
`--prompt-format native` instead joins instruction and data in one `user` turn with base
Llama's own template, for an ablation.

**Position.** The feature is the hidden state at the final prompt token, the `\n\n` that
ends the assistant header. It is the same token in every prompt, so at index 0
(embeddings) the feature is constant and the probe must score exactly 50%. That is a
built-in check, not a bug.

**Intervals.** Accuracy bands and the gap's band are 95% bootstrap intervals over test
pairs (twins are resampled together).

**Two readings of "where the score changes most".** `results.json` reports both the layer
with the largest accuracy gap between the models and the layer where that gap changes
most from the previous layer.

## Placeholder data

`data/placeholder/` is synthetic filler made by `python -m secpo_probe.make_placeholder`.
It exists to exercise the code. Numbers from it are not results.
