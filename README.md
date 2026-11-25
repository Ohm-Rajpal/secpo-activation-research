# SecPO activation probing

CS182 project (Amaury Michaux, Ohm Rajpal, Tamara Nguyen, Weiyang Cheng). We ask at which layer SecPO's defensive
training changes how linearly readable "this prompt contains an injection" is, compared
with the undefended base model.

- Undefended: `meta-llama/Llama-3.1-8B-Instruct`
- Defended: `Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO`

## Method

1. Each prompt is a trusted instruction plus untrusted data. Half the prompts are benign;
   the other half are the same prompts with a harmless injection inserted in the data.
   In the train file the injection is another training example's instruction, appended to
   the data ("crossing"). The test file is SEP as it ships, with SEP's own probe question.
2. For each model, save the hidden state at the last prompt token for all 33 hidden-state
   outputs (width 4096): index 0 is the embeddings, indices 1 to 31 are the raw outputs of
   blocks 1 to 31, and index 32 is the output of block 32 after the model's final norm.
3. For each model and each layer, train a logistic-regression probe (benign vs injected)
   on the train file (SecPO's Alpaca examples) and test it on the SEP file. All probes share
   the same initial weights, seed, and optimizer settings.
4. Compare the two models' test accuracy layer by layer.
5. Run two follow-up checks on the saved activations. Train and test a classifier inside
   the SEP file, to see whether the injection is still readable in each model. And measure,
   with no classifier, how far the injection moves the hidden state away from the clean
   prompt's (the clean-injected distance).

## Run it

```bash
./run_pipeline.sh data/placeholder runs/placeholder "PLACEHOLDER DATA"   # smoke test, a few minutes
python -m secpo_probe.build_real                                         # writes data/real_crossed/
./run_pipeline.sh data/real_crossed runs/real_crossed                    # real data
```

The data directory must contain `train.jsonl` and `test.jsonl` in the format of
[DATA_FORMAT.md](DATA_FORMAT.md), which also describes how the twins are built. Outputs go
to the run directory and are synced to `s3://dl-team-activations-ohmrajpal/<run name>/`.

`build_real` has two modes for the injected train rows; the sampled examples, the benign
rows and the SEP test file are the same in both.

- `--train-injection crossed` (default, writes `data/real_crossed/`): each example's data
  followed by its partner's instruction, in both directions.
- `--train-injection attack` (writes `data/real/`): the attack string SecPO ships. This is
  the dataset behind the committed results in `runs/real` and `runs/real_native`.

Results of the crossed dataset are in `runs/real_crossed`.

| Step | Command | Output |
|---|---|---|
| Build real data | `python -m secpo_probe.build_real` | `data/real_crossed/train.jsonl`, `test.jsonl` (2,000 twin pairs each) |
| Validate | `python -m secpo_probe.data FILE.jsonl` | `OK` or a list of problems |
| Extract | `python -m secpo_probe.extract --model base --data train.jsonl test.jsonl --out RUN` | `RUN/base/train/acts.npy` `[N, 33, 4096]` float16, `meta.jsonl`, `info.json` |
| 2D check | `python -m secpo_probe.plot_2d --acts RUN --layers 4 16 28` | `RUN/sanity2d_pca_train.png` |
| Probes | `python -m secpo_probe.train_probes --acts RUN` | `RUN/probes/results.json`, `results.csv`, probe weights |
| Layer plot | `python -m secpo_probe.plot_layers --results RUN/probes/results.json` | `RUN/probes/test_acc_by_layer.png` and `.pdf` |
| Follow-up checks | `python -m secpo_probe.robustness --acts RUN --data DATA_DIR` | `RUN/probes/robustness.json`, `within_test_by_layer.png`, `clean_injected_distance_by_layer.png` |
| Memorization check | `python -m secpo_probe.control_task --acts RUN` | `RUN/probes/control_results.json`, `control_capacity_by_layer.png` |

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

**Index 32 is not like the other layers.** `transformers` returns the last hidden state
after the final norm, so the step from index 31 to 32 is one block plus the norm, while
every other step is one block. Read a change at index 32 with that in mind.

**Intervals.** Accuracy bands and the gap's band are 95% bootstrap intervals over test
pairs (twins are resampled together).

**Where the score changes.** `results.json` reports three things about the gap between
the models: the first index where it opens (at least 0.02, with an interval that excludes
zero), the index where it is largest, and the index where it changes most from the one
before. The last two are given with and without index 32; use the `blocks_only` entries
to compare layers.

**The probe can memorize.** `control_task` is a sanity check on the method, not a result
about the models. It trains the probe on labels shuffled inside the twin pairs, where
there is nothing real to learn, and the probe still fits most of them (83% to 99% in the
crossed run; the real labels are fitted exactly). Train accuracy therefore says nothing;
only accuracy on rows the probe was not trained on counts.

## The two follow-up checks

The main result is one curve per model: a probe trained on the train file and tested on
the SEP file. When SecPO scores lower than the base model there, two readings are
possible. Injections may be harder to read in SecPO, or the probe may simply carry over
less well from one dataset to the other. `python -m secpo_probe.robustness` reuses the
saved activations (no model is loaded, about three minutes) for two checks.

**1. Inside the test file** (`within_test_by_layer.png`). The classifier is trained and
tested inside the SEP file, always scored on prompts and injection questions it was not
trained on. Two classifiers are used: the logistic probe of the main result (left chart),
and the class-mean rule (right chart), which only asks whether a prompt is closer to the
average injected prompt or to the average benign one. If the probe stays near 100% for
both models, the injection is still readable in SecPO and the gap in the main result is
about carrying over between datasets.

**2. Clean-injected distance** (`clean_injected_distance_by_layer.png`). No classifier.
For every twin pair, take the hidden state at the last prompt token for the clean prompt
and for the same prompt with the injection, and measure the distance between the two.
Average over all pairs and divide by the average distance between two unrelated clean
prompts of the same file, so that 1.0 means the injected prompt is as far from its twin
as a different prompt would be. The top charts show this by layer for each model; the
bottom charts show SecPO's distance divided by the base model's.

**What the crossed run shows.**

- Main result: the two models score the same through index 13. SecPO is lower from index
  14–15, by up to 8.7 points before the final norm (index 26).
- Inside SEP, the logistic probe is right at least 98% of the time in both models from
  index 3 to 31. The injection stays readable in SecPO.
- The class-mean rule separates the models from about index 15: in the later layers it
  scores about 83% on the base model and 65–70% on SecPO. The injection is a less
  prominent part of SecPO's state.
- The clean-injected distance is the same in both models through index 4 and within 5%
  through index 9. SecPO's then falls, fastest over indices 10–16, and ends at about 0.58
  of the base model's on the train file and 0.46 on SEP.

**Limits.** The distance is not specific to instructions: in a one-off control outside
this pipeline, a plain sentence added to the data also moved SecPO's state 20–35% less
than the base model's, against 40–55% less for an instruction. It is read at one token,
the last one of the prompt. And all of this is correlational: nothing here shows that
these changes are what makes SecPO robust.

## Placeholder data

`data/placeholder/` is synthetic filler made by `python -m secpo_probe.make_placeholder`.
It exists to exercise the code. Numbers from it are not results.
