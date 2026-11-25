#!/usr/bin/env bash
# Full pipeline: validate -> extract (both models) -> 2D sanity plots -> probes -> layer plot
# -> follow-up checks -> memorization check -> S3.
#
#   ./run_pipeline.sh data/placeholder runs/placeholder "PLACEHOLDER DATA"
#   ./run_pipeline.sh data/real        runs/real
#
# DATA_DIR must contain train.jsonl and test.jsonl (see DATA_FORMAT.md).
# Set S3_PREFIX="" to skip the upload.
set -euo pipefail

DATA_DIR=${1:?usage: ./run_pipeline.sh DATA_DIR RUN_DIR [TITLE_NOTE]}
RUN_DIR=${2:?usage: ./run_pipeline.sh DATA_DIR RUN_DIR [TITLE_NOTE]}
NOTE=${3:-}
S3_PREFIX=${S3_PREFIX-s3://dl-team-activations-ohmrajpal/$(basename "$RUN_DIR")}

source /opt/pytorch/bin/activate
cd "$(dirname "$0")"

python -m secpo_probe.data "$DATA_DIR/train.jsonl" "$DATA_DIR/test.jsonl"

# One model at a time: each needs about 16 GB of the 23 GB GPU.
for model in base secpo; do
  python -m secpo_probe.extract --model "$model" --out "$RUN_DIR" \
    --data "$DATA_DIR/train.jsonl" "$DATA_DIR/test.jsonl"
done

for method in pca meandiff; do
  python -m secpo_probe.plot_2d --acts "$RUN_DIR" --split train --method "$method" --title-note "$NOTE"
done

python -m secpo_probe.train_probes --acts "$RUN_DIR" --train train --test test
python -m secpo_probe.plot_layers --results "$RUN_DIR/probes/results.json" --title-note "$NOTE"
python -m secpo_probe.robustness --acts "$RUN_DIR" --data "$DATA_DIR" --title-note "$NOTE"
python -m secpo_probe.control_task --acts "$RUN_DIR" --title-note "$NOTE"

if [ -n "$S3_PREFIX" ]; then
  aws s3 sync "$RUN_DIR" "$S3_PREFIX" --only-show-errors
  echo "uploaded to $S3_PREFIX"
fi
