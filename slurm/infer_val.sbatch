#!/bin/bash
#SBATCH --job-name=val_infer
#SBATCH --output=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

set -euo pipefail

cd ~/cultural-qa-llm
source .venv/bin/activate

export WS=/data/horse/ws/sasr043g-cultural-qa-llm
export HF_HOME=$WS/hf_cache
export TRANSFORMERS_CACHE=$WS/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

RUN_DIR=$WS/runs/$(date +%Y%m%d_%H%M%S)_phase2_val_baseline
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

python -u src/infer_mcq.py \
  --input_csv data/mcq_val.csv \
  --out_tsv "$RUN_DIR/mcq_val_pred.tsv"

python -u src/infer_saq.py \
  --input_csv data/saq_val.csv \
  --out_tsv "$RUN_DIR/saq_val_pred.tsv"

echo "Done. Outputs in $RUN_DIR"
ls -lh "$RUN_DIR"
