#!/bin/bash
#SBATCH --job-name=saq_lora_val
#SBATCH --output=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.err
#SBATCH --time=00:40:00
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

ADAPTER_DIR=$WS/models/saq_lora_v1

RUN_DIR=$WS/runs/$(date +%Y%m%d_%H%M%S)_saq_lora_val_v1
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

python -u src/infer_saq_lora.py \
  --adapter_dir "$ADAPTER_DIR" \
  --input_csv data/saq_val.csv \
  --out_tsv "$RUN_DIR/saq_val_pred.tsv"

python -u src/eval_saq.py \
  --truth_csv data/saq_val.csv \
  --pred_tsv "$RUN_DIR/saq_val_pred.tsv" \
  --out_json "$RUN_DIR/saq_val_metrics.json"

cat "$RUN_DIR/saq_val_metrics.json"
