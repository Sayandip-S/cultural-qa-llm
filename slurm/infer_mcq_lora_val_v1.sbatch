#!/bin/bash
#SBATCH --job-name=mcq_lora_val
#SBATCH --output=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.err
#SBATCH --time=00:45:00
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

RUN_DIR=$WS/runs/$(date +%Y%m%d_%H%M%S)_mcq_lora_v1_val
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

python -u src/infer_mcq_lora.py \
  --input_csv data/mcq_val.csv \
  --out_tsv "$RUN_DIR/mcq_val_pred.tsv" \
  --adapter_dir $WS/models/mcq_lora_v1

python -u src/eval_mcq.py \
  --truth_csv data/mcq_val.csv \
  --pred_tsv "$RUN_DIR/mcq_val_pred.tsv" \
  --out_json "$RUN_DIR/mcq_val_metrics.json"

cat "$RUN_DIR/mcq_val_metrics.json"
