#!/bin/bash
#SBATCH --job-name=saq_lora_train
#SBATCH --output=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.err
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

set -euo pipefail

cd ~/cultural-qa-llm
source .venv/bin/activate

export WS=/data/horse/ws/sasr043g-cultural-qa-llm
export HF_HOME=$WS/hf_cache
export TRANSFORMERS_CACHE=$WS/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

# Build training jsonl if not already
python -u src/saq_build_sft_data.py

OUT_DIR=$WS/models/saq_lora_v1
mkdir -p "$OUT_DIR"

python -u src/train_saq_lora.py \
  --train_jsonl data/saq_sft_train.jsonl \
  --val_jsonl data/saq_sft_val.jsonl \
  --out_dir "$OUT_DIR" \
  --epochs 2 \
  --lr 2e-4 \
  --batch_size 1 \
  --grad_accum 16 \
  --max_len 384 \
  --seed 42

echo "Saved adapter to: $OUT_DIR"
