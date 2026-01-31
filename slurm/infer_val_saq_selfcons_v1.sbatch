#!/bin/bash
#SBATCH --job-name=val_saq_sc
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
export HF_HUB_ENABLE_HF_TRANSFER=1

RUN_DIR=$WS/runs/$(date +%Y%m%d_%H%M%S)_phase3C1_saq_selfcons_v1
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

python -u src/infer_saq_selfcons_v1.py \
  --input_csv data/saq_val.csv \
  --out_tsv "$RUN_DIR/saq_val_pred.tsv" \
  --num_samples 5 \
  --max_new_tokens 16 \
  --temperature 0.8 \
  --top_p 0.9 \
  --seed 42

echo "Done. Contents:"
ls -lh "$RUN_DIR"
