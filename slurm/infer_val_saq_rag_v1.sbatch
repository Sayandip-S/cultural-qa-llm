#!/bin/bash
#SBATCH --job-name=val_saq_rag
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

RUN_DIR=$WS/runs/$(date +%Y%m%d_%H%M%S)_phase3C_saq_rag_v1
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

python -u src/infer_saq_rag_v1.py \
  --input_csv data/saq_val.csv \
  --train_csv data/saq_train.csv \
  --k 3 \
  --max_new_tokens 16 \
  --out_tsv "$RUN_DIR/saq_val_pred.tsv"

echo "Done. Contents:"
ls -lh "$RUN_DIR"
