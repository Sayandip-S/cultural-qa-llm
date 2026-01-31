#!/bin/bash
#SBATCH --job-name=val_saq_candrag
#SBATCH --output=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.err
#SBATCH --time=02:00:00
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

RUN_DIR=$WS/runs/$(date +%Y%m%d_%H%M%S)_phase3C2_saq_candidate_rag_v1
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

python -u src/infer_saq_candidate_rag_v1.py \
  --input_csv data/saq_val.csv \
  --train_csv data/saq_train.csv \
  --out_tsv "$RUN_DIR/saq_val_pred.tsv" \
  --top_k 10 \
  --max_new_tokens 8 \
  --seed 42

echo "Done. Contents:"
ls -lh "$RUN_DIR"
