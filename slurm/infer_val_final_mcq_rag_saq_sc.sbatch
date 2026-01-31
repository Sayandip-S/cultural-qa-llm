#!/bin/bash
#SBATCH --job-name=val_final
#SBATCH --output=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.err
#SBATCH --time=02:30:00
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

RUN_DIR=$WS/runs/$(date +%Y%m%d_%H%M%S)_final_val
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

python -u src/infer_mcq_rag_final.py \
  --input_csv data/mcq_val.csv \
  --train_csv data/mcq_train.csv \
  --out_tsv "$RUN_DIR/mcq_val_pred.tsv" \
  --k 6 \
  --vote_n 3 \
  --temperature 0.7 \
  --top_p 0.9

python -u src/infer_saq_sc_final.py \
  --input_csv data/saq_val.csv \
  --out_tsv "$RUN_DIR/saq_val_pred.tsv" \
  --num_samples 5 \
  --temperature 0.8 \
  --top_p 0.9 \
  --max_new_tokens 16

echo "Done:"
ls -lh "$RUN_DIR"
