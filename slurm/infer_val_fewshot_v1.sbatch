#!/bin/bash
#SBATCH --job-name=val_fewshot
#SBATCH --output=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.err
#SBATCH --time=01:00:00
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

RUN_DIR=$WS/runs/$(date +%Y%m%d_%H%M%S)_phase3B_fewshot_v1
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

# FEW-SHOT on VAL
python -u src/infer_mcq_fewshot_v1.py \
  --input_csv data/mcq_val.csv \
  --train_csv data/mcq_train.csv \
  --k 4 \
  --out_tsv "$RUN_DIR/mcq_val_pred.tsv"

python -u src/infer_saq_fewshot_v1.py \
  --input_csv data/saq_val.csv \
  --train_csv data/saq_train.csv \
  --k 4 \
  --out_tsv "$RUN_DIR/saq_val_pred.tsv"

# EVAL (CPU but fine on compute node)
python src/eval_mcq.py \
  --truth_csv data/mcq_val.csv \
  --pred_tsv "$RUN_DIR/mcq_val_pred.tsv" \
  --out_json "$RUN_DIR/mcq_val_metrics.json"

python src/eval_saq.py \
  --truth_csv data/saq_val.csv \
  --pred_tsv "$RUN_DIR/saq_val_pred.tsv" \
  --out_json "$RUN_DIR/saq_val_metrics.json"

echo "Done. Files:"
ls -lh "$RUN_DIR"
