#!/bin/bash
#SBATCH --job-name=test_final
#SBATCH --output=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.err
#SBATCH --time=04:00:00
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

RUN_DIR=$WS/runs/$(date +%Y%m%d_%H%M%S)_final_test
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

# MCQ -> file must be mcq_prediction.tsv
python -u src/infer_mcq_rag_final.py \
  --input_csv data/test_dataset_mcq.csv \
  --train_csv data/mcq_train.csv \
  --out_tsv "$RUN_DIR/mcq_prediction.tsv" \
  --k 6 \
  --vote_n 3 \
  --temperature 0.7 \
  --top_p 0.9

# SAQ -> file must be saq_prediction.tsv
python -u src/infer_saq_sc_final.py \
  --input_csv data/test_dataset_saq.csv \
  --out_tsv "$RUN_DIR/saq_prediction.tsv" \
  --num_samples 5 \
  --temperature 0.8 \
  --top_p 0.9 \
  --max_new_tokens 16

# format sanity: ensure headers exist
head -n 2 "$RUN_DIR/mcq_prediction.tsv"
head -n 2 "$RUN_DIR/saq_prediction.tsv"

# zip for Codabench
cd "$RUN_DIR"
zip -r submission.zip mcq_prediction.tsv saq_prediction.tsv
ls -lh submission.zip
