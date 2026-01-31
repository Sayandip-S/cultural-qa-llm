#!/bin/bash
#SBATCH --job-name=test_mix
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

RUN_DIR=$WS/runs/$(date +%Y%m%d_%H%M%S)_phase3_test_mix
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

# MCQ: prompt v1
python -u src/infer_mcq_prompt_v1.py \
  --input_csv data/test_dataset_mcq.csv \
  --out_tsv "$RUN_DIR/mcq_prediction.tsv"

# SAQ: baseline
python -u src/infer_saq.py \
  --input_csv data/test_dataset_saq.csv \
  --out_tsv "$RUN_DIR/saq_prediction.tsv"

# Zip for Codabench submission
cd "$RUN_DIR"
zip -r submission.zip mcq_prediction.tsv saq_prediction.tsv

echo "Done:"
ls -lh "$RUN_DIR"
