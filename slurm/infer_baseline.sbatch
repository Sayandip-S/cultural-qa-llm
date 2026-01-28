#!/bin/bash
#SBATCH --job-name=baseline_infer
#SBATCH --nodes=1
#SBATCH --output=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

cd ~/cultural-qa-llm
source .venv/bin/activate

export WS=/data/horse/ws/sasr043g-cultural-qa-llm
export HF_HOME=$WS/hf_cache
export TRANSFORMERS_CACHE=$WS/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

RUN_ID=$(date +%Y%m%d_%H%M%S)_baseline
RUN_DIR=$WS/runs/$RUN_ID
mkdir -p "$RUN_DIR"

echo "RUN_DIR=$RUN_DIR"

python -u src/infer_mcq.py --input_csv data/test_dataset_mcq.csv --out_tsv "$RUN_DIR/mcq_prediction.tsv"
python -u src/infer_saq.py --input_csv data/test_dataset_saq.csv --out_tsv "$RUN_DIR/saq_prediction.tsv"

cd "$RUN_DIR"
zip -q submission.zip mcq_prediction.tsv saq_prediction.tsv

echo "Wrote:"
ls -lh "$RUN_DIR"

