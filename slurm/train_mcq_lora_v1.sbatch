#!/bin/bash
#SBATCH --job-name=mcq_lora_train
#SBATCH --output=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/sasr043g-cultural-qa-llm/logs/%x-%j.err
#SBATCH --time=02:00:00
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

mkdir -p $WS/models $WS/tmp

python -u src/mcq_build_sft_data.py \
  --input_csv data/mcq_train.csv \
  --out_jsonl $WS/tmp/mcq_sft_train.jsonl

python -u src/train_mcq_lora.py \
  --train_jsonl $WS/tmp/mcq_sft_train.jsonl \
  --out_dir $WS/models/mcq_lora_v1 \
  --epochs 2 \
  --lr 2e-4 \
  --max_len 512 \
  --batch_size 1 \
  --grad_accum 8
