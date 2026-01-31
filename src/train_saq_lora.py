#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

@dataclass
class Config:
    max_len: int = 384

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/saq_sft_train.jsonl")
    ap.add_argument("--val_jsonl", default="data/saq_sft_val.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config(max_len=args.max_len)

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token  # important for batching

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Training stability
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print("Applying LoRA...")
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print("Loading datasets...")
    ds = load_dataset(
        "json",
        data_files={"train": args.train_jsonl, "validation": args.val_jsonl},
    )

    def tokenize_fn(batch):
        # concatenate prompt + answer (teacher forcing)
        text = [p + " " + a for p, a in zip(batch["prompt"], batch["answer"])]
        return tok(
            text,
            truncation=True,
            max_length=cfg.max_len,
            padding=False,
        )

    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=ds["train"].column_names)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=collator,
    )

    print("Training...")
    trainer.train()

    print("Saving LoRA adapter...")
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Done. Saved to:", args.out_dir)

if __name__ == "__main__":
    main()
