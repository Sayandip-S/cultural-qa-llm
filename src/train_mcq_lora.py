#!/usr/bin/env python3
import argparse, os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files={"train": args.train_jsonl})
    def tokenize(batch):
        return tok(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
            padding=False,
        )
    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
    )

    # LoRA config (good default)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=True,  # works well on A100/H100, if not supported Slurm will still run but may fall back
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        data_collator=collator,
    )
    trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Saved LoRA adapter to:", args.out_dir)

if __name__ == "__main__":
    main()
