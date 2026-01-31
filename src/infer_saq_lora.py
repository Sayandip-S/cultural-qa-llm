#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_ID = "meta-llama/Meta-Llama-3-8B"
MAX_ANSWER_CHARS = 89
FALLBACK = "idk"

def clean_answer(x: str) -> str:
    if not x:
        return ""
    x = x.strip()
    x = x.splitlines()[0].strip()
    x = re.sub(r'^\s*(Answer\s*:\s*)', '', x, flags=re.IGNORECASE)
    x = re.split(r",|;|\band\b|\bor\b|/|\\|\(|\)|\.\s", x, maxsplit=1)[0].strip()
    x = x.strip(' "\'')
    x = re.sub(r"[.?!,:;]+$", "", x).strip()
    if len(x) > MAX_ANSWER_CHARS:
        x = x[:MAX_ANSWER_CHARS].rstrip()
    return x

def build_prompt(q: str) -> str:
    return (
        "You are answering cultural QA.\n"
        "Return EXACTLY ONE short answer (1 to 4 words).\n"
        "No explanation. No list. No commas.\n"
        f"Question: {q}\n"
        "Answer:"
    )

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Loading base + adapter:", args.adapter_dir)

    tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    df = pd.read_csv(args.input_csv)
    rows = []

    for i, r in df.iterrows():
        qid = r["ID"]
        q = r["en_question"]
        prompt = build_prompt(q)

        inputs = tok(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

        text = tok.decode(out[0], skip_special_tokens=True)
        tail = text[len(prompt):].strip() if text.startswith(prompt) else text
        ans = clean_answer(tail)
        if not ans.strip():
            ans = FALLBACK

        rows.append({"ID": qid, "answer": ans})

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df)}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
