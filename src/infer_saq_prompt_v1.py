#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_ANSWER_CHARS = 89
FALLBACK = "idk"

def build_prompt(q: str) -> str:
    return (
        "Answer the question with EXACTLY ONE short phrase (1 to 4 words).\n"
        "No explanation. No list. No commas.\n"
        "Return only the answer text.\n\n"
        f"Question: {q.strip()}\n"
        "Answer:"
    )

def clean_answer(x: str) -> str:
    if not x:
        return ""
    x = x.strip()
    x = x.splitlines()[0].strip()
    x = re.sub(r"^\s*Answer\s*:\s*", "", x, flags=re.IGNORECASE).strip()

    # Keep only first chunk to avoid multi-answer
    x = re.split(r",|;|\band\b|\bor\b|/|\\|\.\s", x, maxsplit=1)[0].strip()

    x = x.strip(' "\'')
    x = re.sub(r"[.?!,:;]+$", "", x).strip()

    # Hard length cap for Codabench
    if len(x) > MAX_ANSWER_CHARS:
        x = x[:MAX_ANSWER_CHARS].rstrip()

    return x

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Input:", args.input_csv)

    out_rows = []
    for i, r in df.iterrows():
        qid = r["ID"]
        q = r.get("en_question", "") or r.get("question", "")
        prompt = build_prompt(q)

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ans = clean_answer(decoded)
        if not ans.strip():
            ans = FALLBACK

        out_rows.append({"ID": qid, "answer": ans})

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df)}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, sep="\t", index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
