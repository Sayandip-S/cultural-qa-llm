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
    # Less restrictive than before: allow short phrases like "fish and chips"
    return (
        "You are answering cultural QA.\n"
        "Return ONLY the answer as a short phrase (1 to 8 words).\n"
        "No explanation. No full sentence.\n\n"
        f"Question: {q.strip()}\n"
        "Answer:"
    )

def clean_answer(x: str) -> str:
    if not x:
        return ""

    x = x.strip()

    # First line only
    x = x.splitlines()[0].strip()

    # Remove common prefix
    x = re.sub(r"^\s*answer\s*:\s*", "", x, flags=re.IGNORECASE).strip()

    # If the model starts explaining, cut it off early
    x = re.split(r"\b(because|since|therefore|so that)\b", x, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    # Remove wrapping quotes
    x = x.strip(' "\'')

    # Remove trailing punctuation
    x = re.sub(r"[.?!;:]+$", "", x).strip()

    # Collapse whitespace
    x = re.sub(r"\s+", " ", x).strip()

    # IMPORTANT: do NOT split on "and/or" (breaks correct answers like "fish and chips")
    # We only split on separators that likely indicate multiple answers / extra text.
    x = re.split(r";|/|\\|\(|\)|\.\s", x, maxsplit=1)[0].strip()

    # Word cap (8 words)
    words = x.split()
    if len(words) > 8:
        x = " ".join(words[:8])

    # Hard char cap for Codabench
    if len(x) > MAX_ANSWER_CHARS:
        x = x[:MAX_ANSWER_CHARS].rstrip()

    return x

def load_model(device: str):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.cpu()
    model.eval()
    return tok, model

@torch.no_grad()
def generate_answer(tok, model, prompt: str, device: str, max_new_tokens: int = 16) -> str:
    inputs = tok(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic
        temperature=1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    tail = text[len(prompt):].strip() if text.startswith(prompt) else text
    return clean_answer(tail)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Input:", args.input_csv)

    df = pd.read_csv(args.input_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    tok, model = load_model(device)

    out_rows = []
    for i, row in df.iterrows():
        qid = row["ID"]
        q = row.get("en_question", "") or row.get("question", "")
        prompt = build_prompt(q)

        ans = generate_answer(tok, model, prompt, device=device, max_new_tokens=args.max_new_tokens)
        ans = clean_answer(ans)

        if not ans.strip():
            ans = FALLBACK

        out_rows.append({"ID": qid, "answer": ans})

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df)}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows, columns=["ID", "answer"]).to_csv(out_path, sep="\t", index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
