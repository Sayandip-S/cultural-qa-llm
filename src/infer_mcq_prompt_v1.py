#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

def build_prompt(prompt: str) -> str:
    # The dataset prompt already includes the options; we only add strict instruction.
    return (
        "You are answering a multiple-choice cultural question.\n"
        "Return ONLY a single letter: A or B or C or D.\n"
        "No JSON. No explanation. No extra text.\n\n"
        f"{prompt.strip()}\n"
    )

def extract_choice(text: str) -> str:
    # Prefer JSON if it exists
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "answer_choice" in obj:
            v = str(obj["answer_choice"]).strip().upper()
            if v in {"A","B","C","D"}:
                return v
    except Exception:
        pass

    # Otherwise find first A/B/C/D token
    m = LETTER_RE.search(text.upper())
    if m:
        c = m.group(1).upper()
        if c in {"A","B","C","D"}:
            return c
    return ""

def choice_to_row(mcqid: str, choice: str) -> dict:
    return {
        "MCQID": mcqid,
        "A": choice == "A",
        "B": choice == "B",
        "C": choice == "C",
        "D": choice == "D",
    }

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=8)
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

    rows = []
    for i, r in df.iterrows():
        mcqid = r["MCQID"]
        prompt = build_prompt(r["prompt"])

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        choice = extract_choice(decoded)
        if choice == "":
            # Fallback: pick A (deterministic) — we could do better later
            choice = "A"
        rows.append(choice_to_row(mcqid, choice))

        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{len(df)}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
