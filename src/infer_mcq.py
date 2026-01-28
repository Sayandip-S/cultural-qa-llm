import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
VALID_CHOICES = ["A", "B", "C", "D"]

def parse_choice(text: str) -> str:
    """
    Robustly extract A/B/C/D from model output.
    Handles:
      - JSON like {"answer_choice":"B"}
      - raw letter B
      - "Answer: B"
      - extra text around it
    """
    if not text:
        return "A"

    # 1) Try to find JSON object substring and parse it
    # Find first {...} block
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        blob = m.group(0)
        try:
            data = json.loads(blob)
            if isinstance(data, dict):
                v = str(data.get("answer_choice", "")).strip().upper()
                if v in VALID_CHOICES:
                    return v
        except Exception:
            pass

    # 2) Direct letter match (prefer standalone A-D)
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)

    # 3) Fallback: last occurrence of A-D anywhere
    letters = re.findall(r"[ABCD]", text.upper())
    if letters:
        return letters[-1]

    return "A"

def load_model(device: str):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.cpu()
    return tok, model

@torch.no_grad()
def generate_choice(tok, model, prompt: str, device: str, max_new_tokens: int = 32) -> str:
    inputs = tok(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # baseline deterministic
        temperature=1.0,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)

    # The decoded text includes the prompt; focus on tail
    tail = text[len(prompt):].strip() if text.startswith(prompt) else text
    return parse_choice(tail)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/test_dataset_mcq.csv")
    ap.add_argument("--out_tsv", default="mcq_prediction.tsv")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Input:", args.input_csv)

    df = pd.read_csv(args.input_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    tok, model = load_model(device)

    preds = []
    for i, row in df.iterrows():
        mcqid = row["MCQID"]
        prompt = row["prompt"]
        choice = generate_choice(tok, model, prompt, device=device)
        preds.append((mcqid, choice))
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df)}")

    # Build output dataframe: MCQID A B C D with True/False
    out_rows = []
    for mcqid, choice in preds:
        out_rows.append({
            "MCQID": mcqid,
            "A": choice == "A",
            "B": choice == "B",
            "C": choice == "C",
            "D": choice == "D",
        })
    out_df = pd.DataFrame(out_rows, columns=["MCQID", "A", "B", "C", "D"])
    out_df.to_csv(args.out_tsv, sep="\t", index=False)
    print("Wrote:", args.out_tsv)

if __name__ == "__main__":
    main()
