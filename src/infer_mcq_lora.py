#!/usr/bin/env python3
import argparse, re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

LETTER_RE = re.compile(r"\b([ABCD])\b")

def extract_letter(text: str) -> str:
    text = (text or "").strip().upper()
    m = LETTER_RE.search(text)
    if m:
        return m.group(1)
    # fallback: first char if it looks like A/B/C/D
    if text[:1] in ["A","B","C","D"]:
        return text[:1]
    return "A"  # safe fallback

@torch.no_grad()
def generate_letter(tok, model, prompt, device, max_new_tokens=5):
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tok.eos_token_id,
    )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_letter(gen)

def build_prompt(row):
    return (
        "You are a helpful assistant.\n\n"
        "Choose the correct option (A, B, C, or D). Output ONLY the single letter.\n\n"
        f"{row['prompt'].strip()}\n"
        "Answer:"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--adapter_dir", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    df = pd.read_csv(args.input_csv)
    rows = []
    for i, row in df.iterrows():
        prompt = build_prompt(row)
        letter = generate_letter(tok, model, prompt, device=device)
        rows.append({
            "MCQID": row["MCQID"],
            "A": letter == "A",
            "B": letter == "B",
            "C": letter == "C",
            "D": letter == "D",
        })
        if (i+1) % 20 == 0:
            print(f"Processed {i+1}/{len(df)}")

    out = pd.DataFrame(rows, columns=["MCQID","A","B","C","D"])
    out.to_csv(args.out_tsv, sep="\t", index=False)
    print("Wrote:", args.out_tsv)

if __name__ == "__main__":
    main()
