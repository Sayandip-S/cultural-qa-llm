import argparse
import re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

def clean_answer(x: str) -> str:
    if not x:
        return ""
    x = x.strip()
    # take first line only (avoid rambling)
    x = x.splitlines()[0].strip()
    # remove wrapping quotes
    x = x.strip(' "\'')
    # remove trailing punctuation
    x = re.sub(r"[.?!,:;]+$", "", x)
    return x

def build_prompt(q: str) -> str:
    # Keep it strict: one short answer, no explanation
    return (
        "Answer the question with a short phrase (1-4 words). "
        "No explanation. Output only the answer.\n\n"
        f"Question: {q}\n"
        "Answer:"
    )

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
def generate_answer(tok, model, prompt: str, device: str, max_new_tokens: int = 16) -> str:
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
    tail = text[len(prompt):].strip() if text.startswith(prompt) else text
    return clean_answer(tail)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/test_dataset_saq.csv")
    ap.add_argument("--out_tsv", default="saq_prediction.tsv")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
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
        q = row["en_question"]
        prompt = build_prompt(q)
        ans = generate_answer(tok, model, prompt, device=device)
        out_rows.append({"ID": qid, "answer": ans})

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df)}")

    out_df = pd.DataFrame(out_rows, columns=["ID", "answer"])
    out_df.to_csv(args.out_tsv, sep="\t", index=False)
    print("Wrote:", args.out_tsv)

if __name__ == "__main__":
    main()
