#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from answer_utils import (
    final_postprocess,
    normalize_for_exact_match,
    FALLBACK_ANSWER,
)

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

def build_prompt(q: str) -> str:
    return (
        "You are answering cultural question answering.\n"
        "Return EXACTLY ONE short answer (1 to 4 words).\n"
        "No explanation. No list. No commas.\n"
        f"Question: {q.strip()}\n"
        "Answer:"
    )

@torch.no_grad()
def generate_one(tok, model, device: str, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> str:
    inputs = tok(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)

    # get tail after prompt
    tail = text[len(prompt):].strip() if text.startswith(prompt) else text
    return tail

def vote(answers: list[str]) -> str:
    """Vote by normalized form but return the shortest original that maps to the winner."""
    norm = [normalize_for_exact_match(a) for a in answers]
    c = Counter(norm)
    winner_norm, _ = c.most_common(1)[0]

    # pick shortest original among those that normalize to winner (helps exact match)
    candidates = [a for a, n in zip(answers, norm) if n == winner_norm]
    candidates = sorted(candidates, key=lambda x: len(x))
    return candidates[0] if candidates else FALLBACK_ANSWER

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    df = pd.read_csv(args.input_csv)
    out_rows = []

    for i, r in df.iterrows():
        qid = r["ID"]
        q = str(r["en_question"])
        prompt = build_prompt(q)

        samples = []
        for _ in range(args.num_samples):
            tail = generate_one(
                tok, model, device, prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            samples.append(tail)

        chosen = vote(samples)
        final = final_postprocess(chosen, q)

        out_rows.append({"ID": qid, "answer": final})

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df)}")

    pd.DataFrame(out_rows, columns=["ID", "answer"]).to_csv(args.out_tsv, sep="\t", index=False)
    print("Wrote:", args.out_tsv)

if __name__ == "__main__":
    main()
