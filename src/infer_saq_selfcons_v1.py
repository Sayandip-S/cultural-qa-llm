#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

MAX_ANSWER_CHARS = 89
FALLBACK_ANSWER = "idk"


def clean_answer(x: str) -> str:
    """Aggressive normalization for exact-match style evaluation."""
    if not x:
        return ""
    x = x.strip()

    # first line only
    lines = x.splitlines()
    x = lines[0].strip() if lines else x.strip()

    # remove common prefixes
    x = re.sub(r"^\s*(Answer\s*:\s*)", "", x, flags=re.IGNORECASE).strip()

    # remove wrapping quotes
    x = x.strip(' "\'')

    # if model spills into sentences, keep first chunk
    x = re.split(r",|;|\band\b|\bor\b|/|\\|\(|\)|\.\s", x, maxsplit=1)[0].strip()

    # remove trailing punctuation
    x = re.sub(r"[.?!,:;]+$", "", x).strip()

    # normalize whitespace
    x = re.sub(r"\s+", " ", x).strip()

    # lowercase to match annotations often being lowercase
    x = x.lower()

    # enforce char limit (Codabench SAQ check)
    if len(x) > MAX_ANSWER_CHARS:
        x = x[:MAX_ANSWER_CHARS].rstrip()

    return x


def build_prompt(q: str) -> str:
    # The tighter the prompt, the less rambling.
    return (
        "Return EXACTLY ONE short answer (1 to 4 words). "
        "No explanation. No list. No commas. "
        "Output only the answer text.\n\n"
        f"Q: {q}\n"
        "A:"
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
    model.eval()
    return tok, model


@torch.no_grad()
def sample_once(tok, model, prompt: str, device: str, max_new_tokens: int,
                temperature: float, top_p: float) -> str:
    inputs = tok(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    tail = text[len(prompt):].strip() if text.startswith(prompt) else text
    return clean_answer(tail)


def vote_answer(samples: list[str]) -> str:
    # Remove empties
    samples = [s for s in samples if s and s.strip()]
    if not samples:
        return ""

    # Most common normalized answer wins
    counts = Counter(samples)
    best, _ = counts.most_common(1)[0]
    return best


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

    # Reproducibility
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Input:", args.input_csv)
    print("Samples per question:", args.num_samples)

    df = pd.read_csv(args.input_csv)

    tok, model = load_model(device)

    out_rows = []
    for i, row in df.iterrows():
        qid = row["ID"]
        q = str(row["en_question"])
        prompt = build_prompt(q)

        samples = []
        for _ in range(args.num_samples):
            s = sample_once(
                tok, model, prompt, device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            samples.append(s)

        ans = vote_answer(samples)
        ans = clean_answer(ans)
        if not ans.strip():
            ans = FALLBACK_ANSWER

        out_rows.append({"ID": qid, "answer": ans})

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df)}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows, columns=["ID", "answer"]).to_csv(out_path, sep="\t", index=False)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
