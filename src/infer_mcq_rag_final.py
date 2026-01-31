#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

def parse_choices(choices_str: str) -> dict:
    # choices column is a JSON-like dict string; safe parse via json loads after normalization
    # but your file shows it as valid JSON with quotes/indentation most of the time
    try:
        return json.loads(choices_str)
    except Exception:
        # fallback: very small heuristic
        out = {}
        for line in str(choices_str).splitlines():
            m = re.match(r'\s*"([ABCD])"\s*:\s*"(.*)"\s*,?\s*$', line)
            if m:
                out[m.group(1)] = m.group(2)
        return out

def build_retriever(train_df: pd.DataFrame) -> Tuple[TfidfVectorizer, any]:
    # Use prompt text as retrieval corpus (it already contains options)
    corpus = train_df["prompt"].astype(str).fillna("").tolist()
    vect = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1, max_df=0.95)
    X = vect.fit_transform(corpus)
    return vect, X

def retrieve(train_df, vect, X, query_text: str, k: int) -> pd.DataFrame:
    qv = vect.transform([query_text])
    sims = linear_kernel(qv, X).ravel()
    idx = sims.argsort()[::-1][:k]
    return train_df.iloc[idx].copy()

def build_fewshot_block(rows: pd.DataFrame) -> str:
    blocks = []
    for _, r in rows.iterrows():
        # training row has answer_idx letter already
        ans = str(r.get("answer_idx", "")).strip()
        prompt = str(r["prompt"]).strip()
        blocks.append(f"{prompt}\n{ans}\n")
    return "\n".join(blocks).strip()

def build_prompt_rag(test_prompt: str, fewshot: str) -> str:
    # enforce strict output
    return (
        "You are answering a multiple-choice cultural question.\n"
        "Return ONLY one letter: A, B, C, or D.\n"
        "No explanation.\n\n"
        "Examples:\n"
        f"{fewshot}\n\n"
        "Now answer this:\n"
        f"{test_prompt}\n"
    )

@torch.no_grad()
def gen_letter(tok, model, device, prompt: str, do_sample: bool, temperature: float, top_p: float, max_new_tokens: int = 4) -> str:
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
    tail = text[len(prompt):].strip() if text.startswith(prompt) else text
    return tail

def extract_letter(s: str) -> str:
    if not s:
        return "A"
    m = LETTER_RE.search(s)
    if m:
        return m.group(1).upper()
    # fallback: if model outputs json like {"answer_choice":"B"}
    m2 = re.search(r'["\']answer_choice["\']\s*:\s*["\']([ABCD])["\']', s, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return "A"

def one_hot_row(mcqid: str, letter: str) -> dict:
    return {
        "MCQID": mcqid,
        "A": letter == "A",
        "B": letter == "B",
        "C": letter == "C",
        "D": letter == "D",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--train_csv", default="data/mcq_train.csv")
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--k", type=int, default=6, help="retrieved examples")
    ap.add_argument("--vote_n", type=int, default=3, help="self-consistency votes for letter")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    df_in = pd.read_csv(args.input_csv)
    df_train = pd.read_csv(args.train_csv)

    vect, X = build_retriever(df_train)

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

    out_rows = []
    for i, r in df_in.iterrows():
        mcqid = r["MCQID"]
        test_prompt = str(r["prompt"]).strip()

        top = retrieve(df_train, vect, X, test_prompt, k=args.k)
        fewshot = build_fewshot_block(top)
        prompt = build_prompt_rag(test_prompt, fewshot)

        letters = []
        for _ in range(args.vote_n):
            tail = gen_letter(tok, model, device, prompt, do_sample=True, temperature=args.temperature, top_p=args.top_p)
            letters.append(extract_letter(tail))

        # vote
        letter = max(set(letters), key=letters.count)
        out_rows.append(one_hot_row(mcqid, letter))

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df_in)}")

    pd.DataFrame(out_rows, columns=["MCQID","A","B","C","D"]).to_csv(args.out_tsv, sep="\t", index=False)
    print("Wrote:", args.out_tsv)

if __name__ == "__main__":
    main()
