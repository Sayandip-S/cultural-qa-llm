#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

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

    m = LETTER_RE.search(text.upper())
    if m:
        c = m.group(1).upper()
        if c in {"A","B","C","D"}:
            return c
    return ""

def choice_to_row(mcqid: str, choice: str) -> dict:
    return {"MCQID": mcqid, "A": choice=="A", "B": choice=="B", "C": choice=="C", "D": choice=="D"}

def build_mcq_fewshot_prompt(target_prompt: str, examples: list[tuple[str,str]]) -> str:
    """
    examples: list of (prompt_text_with_options, answer_letter)
    """
    header = (
        "You are answering a multiple-choice cultural question.\n"
        "Return ONLY a single letter: A or B or C or D.\n"
        "No JSON. No explanation. No extra text.\n\n"
        "Here are examples:\n"
    )
    ex_blocks = []
    for ex_prompt, ex_ans in examples:
        ex_blocks.append(f"{ex_prompt.strip()}\n{ex_ans}\n")
    ex_text = "\n".join(ex_blocks)

    return (
        header
        + ex_text
        + "\nNow answer the next question. Return only A/B/C/D.\n\n"
        + target_prompt.strip()
        + "\n"
    )

def build_country_index(train_df: pd.DataFrame):
    """
    For each country, build a TF-IDF vectorizer over the MCQ prompt text.
    Returns dict country -> (vectorizer, matrix, subset_df)
    """
    index = {}
    for country, sub in train_df.groupby("country"):
        texts = sub["prompt"].astype(str).tolist()
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=50000)
        X = vec.fit_transform(texts)
        index[country] = (vec, X, sub.reset_index(drop=True))
    return index

def retrieve_examples(country_index, country: str, target_text: str, k: int) -> list[tuple[str,str]]:
    if country not in country_index:
        return []

    vec, X, sub = country_index[country]
    qv = vec.transform([str(target_text)])
    sims = cosine_similarity(qv, X).ravel()
    # top-k indices
    top_idx = np.argsort(-sims)[:k]

    examples = []
    for idx in top_idx:
        ex_prompt = sub.loc[idx, "prompt"]
        ex_ans = sub.loc[idx, "answer_idx"]
        examples.append((ex_prompt, ex_ans))
    return examples

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--train_csv", default="data/mcq_train.csv")
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--k", type=int, default=4, help="few-shot examples per question")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    train = pd.read_csv(args.train_csv)

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
    print("Train:", args.train_csv)
    print("k:", args.k)

    country_index = build_country_index(train)

    rows = []
    for i, r in df.iterrows():
        mcqid = r["MCQID"]
        country = r["country"]
        target_prompt = r["prompt"]

        examples = retrieve_examples(country_index, country, target_prompt, k=args.k)
        prompt = build_mcq_fewshot_prompt(target_prompt, examples)

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
        choice = extract_choice(decoded) or "A"
        rows.append(choice_to_row(mcqid, choice))

        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{len(df)}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
