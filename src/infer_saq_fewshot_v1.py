#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_ANSWER_CHARS = 89
FALLBACK_ANSWER = "idk"

def normalize(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    lines = s.splitlines()
    if not lines:
        return ""
    s = lines[0].strip()
    s = re.sub(r"^\s*(Answer\s*:\s*)", "", s, flags=re.IGNORECASE)
    s = re.split(r",|;|\band\b|\bor\b|/|\\|\(|\)|\.\s", s, maxsplit=1)[0].strip()
    s = s.strip(' "\'')
    s = re.sub(r"[.?!,:;]+$", "", s).strip()
    if len(s) > MAX_ANSWER_CHARS:
        s = s[:MAX_ANSWER_CHARS].rstrip()
    return s

def pick_gold_answer(annotations_str: str) -> str:
    """
    annotations is a stringified Python list of dicts with 'en_answers' list.
    Pick the most frequent canonical answer: first entry's first en_answer.
    """
    try:
        ann = ast.literal_eval(annotations_str)
        if isinstance(ann, list) and len(ann) > 0:
            first = ann[0]
            if isinstance(first, dict):
                en = first.get("en_answers", [])
                if isinstance(en, list) and len(en) > 0:
                    return normalize(str(en[0]))
    except Exception:
        pass
    return FALLBACK_ANSWER

def build_saq_fewshot_prompt(question: str, examples: list[tuple[str,str]]) -> str:
    header = (
        "You are answering cultural QA.\n"
        "Return EXACTLY ONE short answer (1 to 4 words).\n"
        "No explanation. No list. No commas.\n\n"
        "Examples:\n"
    )
    ex_blocks = []
    for q, a in examples:
        ex_blocks.append(f"Q: {q}\nA: {a}\n")
    ex_text = "\n".join(ex_blocks)

    return (
        header
        + ex_text
        + "\nNow answer:\n"
        + f"Q: {question}\nA:"
    )

def build_country_index(train_df: pd.DataFrame):
    """
    For each country, TF-IDF over en_question.
    """
    index = {}
    for country, sub in train_df.groupby("country"):
        texts = sub["en_question"].astype(str).tolist()
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=50000)
        X = vec.fit_transform(texts)
        index[country] = (vec, X, sub.reset_index(drop=True))
    return index

def retrieve_examples(country_index, country: str, target_q: str, k: int) -> list[tuple[str,str]]:
    if country not in country_index:
        return []
    vec, X, sub = country_index[country]
    qv = vec.transform([str(target_q)])
    sims = cosine_similarity(qv, X).ravel()
    top_idx = np.argsort(-sims)[:k]

    examples = []
    for idx in top_idx:
        q = sub.loc[idx, "en_question"]
        a = pick_gold_answer(sub.loc[idx, "annotations"])
        examples.append((q, a))
    return examples

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
def generate(tok, model, prompt: str, device: str, max_new_tokens: int = 16) -> str:
    inputs = tok(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    tail = text[len(prompt):].strip() if text.startswith(prompt) else text
    return normalize(tail)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--train_csv", default="data/saq_train.csv")
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Input:", args.input_csv)
    print("Train:", args.train_csv)
    print("k:", args.k)

    df = pd.read_csv(args.input_csv)
    train = pd.read_csv(args.train_csv)

    country_index = build_country_index(train)
    tok, model = load_model(device)

    out_rows = []
    for i, row in df.iterrows():
        qid = row["ID"]
        q = row["en_question"]
        country = row["country"]

        examples = retrieve_examples(country_index, country, q, k=args.k)
        prompt = build_saq_fewshot_prompt(q, examples)

        ans = generate(tok, model, prompt, device=device, max_new_tokens=args.max_new_tokens)
        if not ans.strip():
            ans = FALLBACK_ANSWER
        out_rows.append({"ID": qid, "answer": ans})

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df)}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows, columns=["ID","answer"]).to_csv(out_path, sep="\t", index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
