#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

MAX_ANSWER_CHARS = 89
FALLBACK_ANSWER = "idk"

# --- Answer cleaning (keep it strict + Codabench-safe) ---
def clean_answer(x: str) -> str:
    if not x:
        return ""

    x = x.strip()

    # first line only
    lines = x.splitlines()
    x = lines[0].strip() if lines else x.strip()

    # remove "Answer:" prefix
    x = re.sub(r'^\s*(Answer\s*:\s*)', '', x, flags=re.IGNORECASE).strip()

    # keep only first chunk (single-answer rule)
    x = re.split(r",|;|\band\b|\bor\b|/|\\|\(|\)|\.\s", x, maxsplit=1)[0].strip()

    # remove wrapping quotes
    x = x.strip(' "\'')

    # remove trailing punctuation
    x = re.sub(r"[.?!,:;]+$", "", x).strip()

    # normalize casing for exact match (usually annotations are lowercase-ish)
    x = x.lower()

    # hard char limit
    if len(x) > MAX_ANSWER_CHARS:
        x = x[:MAX_ANSWER_CHARS].rstrip()

    return x


# --- Parse training annotations to get a canonical gold answer ---
def canonical_train_answer(annotations_str: str) -> str:
    """
    annotations in your CSV looks like:
      "[{'answers': [...], 'en_answers': ['football'], 'count': 4}, ...]"
    We'll pick the en_answers[0] from the highest-count entry.
    """
    if not isinstance(annotations_str, str) or not annotations_str.strip():
        return ""

    try:
        data = ast.literal_eval(annotations_str)
        if not isinstance(data, list) or not data:
            return ""
        # sort by count desc, fallback if missing
        data_sorted = sorted(
            data,
            key=lambda d: int(d.get("count", 0)) if isinstance(d, dict) else 0,
            reverse=True,
        )
        top = data_sorted[0]
        if isinstance(top, dict):
            en = top.get("en_answers", None)
            if isinstance(en, list) and len(en) > 0:
                return clean_answer(str(en[0]))
    except Exception:
        return ""

    return ""


# --- Prompt building with retrieved examples ---
def build_prompt(query: str, examples: List[Tuple[str, str]]) -> str:
    # Strict instructions + few retrieved QA pairs
    parts = [
        "You are answering cultural QA.",
        "Return EXACTLY ONE short answer (1 to 4 words).",
        "No explanation. No list. No commas.",
        "Output only the answer text.",
        "",
        "Here are examples:",
    ]
    for i, (q, a) in enumerate(examples, start=1):
        parts.append(f"Example {i}")
        parts.append(f"Q: {q}")
        parts.append(f"A: {a}")
        parts.append("")

    parts.append("Now answer this question.")
    parts.append(f"Q: {query}")
    parts.append("A:")
    return "\n".join(parts)


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
def generate(tok, model, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tok(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic for now
        temperature=1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)

    # focus on tail after prompt
    tail = text[len(prompt):].strip() if text.startswith(prompt) else text
    return clean_answer(tail)


def build_retriever(train_questions: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
    )
    X = vec.fit_transform(train_questions)
    return vec, X


def retrieve_examples(
    vec: TfidfVectorizer,
    X_train,
    train_df: pd.DataFrame,
    query: str,
    country: str,
    k: int,
) -> List[Tuple[str, str]]:
    qv = vec.transform([query])
    sims = (X_train @ qv.T).toarray().ravel()

    # Prefer same-country examples first (if available)
    if "country" in train_df.columns:
        mask_same = (train_df["country"].astype(str) == str(country))
    else:
        mask_same = np.array([False] * len(train_df))

    idx_same = np.where(mask_same.values if hasattr(mask_same, "values") else mask_same)[0]
    idx_all = np.arange(len(train_df))

    def topk_from_indices(indices: np.ndarray, kk: int) -> List[int]:
        if len(indices) == 0:
            return []
        sub = indices[np.argsort(-sims[indices])]
        return sub[:kk].tolist()

    chosen = topk_from_indices(idx_same, k)
    if len(chosen) < k:
        # fill remaining from all (excluding already chosen)
        remaining = k - len(chosen)
        others = idx_all[np.argsort(-sims[idx_all])]
        others = [i for i in others.tolist() if i not in chosen]
        chosen.extend(others[:remaining])

    examples: List[Tuple[str, str]] = []
    for i in chosen:
        q_ex = str(train_df.loc[i, "en_question"])
        a_ex = str(train_df.loc[i, "gold_answer"])
        examples.append((q_ex, a_ex))
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)   # val csv
    ap.add_argument("--train_csv", required=True)   # train csv
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("VAL:", args.input_csv)
    print("TRAIN:", args.train_csv)
    print("k:", args.k)

    val_df = pd.read_csv(args.input_csv)
    train_df = pd.read_csv(args.train_csv)

    # Prepare canonical gold answers for retrieval examples
    train_df = train_df.copy()
    train_df["gold_answer"] = train_df["annotations"].apply(canonical_train_answer)

    # Drop rows with empty gold answers (rare, but possible)
    train_df = train_df[train_df["gold_answer"].astype(str).str.strip() != ""].reset_index(drop=True)

    vec, X_train = build_retriever(train_df["en_question"].astype(str).tolist())
    tok, model = load_model(device)

    out_rows = []
    for i, row in val_df.iterrows():
        qid = row["ID"]
        q = str(row["en_question"])
        c = row.get("country", "")

        examples = retrieve_examples(vec, X_train, train_df, q, str(c), k=args.k)
        prompt = build_prompt(q, examples)

        ans = generate(tok, model, prompt, device=device, max_new_tokens=args.max_new_tokens)
        ans = clean_answer(ans)

        if not ans.strip():
            ans = FALLBACK_ANSWER

        out_rows.append({"ID": qid, "answer": ans})

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(val_df)}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows, columns=["ID", "answer"]).to_csv(out_path, sep="\t", index=False)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
