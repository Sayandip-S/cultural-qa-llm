#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_ANSWER_CHARS = 89
FALLBACK_ANSWER = "idk"

# ---------- text utils ----------

def clamp_len(s: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    s = (s or "").strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s

def simple_clean(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    # first line only
    s = s.splitlines()[0].strip()
    # remove wrapping quotes
    s = s.strip(' "\'')
    # remove trailing punctuation
    s = re.sub(r"[.?!,:;]+$", "", s).strip()
    return clamp_len(s)

def parse_annotations_en_answers(annotations_str: str) -> List[str]:
    """
    annotations column looks like:
    "[{'answers': [...], 'en_answers': ['football'], 'count': 4}, ...]"
    We'll extract all en_answers and keep unique order.
    """
    out: List[str] = []
    if not annotations_str or not isinstance(annotations_str, str):
        return out
    try:
        data = ast.literal_eval(annotations_str)
        if not isinstance(data, list):
            return out
        for item in data:
            if isinstance(item, dict) and "en_answers" in item:
                ans_list = item["en_answers"]
                if isinstance(ans_list, list):
                    for a in ans_list:
                        a = simple_clean(str(a))
                        if a:
                            out.append(a)
    except Exception:
        return out

    # unique preserve order
    uniq = list(OrderedDict.fromkeys(out).keys())
    return uniq

# ---------- retrieval ----------

def build_retriever(corpus_questions: List[str]) -> Tuple[TfidfVectorizer, "scipy.sparse.csr_matrix"]:
    """
    TF-IDF cosine retrieval. Fast and good enough for this dataset size.
    """
    vect = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X = vect.fit_transform(corpus_questions)
    return vect, X

def topk_indices(vect: TfidfVectorizer, X, query: str, k: int) -> List[int]:
    qv = vect.transform([query])
    # cosine similarity for TF-IDF normalized vectors
    sims = linear_kernel(qv, X).ravel()
    # argsort descending
    idx = sims.argsort()[::-1][:k]
    return idx.tolist()

# ---------- LLM choose-from-candidates ----------

def build_choice_prompt(question: str, candidates: List[str]) -> str:
    """
    Ask model to output ONLY a number 1..N.
    This makes parsing reliable and keeps answers short.
    """
    # Ensure candidates are short and safe
    cand_lines = []
    for i, c in enumerate(candidates, 1):
        cand_lines.append(f"{i}. {c}")

    return (
        "You are answering a cultural question.\n"
        "Choose the best answer from the candidate list.\n"
        "IMPORTANT: Output ONLY the number of the best candidate (e.g., 1).\n"
        "No explanation. No extra text.\n\n"
        f"Question: {question.strip()}\n\n"
        "Candidates:\n"
        + "\n".join(cand_lines)
        + "\n\nAnswer number:"
    )

def parse_choice_number(text: str, n: int) -> int:
    """
    Extract first integer 1..n from model output.
    """
    if not text:
        return 1
    m = re.search(r"(\d+)", text)
    if not m:
        return 1
    try:
        v = int(m.group(1))
        if 1 <= v <= n:
            return v
    except Exception:
        pass
    return 1

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.cpu()
    model.eval()
    return device, tok, model

@torch.no_grad()
def choose_candidate(tok, model, device: str, prompt: str, max_new_tokens: int = 8) -> str:
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
    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return decoded

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="e.g., data/saq_val.csv or data/test_dataset_saq.csv")
    ap.add_argument("--train_csv", required=True, help="e.g., data/saq_train.csv")
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    df_in = pd.read_csv(args.input_csv)
    df_train = pd.read_csv(args.train_csv)

    # Build corpus for retrieval
    train_questions = df_train["en_question"].astype(str).fillna("").tolist()
    vect, X = build_retriever(train_questions)

    device, tok, model = load_model()
    print("Device:", device)
    print("Input:", args.input_csv)
    print("Train corpus:", args.train_csv, "rows=", len(df_train))
    print("top_k:", args.top_k)

    out_rows = []
    for i, row in df_in.iterrows():
        qid = row["ID"]
        q = str(row["en_question"])

        idxs = topk_indices(vect, X, q, k=args.top_k)

        # Build candidate list from retrieved train rows' gold en_answers
        candidates: List[str] = []
        for j in idxs:
            ann = df_train.iloc[j].get("annotations", "")
            for a in parse_annotations_en_answers(str(ann)):
                candidates.append(a)

        # de-dup and keep order
        candidates = list(OrderedDict.fromkeys([simple_clean(c) for c in candidates if c]).keys())

        # Fallback if retrieval produced nothing
        if not candidates:
            ans = FALLBACK_ANSWER
            out_rows.append({"ID": qid, "answer": ans})
            continue

        # Keep candidates to a manageable size (still based on top-10 retrieval)
        # Usually you’ll end up with ~10-30 unique answers.
        candidates = candidates[:30]

        prompt = build_choice_prompt(q, candidates)
        raw = choose_candidate(tok, model, device, prompt, max_new_tokens=args.max_new_tokens)
        choice = parse_choice_number(raw, n=len(candidates))
        ans = candidates[choice - 1]

        ans = simple_clean(ans)
        if not ans.strip():
            ans = FALLBACK_ANSWER

        out_rows.append({"ID": qid, "answer": ans})

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df_in)}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows, columns=["ID", "answer"]).to_csv(out_path, sep="\t", index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
