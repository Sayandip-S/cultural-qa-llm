#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_ANSWER_CHARS = 89
FALLBACK_ANSWER = "idk"


# ---------- utils ----------

def clamp_len(s: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    s = (s or "").strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.splitlines()[0].strip()
    s = s.strip(' "\'')
    s = re.sub(r"[.?!,:;]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return clamp_len(s)

def is_numeric_question(q: str) -> bool:
    ql = (q or "").lower()
    # common patterns in your dataset
    if "arabic numerals" in ql or "numerals" in ql:
        return True
    if ql.startswith("how many"):
        return True
    # sometimes "Provide in Arabic numerals (e.g., 7, 8) only."
    if "e.g." in ql and "only" in ql and any(ch.isdigit() for ch in ql):
        return True
    return False

def parse_en_answers(annotations_str: str) -> List[str]:
    out: List[str] = []
    if not annotations_str or not isinstance(annotations_str, str):
        return out
    try:
        data = ast.literal_eval(annotations_str)
        if not isinstance(data, list):
            return out
        for item in data:
            if isinstance(item, dict):
                for a in item.get("en_answers", []) or []:
                    a = clean_text(str(a)).lower()
                    if a:
                        out.append(a)
    except Exception:
        return out
    # unique preserve order
    return list(OrderedDict.fromkeys(out).keys())

def build_retriever(corpus_questions: List[str]) -> Tuple[TfidfVectorizer, "scipy.sparse.csr_matrix"]:
    vect = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95)
    X = vect.fit_transform(corpus_questions)
    return vect, X

def topk_with_scores(vect: TfidfVectorizer, X, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
    qv = vect.transform([query])
    sims = linear_kernel(qv, X).ravel()
    idx = sims.argsort()[::-1][:k]
    return idx, sims[idx]


# ---------- LLM chooser (only as tiebreak) ----------

def build_choice_prompt(question: str, candidates: List[str]) -> str:
    lines = [f"{i}. {c}" for i, c in enumerate(candidates, 1)]
    return (
        "Choose the best answer from the candidate list.\n"
        "IMPORTANT: Output ONLY the number of the best candidate (e.g., 1).\n"
        "No explanation.\n\n"
        f"Question: {question.strip()}\n\n"
        "Candidates:\n" + "\n".join(lines) + "\n\nAnswer number:"
    )

def parse_choice_number(text: str, n: int) -> int:
    if not text:
        return 1
    m = re.search(r"(\d+)", text)
    if not m:
        return 1
    v = int(m.group(1))
    if 1 <= v <= n:
        return v
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
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--top_k", type=int, default=50, help="retrieve more, then score/filter")
    ap.add_argument("--max_candidates", type=int, default=30, help="cap unique candidates before scoring")
    ap.add_argument("--llm_top_m", type=int, default=10, help="LLM chooses among top M scored candidates")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    df_in = pd.read_csv(args.input_csv)
    df_train = pd.read_csv(args.train_csv)

    # group train by country for cleaner retrieval
    train_by_country: Dict[str, pd.DataFrame] = dict(tuple(df_train.groupby("country")))
    retrievers = {}  # country -> (vect, X, questions_list, df_country)

    for c, dfc in train_by_country.items():
        qs = dfc["en_question"].astype(str).fillna("").tolist()
        vect, X = build_retriever(qs)
        retrievers[c] = (vect, X, qs, dfc.reset_index(drop=True))

    device, tok, model = load_model()
    print("Device:", device)
    print("Input:", args.input_csv, "rows=", len(df_in))
    print("Train:", args.train_csv, "rows=", len(df_train))
    print("Retrieve top_k:", args.top_k, "LLM top_m:", args.llm_top_m)

    out_rows = []
    for i, row in df_in.iterrows():
        qid = row["ID"]
        q = str(row["en_question"])
        country = row.get("country", None)
        country = str(country) if country is not None else ""

        numeric = is_numeric_question(q)

        # pick country retriever; if unknown, fallback to full train (rare)
        if country in retrievers:
            vect, X, _, dfc = retrievers[country]
        else:
            # fallback: build once on full train (only if needed)
            qs_all = df_train["en_question"].astype(str).fillna("").tolist()
            vect_all, X_all = build_retriever(qs_all)
            vect, X, dfc = vect_all, X_all, df_train.reset_index(drop=True)

        idxs, sims = topk_with_scores(vect, X, q, k=args.top_k)

        # similarity-weighted candidate scoring
        cand_score = defaultdict(float)
        for rank, (j, sim) in enumerate(zip(idxs, sims)):
            ann = dfc.iloc[int(j)].get("annotations", "")
            answers = parse_en_answers(str(ann))
            # weight: similarity + slight preference for top ranks
            w = float(sim) + (1.0 / (1.0 + rank))
            for a in answers:
                if numeric and not re.fullmatch(r"\d+", a):
                    continue
                if (not numeric) and re.fullmatch(r"\d+", a):
                    # avoid numeric-only candidates for non-numeric questions
                    continue
                cand_score[a] += w

        # if no candidates, fallback
        if not cand_score:
            out_rows.append({"ID": qid, "answer": FALLBACK_ANSWER})
            continue

        # pick top candidates by score
        ranked = sorted(cand_score.items(), key=lambda x: x[1], reverse=True)
        candidates = [a for a, _ in ranked[:args.max_candidates]]

        # If the top candidate is clearly dominant, skip LLM
        if len(ranked) == 1 or (ranked[0][1] >= 1.35 * ranked[1][1]):
            ans = candidates[0]
        else:
            # LLM tiebreak among top M candidates only
            top_m = candidates[: max(2, min(args.llm_top_m, len(candidates)))]
            prompt = build_choice_prompt(q, top_m)
            raw = choose_candidate(tok, model, device, prompt, max_new_tokens=args.max_new_tokens)
            choice = parse_choice_number(raw, n=len(top_m))
            ans = top_m[choice - 1]

        ans = clean_text(ans).lower()
        if not ans:
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
