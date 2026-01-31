#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd

SYSTEM = "You are a helpful assistant for cultural question answering."
INSTRUCTION = (
    "Answer the question with EXACTLY ONE short answer (1 to 4 words). "
    "No explanation. No list. No commas."
)

def pick_target_answer(ann_str: str) -> str:
    """
    annotations is a stringified python list like:
      [{'en_answers': ['football'], 'count': 4}, ...]
    We pick the most frequent en_answer across entries using counts.
    """
    data = ast.literal_eval(ann_str)
    c = Counter()
    for item in data:
        en_answers = item.get("en_answers", []) or []
        count = int(item.get("count", 1))
        for a in en_answers:
            a = str(a).strip()
            if a:
                c[a.lower()] += count
    if not c:
        return "idk"
    # return the most common (but keep original casing simple: lowercase)
    return c.most_common(1)[0][0]

def to_record(row) -> dict:
    q = str(row["en_question"]).strip()
    a = pick_target_answer(row["annotations"])

    # Simple instruct format (works well with Llama 3)
    prompt = (
        f"{SYSTEM}\n\n"
        f"Instruction: {INSTRUCTION}\n"
        f"Question: {q}\n"
        f"Answer:"
    )
    return {"prompt": prompt, "answer": a}

def write_jsonl(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            rec = to_record(r)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/saq_train.csv")
    ap.add_argument("--val_csv", default="data/saq_val.csv")
    ap.add_argument("--out_train", default="data/saq_sft_train.jsonl")
    ap.add_argument("--out_val", default="data/saq_sft_val.jsonl")
    args = ap.parse_args()

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    write_jsonl(train_df, Path(args.out_train))
    write_jsonl(val_df, Path(args.out_val))

    print("Wrote:", args.out_train, "rows=", len(train_df))
    print("Wrote:", args.out_val, "rows=", len(val_df))

if __name__ == "__main__":
    main()
