#!/usr/bin/env python3
from __future__ import annotations
import argparse
import ast
import json
import re
import pandas as pd

def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \"'")
    return s

def extract_gold_answers(annotations_str: str) -> set[str]:
    """
    annotations column looks like a Python list of dicts as a string.
    Each dict contains 'en_answers': [..].
    We'll accept match if prediction equals ANY normalized en_answer.
    """
    try:
        ann = ast.literal_eval(annotations_str)
    except Exception:
        return set()

    gold = set()
    if isinstance(ann, list):
        for item in ann:
            if isinstance(item, dict) and "en_answers" in item:
                for a in item.get("en_answers", []):
                    gold.add(normalize(str(a)))
    return gold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth_csv", default="data/saq_val.csv")
    ap.add_argument("--pred_tsv", required=True)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    truth = pd.read_csv(args.truth_csv)
    pred = pd.read_csv(args.pred_tsv, sep="\t")

    # IMPORTANT: SAQ may have duplicate IDs; evaluation should be row-by-row.
    # We'll align by row order, but also check lengths match.
    if len(truth) != len(pred):
        raise ValueError(f"Row count mismatch: truth={len(truth)} pred={len(pred)}")

    truth = truth.reset_index(drop=True)
    pred = pred.reset_index(drop=True)

    pred_ans = pred["answer"].fillna("").astype(str).map(normalize)
    gold_sets = truth["annotations"].astype(str).map(extract_gold_answers)

    correct = []
    for p, gold in zip(pred_ans, gold_sets):
        correct.append(p in gold)

    truth["correct"] = correct
    overall = float(truth["correct"].mean())
    by_country = truth.groupby("country")["correct"].mean().to_dict()
    by_country = {k: float(v) for k, v in by_country.items()}

    metrics = {
        "saq_overall_accuracy": overall,
        "saq_by_country_accuracy": by_country,
        "rows_truth": int(len(truth)),
        "rows_pred": int(len(pred)),
    }

    print(json.dumps(metrics, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
