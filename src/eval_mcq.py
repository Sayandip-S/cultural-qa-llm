#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import pandas as pd

def pred_choice_from_row(row) -> str:
    # row has A,B,C,D as booleans (or strings)
    for c in ["A","B","C","D"]:
        v = row[c]
        if isinstance(v, bool) and v:
            return c
        if isinstance(v, str) and v.strip().lower() == "true":
            return c
    return ""  # invalid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth_csv", default="data/mcq_val.csv")
    ap.add_argument("--pred_tsv", required=True)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    truth = pd.read_csv(args.truth_csv)
    pred = pd.read_csv(args.pred_tsv, sep="\t")

    # join by MCQID to be safe
    merged = truth.merge(pred, on="MCQID", how="inner", validate="one_to_one")
    merged["pred_choice"] = merged.apply(pred_choice_from_row, axis=1)
    merged["correct"] = merged["pred_choice"] == merged["answer_idx"]

    overall = float(merged["correct"].mean())
    by_country = merged.groupby("country")["correct"].mean().to_dict()
    by_country = {k: float(v) for k, v in by_country.items()}

    metrics = {
        "mcq_overall_accuracy": overall,
        "mcq_by_country_accuracy": by_country,
        "rows_truth": int(len(truth)),
        "rows_pred": int(len(pred)),
        "rows_merged": int(len(merged)),
    }

    print(json.dumps(metrics, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
