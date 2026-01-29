#!/usr/bin/env python3
"""
Create reproducible train/val splits for MCQ and SAQ from the provided training CSVs.

Outputs:
  data/mcq_train.csv
  data/mcq_val.csv
  data/saq_train.csv
  data/saq_val.csv

Rules:
- Use ONLY train datasets.
- Stratify by country.
- Fixed random seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_stratified(
    df: pd.DataFrame,
    stratify_col: str,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stratify_col not in df.columns:
        raise ValueError(f"Missing column '{stratify_col}' in df. Columns={df.columns.tolist()}")

    # If any class has too few samples, sklearn stratify may fail.
    # We'll detect and fall back to non-stratified split with a warning.
    counts = df[stratify_col].value_counts(dropna=False)
    too_small = counts[counts < 2]
    if len(too_small) > 0:
        print(
            f"[WARN] Some '{stratify_col}' groups have <2 samples. "
            f"Falling back to non-stratified split for these data.\n"
            f"Groups:\n{too_small.to_string()}\n"
        )
        train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed, shuffle=True)
        return train_df, val_df

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=df[stratify_col],
    )
    return train_df, val_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcq_in", default="data/train_dataset_mcq.csv", help="Path to MCQ train CSV")
    ap.add_argument("--saq_in", default="data/train_dataset_saq.csv", help="Path to SAQ train CSV")
    ap.add_argument("--out_dir", default="data", help="Output directory for split CSVs")
    ap.add_argument("--val_ratio", type=float, default=0.10, help="Validation ratio (e.g. 0.10)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- MCQ ---
    mcq = pd.read_csv(args.mcq_in)
    # Train MCQ has column 'country' (e.g., China, Iran, UK, US) in your earlier printout.
    if "country" not in mcq.columns:
        raise ValueError(f"MCQ file missing 'country'. Columns={mcq.columns.tolist()}")

    mcq_train, mcq_val = split_stratified(mcq, "country", args.val_ratio, args.seed)

    mcq_train_path = out_dir / "mcq_train.csv"
    mcq_val_path = out_dir / "mcq_val.csv"
    mcq_train.to_csv(mcq_train_path, index=False)
    mcq_val.to_csv(mcq_val_path, index=False)

    # --- SAQ ---
    saq = pd.read_csv(args.saq_in)
    # Train SAQ has 'country' (e.g., IR, GB, US, CN) in your earlier printout.
    if "country" not in saq.columns:
        raise ValueError(f"SAQ file missing 'country'. Columns={saq.columns.tolist()}")

    saq_train, saq_val = split_stratified(saq, "country", args.val_ratio, args.seed)

    saq_train_path = out_dir / "saq_train.csv"
    saq_val_path = out_dir / "saq_val.csv"
    saq_train.to_csv(saq_train_path, index=False)
    saq_val.to_csv(saq_val_path, index=False)

    # --- Print summary ---
    print("=== Split complete ===")
    print(f"Seed: {args.seed}  Val ratio: {args.val_ratio}")
    print("\nMCQ:")
    print(f"  input: {args.mcq_in}")
    print(f"  train: {mcq_train_path}  rows={len(mcq_train)}")
    print(f"  val:   {mcq_val_path}    rows={len(mcq_val)}")
    print("  by-country (val):")
    print(mcq_val["country"].value_counts().to_string())

    print("\nSAQ:")
    print(f"  input: {args.saq_in}")
    print(f"  train: {saq_train_path}  rows={len(saq_train)}")
    print(f"  val:   {saq_val_path}    rows={len(saq_val)}")
    print("  by-country (val):")
    print(saq_val["country"].value_counts().to_string())


if __name__ == "__main__":
    main()
