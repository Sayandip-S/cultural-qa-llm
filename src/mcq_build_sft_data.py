#!/usr/bin/env python3
import argparse, json
import pandas as pd

SYSTEM = "You are a helpful assistant that answers multiple-choice cultural questions."

def build_prompt(row):
    # row["prompt"] already contains question + options A-D and ends with "Answer:"
    # We just enforce strict output format.
    return (
        f"{SYSTEM}\n\n"
        "Instruction: Choose the correct option (A, B, C, or D). "
        "Output ONLY the single letter.\n\n"
        f"{row['prompt'].strip()}\n"
        "Answer:"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    # answer_idx in your train MCQ is already a letter like "A"/"B"/"C"/"D"
    df["answer_idx"] = df["answer_idx"].astype(str).str.strip()

    out_path = args.out_jsonl
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            prompt = build_prompt(row)
            target = row["answer_idx"]
            # Full text for causal LM training:
            # model learns to continue from "Answer:" to the target letter.
            text = f"{prompt} {target}\n"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} training examples to {out_path}")

if __name__ == "__main__":
    main()
