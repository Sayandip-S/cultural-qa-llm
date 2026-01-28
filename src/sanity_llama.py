import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

def main():
    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
    print("CUDA available:", torch.cuda.is_available())

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.cpu()

    prompt = "Q: On which holiday do all family members tend to reunite in the US?\nA:"
    inputs = tok(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
        )

    print("=== OUTPUT ===")
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
