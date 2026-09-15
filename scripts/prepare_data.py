import os, argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from src.data.preprocessor import tokenize_dataset, multi_hot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="goemotions")
    ap.add_argument("--tokenizer", default="microsoft/deberta-v3-xsmall")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.dataset} ...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = tokenize_dataset(ds, tok, args.max_length)

    def enc(ex):
        return {"multi_hot_labels": multi_hot(ex["labels"], 27).tolist()}
    ds = ds.map(enc, batched=True)
    ds.save_to_disk(args.out)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
