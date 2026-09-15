import argparse, torch, os
from datasets import load_from_disk
from src.models.teacher import LLMTeacher

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="data/teacher_labels")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_from_disk(args.data)
    texts = ds["train"]["text"]
    if args.limit > 0: texts = texts[:args.limit]

    print(f"Loading teacher model: {args.model} ...")
    teacher = LLMTeacher(args.model)
    print(f"Generating soft labels for {len(texts)} samples ...")
    probs, hidden = teacher.generate_soft_labels(texts, batch_size=args.batch_size)
    torch.save(probs, os.path.join(args.out, "teacher_soft.pt"))
    torch.save(hidden, os.path.join(args.out, "teacher_hidden.pt"))
    print(f"Saved teacher labels to {args.out}")

if __name__ == "__main__":
    main()
