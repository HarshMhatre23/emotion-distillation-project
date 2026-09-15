import argparse, torch, numpy as np
from datasets import load_from_disk
from src.models.student import EmotionStudent
from src.evaluation.metrics import compute_metrics
from src.evaluation.efficiency import measure_latency, model_size_mb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/checkpoints/student_ep3.pt")
    args = ap.parse_args()
    
    ds = load_from_disk("data/processed")["train"]
    N = min(len(ds), 500)
    ids = torch.tensor(ds["input_ids"][:N])
    attn = torch.tensor(ds["attention_mask"][:N])
    hard = np.array(ds["multi_hot_labels"][:N], dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmotionStudent().to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    except:
        print(f"WARNING: Could not load {args.checkpoint}. Using random weights.")
    model.eval()

    all_preds = []
    with torch.no_grad():
        for i in range(0, N, 64):
            lg, _ = model(ids[i:i+64].to(device), attn[i:i+64].to(device))
            all_preds.append(torch.sigmoid(lg).cpu().numpy())
    preds = np.concatenate(all_preds)

    metrics = compute_metrics(preds, hard)
    eff = measure_latency(model, ids[:8].to(device), attn[:8].to(device), device=device)
    size = model_size_mb(model)

    print("\n--- Evaluation Report ---")
    print(f"Accuracy (Macro F1): {metrics['macro_f1']:.4f}")
    print(f"Accuracy (Micro F1): {metrics['micro_f1']:.4f}")
    print(f"Latency: {eff['latency_ms']:.2f} ms")
    print(f"Throughput: {eff['throughput']:.1f} samples/sec")
    print(f"Model Size: {size['int8_mb']:.1f} MB (INT8)")
    print("-------------------------")

if __name__ == "__main__":
    main()
