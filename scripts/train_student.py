import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_from_disk
from src.models.student import EmotionStudent
from src.utils.seed import set_seed

#Hope you can make it more better

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)   # high LR OK because encoder frozen
    ap.add_argument("--n-samples", type=int, default=100)
    args = ap.parse_args()

    set_seed(42)
    print("Loading dataset...")
    ds = load_from_disk("data/processed")["train"]

    # Use teacher labels as soft targets
    teacher = torch.load("data/teacher_labels/teacher_soft.pt", map_location="cpu").float()
    N = min(len(ds), teacher.size(0), args.n_samples)
    print(f"Training on {N} samples (matches teacher label count)")

    input_ids = torch.tensor(ds["input_ids"][:N], dtype=torch.long)
    attn      = torch.tensor(ds["attention_mask"][:N], dtype=torch.long)
    teacher   = teacher[:N]
    teacher   = torch.nan_to_num(teacher, nan=1.0/27.0).clamp(min=1e-6)
    teacher   = teacher / teacher.sum(dim=-1, keepdim=True)

    print(f"Teacher shape: {teacher.shape} | max prob mean: {teacher.max(-1).values.mean():.3f}")

    loader = DataLoader(TensorDataset(input_ids, attn, teacher),
                        batch_size=args.batch_size, shuffle=True)

    print("Building student (FROZEN encoder)...")
    model = EmotionStudent(num_emotions=27, freeze_encoder=True)

    # Only optimize the trainable head parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    os.makedirs("outputs/checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        # Force encoder to eval mode even in train (batch norm safety)
        if model.freeze_encoder:
            model.encoder.eval()
        total, n, skipped = 0.0, 0, 0
        for ids, msk, tgt in loader:
            logits, _ = model(ids, msk)
            if not torch.isfinite(logits).all():
                skipped += 1
                optimizer.zero_grad()
                continue
            # Soft cross-entropy (KL with teacher as soft target)
            log_p = F.log_softmax(logits, dim=-1)
            loss = -(tgt * log_p).sum(dim=-1).mean()
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += loss.item(); n += 1
        avg = total / max(n, 1)
        print(f"Epoch {epoch+1} | loss = {avg:.4f} | batches={n} skipped={skipped}")
        torch.save(model.state_dict(),
                   f"outputs/checkpoints/student_ep{epoch+1}.pt")

    print("Training done!")


if __name__ == "__main__":
    main()
