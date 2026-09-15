import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from datasets import load_from_disk

ds = load_from_disk("data/processed")["train"]
multi = np.array(ds["multi_hot_labels"][:2000])
first = np.array([np.where(r > 0.5)[0][0] if (r > 0.5).any() else 0 for r in multi])
print("First-positive label distribution (top 10):")
vals, counts = np.unique(first, return_counts=True)
for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])[:10]:
    print(f"  label {v:2d}: {c} samples ({100*c/len(first):.1f}%)")

soft = __import__("torch").load("data/teacher_labels/teacher_soft.pt", map_location="cpu")
print(f"\nTeacher soft shape: {soft.shape}")
print(f"Teacher soft NaN: {__import__('torch').isnan(soft).any().item()}")
print(f"Teacher soft sum (first 5): {soft[:5].sum(dim=-1).tolist()}")
print(f"Teacher soft max (first 5): {soft[:5].max(dim=-1).values.tolist()}")
