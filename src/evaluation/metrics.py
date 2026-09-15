import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
def compute_metrics(preds, truth, threshold=0.5):
    pb = (preds > threshold).astype(int)
    return {"macro_f1": f1_score(truth, pb, average="macro", zero_division=0),
            "micro_f1": f1_score(truth, pb, average="micro", zero_division=0),
            "precision": precision_score(truth, pb, average="macro", zero_division=0),
            "recall": recall_score(truth, pb, average="macro", zero_division=0),
            "subset_acc": accuracy_score(truth, pb)}
