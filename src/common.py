import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


# Seed every random number generator we use.
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# Read train/val/test CSVs from a processed data folder.
def load_splits(data_dir):
    folder = Path(data_dir)
    if not folder.is_absolute():
        folder = ROOT / folder
    return [pd.read_csv(folder / f"{name}.csv") for name in ("train", "val", "test")]


# Metrics with "fake" (label 1) as the positive class.
def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# Write a result dictionary to results/<name>.json.
def save_result(name, result):
    folder = ROOT / "results"
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"saved {path}")
