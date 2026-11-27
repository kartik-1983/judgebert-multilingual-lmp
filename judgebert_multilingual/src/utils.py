"""Utility functions for the multilingual JUDGEBERT project."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_dataset(path: str = "data/lmp_dataset.jsonl") -> pd.DataFrame:
    """Load the LMP dataset from JSONL or CSV."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics with safe handling of tiny/constant samples."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) < 2:
        raise ValueError("At least two paired observations are required.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metric inputs must contain only finite numeric values.")

    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
    else:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return {
        "pearson": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman": float(spearman_r),
        "spearman_p": float(spearman_p),
        "rmse": rmse,
        "mae": mae,
        "n": int(len(y_true)),
    }


def decision_from_score(score: float) -> str:
    """Map continuous LMP score to discrete decision."""
    if score >= 8.0:
        return "Preserved"
    elif score >= 5.0:
        return "Partially Preserved"
    else:
        return "Not Preserved"


def save_results(results: Dict[str, Any], path: str) -> None:
    """Save results dictionary to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Pretty-print metrics."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"  Pearson r   : {metrics['pearson']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"  Spearman ρ  : {metrics['spearman']:.4f} (p={metrics['spearman_p']:.2e})")
    print(f"  RMSE        : {metrics['rmse']:.4f}")
    print(f"  MAE         : {metrics['mae']:.4f}")
    print(f"  N           : {metrics['n']}")
    print(f"{'='*50}\n")
