"""
Full evaluation suite:
- Load trained model or use baselines
- Pearson / RMSE / MAE / Spearman
- Per-language analysis
- Sanity checks (identical & unrelated pairs)
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import JudgeBERTRegressor, get_tokenizer
from utils import load_dataset, compute_metrics, print_metrics, save_results, decision_from_score
from train import LMPDataset


@torch.no_grad()
def predict_with_model(
    model,
    tokenizer,
    originals: List[str],
    simplifieds: List[str],
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 256,
) -> np.ndarray:
    """Run the JUDGEBERT model on pairs and return predicted LMP scores."""
    model.eval()
    # Build a temporary DataFrame for the Dataset class
    tmp = pd.DataFrame({
        "original": originals,
        "simplified": simplifieds,
        "lmp_score": [5.0] * len(originals),  # dummy
    })
    loader = DataLoader(
        LMPDataset(tmp, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []
    for batch in tqdm(loader, desc="Predict"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        preds.extend(out["predictions"].cpu().numpy().tolist())
    return np.array(preds)


def run_sanity_checks(df: pd.DataFrame, pred_col: str = "judgebert_score") -> Dict:
    """Evaluate identical and unrelated pairs."""
    results = {}

    identical = df[df["domain"] == "sanity_identical"]
    if len(identical) > 0:
        scores = identical[pred_col].values
        results["identical"] = {
            "n": len(identical),
            "mean_pred": float(np.mean(scores)),
            "min_pred": float(np.min(scores)),
            "max_pred": float(np.max(scores)),
            "pct_above_9": float(np.mean(scores >= 9.0) * 100),
        }
        print("\n--- Sanity Check: Identical pairs ---")
        print(f"  N = {results['identical']['n']}")
        print(f"  Mean predicted LMP = {results['identical']['mean_pred']:.2f}")
        print(f"  % ≥ 9.0             = {results['identical']['pct_above_9']:.1f}%")

    unrelated = df[df["domain"] == "sanity_unrelated"]
    if len(unrelated) > 0:
        scores = unrelated[pred_col].values
        results["unrelated"] = {
            "n": len(unrelated),
            "mean_pred": float(np.mean(scores)),
            "min_pred": float(np.min(scores)),
            "max_pred": float(np.max(scores)),
            "pct_below_2": float(np.mean(scores <= 2.0) * 100),
        }
        print("\n--- Sanity Check: Unrelated pairs ---")
        print(f"  N = {results['unrelated']['n']}")
        print(f"  Mean predicted LMP = {results['unrelated']['mean_pred']:.2f}")
        print(f"  % ≤ 2.0             = {results['unrelated']['pct_below_2']:.1f}%")

    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_dataset(args.data)

    # Load model if checkpoint exists
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = ckpt.get("model_name", "xlm-roberta-base")
        model = JudgeBERTRegressor(model_name=model_name).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        tokenizer = get_tokenizer(model_name)

        preds = predict_with_model(
            model, tokenizer,
            df["original"].tolist(),
            df["simplified"].tolist(),
            device,
            batch_size=args.batch_size,
        )
        df["judgebert_score"] = preds
        df["judgebert_decision"] = [decision_from_score(s) for s in preds]
    else:
        print(f"Checkpoint {ckpt_path} not found. Using SBERT as proxy for demonstration.")
        from baselines import run_sbert
        preds = run_sbert(df["original"].tolist(), df["simplified"].tolist())
        df["judgebert_score"] = preds
        df["judgebert_decision"] = [decision_from_score(s) for s in preds]

    # Overall metrics
    metrics = compute_metrics(df["lmp_score"].values, df["judgebert_score"].values)
    print_metrics(metrics, "JUDGEBERT-style Model vs Human LMP (Overall)")

    results = {"overall": metrics, "by_language": {}, "sanity": {}}

    # Per language
    for lang in ["en", "hi"]:
        mask = df["language"] == lang
        if mask.sum() == 0:
            continue
        m = compute_metrics(
            df.loc[mask, "lmp_score"].values,
            df.loc[mask, "judgebert_score"].values,
        )
        print_metrics(m, f"Language = {lang.upper()}")
        results["by_language"][lang] = m

    # Sanity checks
    results["sanity"] = run_sanity_checks(df, "judgebert_score")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "full_predictions.csv", index=False)
    save_results(results, str(out_dir / "evaluation_metrics.json"))
    print(f"\nEvaluation complete. Results in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/lmp_dataset.jsonl")
    parser.add_argument("--checkpoint", default="models/judgebert_xlmr/best_model.pt")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
