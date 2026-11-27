"""
Baseline semantic similarity methods for Legal Meaning Preservation.

Implements:
1. Sentence-BERT cosine similarity (multilingual)
2. BERTScore (F1)

These serve as generic semantic-similarity baselines against which the
JUDGEBERT-inspired model is compared.
"""

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from tqdm import tqdm

from utils import load_dataset, compute_metrics, print_metrics, save_results, decision_from_score


def run_sbert(
    originals: List[str],
    simplifieds: List[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute cosine similarity with a multilingual Sentence-BERT model.
    Scores are scaled to approximately [1, 10] for comparison with LMP.
    """
    print(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    emb_orig = model.encode(originals, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    emb_simp = model.encode(simplifieds, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)

    # Cosine similarity in [-1, 1] → map to [1, 10]
    cos = util.cos_sim(emb_orig, emb_simp).diagonal().cpu().numpy()
    # Linear map: -1 → 1, +1 → 10
    scores = 1.0 + 4.5 * (cos + 1.0)  # maps [-1,1] → [1,10]
    return scores.astype(float)


def run_bertscore(
    originals: List[str],
    simplifieds: List[str],
    lang: str = "en",
    model_type: str = "xlm-roberta-base",
) -> np.ndarray:
    """Compute multilingual XLM-R BERTScore F1 and scale to [1, 10].

    Pass lang=None because model_type explicitly selects a multilingual encoder;
    do not silently score Hindi with the English-only fallback.
    """
    print(f"Computing multilingual BERTScore (model={model_type}, lang={lang})...")
    _, _, f1 = bert_score(
        simplifieds,
        originals,
        lang=None,
        model_type=model_type,
        verbose=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        rescale_with_baseline=False,
    )
    scores = 1.0 + 9.0 * f1.cpu().numpy()
    return scores.astype(float)

def evaluate_baselines(
    data_path: str = "data/lmp_dataset.jsonl",
    output_dir: str = "results",
) -> Dict:
    """Run both baselines and compute metrics overall and by language."""
    df = load_dataset(data_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {"overall": {}, "by_language": {}}

    # ---- SBERT ----
    print("\n>>> Running SBERT baseline...")
    sbert_scores = run_sbert(df["original"].tolist(), df["simplified"].tolist())
    df["sbert_score"] = sbert_scores
    df["sbert_decision"] = [decision_from_score(s) for s in sbert_scores]

    metrics_sbert = compute_metrics(df["lmp_score"].values, sbert_scores)
    print_metrics(metrics_sbert, "SBERT vs Human LMP (Overall)")
    results["overall"]["sbert"] = metrics_sbert

    # ---- Multilingual BERTScore, evaluated independently per language ----
    print("\n>>> Running multilingual XLM-R BERTScore baseline...")
    df["bertscore"] = np.nan
    results["overall"]["bertscore"] = {}
    results["by_language"].setdefault("en", {})
    results["by_language"].setdefault("hi", {})
    for lang in ["en", "hi"]:
        mask = df["language"] == lang
        if not mask.any():
            continue
        try:
            scores = run_bertscore(
                df.loc[mask, "original"].tolist(),
                df.loc[mask, "simplified"].tolist(),
                lang=lang,
                model_type="xlm-roberta-base",
            )
            df.loc[mask, "bertscore"] = scores
            metrics = compute_metrics(df.loc[mask, "lmp_score"].values, scores)
            print_metrics(metrics, f"BERTScore – Language = {lang.upper()}")
            results["by_language"][lang]["bertscore"] = metrics
        except Exception as e:
            print(f"BERTScore failed for {lang}: {e}")
            results["by_language"][lang]["bertscore"] = {"error": str(e)}
    valid = df["bertscore"].notna()
    if valid.any():
        results["overall"]["bertscore"] = compute_metrics(
            df.loc[valid, "lmp_score"].values, df.loc[valid, "bertscore"].values
        )
        print_metrics(results["overall"]["bertscore"], "BERTScore vs LMP labels (available language rows)")

    # Per-language breakdown for SBERT
    for lang in ["en", "hi"]:
        mask = df["language"] == lang
        if mask.sum() == 0:
            continue
        m = compute_metrics(df.loc[mask, "lmp_score"].values, df.loc[mask, "sbert_score"].values)
        print_metrics(m, f"SBERT – Language = {lang.upper()}")
        results["by_language"][lang] = {"sbert": m}

    # Save predictions
    pred_path = Path(output_dir) / "baseline_predictions.csv"
    df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    save_results(results, str(Path(output_dir) / "baseline_metrics.json"))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/lmp_dataset.jsonl")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    evaluate_baselines(args.data, args.out)
