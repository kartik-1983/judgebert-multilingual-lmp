"""
Baseline semantic similarity methods for Legal Meaning Preservation.

Implements:
1. TF-IDF cosine similarity (lexical baseline)
2. Sentence-BERT cosine similarity (multilingual)
3. BERTScore (F1)

These serve as generic semantic-similarity and lexical baselines against which the
JUDGEBERT-inspired / PreserveLex model is compared.
"""

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from tqdm import tqdm

from utils import load_dataset, compute_metrics, print_metrics, save_results, decision_from_score


def run_tfidf(
    originals: List[str],
    simplifieds: List[str],
    ngram_range: tuple = (1, 2),
) -> np.ndarray:
    """Compute TF-IDF cosine similarity and scale to [1, 10].

    This is a simple lexical overlap baseline. It often performs well on
    general-domain text where surface form is informative, but is expected
    to be weaker than a supervised legal LMP model on obligation / condition
    / exception changes.
    """
    print("Computing TF-IDF cosine similarity baseline...")
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=1, sublinear_tf=True)
    # Fit on the union of originals and simplifieds
    all_texts = list(originals) + list(simplifieds)
    vectorizer.fit(all_texts)

    orig_vecs = vectorizer.transform(originals)
    simp_vecs = vectorizer.transform(simplifieds)

    # Pairwise cosine for corresponding pairs
    cos_sims = np.array([
        cosine_similarity(orig_vecs[i], simp_vecs[i])[0, 0]
        for i in range(len(originals))
    ])
    # Cosine is in [0, 1] for non-negative TF-IDF → map to [1, 10]
    scores = 1.0 + 9.0 * cos_sims
    return scores.astype(float)


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
    """Compute BERTScore F1 and scale to [1, 10]."""
    print(f"Computing BERTScore (model={model_type})...")
    # BERTScore returns P, R, F1
    _, _, f1 = bert_score(
        simplifieds,
        originals,
        lang=lang if lang in ("en", "zh", "de", "fr") else "en",  # fallback
        model_type=model_type,
        verbose=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    f1 = f1.numpy()
    # F1 is roughly [0,1]; map to [1,10]
    scores = 1.0 + 9.0 * f1
    return scores.astype(float)


def evaluate_baselines(
    data_path: str = "data/lmp_dataset.jsonl",
    output_dir: str = "results",
) -> Dict:
    """Run TF-IDF, SBERT and BERTScore baselines and compute metrics."""
    df = load_dataset(data_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {"overall": {}, "by_language": {}}

    originals = df["original"].tolist()
    simplifieds = df["simplified"].tolist()

    # ---- TF-IDF (lexical baseline) ----
    print("\n>>> Running TF-IDF baseline...")
    tfidf_scores = run_tfidf(originals, simplifieds)
    df["tfidf_score"] = tfidf_scores
    df["tfidf_decision"] = [decision_from_score(s) for s in tfidf_scores]
    metrics_tfidf = compute_metrics(df["lmp_score"].values, tfidf_scores)
    print_metrics(metrics_tfidf, "TF-IDF vs Human LMP (Overall)")
    results["overall"]["tfidf"] = metrics_tfidf

    # ---- SBERT ----
    print("\n>>> Running SBERT baseline...")
    sbert_scores = run_sbert(originals, simplifieds)
    df["sbert_score"] = sbert_scores
    df["sbert_decision"] = [decision_from_score(s) for s in sbert_scores]
    metrics_sbert = compute_metrics(df["lmp_score"].values, sbert_scores)
    print_metrics(metrics_sbert, "SBERT vs Human LMP (Overall)")
    results["overall"]["sbert"] = metrics_sbert

    # ---- BERTScore ----
    print("\n>>> Running BERTScore baseline (may take time)...")
    try:
        bs_scores = run_bertscore(
            originals,
            simplifieds,
            model_type="xlm-roberta-base",
        )
        df["bertscore"] = bs_scores
        metrics_bs = compute_metrics(df["lmp_score"].values, bs_scores)
        print_metrics(metrics_bs, "BERTScore vs Human LMP (Overall)")
        results["overall"]["bertscore"] = metrics_bs
    except Exception as e:
        print(f"BERTScore failed or skipped: {e}")
        df["bertscore"] = np.nan
        results["overall"]["bertscore"] = {"error": str(e)}

    # Per-language breakdown
    for lang in ["en", "hi"]:
        mask = df["language"] == lang
        if mask.sum() == 0:
            continue
        lang_results = {}
        for col, name in [("tfidf_score", "tfidf"), ("sbert_score", "sbert")]:
            if col in df.columns:
                m = compute_metrics(
                    df.loc[mask, "lmp_score"].values,
                    df.loc[mask, col].values,
                )
                print_metrics(m, f"{name.upper()} – Language = {lang.upper()}")
                lang_results[name] = m
        results["by_language"][lang] = lang_results

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
