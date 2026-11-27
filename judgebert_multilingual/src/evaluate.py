"""
Research-safe evaluation for the JUDGEBERT-inspired LMP regressor.

This script never substitutes SBERT for the trained regression model.
By default it refuses to compute supervised metrics on the repository's
known synthetic/template-generated dataset. Use --allow-simulated-label-metrics
only for a clearly labelled pipeline smoke test, never as research evidence.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import JudgeBERTRegressor, get_tokenizer
from utils import load_dataset, compute_metrics, print_metrics, save_results, decision_from_score
from train import LMPDataset
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_with_model(model, tokenizer, df, device, batch_size=16, max_length=256):
    model.eval()
    loader = DataLoader(
        LMPDataset(df, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = []
    for batch in tqdm(loader, desc="Predict"):
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        predictions.extend(out["predictions"].cpu().numpy().tolist())
    return np.asarray(predictions, dtype=float)


def main(args):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(
            f"Trained checkpoint not found: {checkpoint}. "
            "Evaluation stopped: SBERT is not a substitute for the LMP regressor. "
            "Train the model first or provide --checkpoint."
        )

    data_path = Path(args.data)
    known_synthetic = data_path.as_posix().endswith("data/lmp_dataset.jsonl")
    if known_synthetic and not args.allow_simulated_label_metrics:
        raise SystemExit(
            "The default lmp_dataset.jsonl is documented as template-generated "
            "with simulated scores. No human-performance metrics will be computed. "
            "Use a genuinely human-annotated held-out CSV/JSONL, or pass "
            "--allow-simulated-label-metrics only for a labelled smoke test."
        )
    if not known_synthetic and not args.human_labels_verified:
        raise SystemExit(
            "For a non-default dataset, confirm that labels are genuinely human-rated, "
            "independently reviewed/adjudicated, and held out by source clause using "
            "--human-labels-verified. This flag is an explicit user attestation, not an audit."
        )

    df = load_dataset(args.data)
    required = {"original", "simplified", "lmp_score", "language"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit("Dataset missing required columns: " + ", ".join(sorted(missing)))
    if df.empty:
        raise SystemExit("Dataset is empty.")
    if df[["original", "simplified"]].isna().any().any():
        raise SystemExit("Dataset contains missing original/simplified text.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model_state_dict"), dict):
        raise SystemExit("Checkpoint must contain a model_state_dict mapping.")
    model_name = ckpt.get("model_name", "xlm-roberta-base")
    model = JudgeBERTRegressor(model_name=model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    tokenizer = get_tokenizer(model_name)

    predictions = predict_with_model(
        model, tokenizer, df, device,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    df["preservelex_lmp_prediction"] = predictions
    df["predicted_decision"] = [decision_from_score(x) for x in predictions]

    metrics = compute_metrics(
        pd.to_numeric(df["lmp_score"], errors="raise").to_numpy(),
        predictions,
    )
    evidence_type = (
        "SIMULATED_LABEL_SMOKE_TEST_ONLY"
        if known_synthetic
        else "USER_ATTESTED_HUMAN_LABELS_REQUIRES_AUDIT"
    )
    results = {
        "model_name": model_name,
        "checkpoint": str(checkpoint),
        "dataset": str(data_path),
        "evidence_type": evidence_type,
        "n": int(len(df)),
        "metrics_against_dataset_labels": metrics,
        "warning": (
            "These metrics are not human-validated research evidence."
            if evidence_type != "LABEL_PROVENANCE_REQUIRES_EXTERNAL_VERIFICATION"
            else "Verify annotation provenance, adjudication and split integrity before publication."
        ),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "predictions.csv", index=False)
    save_results(results, str(output / "evaluation_metrics.json"))
    print_metrics(metrics, "Model vs dataset labels (inspect evidence_type before reporting)")
    print(f"Evidence type: {evidence_type}")
    print(f"Saved outputs to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research-safe PreserveLex model evaluation")
    parser.add_argument("--data", default="data/lmp_dataset.jsonl")
    parser.add_argument("--checkpoint", default="models/judgebert_xlmr/best_model.pt")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--human-labels-verified",
        action="store_true",
        help="Attest that the input contains genuinely human-rated, adjudicated, held-out labels.",
    )
    parser.add_argument(
        "--allow-simulated-label-metrics",
        action="store_true",
        help="Allow metrics against known simulated labels for smoke testing only.",
    )
    main(parser.parse_args())
