"""Prepare and aggregate genuine human LMP annotations.

Existing generated labels are synthetic. This tool creates a blank annotation
sheet and aggregates actual ratings supplied by independent human annotators.
"""
import argparse
from pathlib import Path
import pandas as pd


def prepare_template(data_path, output_path):
    df = pd.read_json(data_path, lines=True)
    required = {"id", "language", "domain", "original", "simplified"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    out = df[["id", "language", "domain", "original", "simplified"]].copy()
    out = out.rename(columns={"id": "pair_id"})
    for col in ["annotator_id", "score", "error_types", "notes"]:
        out[col] = ""
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Blank annotation template written: {output_path}")
    print("Enter actual independent ratings. Do not copy synthetic lmp_score labels.")


def aggregate_annotations(input_path, output_path, min_annotators=2):
    ann = pd.read_csv(input_path, keep_default_na=False)
    required = {"pair_id", "annotator_id", "score", "error_types", "notes"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"Annotation file missing columns: {sorted(missing)}")
    ann["score"] = pd.to_numeric(ann["score"], errors="coerce")
    if ann["score"].isna().any() or not ann["score"].between(1, 10).all():
        raise ValueError("Every annotation needs a numeric score from 1 to 10.")
    if (ann["annotator_id"].astype(str).str.strip() == "").any():
        raise ValueError("annotator_id cannot be blank.")
    if ann.duplicated(["pair_id", "annotator_id"]).any():
        raise ValueError("Each annotator may score a pair only once.")
    counts = ann.groupby("pair_id")["annotator_id"].nunique()
    eligible = counts[counts >= min_annotators].index
    ann = ann[ann["pair_id"].isin(eligible)].copy()
    if ann.empty:
        raise ValueError(f"No pair has at least {min_annotators} independent annotators.")
    grouped = ann.groupby("pair_id").agg(
        human_lmp_score=("score", "mean"),
        human_lmp_median=("score", "median"),
        annotator_count=("annotator_id", "nunique"),
        individual_scores=("score", lambda x: ";".join(f"{v:g}" for v in x)),
        error_types=("error_types", lambda x: ";".join(sorted({e.strip() for cell in x for e in str(cell).split(";") if e.strip()}))),
        annotation_notes=("notes", lambda x: " | ".join(str(v) for v in x if str(v).strip())),
    ).reset_index()
    grouped["human_decision"] = pd.cut(
        grouped["human_lmp_score"], bins=[0, 5, 8, 10],
        labels=["Not Preserved", "Partially Preserved", "Preserved"], include_lowest=True
    ).astype(str)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Aggregated {len(grouped)} pairs from {len(ann)} human ratings: {output_path}")
    print("Retain raw ratings and report annotator agreement separately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human LMP annotation utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("template")
    p.add_argument("--data", default="data/lmp_dataset.jsonl")
    p.add_argument("--out", default="data/human_annotation_template.csv")
    a = sub.add_parser("aggregate")
    a.add_argument("--input", default="data/human_annotations_filled.csv")
    a.add_argument("--out", default="data/human_lmp_reference.csv")
    a.add_argument("--min-annotators", type=int, default=2)
    args = parser.parse_args()
    if args.command == "template":
        prepare_template(args.data, args.out)
    else:
        aggregate_annotations(args.input, args.out, args.min_annotators)
