"""Validate independently annotated LMP pairs; never generate or infer labels."""
import argparse
from pathlib import Path
import pandas as pd

REQUIRED = [
    "pair_id", "source_document_id", "language", "jurisdiction",
    "original_text", "simplified_text", "source_document_title",
    "source_url", "source_license_or_reuse_basis", "source_accessed_date",
    "simplification_provenance", "reviewer_1_id", "reviewer_1_lmp_score",
    "reviewer_2_id", "reviewer_2_lmp_score", "adjudicator_id",
    "adjudicated_lmp_score", "review_status",
]
SCORE_COLUMNS = ["reviewer_1_lmp_score", "reviewer_2_lmp_score", "adjudicated_lmp_score"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Completed annotation CSV")
    parser.add_argument("--output", default="data/human_lmp_annotations/validated_lmp_pairs.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str).fillna("")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing))

    for col in SCORE_COLUMNS:
        df[col] = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")

    valid_score = df[SCORE_COLUMNS].apply(lambda col: col.between(1, 10) & col.notna()).all(axis=1)
    required_text = [
        "pair_id", "source_document_id", "language", "jurisdiction",
        "original_text", "simplified_text", "source_document_title",
        "source_url", "source_license_or_reuse_basis", "source_accessed_date",
        "simplification_provenance", "reviewer_1_id", "reviewer_2_id", "adjudicator_id",
    ]
    complete = df[required_text].apply(lambda col: col.str.strip().ne("")).all(axis=1)
    distinct_reviewers = df["reviewer_1_id"].str.strip().ne(df["reviewer_2_id"].str.strip())
    adjudicated = df["review_status"].str.strip().str.lower().eq("adjudicated")
    accepted = valid_score & complete & distinct_reviewers & adjudicated

    accepted_df = df.loc[accepted].copy()
    rejected = df.loc[~accepted].copy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted_df["lmp_score"] = accepted_df["adjudicated_lmp_score"].astype(int)
    accepted_df["label_provenance"] = "independent_two_reviewer_plus_adjudication"
    accepted_df.to_csv(output, index=False, encoding="utf-8-sig")
    rejected_path = output.with_name(output.stem + "_rejected.csv")
    rejected.to_csv(rejected_path, index=False, encoding="utf-8-sig")

    print(f"Input rows: {len(df)}")
    print(f"Accepted adjudicated rows: {len(accepted_df)}")
    print(f"Rejected/pending rows: {len(rejected)}")
    print(f"Validated output: {output}")
    print(f"Review queue: {rejected_path}")
    print("No missing ratings were inferred or filled.")


if __name__ == "__main__":
    main()
