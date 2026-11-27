"""Compute inter-reviewer agreement for completed PreserveLex annotations.

Input: CSV exported from the adjudication workbook with one row per pair.
Required columns: pair_id, reviewer_1_lmp_score, reviewer_2_lmp_score.
Optional dimension columns: reviewer_1_<dimension>, reviewer_2_<dimension>
for obligations, rights_permissions, conditions_exceptions, scope_parties,
dates_amounts. Optional review_status.

This script never creates, fills, averages, or adjudicates labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


DIMENSIONS = [
    "obligations",
    "rights_permissions",
    "conditions_exceptions",
    "scope_parties",
    "dates_amounts",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV exported from completed workbook")
    parser.add_argument("--output", default="results/human_annotation_agreement.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str).fillna("")
    required = {"pair_id", "reviewer_1_lmp_score", "reviewer_2_lmp_score"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit("Missing columns: " + ", ".join(sorted(missing)))

    if df["pair_id"].duplicated().any():
        raise SystemExit("Duplicate pair_id rows found; expected one row per pair.")

    a = pd.to_numeric(df["reviewer_1_lmp_score"], errors="coerce")
    b = pd.to_numeric(df["reviewer_2_lmp_score"], errors="coerce")
    valid = a.between(1, 10) & b.between(1, 10)
    if not valid.any():
        raise SystemExit("No rows contain two valid numeric scores from 1 to 10.")

    av, bv = a[valid].astype(int), b[valid].astype(int)
    diff = (av - bv).abs()
    result = {
        "input_rows": int(len(df)),
        "pairs_with_two_valid_scores": int(valid.sum()),
        "pairs_missing_or_invalid_scores": int((~valid).sum()),
        "exact_score_agreement": float((diff == 0).mean()),
        "within_one_point_agreement": float((diff <= 1).mean()),
        "mean_absolute_score_difference": float(diff.mean()),
        "quadratic_weighted_cohen_kappa": float(
            cohen_kappa_score(av, bv, labels=list(range(1, 11)), weights="quadratic")
        ),
        "dimension_agreement": {},
        "adjudication_rate": None,
        "interpretation_note": (
            "Weighted kappa measures ordinal agreement; it is not accuracy. "
            "Inspect confidence intervals and language/domain subgroups separately. "
            "No missing ratings were imputed."
        ),
    }

    for dim in DIMENSIONS:
        c1, c2 = f"reviewer_1_{dim}", f"reviewer_2_{dim}"
        if c1 in df.columns and c2 in df.columns:
            m = df[c1].str.strip().ne("") & df[c2].str.strip().ne("")
            result["dimension_agreement"][dim] = {
                "n": int(m.sum()),
                "exact_agreement": float((df.loc[m, c1].str.strip().str.lower()
                                          == df.loc[m, c2].str.strip().str.lower()).mean())
                if m.any() else None,
            }

    if "review_status" in df.columns:
        statuses = df["review_status"].str.strip().str.lower()
        result["adjudication_rate"] = float(statuses.isin(
            ["adjudicated", "third_reviewer_adjudicated"]
        ).mean())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
