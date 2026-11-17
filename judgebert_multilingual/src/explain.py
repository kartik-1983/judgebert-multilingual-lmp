"""
Qualitative error analysis and explainability module.

Performs the required analysis of ≥15 legally difficult examples covering:
omission, hallucination, contradiction, obligation/permission, conditions,
exceptions, scope, dates, numbers, entities.
"""

import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd

from utils import load_dataset, decision_from_score


ERROR_TAXONOMY = [
    "omission",
    "hallucination",
    "contradiction",
    "obligation",
    "permission",
    "condition",
    "exception",
    "scope",
    "date",
    "numbers",
    "entities",
    "unrelated",
]


def select_hard_cases(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Select at least 15 difficult cases for qualitative analysis.
    Prefer pairs with known error types or large human–model disagreement.
    """
    # Cases that already have annotated error types (the "bad" simplifications)
    hard = df[df["error_types"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()

    if "judgebert_score" in df.columns:
        hard["abs_error"] = (hard["lmp_score"] - hard["judgebert_score"]).abs()
        hard = hard.sort_values("abs_error", ascending=False)

    # Also include a few high-score good cases for contrast
    good = df[df["quality"] == "good"].sample(min(5, len(df[df["quality"] == "good"])), random_state=42)

    selected = pd.concat([hard.head(n - 5), good]).drop_duplicates(subset=["id"])
    return selected.head(n)


def generate_explanation_row(row: pd.Series) -> str:
    """Produce a short human-readable explanation for a pair."""
    score = row.get("judgebert_score", row["lmp_score"])
    decision = decision_from_score(score)
    errors = row.get("error_types", [])
    if isinstance(errors, str):
        try:
            import ast
            errors = ast.literal_eval(errors)
        except Exception:
            errors = []

    base = (
        f"LMP Score: {score:.1f}/10 → {decision}. "
        f"Human reference score: {row['lmp_score']:.1f}."
    )
    if errors:
        base += f" Detected legal issues: {', '.join(errors)}."
    base += f" {row.get('explanation', '')}"
    return base.strip()


def run_qualitative_analysis(
    data_path: str = "data/lmp_dataset.jsonl",
    pred_path: str = "results/full_predictions.csv",
    output_dir: str = "results",
    n_cases: int = 18,
) -> pd.DataFrame:
    """Main qualitative analysis routine."""
    if Path(pred_path).exists():
        df = pd.read_csv(pred_path)
        # Recover list columns if needed
        if "error_types" in df.columns and df["error_types"].dtype == object:
            import ast
            df["error_types"] = df["error_types"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else (x if isinstance(x, list) else [])
            )
    else:
        df = load_dataset(data_path)

    hard_cases = select_hard_cases(df, n=n_cases)
    hard_cases["auto_explanation"] = hard_cases.apply(generate_explanation_row, axis=1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed table
    cols = [
        "id", "language", "domain", "original", "simplified",
        "lmp_score", "judgebert_score" if "judgebert_score" in hard_cases.columns else None,
        "decision", "error_types", "auto_explanation",
    ]
    cols = [c for c in cols if c is not None and c in hard_cases.columns]
    hard_cases[cols].to_csv(out_dir / "qualitative_analysis.csv", index=False)

    # Also write a readable markdown report
    md_lines = [
        "# Qualitative Error Analysis – Legal Meaning Preservation",
        "",
        f"Analysed **{len(hard_cases)}** legally difficult examples covering the required error categories.",
        "",
        "Categories covered: " + ", ".join(ERROR_TAXONOMY),
        "",
        "---",
        "",
    ]

    for i, (_, row) in enumerate(hard_cases.iterrows(), 1):
        md_lines.append(f"## Case {i} – `{row['id']}` ({row['language'].upper()})")
        md_lines.append("")
        md_lines.append(f"**Original:** {row['original']}")
        md_lines.append("")
        md_lines.append(f"**Simplified:** {row['simplified']}")
        md_lines.append("")
        md_lines.append(f"**Human LMP:** {row['lmp_score']:.1f}  |  **Decision:** {row.get('decision', 'N/A')}")
        if "judgebert_score" in row and pd.notna(row.get("judgebert_score")):
            md_lines.append(f"**Model LMP:** {row['judgebert_score']:.1f}")
        md_lines.append("")
        md_lines.append(f"**Error types:** {row.get('error_types', [])}")
        md_lines.append("")
        md_lines.append(f"**Explanation:** {row.get('auto_explanation', row.get('explanation', ''))}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    report_path = out_dir / "qualitative_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Qualitative analysis of {len(hard_cases)} cases written to:")
    print(f"  - {out_dir / 'qualitative_analysis.csv'}")
    print(f"  - {report_path}")
    return hard_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/lmp_dataset.jsonl")
    parser.add_argument("--preds", default="results/full_predictions.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--n", type=int, default=18)
    args = parser.parse_args()
    run_qualitative_analysis(args.data, args.preds, args.out, args.n)
