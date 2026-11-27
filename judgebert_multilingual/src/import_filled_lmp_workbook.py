"""Convert the PreserveLex 100-pair workbook into a provenance-labelled dev dataset.

This workbook is synthetic reference data. Outputs are for development/training
smoke tests only, never human-validated evaluation or publication metrics.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED = {
    "pair_id", "domain", "original_text", "simplified_text", "lmp_score",
    "obligations", "rights_permissions", "deadlines_amounts",
    "conditions_exceptions", "scope_parties", "change_description",
    "annotator_notes", "confidence",
}
DIMENSIONS = [
    "obligations", "rights_permissions", "deadlines_amounts",
    "conditions_exceptions", "scope_parties",
]
ALLOWED_FLAGS = {"Preserved", "Changed", "Potential Change"}


def detect_language(text: str) -> str:
    """Detect supported script conservatively; workbook currently contains English."""
    devanagari = sum("\u0900" <= ch <= "\u097f" for ch in text)
    latin = sum(("a" <= ch.lower() <= "z") for ch in text)
    if devanagari and latin:
        raise ValueError("Mixed Devanagari/Latin text requires manual language review.")
    if devanagari:
        return "hi"
    if latin:
        return "en"
    raise ValueError("Could not infer en/hi from text; add an explicit language label.")


def convert(source: Path, output: Path) -> pd.DataFrame:
    df = pd.read_excel(source, sheet_name="LMP_Dataset_100_Pairs")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Workbook missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Workbook contains no data rows.")
    if df["pair_id"].duplicated().any():
        raise ValueError("pair_id values must be unique.")
    for col in ["original_text", "simplified_text"]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} contains blank text.")
    scores = pd.to_numeric(df["lmp_score"], errors="raise")
    if not scores.between(1, 10).all():
        raise ValueError("lmp_score must be within [1, 10].")
    for col in DIMENSIONS:
        invalid = set(df[col].dropna().astype(str)) - ALLOWED_FLAGS
        if invalid or df[col].isna().any():
            raise ValueError(f"Invalid {col} values: {sorted(invalid)}")
    if df[["domain", "change_description", "annotator_notes", "confidence"]].isna().any().any():
        raise ValueError("Metadata fields contain missing values.")
    if df.duplicated(subset=["original_text", "simplified_text"]).any():
        raise ValueError("Duplicate original/simplified pairs detected.")

    languages = []
    for row in df.itertuples(index=False):
        src_lang = detect_language(str(row.original_text))
        dst_lang = detect_language(str(row.simplified_text))
        if src_lang != dst_lang:
            raise ValueError(f"Language mismatch in pair_id={row.pair_id}: {src_lang}/{dst_lang}")
        languages.append(src_lang)

    out = df.rename(columns={
        "original_text": "original",
        "simplified_text": "simplified",
    }).copy()
    out["language"] = languages
    out["id"] = out["pair_id"].astype(str).map(lambda x: f"synthetic_workbook_{x}")
    out["label_provenance"] = "synthetic_reference"
    out["evidence_type"] = "synthetic_development_only"
    out["split_group"] = out["language"].astype(str) + "::" + out["original"].astype(str).str.strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8")
    jsonl = output.with_suffix(".jsonl")
    out.to_json(jsonl, orient="records", lines=True, force_ascii=False)
    metadata = {
        "source_workbook": source.name,
        "rows": int(len(out)),
        "language_counts": out["language"].value_counts().to_dict(),
        "domain_counts": out["domain"].value_counts().to_dict(),
        "label_provenance": "synthetic_reference",
        "permitted_use": ["development", "pipeline testing", "exploratory training"],
        "prohibited_claims": [
            "human annotation agreement", "human-validated accuracy",
            "legal correctness", "English-Hindi performance", "domain generalization",
            "official JUDGEBERT comparison",
        ],
        "note": "The workbook README states that labels are synthetic. Do not present them as human ratings.",
    }
    output.with_name("DATA_PROVENANCE.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/PreserveLex_LMP_100_Pairs_Dataset_Filled.xlsx")
    parser.add_argument("--output", default="data/synthetic_development/preservelex_100_pairs.csv")
    args = parser.parse_args()
    result = convert(Path(args.input), Path(args.output))
    print(f"Converted {len(result)} rows to {args.output}")
    print("Provenance: SYNTHETIC — development only; not human-validated evaluation data.")
