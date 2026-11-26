"""
Import an openly licensed English-Hindi legal parallel corpus from Hugging Face.

This is an AUXILIARY multilingual legal-domain resource. Its rows are translations,
not original-vs-simplified pairs and it contains no LMP ratings. Therefore this
script deliberately does NOT create lmp_score labels or add these rows to LMP
regression training. It exports a reviewable corpus and a blank human-review sheet.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_ID = "Wasserstoff-AI/legalTransEn_Indic"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
LICENSE = "MIT"


def first_present(row, candidates):
    for key in candidates:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/external_legal_parallel")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="0 imports the full available split")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID, split=args.split)
    if args.max_rows > 0:
        dataset = dataset.select(range(min(args.max_rows, len(dataset))))

    rows = []
    for idx, item in enumerate(dataset):
        english = first_present(item, ["english", "English", "en", "source"])
        hindi = first_present(item, ["hindi", "Hindi", "hi", "target"])
        if not english or not hindi:
            continue
        rows.append({
            "record_id": f"legaltrans_en_hi_{idx:06d}",
            "english": english,
            "hindi": hindi,
            "pair_type": "cross_lingual_legal_translation",
            "source_dataset": DATASET_ID,
            "source_url": DATASET_URL,
            "source_license": LICENSE,
            "human_review_status": "not_reviewed",
            "human_meaning_equivalence": "",
            "reviewer_notes": "",
        })

    if not rows:
        raise RuntimeError(
            "No usable English/Hindi fields found. Inspect dataset columns before proceeding."
        )

    df = pd.DataFrame(rows)
    jsonl_path = out_dir / "legaltrans_en_hi.jsonl"
    csv_path = out_dir / "legaltrans_en_hi.csv"
    review_path = out_dir / "legaltrans_en_hi_human_review.csv"
    metadata_path = out_dir / "SOURCE_METADATA.json"

    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    review = df[[
        "record_id", "english", "hindi", "source_dataset", "source_url",
        "source_license"
    ]].copy()
    review["reviewer_1_equivalence"] = ""
    review["reviewer_2_equivalence"] = ""
    review["adjudicated_equivalence"] = ""
    review["reviewer_notes"] = ""
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "license": LICENSE,
        "upstream_reported_rows": 89951,
        "imported_rows": len(df),
        "split": args.split,
        "pair_type": "English-Hindi legal translation pairs",
        "intended_use": (
            "Auxiliary multilingual legal-domain corpus and cross-lingual review. "
            "Not an LMP simplification dataset; no LMP scores are assigned."
        ),
        "citation_note": "Cite the Hugging Face dataset card and retain MIT attribution.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    print(f"Imported {len(df):,} English-Hindi legal pairs from {DATASET_ID}")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    print(f"Human review template: {review_path}")
    print("No LMP scores were invented. Do not feed this translation corpus to LMP regression training.")


if __name__ == "__main__":
    main()
