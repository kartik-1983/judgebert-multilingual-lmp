"""Import ASSET human simplification ratings as an auxiliary benchmark.

ASSET is general-domain English sentence simplification, not legal text.
Keep its 0-100 crowd ratings on their native scale; never relabel them as
legal LMP 1-10 scores or merge them into legal training labels.
"""
import argparse
import json
import hashlib
from pathlib import Path
import pandas as pd
from datasets import load_dataset

DATASET_ID = "facebook/asset"
DATASET_URL = "https://huggingface.co/datasets/facebook/asset"
REPOSITORY_URL = "https://github.com/facebookresearch/asset"
LICENSE = "CC BY-SA 4.0"
CITATION = (
    "Alva-Manchego, F., Martin, L., Bordes, A., Scarton, C., Sagot, B., "
    "& Specia, L. (2020). ASSET: A Dataset for Tuning and Evaluation of "
    "Sentence Simplification Models with Multiple Rewriting Transformations. "
    "Proceedings of ACL 2020."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/external_asset")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="0 imports all rows from the ratings split")
    args = parser.parse_args()

    dataset = load_dataset(DATASET_ID, "ratings", split="full")
    if args.max_rows:
        dataset = dataset.select(range(min(args.max_rows, len(dataset))))

    df = dataset.to_pandas()
    # ClassLabel values can appear as strings, Python ints, or NumPy ints
    # after conversion to pandas. Decode using the dataset feature schema.
    aspect_feature = dataset.features["aspect"]
    def decode_aspect(value):
        if pd.isna(value):
            return ""
        if isinstance(value, str) and not value.strip().isdigit():
            return value.strip().lower()
        try:
            return aspect_feature.int2str(int(value)).strip().lower()
        except (ValueError, TypeError, AttributeError):
            return str(value).strip().lower()

    df["aspect"] = df["aspect"].map(decode_aspect)
    print("ASSET aspect counts:", df["aspect"].value_counts(dropna=False).to_dict())
    df = df[df["aspect"].eq("meaning")].copy()
    if df.empty:
        raise RuntimeError("No meaning-rating rows found in ASSET ratings config.")

    # Preserve every human rating, plus a per-pair aggregate. The aggregate
    # is a mean of crowd ratings, not a legal LMP score.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating", "original", "simplification"])
    df["rating"] = df["rating"].astype(float)

    grouped = (df.groupby(["original_sentence_id", "original", "simplification"], dropna=False)
                 .agg(mean_meaning_rating=("rating", "mean"),
                      rating_count=("rating", "count"),
                      ratings=("rating", lambda x: json.dumps([float(v) for v in x])))
                 .reset_index())
    grouped["pair_id"] = grouped.apply(
        lambda r: "asset_" + hashlib.sha1(
            (str(r["original_sentence_id"]) + "|" + r["simplification"]).encode("utf-8")
        ).hexdigest()[:16],
        axis=1
    )
    grouped["rating_scale"] = "0-100"
    grouped["rating_type"] = "crowd_human_meaning_preservation"
    grouped["domain"] = "general_domain_not_legal"
    grouped["source_dataset"] = DATASET_ID
    grouped["source_url"] = DATASET_URL
    grouped["source_repository"] = REPOSITORY_URL
    grouped["source_license"] = LICENSE
    grouped["lmp_score_1_to_10"] = ""
    grouped["legal_lmp_eligible"] = False

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "asset_meaning_ratings_raw.csv"
    pair_path = out / "asset_meaning_ratings_by_pair.csv"
    meta_path = out / "SOURCE_METADATA.json"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    grouped.to_csv(pair_path, index=False, encoding="utf-8-sig")
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "repository_url": REPOSITORY_URL,
        "license": LICENSE,
        "configuration": "ratings",
        "rating_aspect": "meaning",
        "native_rating_scale": "0-100",
        "raw_meaning_rating_rows_imported": int(len(df)),
        "unique_original_simplification_pairs": int(len(grouped)),
        "rating_interpretation": "crowd human judgments of meaning preservation",
        "domain": "general-domain English sentence simplification",
        "not_legal_lmp": True,
        "not_converted_to_project_1_to_10_scale": True,
        "citation": CITATION,
        "use_restriction": "Comply with CC BY-SA 4.0 attribution and share-alike terms.",
        "note": "Use as an auxiliary external benchmark only; do not merge into legal LMP regression labels."
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Imported {len(df):,} human meaning-rating rows.")
    print(f"Aggregated into {len(grouped):,} original/simplification pairs.")
    print(f"Pair-level CSV: {pair_path}")
    print("Ratings remain on the original 0-100 scale. They are not legal LMP labels.")


if __name__ == "__main__":
    main()
