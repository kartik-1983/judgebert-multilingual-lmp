from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent
pilot = ROOT / "preservelex_pilot10_corrected.csv"
manifest = ROOT / "preservelex_pilot10_source_manifest.csv"
r1 = ROOT / "preservelex_R1_annotation_sheet.csv"
r2 = ROOT / "preservelex_R2_annotation_sheet.csv"

required = {
    "pair_id", "language", "source_title", "issuing_authority",
    "jurisdiction", "canonical_source_url", "source_page_printed",
    "source_clause", "candidate_simplification",
    "meaning_changing_contrast", "review_status"
}

for path in (pilot, manifest, r1, r2):
    if not path.exists():
        print(f"MISSING: {path.name}")
        sys.exit(1)

df = pd.read_csv(pilot, dtype=str).fillna("")
missing = sorted(required - set(df.columns))
if missing:
    print("Pilot missing columns:", ", ".join(missing))
    sys.exit(1)

if df["pair_id"].duplicated().any():
    print("ERROR: duplicate pair_id in pilot")
    sys.exit(1)

m = pd.read_csv(manifest, dtype=str).fillna("")
if set(df["pair_id"]) != set(m["pair_id"]):
    print("ERROR: pilot and manifest pair_id sets differ")
    sys.exit(1)

for path in (r1, r2):
    reviewer = pd.read_csv(path, dtype=str).fillna("")
    if len(reviewer) != len(df):
        print(f"ERROR: {path.name} has {len(reviewer)} rows; expected {len(df)}")
        sys.exit(1)
    if "simplification_score_1_to_10" not in reviewer.columns:
        print(f"ERROR: score column missing in {path.name}")
        sys.exit(1)
    if reviewer["simplification_score_1_to_10"].str.strip().ne("").any():
        print(f"WARNING: {path.name} already contains scores; inspect before use.")

print(f"PASS: {len(df)} unique pilot records")
print(f"PASS: {len(m)} manifest records match pilot IDs")
print("PASS: both reviewer sheets have matching row counts")
print("Reminder: verify exact source text and page/clause manually before annotation.")
