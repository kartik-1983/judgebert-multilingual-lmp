# Importing the PreserveLex 100-pair workbook

The supplied workbook explicitly labels its 100 scores as synthetic reference annotations. Use it for pipeline development and exploratory training only.

## 1. Place the workbook

From the `judgebert_multilingual` project directory, copy the workbook to:

```text
data/PreserveLex_LMP_100_Pairs_Dataset_Filled.xlsx
```

Install the Excel reader if needed:

```powershell
pip install openpyxl
```

## 2. Convert and validate

```powershell
python src/import_filled_lmp_workbook.py --input data/PreserveLex_LMP_100_Pairs_Dataset_Filled.xlsx
```

Expected outputs:

- `data/synthetic_development/preservelex_100_pairs.csv`
- `data/synthetic_development/preservelex_100_pairs.jsonl`
- `data/synthetic_development/DATA_PROVENANCE.json`

The importer validates required columns, IDs, scores, dimension flags, duplicate pairs, non-empty text, and same-language pairs. Language is inferred from script; the supplied workbook currently resolves to English for all 100 pairs.

## 3. Optional exploratory training

```powershell
python src/train.py --data data/synthetic_development/preservelex_100_pairs.csv --allow-synthetic-training --no-augment
```

The explicit flag is required because the workbook's labels are synthetic. `--no-augment` avoids creating artificial unrelated pairs from a single-language set.

## 4. Smoke-test evaluation only

After a checkpoint exists:

```powershell
python src/evaluate.py --data data/synthetic_development/preservelex_100_pairs.csv --allow-simulated-label-metrics
```

This produces a synthetic-label smoke-test result. Do not report its Pearson/Spearman/MAE/RMSE as human-validated performance, legal correctness, or a comparison with JUDGEBERT.

## 5. What this dataset does not establish

- No human annotation agreement or adjudication.
- No Hindi examples or English-Hindi performance.
- No independently verified document-level train/test separation.
- No domain generalization evidence.
