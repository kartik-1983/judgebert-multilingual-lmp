# External legal dataset integration

## Dataset selected

- **Name:** Supreme Court and High Court Orders Parallel Corpus (English-Hindi)
- **Hugging Face ID:** `Wasserstoff-AI/legalTransEn_Indic`
- **Source:** https://huggingface.co/datasets/Wasserstoff-AI/legalTransEn_Indic
- **License shown on dataset card:** MIT
- **Upstream row count shown on the card:** approximately 89,951
- **Fields shown in the viewer:** `english`, `hindi`
- **Source description:** aligned English-Hindi sentence pairs extracted from publicly available Supreme Court of India and Punjab & Haryana High Court orders.

## Why this dataset is being added

It supplies real, externally sourced Indian legal text in English and Hindi. This strengthens the multilingual legal-domain data resources and enables a separate cross-lingual legal equivalence review.

## Important task distinction

These records are **translations**, not original-versus-simplified sentence pairs. The dataset card does not provide Legal Meaning Preservation (LMP) scores. Therefore:

- Do not assign an LMP score automatically.
- Do not merge these rows into the supervised LMP regression training file.
- Use the generated human-review sheet for independent review of cross-lingual meaning equivalence.
- Report this as an auxiliary multilingual legal corpus, not as the main LMP simplification benchmark.

The existing 334 synthetic English/Hindi development pairs remain synthetic. The 10 English insurance pilot clauses remain a separate pilot resource.

## Import

From the `judgebert_multilingual` directory:

```powershell
pip install datasets pandas
python src/import_legal_parallel_corpus.py
```

For a small test import:

```powershell
python src/import_legal_parallel_corpus.py --max-rows 100
```

The script writes:

- `data/external_legal_parallel/legaltrans_en_hi.csv`
- `data/external_legal_parallel/legaltrans_en_hi.jsonl`
- `data/external_legal_parallel/legaltrans_en_hi_human_review.csv`
- `data/external_legal_parallel/SOURCE_METADATA.json`

The raw dataset is downloaded by the Hugging Face Datasets library at run time and is not committed to Git.

## Attribution

Wasserstoff-AI, *Supreme Court and High Court Orders Parallel Translation Corpus (English-Hindi)*, Hugging Face Datasets. License: MIT. Dataset URL: https://huggingface.co/datasets/Wasserstoff-AI/legalTransEn_Indic.

Retain the upstream dataset card and license when using or redistributing the downloaded files. Verify the current dataset card before final submission because dataset contents and metadata may change.
