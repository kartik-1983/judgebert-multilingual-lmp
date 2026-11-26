# Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT

**CSET 346 – Natural Language Processing**  
**Project Title:** Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT

This repository implements a JUDGEBERT-inspired extension for English and Hindi legal text, focusing on Legal Meaning Preservation (LMP) between original and simplified legal sentences.

> **Dataset integrity status:** The current 334 English/Hindi development pairs are synthetic/template-generated and use simulated scores. The English-Hindi legal parallel corpus is translation data, not simplification data. A validated independent human-rated LMP dataset is still pending. Do not claim that external legal simplification data has trained or improved the LMP model.

## Project Overview

Legal text simplification must preserve legally critical elements: obligations, conditions, exceptions, scope, dates, permissions and amounts. The project:
- Implements SBERT cosine similarity and BERTScore baselines.
- Fine-tunes XLM-RoBERTa as a JUDGEBERT-inspired regression model.
- Adds a legal error taxonomy and review checklist.
- Provides separate English-Hindi legal translation and ASSET human-rated meaning-preservation benchmark importers.
- Adds a two-reviewer plus adjudicator protocol for building an independently annotated LMP dataset.

**Research Question:** How effectively can a JUDGEBERT-inspired model assess legal meaning preservation in English and Hindi, and how does it compare with generic semantic-similarity approaches on an independently annotated held-out set?

## Key Features

- **Languages:** English + Hindi
- **Model:** XLM-RoBERTa-base regression model
- **Baselines:** Sentence-BERT and BERTScore
- **Explainability:** omission, addition/hallucination, contradiction, obligation/permission, condition/exception, scope/party, date/amount and entity changes
- **Evaluation plan:** Pearson/Spearman correlation, RMSE, MAE, sanity checks and qualitative error analysis
- **Human annotation plan:** two independent reviewers and third-party adjudication

## Dataset Status

| Resource | Current status | Permitted role |
|---|---|---|
| 334 generated English/Hindi pairs | Synthetic; simulated labels | Development/debugging only; not human gold labels |
| 10 English insurance pilot clauses | Pilot material; human validation pending | Annotation pilot only |
| ASSET English simplification ratings | External CC BY-SA 4.0 general-domain pairs with human meaning ratings (0–100); importer committed, local download pending | Auxiliary general-domain benchmark only; not legal LMP labels |
| legalTransEn_Indic English-Hindi corpus | External MIT-licensed legal translation pairs; local import must be run | Auxiliary multilingual legal-domain analysis; not LMP regression labels |
| Independent human-rated simplification pairs | Pending | Required before claiming validated LMP supervision |

No external simplification dataset is currently approved for import because the closest candidates found did not have a clearly verified reuse license and/or row-level validated LMP ratings. See docs/DATASET_STATUS_AND_LICENSE_REVIEW.md.

## Repository Structure

```
judgebert_multilingual/
├── data/
│   ├── external_legal_parallel/       # Runtime import output; not committed
│   └── human_lmp_annotations/         # Blank human annotation template
├── src/
│   ├── data_generation.py             # Synthetic development data
│   ├── import_legal_parallel_corpus.py
│   ├── validate_human_lmp_annotations.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── docs/
│   ├── HUMAN_LMP_ANNOTATION_PROTOCOL.md
│   └── DATASET_STATUS_AND_LICENSE_REVIEW.md
└── README.md
```

## Quick Start

From the judgebert_multilingual directory:

```powershell
pip install -r requirements.txt
python src/data_generation.py
python src/baselines.py
python src/train.py --epochs 5 --batch_size 8
python src/evaluate.py
python src/explain.py
```

The training command above reflects the existing development workflow. For research claims, train/evaluate only with clearly identified data splits and do not mix synthetic labels into the human-annotated benchmark.

## ASSET Human Meaning-Preservation Benchmark

ASSET provides crowd-worker meaning ratings on a native 0–100 scale. It is general-domain English simplification, not legal text. Import it separately:

```powershell
pip install datasets pandas
python src/import_asset_meaning_ratings.py --max-rows 100
```

Remove `--max-rows 100` for the full ratings split. See `docs/ASSET_EXTERNAL_BENCHMARK.md`. Do not convert these scores into legal LMP 1–10 labels or merge them into the legal training set.

## Auxiliary English-Hindi Legal Corpus

The legalTransEn_Indic importer is committed, but the data must be downloaded locally:

```powershell
pip install datasets pandas
python src/import_legal_parallel_corpus.py --max-rows 100
```

Remove --max-rows 100 to import the available split. The importer writes CSV/JSONL, a review sheet and source metadata under data/external_legal_parallel/. These are translations, not simplifications; no LMP score is assigned.

## Independent Human LMP Annotation

1. Copy data/human_lmp_annotations/annotation_template.csv to a working CSV.
2. Add original/simplified pairs with source URL, provision, jurisdiction, license/reuse basis and simplification provenance.
3. Obtain two independent ratings using docs/HUMAN_LMP_ANNOTATION_PROTOCOL.md.
4. Have a third reviewer adjudicate disagreements and complete the adjudicated fields.
5. Validate the completed file:

```powershell
python src/validate_human_lmp_annotations.py --input data/human_lmp_annotations/your_annotated_pairs.csv
```

Only fully reviewed, adjudicated rows are written to validated_lmp_pairs.csv. The script never invents or fills ratings.

## Research Integrity and Limitations

- Synthetic scores are not human judgments.
- Translation pairs are not simplification pairs.
- Similarity metrics are not validated LMP ratings.
- Do not claim external-corpus training or model improvement until the corpus is licensed, annotated, imported, trained and evaluated on the same untouched human-rated test set.
- Split by source document to avoid leakage.
- Preserve dataset attribution and license terms.

## Reference Paper

Beauchemin, D., Albert-Rochette, M., Khoury, R., & Déziel, P. L. (2025). JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences. Proceedings of EMNLP 2025, pp. 92–118.  
https://aclanthology.org/2025.emnlp-main.5/  
Code: https://github.com/GRAAL-Research/JUDGEBERT

## License & Citation

Cite the original JUDGEBERT paper and every external dataset used. Check each dataset's license independently; the project code license does not grant rights to third-party datasets.
