# Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT (PreserveLex)

**CSET 346 – Natural Language Processing**  
**Project Title:** Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT / PreserveLex

This repository implements a professional extension of the JUDGEBERT framework (Beauchemin et al., EMNLP 2025) to **English + Hindi** legal text, focusing on Legal Meaning Preservation (LMP) between original and simplified legal sentences.

## Project Overview

Legal text simplification must preserve legally critical elements (obligations, conditions, exceptions, scope, dates, permissions). Generic semantic similarity metrics often fail on these. This project:

- Creates a bilingual (English + Hindi) dataset of original–simplified legal sentence pairs with LMP annotations (1–10 scale).
- Implements baselines: **TF-IDF cosine**, multilingual Sentence-BERT, and BERTScore.
- Fine-tunes a multilingual transformer (XLM-RoBERTa) as a JUDGEBERT-inspired regression model for LMP prediction.
- Adds explainability (rule-based advisory dimensions + error-type analysis).
- Conducts the required experiments: language comparison, domain generalization, sanity checks, and qualitative error analysis.

**Research Question:** How effectively can a JUDGEBERT-based model assess legal meaning preservation in a new language (Hindi) and legal domain, and does it outperform generic semantic-similarity / lexical approaches?

## Key Features

- **Languages:** English + Hindi
- **Domain:** Property / General Insurance + Indian Contract Act style clauses
- **Dataset:** ≥ 300 original–simplified pairs with simulated human LMP scores + error type annotations (synthetic / template-based; not human gold)
- **Models / Baselines:** 
  - Lexical: TF-IDF (1–2 gram) cosine similarity
  - Semantic: Sentence-BERT (paraphrase-multilingual), BERTScore (XLM-RoBERTa)
  - Main model: XLM-RoBERTa-base fine-tuned for regression on sentence pairs
- **Evaluation:** Pearson, Spearman, RMSE, MAE; sanity checks; qualitative analysis
- **Explainability:** Five advisory review dimensions (obligations, rights/permissions, deadlines/amounts, conditions/exceptions, scope/parties) + short explanations

## Repository Structure

```
judgebert_multilingual/
├── data/                   # Dataset files (JSONL / CSV)
├── src/                    # Source code
│   ├── data_generation.py  # Synthetic data + annotation simulation
│   ├── baselines.py        # TF-IDF, SBERT & BERTScore baselines
│   ├── model.py            # JUDGEBERT-style regression model
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation + metrics
│   ├── explain.py          # Explainability & error analysis
│   └── utils.py            # Helpers
├── notebooks/              # Exploration & analysis notebooks
├── models/                 # Saved checkpoints
├── results/                # Metrics, predictions, qualitative analysis
├── docs/                   # Paper, slides, annotation guidelines
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate / load the dataset
python src/data_generation.py

# 3. Run baselines (TF-IDF + SBERT + BERTScore)
python src/baselines.py

# 4. Train the JUDGEBERT-inspired model
python src/train.py --epochs 5 --batch_size 8

# 5. Full evaluation + sanity checks + qualitative analysis
python src/evaluate.py
python src/explain.py
```

## Dataset

The dataset contains original–simplified pairs in English and Hindi drawn from insurance / contract language. Each instance includes:

- `id`, `language`, `domain`
- `original`, `simplified`
- `lmp_score` (1–10, average of simulated annotators)
- `decision` (Preserved / Partially Preserved / Not Preserved)
- `error_types` (list)
- `explanation` (short text)

**Annotation Procedure (documented):**  
Simulated following the original paper’s methodology. Real student projects should replace scores with genuine human annotations from law students or domain experts using the guidelines in `docs/annotation_guidelines.md`.

## Experiments (as required)

1. Generic semantic / lexical similarity vs. JUDGEBERT-based LMP prediction
2. Performance comparison across English and Hindi
3. Generalization / out-of-domain diagnostic (e.g., ASSET)
4. Sanity checks: identical pairs → high LMP; unrelated pairs → low LMP
5. Qualitative error analysis on legally difficult examples

## Current Evidence Status (Important)

- Primary development corpus uses **simulated / template-generated labels**.
- No independently annotated human gold test set is claimed.
- Out-of-domain diagnostic on ASSET (general text) shows TF-IDF currently stronger than the legal-specialized model (expected, because ASSET is not legal).
- Superiority claims over JUDGEBERT or generic baselines require a common, source-clause-disjoint, human-annotated legal test set.

## Expected System Output (per pair)

- **LMP Score:** 1–10  
- **Decision:** Preserved / Partially Preserved / Not Preserved  
- **Optional legal error type(s)** / advisory dimension warnings  
- **Short explanation** of the legal meaning change

## Deliverables

1. Source Code (this repository)
2. Research Paper (IEEE style) – see `docs/`
3. Presentation Slides – see `docs/`

## Reference Paper

Beauchemin, D., Albert-Rochette, M., Khoury, R., & Déziel, P. L. (2025).  
JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences.  
*Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, pp. 92–118.  
https://aclanthology.org/2025.emnlp-main.5.pdf  
Code: https://github.com/GRAAL-Research/JUDGEBERT

## Project Instructor

**Dr. Surabhi Solanki**  
N2-CUB-012 · surabhi.solanki@bennett.edu.in

## License & Citation

This educational project is released for academic use. When using the code or ideas, please cite both the original JUDGEBERT paper and this project.

---

*Built as a professional, reproducible extension of JUDGEBERT for multilingual (English–Hindi) legal meaning preservation. Claims of superiority are reserved for future human-annotated evaluation.*
