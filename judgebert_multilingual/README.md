# Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT

> **Research integrity note (read before reporting results):** The checked-in 334-pair dataset is template-generated and its LMP scores are simulated, not human annotations. Its domain labels are randomly assigned and cannot support a domain-generalization claim. Treat it as development/smoke-test data only. A blank human annotation sheet and annotation tools are provided; empirical claims require genuine independent ratings and saved held-out evaluation outputs.

**CSET 346 – Natural Language Processing**  
**Project Title:** Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT

This repository implements a professional extension of the JUDGEBERT framework (Beauchemin et al., EMNLP 2025) to **English + Hindi** legal text, focusing on Legal Meaning Preservation (LMP) between original and simplified legal sentences.

## Project Overview

Legal text simplification must preserve legally critical elements (obligations, conditions, exceptions, scope, dates, permissions). Generic semantic similarity metrics often fail on these. This project:

- Creates a bilingual (English + Hindi) dataset of original–simplified legal sentence pairs with LMP annotations (1–10 scale).
- Implements strong baselines (SBERT cosine similarity, BERTScore).
- Fine-tunes a multilingual transformer (XLM-RoBERTa) as a JUDGEBERT-inspired regression model for LMP prediction.
- Adds explainability (attention-based and error-type analysis).
- Conducts the required experiments: language comparison, domain generalization, sanity checks, and qualitative error analysis.

**Research Question:** How effectively can a JUDGEBERT-based model assess legal meaning preservation in a new language (Hindi) and legal domain, and does it outperform generic semantic-similarity approaches?

## Key Features

- **Languages:** English + Hindi (Indian language)
- **Domain:** Property / General Insurance + Indian Contract Act style clauses
- **Dataset:** ≥ 300 original–simplified pairs with simulated human LMP scores + error type annotations
- **Models:** 
  - Baselines: Sentence-BERT (all-MiniLM / paraphrase-multilingual), BERTScore
  - Main model: XLM-RoBERTa-base fine-tuned for regression on sentence pairs
- **Evaluation:** Pearson correlation, RMSE, MAE, Spearman; sanity checks; qualitative analysis of ≥15 hard cases
- **Explainability:** Error taxonomy (omission, hallucination, contradiction, obligation/permission change, condition/exception/scope/date/entity errors) + short natural-language explanations

## Repository Structure

```
judgebert_multilingual/
├── data/                   # Dataset files (JSONL / CSV)
├── src/                    # Source code
│   ├── data_generation.py  # Synthetic data + annotation simulation
│   ├── baselines.py        # SBERT & BERTScore baselines
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

# 3. Run baselines
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
Simulated following the original paper’s methodology (majority / average of multiple raters). Real student projects should replace scores with genuine human annotations from law students or domain experts using the provided guidelines in `docs/annotation_guidelines.md`.

## Experiments (as required)

1. **Generic semantic similarity vs. JUDGEBERT-based LMP prediction**
2. **Performance comparison across English and Hindi**
3. **Generalization** to a second legal domain or contextual vs. sentence-only input
4. **Sanity checks**: identical pairs → high LMP; unrelated pairs → low LMP
5. **Qualitative error analysis** on ≥15 legally difficult examples (omissions, contradictions, obligation changes, etc.)

## Expected System Output (per pair)

- **LMP Score:** 1–10  
- **Decision:** Preserved / Partially Preserved / Not Preserved  
- **Optional legal error type(s)**  
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

*Built as a professional, reproducible extension of JUDGEBERT for multilingual (English–Hindi) legal meaning preservation.*


## Human annotation and research-compliance workflow

- Annotation rubric: `docs/annotation_guidelines.md`
- Blank sheet (334 pairs): `data/human_annotation_template.csv`
- Prepare a fresh sheet: `python src/annotation_tools.py template`
- Aggregate completed ratings (at least two independent annotators per pair):
  `python src/annotation_tools.py aggregate --input data/human_annotations_filled.csv`
- Experiment requirements and evidence checklist: `docs/experiment_protocol.md`
- Starting bilingual sanity suite: `docs/sanity_test_cases.csv`
- Research status and claim limits: `README_RESEARCH_STATUS.md`

Do not present synthetic labels as human judgments, randomly assigned domains as domain generalization, or model superiority as established without a fair held-out comparison.

## Running the PreserveLex application with the trained model

The Gradio application entry point is `app.py`. Start it from the
`judgebert_multilingual/` directory:

```bash
python app.py
```

The application loads the fine-tuned checkpoint from
`models/judgebert_xlmr/best_model.pt` by default. The checkpoint is intentionally
not stored in Git because of its size. If it is stored elsewhere, configure an
absolute path before starting the application:

PowerShell:

```powershell
$env:PLX_CHECKPOINT_PATH = "D:\\models\\best_model.pt"
python app.py
```

Linux/macOS:

```bash
export PLX_CHECKPOINT_PATH="/path/to/best_model.pt"
python app.py
```

**Scoring safety:** PreserveLex now fails closed if the fine-tuned checkpoint is
missing, unreadable, or incompatible. It displays the reason and does not produce
an LMP score or a Preserved/Partially preserved/Not preserved verdict. Generic
SBERT cosine similarity is not used as a substitute for legal meaning-preservation
prediction. The model-status message in the application shows whether the trained
checkpoint loaded successfully.

Only use a checkpoint produced for this repository's `JudgeBERTRegressor`
architecture and matching tokenizer/base model. Do not load untrusted checkpoint
files.

