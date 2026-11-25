# PreserveLex Immediate Execution Plan

This plan separates work that can be completed in code from work requiring real-world source verification and human review.

## Phase A — Protect research integrity

- Keep the existing 334-row generated dataset and checkpoint as an exploratory baseline.
- Keep AI preannotations and simulated reviewer files in separate, explicitly named files.
- Never feed simulated reviewer scores into the human gold-label file.
- Do not report the previous validation metrics as real-world legal performance.

## Phase B — Source verification

For each candidate pair:
1. Identify the exact legal document and issuing authority.
2. Record jurisdiction, language, document version/date, canonical URL, page and clause.
3. Verify that the source text is exact.
4. Record how the simplified version was produced and by whom.
5. Confirm the lawful basis for retaining and redistributing the text.
6. Exclude items with unverifiable provenance from the source-verified benchmark.

Use `data/source_manifest_template.csv` as the provenance record. Never invent missing metadata.

## Phase C — Human annotation

- Follow `docs/ANNOTATION_GUIDELINES.md`.
- Two reviewers independently score the same source-verified pairs.
- Preserve raw ratings; do not overwrite them with adjudicated scores.
- Record context-needed items and disagreements.
- Calculate ordinal agreement and score differences only after both reviews are complete.

## Phase D — Dataset and model evaluation

- Deduplicate by source clause and pair.
- Group variants and parallel translations together before splitting.
- Create and save train/validation/test split IDs.
- Keep the final test set human-annotated and untouched during model selection.
- Compare lexical/TF-IDF, multilingual embedding, BERTScore-style, and fine-tuned XLM-R baselines on the same split.
- Report Pearson, Spearman, MAE, RMSE, categorical metrics, sample sizes, and separate English/Hindi results.
- Add qualitative error analysis for obligations, rights, deadlines, conditions, and scope.

## Phase E — Demonstration and report

- Clearly label the system as a research prototype, not legal advice.
- Show input clauses, score, category-level evidence, and uncertainty.
- Include provenance, methodology, dataset card, limitations, reproducibility details, and only executed results in the report.
