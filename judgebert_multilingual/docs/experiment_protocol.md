# Required Experiment Protocol and Evidence Checklist

## Current data caveat
The checked-in 334-row dataset is programmatically generated from a small set of templates. Scores are sampled from predefined ranges, not collected from human annotators. Domain labels are assigned during generation and do not establish independent source-domain corpora. The dataset is useful for pipeline development and smoke tests only.

## Experiment A — Baselines versus regression
1. Collect genuine human reference scores using annotation_guidelines.md.
2. Split by source clause/document before training; no variants of one source clause may cross partitions.
3. Run SBERT, multilingual XLM-R BERTScore, and the fine-tuned XLM-R regression model on the same held-out pairs.
4. Report N, Pearson r, Spearman rho, RMSE and MAE overall and by language.
5. Record the exact dataset version and checkpoint.

## Experiment B — English versus Hindi
Use held-out human-annotated examples in both languages. Report sample counts and metrics separately. Do not infer Hindi performance from English-only results or from the encoder's multilingual label.

## Experiment C — Domain generalization
Collect a real, independently sourced target legal domain and document jurisdiction, source and licence. Train/validate without target-domain examples, then evaluate once on that target domain. The current synthetic domain tags are randomly assigned and must not be used as evidence of domain generalization.

## Experiment D — Sanity checks
Use a separate test suite for identical, unrelated, negation-reversed, modality-changed, deadline/amount-changed, exception-deleted, party-substituted and scope-changed pairs. Report each category, N, mean score and expected direction. Keep test cases out of training.

## Experiment E — Qualitative analysis
Manually inspect at least 15 difficult cases. Include original, simplified, human score, model score, legal issue, evidence span, reviewer explanation and whether the model flag was correct. Automatically generated explanations do not count as human-reviewed analysis.

## Required reporting
Data provenance, jurisdiction, language, licence, annotation procedure, annotator count and agreement; split strategy and leakage controls; model/checkpoint versions; N, Pearson, Spearman, RMSE and MAE; per-language/domain results; confidence intervals or paired significance analysis where feasible; failure cases and limitations.

## Claims to avoid
Do not claim human annotation for simulated labels, domain generalization from randomly assigned domain tags, model superiority without a same-test-set comparison, or legal correctness from a demo score.
