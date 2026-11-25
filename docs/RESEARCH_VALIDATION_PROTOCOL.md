# PreserveLex Research Validation Protocol

**Project:** Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT  
**Model under study:** XLM-RoBERTa-based regression model  
**Languages:** English and Hindi  
**Status:** Research protocol; results must be populated only after experiments are run.

## 1. Research objective

Evaluate whether a multilingual model can estimate preservation of legally material meaning between an original legal clause and a simplified version. The system is a research prototype, not a legal decision-maker, and must not be presented as legal advice.

The project is inspired by JUDGEBERT (Beauchemin et al., EMNLP 2025), which introduces FrJUDGE and evaluates legal meaning preservation for French insurance text. PreserveLex investigates a distinct multilingual English–Hindi setting; it must not imply that the original paper validated this adaptation.

References:
- Paper: https://aclanthology.org/2025.emnlp-main.5/
- Official implementation: https://github.com/GRAAL-Research/JUDGEBERT

## 2. Definition and annotation dimensions

A simplified clause preserves legal meaning when it retains the material legal effects of the source. Reviewers should compare the clauses across these dimensions:

1. **Obligations:** duties, prohibitions, permissions to perform required acts, and responsible actors.
2. **Rights and permissions:** entitlements, discretion, consent, remedies, and who may exercise them.
3. **Deadlines and amounts:** dates, durations, notice periods, monetary values, limits, caps, percentages, and thresholds.
4. **Conditions and exceptions:** prerequisites, triggers, exclusions, provisos, and circumstances in which a rule does or does not apply.
5. **Scope and parties:** covered persons, property, events, territory, claims, definitions, and the reach of a clause.

Reviewers must identify the affected dimension(s), point to the relevant text, and explain the legal effect of the change. A fluent paraphrase is not necessarily meaning-preserving.

## 3. Scoring rubric

Use an integer score from 1 to 10. Score preservation of legal effect, not fluency or stylistic quality.

| Score | Operational interpretation |
|---:|---|
| 10 | Material legal meaning is fully preserved; only harmless wording changes. |
| 8–9 | Meaning is substantially preserved; at most a minor nuance is uncertain. |
| 6–7 | A material detail is weakened, ambiguous, or incompletely expressed. |
| 4–5 | One or more important conditions, limits, parties, or duties are changed or omitted. |
| 2–3 | Major legal effect is reversed, broadened, narrowed, or materially misrepresented. |
| 1 | The clauses are unrelated or the core legal effect is fundamentally reversed. |

These bands are annotation guidance, not an automatically validated legal standard. If reviewers cannot judge a clause because context or definitions are missing, mark NEEDS_CONTEXT in notes and do not force a confident score.

## 4. Source provenance and eligibility

Every real legal example must have a traceable source record before it enters the gold evaluation set. Record:
- source title and issuing authority;
- canonical URL or repository identifier;
- jurisdiction and language;
- document date/version and access date;
- page, section, article, rule, or clause identifier;
- whether the text is public domain, licensed, or used under permission;
- exact source text and the transformation/simplification provenance.

Do not invent URLs, page numbers, citations, or source attributions. Do not label template-generated examples as sourced legal text. Keep generated sanity examples in a separate test-only file.

## 5. Independent annotation workflow

1. Freeze a versioned annotation batch and assign stable pair IDs.
2. Give each reviewer the same source and simplified text, provenance/context, rubric, and annotation instructions.
3. Reviewers work independently and cannot see one another's scores or notes.
4. Each reviewer records: pair ID, reviewer ID, score (1–10), affected dimensions, rationale, confidence, and NEEDS_CONTEXT status.
5. Resolve disagreements only after independent ratings are locked. Preserve original ratings and record adjudicated labels separately.
6. Never generate reviewer identities or human scores using an AI model. AI suggestions, if used, must be stored separately and explicitly labelled as model-generated.

Recommended minimum for a pilot: two independent reviewers per item. Report the number of items and reviewers actually completed; do not imply a larger annotation study than was conducted.

## 6. Agreement and adjudication

For ordinal 1–10 scores, report:
- exact agreement and within-one-point agreement;
- mean absolute reviewer score difference;
- an ordinal agreement statistic such as quadratic-weighted Cohen's kappa for two reviewers, or Krippendorff's alpha with an explicitly stated ordinal distance function when there are multiple reviewers or missing ratings;
- confidence intervals where feasible.

Do not call Pearson correlation alone inter-annotator agreement: correlation measures association and can remain high despite systematic reviewer score offsets. Report the statistic, sample size, missingness, and calculation method. Keep raw reviewer scores and adjudicated scores.

## 7. Dataset construction and leakage controls

- Deduplicate exact pairs and inspect repeated source clauses.
- Group all variants of the same source clause together before splitting.
- Where translations or parallel clauses exist, group them across languages so equivalent content cannot appear in different splits.
- Split by source document/clause group, not by row. Keep train, validation, and test groups disjoint.
- Freeze split IDs and random seeds; save a split manifest.
- Fit preprocessing and any learned transformations on training data only.
- Add synthetic identical/unrelated examples to a separate sanity suite or training-only augmentation; never let generated examples contaminate the held-out real test set.
- Do not use test labels for model selection, threshold tuning, or early stopping.

The currently audited 334-row dataset is template-generated, with 12 unique original clauses per language and randomly sampled score ranges. It is useful for code smoke tests and controlled sanity checks, but it is not human ground truth and cannot establish real-world legal generalization.

## 8. Model experiments and baselines

Use the same frozen train/validation/test groups for all model comparisons. Document exact model identifiers, library versions, tokenization, maximum sequence length, hyperparameters, seeds, hardware, and checkpoint hashes.

At minimum compare:
1. a simple lexical or TF-IDF baseline;
2. a multilingual sentence-embedding cosine-similarity baseline;
3. a multilingual BERTScore-style baseline, if correctly configured;
4. the fine-tuned XLM-RoBERTa regression model.

If reproducing original JUDGEBERT is feasible, document its language/domain mismatch and adaptation method. Do not claim superiority over JUDGEBERT unless a fair comparison uses the same test examples, labels, and evaluation protocol.

## 9. Evaluation

For the continuous score, report Pearson correlation, Spearman correlation, MAE, and RMSE with sample size. Include confidence intervals where feasible. Report results separately for English and Hindi, plus a clearly described macro or pooled summary. Do not compare pooled metrics alone if language composition differs.

For categorical decisions, report the confusion matrix, macro-F1, per-class precision/recall/F1, and class counts. Thresholds must be specified before final test evaluation.

Required error analysis:
- obligations added, removed, or reversed;
- rights/permissions changed;
- deadlines, amounts, or limits altered;
- conditions/exceptions omitted;
- scope/party changes;
- negation and modality errors;
- Hindi-specific terminology, morphology, and translation issues.

Required sanity checks:
- identical pairs should receive the maximum score;
- unrelated pairs should receive a very low score;
- changed negation, party, deadline, amount, exception, or obligation should be detected;
- paraphrases with preserved legal effect should not be penalized merely for lexical differences.

## 10. Explainability and user-facing safeguards

Explanations should point to evidence in the pair and identify the dimension at risk. Do not present attention weights or similarity scores as proof of legal reasoning. Show uncertainty and encourage expert review for ambiguous or high-impact cases.

The interface must state that PreserveLex is an experimental research aid, not legal advice, and that its scores do not determine legal validity or replace qualified legal review.

## 11. Reproducibility and reporting checklist

Before reporting a result, archive:
- dataset version and source manifest;
- annotation guideline version and raw independent ratings;
- adjudication log;
- train/validation/test split manifest;
- code commit and environment/dependency versions;
- model checkpoint and hash;
- full configuration and random seeds;
- per-language metrics, confidence intervals, and error analysis;
- limitations and any deviations from this protocol.

Never report an experiment as completed unless its logs and outputs exist. Distinguish implemented, executed, human-annotated, and independently validated functionality in the final report.
