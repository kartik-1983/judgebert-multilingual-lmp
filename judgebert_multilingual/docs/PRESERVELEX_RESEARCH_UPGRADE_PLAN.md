# PreserveLex Research Upgrade Plan

Status: protocol and implementation plan only. No new human results are claimed here.

## 1. Current evidence boundary

The current paper describes a 334-pair template-generated development corpus with simulated labels, a 10-clause English insurance pilot awaiting independent annotation, and an 86-entry legal navigation library. The 100-pair reviewer workbook is a separate constructed annotation batch (80 English, 20 Hindi; 25 underlying clauses with four variants each). It is not human-annotated until reviewers complete it, and it is not source-verified legal data until each source and reuse basis is checked.

Do not combine these resources or report them as one human-validated benchmark. Do not use design-intent bands or synthetic scores as gold labels.

## 2. Immediate annotation protocol

1. Two reviewers independently annotate all 100 pairs without seeing each other's judgments.
2. Each reviewer supplies an overall integer LMP score (1-10), five invariant verdicts (obligations; rights/permissions; conditions/exceptions; scope/parties; dates/amounts), confidence, and rationale where required.
3. Use a third reviewer when the overall scores differ by at least two points OR any invariant verdict differs. Do not silently average a material disagreement.
4. Accept a row for supervised learning only after provenance checks, two independent reviews, and adjudication where required.
5. Keep reviewer identity and consent/audit records separately; pseudonymize released data.

## 3. Benchmark and split design

- Deduplicate exact and near-duplicate pairs.
- Keep every variant derived from the same base clause in one partition.
- Prefer document-level grouping when multiple clauses come from one source document; otherwise group by source-clause ID.
- Freeze an untouched test set before model selection.
- Report pair counts, source-document counts, clause counts, language, jurisdiction, domain, and missingness for every split.
- Treat 20 Hindi pairs as a pilot, not sufficient evidence of broad Hindi generalization.

## 4. Baselines and controlled comparison

Run all systems on the same adjudicated held-out pairs:
- Original JUDGEBERT, where its released implementation and compatible evaluation inputs can be reproduced.
- Multilingual Sentence-BERT cosine similarity.
- BERTScore with a documented multilingual encoder.
- PreserveLex XLM-RoBERTa regression model.
- PreserveLex advisory rule layer alone.
- PreserveLex neural model plus advisory rule layer (evaluate warnings separately; do not collapse them into an undocumented score).

Record exact checkpoint/version, tokenizer, maximum length, preprocessing, random seeds, hyperparameters, and inference settings. If the original JUDGEBERT cannot be run fairly on English/Hindi, report that limitation rather than substituting an unrelated model under its name.

## 5. Metrics and statistical reporting

For continuous human scores, report Pearson r, Spearman rho, MAE, and RMSE with confidence intervals (bootstrap by source document/clause group, not by individual variant). Report results separately by language and domain.

For inter-annotator reliability, report an appropriate two-way absolute-agreement ICC for numeric ratings and/or weighted Cohen's kappa for ordinal ratings, with the exact variant and confidence interval stated. Also report mean absolute reviewer score difference, exact agreement, within-one-point agreement, dimension-level agreement, and adjudication rate. Correlation alone is not an agreement statistic.

## 6. Legal-invariant challenge evaluation

Build a separately reported challenge set with controlled minimal edits:
- negation insertion/deletion;
- obligation/permission modal changes (shall, must, may, should);
- deadline/duration changes;
- amount/quantity changes;
- condition or exception deletion/addition;
- party substitution;
- scope expansion/restriction.

Report per-category detection precision, recall, F1, and false-positive examples. Keep challenge examples out of training and tune thresholds only on validation data.

## 7. Ablation and generalization

Compare:
1. XLM-R regression alone;
2. advisory rules alone;
3. XLM-R plus advisory rules.

Evaluate whether the rule layer improves detection of material invariant changes, while separately measuring false alarms on meaning-preserving paraphrases. Run cross-domain evaluation using a domain excluded from training. Report English and Hindi independently and avoid pooled claims that hide language imbalance.

## 8. Error analysis and reproducibility

Manually inspect at least 30 difficult test cases, sampled by error type and language, with reviewer adjudication. Include examples of false preservation, false alarm, and model/rule disagreement.

Release (subject to source licenses and privacy constraints): annotation protocol, provenance manifest, split IDs, preprocessing and evaluation scripts, model configuration, seeds, environment versions, and a model card/dataset card.

## 9. Paper revision checklist

- Replace simulated-label metrics in the main evidence narrative with human-test results when available; retain simulated results only as clearly labelled pipeline diagnostics.
- Add an annotation and source-provenance subsection.
- Add a benchmark split table and reviewer-agreement table.
- Add a common-test-set baseline comparison table with confidence intervals.
- Add challenge-set and ablation tables.
- Add per-language/domain results and error analysis.
- State limitations: constructed pairs, source coverage, Hindi sample size, rule coverage, domain shift, and prototype deployment constraints.
- Make no superiority claim unless the controlled human-annotated comparison supports it.

## 10. Go/no-go rule for a superiority claim

A superiority claim is justified only if PreserveLex is evaluated on the same untouched human-annotated test set as relevant baselines, has adequate source-level separation, reports uncertainty and reviewer reliability, and demonstrates a meaningful improvement on pre-specified metrics without concealing weak language/domain subgroups. Otherwise describe the work as a prototype and report the findings as preliminary.
