# Human Annotation Agreement: Running the Analysis

Run this only after both reviewers have completed their independent ratings and the workbook has been exported to CSV with one row per pair.

From the `judgebert_multilingual` directory:

```powershell
python src/compute_annotation_agreement.py --input data/human_lmp_annotations/completed_annotations.csv --output results/human_annotation_agreement.json
```

Required columns:
- `pair_id`
- `reviewer_1_lmp_score`
- `reviewer_2_lmp_score`

Optional dimension columns use the names `reviewer_1_obligations` and `reviewer_2_obligations`, with equivalent pairs for `rights_permissions`, `conditions_exceptions`, `scope_parties`, and `dates_amounts`. If present, `review_status` is used to calculate the adjudication rate.

The script reports the number of complete score pairs, exact agreement, agreement within one point, mean absolute score difference, quadratic weighted Cohen's kappa, dimension-level exact agreement, and adjudication rate when available.

It does not fill missing scores, decide which reviewer is correct, adjudicate disagreements, or create training labels. Kappa is an agreement statistic, not a measure of model accuracy. For publication, report uncertainty intervals and stratify by language/domain where sample sizes permit.

Do not run this on the blank template or on simulated labels and describe the output as human reliability.
