# Evaluation safety and reproducibility

## Why the evaluation entry point is strict

The repository's default `data/lmp_dataset.jsonl` is template-generated and its
scores are simulated. It is useful for pipeline development, not for measuring
agreement with human judgments or making benchmark claims.

`src/evaluate.py` now follows these rules:

1. It requires a trained PreserveLex regression checkpoint. It never substitutes
   SBERT cosine similarity and labels it as the main model.
2. It refuses to compute supervised metrics against the known synthetic default
   dataset unless the explicit smoke-test flag is supplied.
3. For a different input dataset, it requires an explicit
   `--human-labels-verified` attestation. This is a user assertion, not an
   automatic provenance audit.
4. It writes the evidence type and a warning into `evaluation_metrics.json`.
5. Regression metric utilities reject mismatched, too-small, or non-finite
   arrays and return undefined correlations for constant vectors rather than
   crashing.

## Commands

Run from `judgebert_multilingual/`.

Research evaluation (only after real labels are independently reviewed,
adjudicated, held out by source clause, and provenance is checked):

```powershell
python src/evaluate.py --data data/human_lmp_annotations/validated_heldout.csv --checkpoint models/judgebert_xlmr/best_model.pt --human-labels-verified --output_dir results/human_heldout
```

Synthetic pipeline smoke test (not publishable performance evidence):

```powershell
python src/evaluate.py --data data/lmp_dataset.jsonl --checkpoint models/judgebert_xlmr/best_model.pt --allow-simulated-label-metrics --output_dir results/synthetic_smoke_test
```

If no trained checkpoint exists, evaluation stops with an actionable error. Do
not create or rename a proxy checkpoint to bypass this guard.

## What can be demonstrated without human annotations

- Application workflow and advisory legal-invariant checks.
- Data loading, tokenization and training-pipeline execution.
- Identical/unrelated and legal-invariant challenge examples as qualitative
  sanity tests, provided their outputs are described as tests, not validated
  accuracy.
- Reproducibility controls, grouped split design and annotation workflow.

What cannot be established without human-rated held-out data:

- Correlation with human legal-meaning judgments.
- Human-level accuracy or reliability.
- Superiority over JUDGEBERT, SBERT or other baselines.
- Cross-domain or multilingual generalization claims.

## Manual verification checklist

```powershell
python -m compileall src
python -c "from src.utils import compute_metrics; print(compute_metrics([1,2,3],[1,2,2]))"
python src/evaluate.py --help
```

Record the actual terminal output and environment versions when these checks are
run. A documented command is not evidence that it has already passed.
