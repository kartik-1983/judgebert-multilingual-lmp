# Research status and truthful claims

- The checked-in 334-pair dataset is synthetic/template-generated, not a human-annotated benchmark.
- Domain labels are randomly assigned during generation and are not evidence of held-out domain generalization.
- Model, training, baseline, metric, sanity-check and explainability scripts are implementation components; successful execution and empirical outcomes must be evidenced by saved artifacts.
- Use src/annotation_tools.py to prepare a blank annotation sheet and aggregate actual independent human ratings.
- Follow docs/annotation_guidelines.md and docs/experiment_protocol.md.
- docs/sanity_test_cases.csv is a starting suite; have a legally knowledgeable reviewer validate and expand it.
- Do not claim PreserveLex outperforms original JUDGEBERT without a valid same-test-set comparison.


## Evaluation safety update

- `src/evaluate.py` now fails closed when the trained regression checkpoint is absent; it does not substitute SBERT and mislabel the output.
- The known synthetic default dataset is blocked from supervised metric reporting unless the user explicitly opts into a smoke test.
- Non-default datasets require an explicit human-label provenance attestation; this is not an automatic audit and the provenance must still be checked.
- Metric utilities now handle too-small, non-finite, and constant-vector inputs safely.
- See `docs/EVALUATION_SAFETY.md` for commands and the boundary between demonstrable pipeline checks and research claims.
- These code changes are committed, but local execution, model training, and empirical evaluation have not been performed as part of this update.
