# Research status and truthful claims

- The checked-in 334-pair dataset is synthetic/template-generated, not a human-annotated benchmark.
- Domain labels are randomly assigned during generation and are not evidence of held-out domain generalization.
- Model, training, baseline, metric, sanity-check and explainability scripts are implementation components; successful execution and empirical outcomes must be evidenced by saved artifacts.
- Use src/annotation_tools.py to prepare a blank annotation sheet and aggregate actual independent human ratings.
- Follow docs/annotation_guidelines.md and docs/experiment_protocol.md.
- docs/sanity_test_cases.csv is a starting suite; have a legally knowledgeable reviewer validate and expand it.
- Do not claim PreserveLex outperforms original JUDGEBERT without a valid same-test-set comparison.
