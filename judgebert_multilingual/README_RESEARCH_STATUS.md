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

## Additional 100-pair workbook (synthetic development data)

- The supplied `PreserveLex_LMP_100_Pairs_Dataset_Filled.xlsx` contains 100 English-only pairs and synthetic reference labels. It is not a human-annotated benchmark.
- Convert it with `python src/import_filled_lmp_workbook.py --input data/PreserveLex_LMP_100_Pairs_Dataset_Filled.xlsx`.
- The importer writes CSV/JSONL and `DATA_PROVENANCE.json` under `data/synthetic_development/`; it maps the workbook's `original_text`/`simplified_text` fields to the model schema and records `label_provenance=synthetic_reference`.
- Exploratory training requires the explicit flag `--allow-synthetic-training`. Example:
  `python src/train.py --data data/synthetic_development/preservelex_100_pairs.csv --allow-synthetic-training --no-augment`
- Evaluation detects synthetic provenance and blocks human-performance metrics. To run a labelled pipeline smoke test only:
  `python src/evaluate.py --data data/synthetic_development/preservelex_100_pairs.csv --allow-simulated-label-metrics`
- The 100-pair workbook is English-only (100/100). It does not establish Hindi coverage or bilingual performance. Do not combine it with the 334-pair corpus and call the combined data human-labelled.
- The workbook has no source-clause/document provenance identifier. The importer uses each original clause as `split_group`; this prevents exact-original variants leaking across splits, but cannot guarantee document-level separation where source documents are unknown.
