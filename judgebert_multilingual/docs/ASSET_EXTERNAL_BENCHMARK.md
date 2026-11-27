# ASSET External Human Meaning-Preservation Benchmark

## Verified source and license

- Dataset: ASSET (Alva-Manchego et al., ACL 2020)
- Hugging Face dataset: https://huggingface.co/datasets/facebook/asset
- Official repository: https://github.com/facebookresearch/asset
- License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0), as stated by the Hugging Face dataset card and the official repository LICENSE.
- Citation: Alva-Manchego, F., Martin, L., Bordes, A., Scarton, C., Sagot, B., & Specia, L. (2020). ASSET: A Dataset for Tuning and Evaluation of Sentence Simplification Models with Multiple Rewriting Transformations. Proceedings of ACL 2020.

## What it contains

ASSET is an English, general-domain sentence simplification resource. Its `ratings` configuration contains original sentences, system-produced simplifications, the rating aspect (meaning, fluency or simplicity), worker identifiers and human quality ratings. Meaning ratings use the dataset's native 0–100 scale. The dataset card describes these as crowd-worker judgments.

## PreserveLex use and strict limitation

This is **not a legal dataset**. The importer selects only rows where aspect=meaning and aggregates multiple human ratings for each original/simplification pair. It preserves the native 0–100 values and count of ratings. It does not convert them to 1–10, call them legal LMP scores, or add them to legal regression training.

Use this dataset as an external general-domain meaning-preservation benchmark or for a clearly separate auxiliary experiment. It cannot establish legal-domain performance or replace independent legal annotation.

## Import

From `judgebert_multilingual`:

```powershell
pip install datasets pandas
python src/import_asset_meaning_ratings.py --max-rows 100
```

Remove `--max-rows 100` for the full ratings split.

Outputs:
- `data/external_asset/asset_meaning_ratings_raw.csv`
- `data/external_asset/asset_meaning_ratings_by_pair.csv`
- `data/external_asset/SOURCE_METADATA.json`

## License compliance

The current Hugging Face dataset card identifies CC BY-SA 4.0. Preserve attribution and comply with the share-alike terms when distributing adaptations. Review the full license and source materials before redistribution. This note is not legal advice.

## Citation

Alva-Manchego et al. (2020), ACL 2020. Dataset and official repository links above.
