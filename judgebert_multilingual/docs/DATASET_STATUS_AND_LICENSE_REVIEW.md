# Legal Simplification Dataset: License and Suitability Review

Review date: 27 November 2026

| Resource | Contents | License status checked | PreserveLex suitability |
|---|---|---|---|
| SIMPLE-LAW (Rabbani et al., LREC 2026) | 6,230 aligned original/simplified legal passages; India, US, UK | No explicit dataset license established in repository listing reviewed | Strong task match, but do not import/train/redistribute until reuse terms are confirmed |
| Lex-Simple | 500 US Supreme Court sentences with three simplified references; legal expert/student involvement reported | Clear license not established in evidence reviewed | Strong task match, but permission and row-level ratings need confirmation |
| JUDGEBERT FrJUDGE | Single French sentences with grammaticality labels | CC-BY-NC-SA 4.0 | Not LMP pairs; do not repurpose labels |
| legalTransEn_Indic | English-Hindi legal translations | MIT shown on dataset card | Auxiliary multilingual corpus only; not simplification and no LMP ratings |
| ASSET (Alva-Manchego et al., ACL 2020) | English general-domain simplification; human crowd ratings for meaning/fluency/simplicity; meaning ratings 0–100 | CC BY-NC 4.0 stated in official repo LICENSE and Hugging Face ratings metadata | Suitable as a separate general-domain meaning-preservation benchmark; not legal LMP labels |
| Existing 334 project pairs | Template-generated English/Hindi pairs | Project-generated | Synthetic development data; not external gold labels |

## Additional verified external benchmark

ASSET is not legal-domain data, but it provides human crowd-worker meaning-preservation ratings for original/simplification pairs. The importer in `src/import_asset_meaning_ratings.py` selects the `meaning` aspect, preserves the native 0–100 scale and aggregates ratings by pair. It does not assign legal LMP scores or train the legal model. Official repository and dataset card identify CC BY-NC 4.0 for the ratings configuration.

- ASSET dataset card: https://huggingface.co/datasets/facebook/asset
- ASSET official repository/license: https://github.com/facebookresearch/asset
- ASSET paper: https://aclanthology.org/2020.acl-main.424/

## Sources
- SIMPLE-LAW paper: https://aclanthology.org/2026.lrec-1.45/
- SIMPLE-LAW repository: https://github.com/mohammeddanishrabbani/Legal-Simplification-mrabbani
- Lex-Simple implementation: https://github.com/koc-lab/uslt-lex-simple
- JUDGEBERT repository/license: https://github.com/GRAAL-Research/JUDGEBERT
- English-Hindi legal translation corpus: https://huggingface.co/datasets/Wasserstoff-AI/legalTransEn_Indic

## Decision
No external legal simplification dataset with verified reuse terms and validated legal LMP ratings is currently approved. ASSET is added as a separately identified, licensed general-domain benchmark with human meaning ratings; it does not satisfy the legal-domain annotation requirement. The repository now includes a human annotation protocol, blank schema and strict validator. Human annotation must be completed before any rows are called validated or used as LMP regression labels.
