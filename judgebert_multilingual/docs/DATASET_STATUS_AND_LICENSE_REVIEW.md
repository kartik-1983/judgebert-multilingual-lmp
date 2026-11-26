# Legal Simplification Dataset: License and Suitability Review

Review date: 27 November 2026

| Resource | Contents | License status checked | PreserveLex suitability |
|---|---|---|---|
| SIMPLE-LAW (Rabbani et al., LREC 2026) | 6,230 aligned original/simplified legal passages; India, US, UK | No explicit dataset license established in repository listing reviewed | Strong task match, but do not import/train/redistribute until reuse terms are confirmed |
| Lex-Simple | 500 US Supreme Court sentences with three simplified references; legal expert/student involvement reported | Clear license not established in evidence reviewed | Strong task match, but permission and row-level ratings need confirmation |
| JUDGEBERT FrJUDGE | Single French sentences with grammaticality labels | CC-BY-NC-SA 4.0 | Not LMP pairs; do not repurpose labels |
| legalTransEn_Indic | English-Hindi legal translations | MIT shown on dataset card | Auxiliary multilingual corpus only; not simplification and no LMP ratings |
| Existing 334 project pairs | Template-generated English/Hindi pairs | Project-generated | Synthetic development data; not external gold labels |

## Sources
- SIMPLE-LAW paper: https://aclanthology.org/2026.lrec-1.45/
- SIMPLE-LAW repository: https://github.com/mohammeddanishrabbani/Legal-Simplification-mrabbani
- Lex-Simple implementation: https://github.com/koc-lab/uslt-lex-simple
- JUDGEBERT repository/license: https://github.com/GRAAL-Research/JUDGEBERT
- English-Hindi legal translation corpus: https://huggingface.co/datasets/Wasserstoff-AI/legalTransEn_Indic

## Decision
No external legal simplification dataset is imported by this commit because clear reuse terms and/or validated row-level LMP ratings could not be verified for the closest matching candidates. The repository now includes a human annotation protocol, blank schema and strict validator. Human annotation must be completed before any rows are called validated or used as LMP regression labels.
