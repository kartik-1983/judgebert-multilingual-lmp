# Independent Human LMP Annotation Protocol

## Purpose
Create a defensible English/Hindi legal meaning-preservation dataset from independently reviewed original–simplified legal text pairs. Synthetic scores and translation-only pairs are not accepted as LMP supervision.

## Dataset screening (27 November 2026)
- SIMPLE-LAW (6,230 pairs): the LREC 2026 paper describes aligned original/simplified legal passages from Indian, US and UK laws. The public repository listing reviewed does not expose a clear dataset license. Do not import, redistribute or train on it until the authors confirm reuse terms or publish an explicit license.
- Lex-Simple (500 US Supreme Court sentences, three simplified references): legal-expert/student involvement and human evaluation are described, but the evidence checked does not establish a clear dataset license or row-level numeric LMP ratings. Do not import without permission and verification.
- JUDGEBERT FrJUDGE: CC-BY-NC-SA 4.0, but its published fields are single-sentence grammaticality labels, not original/simplified pairs or LMP ratings.
- legalTransEn_Indic: MIT English-Hindi legal translations, useful as auxiliary data, but not simplification pairs and not LMP-rated.

Therefore this repository uses a human annotation protocol rather than claiming an unverified dataset license. Re-check licenses before any future import.

## Pair sourcing
Use legal text whose source and reuse status can be documented. Prefer official government publications and record document title, section/article, URL, jurisdiction, access date and applicable reuse/license statement. Do not assume every judgment, website transcription or third-party dataset is public domain.

Each pair needs exact original clause, a plain-language simplification, stable source-document ID, source URL, provision, jurisdiction, language and simplification provenance. Do not use model-generated simplifications as gold references unless independently reviewed and accepted; record model assistance explicitly.

## Annotation rubric
Two reviewers independently assess every pair. At least one reviewer should have legal training or documented legal-domain experience. A third reviewer adjudicates disagreements. Reviewers must not see each other's scores before submitting.

Overall LMP score (1–10):
- 10: all material legal meaning preserved; wording/readability only changed.
- 8–9: meaning preserved; negligible nuance loss only.
- 6–7: mostly preserved, but limited ambiguity or minor detail loss.
- 4–5: material detail weakened, omitted or made ambiguous.
- 2–3: major legal meaning changed; important obligation, right, condition, exception, scope, party, date or amount lost/altered.
- 1: contradiction, reversal, fabricated legal effect or unrelated simplification.

Review each invariant as preserved, changed, omitted, added or not_applicable:
1. obligations and prohibitions;
2. rights and permissions;
3. conditions and exceptions;
4. scope and parties;
5. dates, deadlines, durations and amounts.

Error tags: omission, addition_or_hallucination, contradiction, obligation_change, permission_change, condition_change, exception_change, scope_or_party_change, date_or_duration_change, number_or_amount_change, entity_change, other.

A high score is not allowed when a material legal invariant is changed, omitted or fabricated. Explain any score of 7 or below and any changed/omitted/added invariant.

## Adjudication and acceptance
- Two independent overall scores are required.
- If scores differ by more than 2 points, or any material invariant judgments conflict, a third reviewer adjudicates.
- The adjudicator records the final score and rationale; do not silently average a material disagreement.
- adjudicated_lmp_score is the only field eligible as regression target.
- Rows lacking source provenance, two independent reviews or adjudication are not accepted for supervised training.
- Pseudonymize reviewer identities in released research data; retain consent/audit records separately.

## Quality checks and splits
Deduplicate exact/near-duplicate pairs. Split by source-document ID, never randomly by row. Keep a held-out test set untouched. Report language, jurisdiction, source-document counts, reviewer count, missingness, score distribution, weighted Cohen's kappa where suitable, and adjudication rate. Do not claim improvement until baseline and updated model are compared on the same untouched, independently annotated test set.

## Workflow
1. Copy data/human_lmp_annotations/annotation_template.csv to a working file.
2. Add legally sourced original/simplified pairs and provenance.
3. Have two reviewers fill independent fields.
4. Have an adjudicator complete adjudication fields.
5. Run:
   
   python src/validate_human_lmp_annotations.py --input data/human_lmp_annotations/your_annotated_pairs.csv

The validator writes only fully reviewed rows to data/human_lmp_annotations/validated_lmp_pairs.csv. It does not invent scores or create labels.

## Research integrity
Until real rows pass this protocol, the human-annotated LMP dataset remains pending. The 334 template-generated pairs remain synthetic development data. The English-Hindi legal translation corpus remains auxiliary. Neither is evidence that the LMP model has been trained or improved using externally validated simplification ratings.
