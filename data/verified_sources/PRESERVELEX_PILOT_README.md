# PreserveLex pilot package

This package contains a 10-clause English pilot based on the Generali Central Standard Fire & Special Perils Policy wording (UIN IRDAN132CPPR0013V01201920).

## Contents
- preservelex_pilot10_corrected.csv: clause summaries, draft simplifications, meaning-changing contrasts, dimensions and review status.
- preservelex_pilot10_source_manifest.csv: provenance records using the repository manifest schema.
- preservelex_R1_annotation_sheet.csv and preservelex_R2_annotation_sheet.csv: separate independent reviewer sheets.

## Research status and limitations
- The source document and page/clause locators were identified from the official insurer-hosted policy wording.
- Exact source text is intentionally not reproduced in these CSVs. Reviewers must inspect the source document and confirm exact wording and page/clause before scoring.
- Simplifications are AI-assisted researcher drafts, not human-validated.
- No gold labels or model predictions are supplied. Do not use this pilot for training or claim evaluation performance until source text is checked and independent human annotations are completed.
- Do not merge this pilot into the existing synthetic dataset. Keep it separate.
- Confirm applicable reuse terms before redistributing verbatim policy text.

## Suggested destination
Place the four CSV files in:
data/verified_sources/

Keep the README with them for provenance and handling instructions.
