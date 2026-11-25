# PreserveLex Human Annotation Guidelines

**Version:** 1.0  
**Task:** Rate legal meaning preservation between an original clause and a simplified clause.  
**Languages:** English and Hindi  
**Use:** Research annotation only; not legal advice.

## Before rating

- Read the full original and simplified clauses.
- Read any supplied surrounding provision, definitions, schedule, or source context.
- Check that both texts concern the same legal provision and language/version.
- If essential context is missing or the text is not understandable, do not guess. Mark NEEDS_CONTEXT and explain what is missing.
- Do not use the model's predicted score or another reviewer's rating to decide your score.

## What to compare

Assess the legal effect across five dimensions:

1. **Obligations:** Who must or must not do what? Has a duty, prohibition, or responsible party changed?
2. **Rights and permissions:** Who may do what? Are consent requirements, discretion, entitlements, or remedies changed?
3. **Deadlines and amounts:** Are dates, notice periods, time limits, sums, caps, percentages, or thresholds preserved?
4. **Conditions and exceptions:** Are prerequisites, triggers, exclusions, provisos, and exceptions preserved?
5. **Scope and parties:** Are covered persons, property, events, claims, territory, and defined terms preserved?

A simplification can be grammatically clear but legally wrong. Pay particular attention to negation, modal verbs (shall/may/must), exceptions, and who performs an action.

## Score each pair

Use an integer from 1 to 10:

| Score | Guidance |
|---:|---|
| 10 | All material legal effects are preserved. |
| 8–9 | Substantially preserved; only minor nuance or wording uncertainty. |
| 6–7 | A material detail is weakened, ambiguous, or incompletely expressed. |
| 4–5 | Important condition, limit, party, or duty is changed or omitted. |
| 2–3 | Major legal effect is reversed, broadened, narrowed, or misrepresented. |
| 1 | Unrelated clauses or fundamentally reversed core meaning. |

Score the legal meaning, not writing quality. Do not automatically assign a high score because words are similar or a low score because wording is different.

## Error categories

Select every applicable category:
- Obligations
- Rights and permissions
- Deadlines and amounts
- Conditions and exceptions
- Scope and parties
- None (only if no material meaning error is identified)

If the simplification introduces a new legal effect, note that explicitly.

## Required reviewer fields

For each pair, complete:
- pair_id: retain the supplied identifier;
- annotator_id: use the reviewer ID assigned by the study coordinator;
- score: integer 1–10;
- error_types: one or more categories above, or None;
- notes: briefly identify the preserved/changed legal effect and the relevant wording;
- confidence: High, Medium, or Low;
- needs_context: Yes or No.

Never fill another reviewer's file. Do not discuss scores with the other reviewer until both independent files have been submitted and locked.

## Examples of issues to flag

- “within 30 days” changed to “as soon as convenient” (deadline weakened/removed);
- “shall not settle without written consent” changed to “may settle independently” (prohibition and consent condition reversed);
- a stated liability cap omitted (amount/limit changed);
- an exclusion or exception omitted (coverage scope broadened);
- the insured and insurer roles swapped (party/obligation changed).

These are illustrative patterns, not ratings for any current dataset item.

## Uncertainty and escalation

Use NEEDS_CONTEXT when the meaning depends on an unavailable definition, schedule, cross-reference, jurisdiction-specific term, or missing surrounding clause. Explain the missing information. Do not invent legal context or resolve ambiguity by assumption.

## Independence and integrity

- Complete ratings independently.
- Do not copy or derive ratings from AI predictions.
- AI-generated suggestions, if any, must be stored separately and must never be relabelled as human annotations.
- Preserve the original submitted ratings. Adjudication must be a separate file with a documented reason.
- Do not claim human agreement until two genuinely independent reviewers have completed the same items.
