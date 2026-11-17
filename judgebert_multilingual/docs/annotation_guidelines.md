# Annotation Guidelines – Legal Meaning Preservation (LMP)

These guidelines follow the spirit of the FrJUDGE / JUDGEBERT annotation protocol and are adapted for English and Hindi legal (insurance / contract) sentences.

## Task Definition

Given an **original legal sentence** and a **simplified / reformulated version**, assign a **Legal Meaning Preservation (LMP) score from 1 to 10** that reflects how well the simplified text preserves the *legal* meaning of the original.

### What is “Legal Meaning”?

Legal meaning is **not** the same as ordinary semantic similarity. It specifically concerns:

- Whether **obligations**, **permissions**, and **prohibitions** remain the same.
- Whether **conditions**, **exceptions**, and **limitations** are retained.
- Whether **scope**, **dates**, **numbers**, **parties**, and **entities** are unchanged.
- Whether the simplification introduces **hallucinations** or **contradictions** relative to the original legal effect.

A pair can be linguistically very similar yet receive a low LMP score if a legally critical element is altered or omitted.

## Score Scale

| Score | Meaning |
|-------|---------|
| **9–10** | Fully preserved. All legally relevant elements are retained; wording changes do not affect legal effect. |
| **7–8**  | Largely preserved. Minor omissions or rephrasings that do not change the core legal meaning. |
| **5–6**  | Partially preserved. Some important conditions, exceptions or scope elements are weakened or missing. |
| **3–4**  | Poorly preserved. Key legal obligations, conditions or exceptions are substantially altered. |
| **1–2**  | Not preserved. Contradiction, major hallucination, or complete loss of legal meaning. |

## Decision Labels (derived from score)

- **Preserved**: score ≥ 8.0  
- **Partially Preserved**: 5.0 ≤ score < 8.0  
- **Not Preserved**: score < 5.0  

## Error Categories (mark all that apply)

When the score is below 8, annotate the type(s) of legal meaning change:

- **omission** – important legal detail is missing
- **hallucination** – new content not present in the original is introduced
- **contradiction** – simplified text asserts the opposite of the original
- **obligation** – duty / “shall” / “must” is weakened or removed
- **permission** – right / “may” is changed
- **condition** – “if / subject to / provided that” is altered
- **exception** – exclusion or exception is dropped or changed
- **scope** – breadth of coverage is expanded or narrowed incorrectly
- **date** – time limits or notice periods are changed
- **numbers** – quantities, percentages, monetary amounts are altered
- **entities** – parties, insured objects, or defined terms are changed

## Practical Instructions for Annotators

1. Read the original sentence carefully, identifying every legal operator (shall, may, subject to, except, within X days, etc.).
2. Read the simplified sentence and check whether each operator and its arguments are preserved.
3. Assign the LMP score independently of fluency or simplicity.
4. Spend no more than 5 minutes per pair (as in the original protocol).
5. When in doubt between two adjacent scores, prefer the lower score (conservative with respect to legal risk).

## Sanity Checks (for quality control)

- Identical sentences must receive score = 10.
- Completely unrelated sentences must receive score ≤ 2.

## Notes for This Project

The current dataset uses **simulated scores** generated from controlled good/bad simplification templates that follow the above principles.  
**For a full academic submission, replace the simulated scores with genuine annotations from at least two independent annotators** (preferably law students or legal professionals) and report inter-annotator agreement (e.g., Krippendorff’s α or Pearson correlation between annotators).
