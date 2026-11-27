#!/usr/bin/env python3
"""
Generate a professional IEEE-style research paper (Word .docx)
for the multilingual JUDGEBERT / PreserveLex LMP project.

This version follows a careful evidence-aligned style:
- No unsubstantiated superiority claims
- Explicitly marks simulated labels
- Includes TF-IDF lexical baseline
- Reports ASSET as out-of-domain diagnostic only
"""

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import os


def set_run_font(run, name="Times New Roman", size=10, bold=False, italic=False):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_heading_styled(doc, text, level=1):
    p = doc.add_paragraph()
    run = p.add_run(text)
    if level == 1:
        set_run_font(run, size=12, bold=True)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
    else:
        set_run_font(run, size=11, bold=True)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
    return p


def add_body(doc, text, first_indent=True):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_run_font(run, size=10)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(6)
    if first_indent:
        p.paragraph_format.first_line_indent = Inches(0.25)
    return p


def create_paper(output_path="docs/IEEE_Paper_Multilingual_JUDGEBERT.docx"):
    doc = Document()

    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)

    # ---------- Title ----------
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(
        "PreserveLex: An Explainable Multilingual Framework for\n"
        "Legal Meaning Preservation in English and Hindi Contracts"
    )
    set_run_font(run, size=14, bold=True)
    title.paragraph_format.space_after = Pt(12)

    authors = doc.add_paragraph()
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = authors.add_run(
        "Kartik Nagar · Mohita Purohit · Samriddhi Sharma · Parth Sharma\n"
        "Department of Computer Science and Engineering\n"
        "Bennett University, Greater Noida, India\n"
        "CSET 346 – Natural Language Processing\n"
        "Instructor: Dr. Surabhi Solanki"
    )
    set_run_font(run, size=10)
    authors.paragraph_format.space_after = Pt(12)

    # ---------- Abstract ----------
    abs_head = doc.add_paragraph()
    run = abs_head.add_run("Abstract")
    set_run_font(run, size=10, bold=True)
    abs_head.alignment = WD_ALIGN_PARAGRAPH.CENTER

    abstract_text = (
        "Legal text simplification must improve readability without changing obligations, permissions, conditions, "
        "exceptions, parties, amounts, or deadlines. General semantic similarity can reward topical overlap even when "
        "a legally operative relation is reversed. This paper presents PreserveLex, a research prototype for "
        "sentence-pair legal meaning preservation (LMP) in English and Hindi. Its learned scorer uses XLM-RoBERTa-base "
        "with a regression head to estimate a 1–10 preservation score. A separate advisory rule layer reports potential "
        "changes across five dimensions. We describe the architecture, data resources, provenance controls, and a "
        "reproducible comparison protocol against original JUDGEBERT and multilingual baselines (including a TF-IDF "
        "lexical baseline). The available development corpus and workbook contain synthetic or simulated labels. "
        "An out-of-domain diagnostic on the ASSET simplification dataset shows that TF-IDF currently correlates more "
        "strongly with human meaning ratings than the legal-specialized model, as expected for general-domain text. "
        "Accordingly, this paper does not claim human-validated accuracy, Hindi performance superiority, or overall "
        "superiority over JUDGEBERT. It defines the experiments required to test those claims."
    )
    add_body(doc, abstract_text, first_indent=False)

    kw = doc.add_paragraph()
    run = kw.add_run("Index Terms—")
    set_run_font(run, size=10, bold=True)
    run = kw.add_run(
        "Legal meaning preservation, legal text simplification, multilingual NLP, XLM-RoBERTa, "
        "JUDGEBERT, TF-IDF baseline, explainable AI, contract analysis."
    )
    set_run_font(run, size=10, italic=True)

    # ---------- 1. Introduction ----------
    add_heading_styled(doc, "I. INTRODUCTION", level=1)
    add_body(doc,
        "Legal clauses encode operative relationships, not merely topics. A paraphrase can preserve most words yet "
        "reverse who owes a duty, remove a condition precedent, weaken a prohibition, or change a deadline. The "
        "problem matters in multilingual settings such as India, where contracts may be drafted in English while "
        "readers prefer Hindi or other languages. JUDGEBERT introduced a learned legal meaning preservation metric "
        "for French legal sentences and the FrJUDGE dataset. PreserveLex adopts the task framing and combines an "
        "XLM-RoBERTa pair regressor with a separate advisory review layer."
    )
    add_body(doc,
        "Research questions: (RQ1) Can an XLM-RoBERTa pair regressor provide an LMP score interface for English and "
        "Hindi? (RQ2) Can separate rule checks expose legally salient changes that a scalar score may overlook? "
        "(RQ3) How does PreserveLex compare with original JUDGEBERT and multilingual / lexical baselines on a common "
        "human-annotated test set? (RQ4) How robust are the systems to targeted legal-boundary perturbations?"
    )

    # ---------- 2. Related Work ----------
    add_heading_styled(doc, "II. BACKGROUND AND RELATED WORK", level=1)
    add_body(doc,
        "Beauchemin et al. introduced FrJUDGE and JUDGEBERT (CamemBERT-baseV2 regression). XLM-RoBERTa provides "
        "multilingual encoding. Sentence-BERT and BERTScore are strong generic similarity baselines. TF-IDF cosine "
        "similarity remains a useful lexical baseline that often performs well when surface form is informative. "
        "None of the generic methods inherently enforces party-role, modality, or exception invariants."
    )

    # ---------- 3. Problem Formulation ----------
    add_heading_styled(doc, "III. PROBLEM FORMULATION", level=1)
    add_body(doc,
        "Given an original clause x, a candidate simplification z, and a language indicator, the neural component "
        "estimates a score ŷ ∈ [1, 10]. In parallel, an advisory function returns possible findings across five "
        "dimensions: Obligations, Rights and permissions, Deadlines and amounts, Conditions and exceptions, and "
        "Scope and parties. The displayed result is the pair (score, findings); the two paths are intentionally not "
        "fused into a single hidden weighted score."
    )

    # ---------- 4. Architecture ----------
    add_heading_styled(doc, "IV. PRESERVELEX ARCHITECTURE", level=1)
    add_body(doc,
        "Neural path: XLM-RoBERTa-base encodes the sentence pair (max 256 tokens). A regression head (dropout → "
        "linear → ReLU → dropout → scalar) produces the LMP score, clamped to [1, 10]. Training uses a SmoothL1-style "
        "loss. Advisory path: explicit patterns and lexical cues flag possible changes on the five dimensions. "
        "A warning means “inspect this change,” not “the candidate is legally wrong.”"
    )

    # ---------- 5. Data ----------
    add_heading_styled(doc, "V. DATASETS AND PROVENANCE", level=1)
    add_body(doc,
        "Development corpus: 334 pairs (English/Hindi), template-generated with simulated labels. Filled workbook: "
        "100 English pairs with synthetic reference annotations across Insurance, Contract, Employment, Tenancy and "
        "Consumer domains. Insurance pilot: 10 source-linked clauses awaiting independent annotation. All current "
        "labels are synthetic or simulated; they support pipeline testing but must not be presented as human gold."
    )
    add_body(doc,
        "Recommended split policy: pairs derived from the same source clause must remain in one partition. The final "
        "evaluation set must be held out from model selection, threshold tuning and rule development."
    )

    # ---------- 6. Experimental Design ----------
    add_heading_styled(doc, "VI. EXPERIMENTAL DESIGN", level=1)
    add_heading_styled(doc, "A. Baselines", level=2)
    add_body(doc,
        "We compare against: (i) TF-IDF cosine similarity (1–2 gram, scaled to [1, 10]) as a simple lexical baseline; "
        "(ii) multilingual Sentence-BERT cosine similarity; (iii) BERTScore with XLM-RoBERTa-base; and (iv) the "
        "original JUDGEBERT configuration when a common test set becomes available. All systems receive the same "
        "source/candidate pairs. Scale calibration, if needed, is fitted only on training/development data."
    )
    add_heading_styled(doc, "B. Metrics", level=2)
    add_body(doc,
        "Report Pearson correlation, Spearman rank correlation, MAE and RMSE against adjudicated human scores, with "
        "bootstrap confidence intervals where possible. Report results separately for English and Hindi and by domain. "
        "Correlation alone is insufficient: a system can rank pairs well while making large absolute score errors."
    )
    add_heading_styled(doc, "C. Out-of-Domain Diagnostic (ASSET)", level=2)
    add_body(doc,
        "As a diagnostic on general-domain text, we evaluated on 100 pairs from the ASSET simplification dataset "
        "(human meaning ratings rescaled to 1–10). Observed metrics:"
    )
    add_body(doc,
        "PreserveLex (current checkpoint): Pearson ≈ 0.078, Spearman ≈ 0.062, MAE ≈ 2.53, RMSE ≈ 3.17. "
        "TF-IDF (1–2 gram): Pearson ≈ 0.642, Spearman ≈ 0.629, MAE ≈ 2.03, RMSE ≈ 2.47."
    )
    add_body(doc,
        "Interpretation: ASSET is not legal text. Lexical overlap is a strong signal on general-domain simplification, "
        "so TF-IDF outperforms a model specialized on simulated legal pairs. This result is reported as an "
        "out-of-domain diagnostic only; it does not constitute a claim about legal LMP performance. Primary evaluation "
        "must use independently annotated legal pairs with source-clause-disjoint splits."
    )

    # ---------- 7. Current Evidence Status ----------
    add_heading_styled(doc, "VII. CURRENT EVIDENCE AND RESULTS STATUS", level=1)
    add_body(doc,
        "At the time of writing, available data support implementation and pipeline testing, not a benchmark "
        "performance claim. No verified bilingual human test set or same-test-set comparison against original "
        "JUDGEBERT, Sentence-BERT, BERTScore or TF-IDF on legal text is available. Earlier internal metrics computed "
        "against simulated labels with unverified split provenance are excluded from the evidence base of this paper."
    )

    # ---------- 8. Limitations ----------
    add_heading_styled(doc, "VIII. LIMITATIONS, ETHICS AND FUTURE WORK", level=1)
    add_body(doc,
        "Current development labels are simulated or synthetic. The advisory layer relies on lexical patterns and may "
        "miss indirect or paraphrased changes. PreserveLex is a research and decision-support prototype, not a source "
        "of legal advice. Future work includes independent double annotation, a source-verified Hindi test set, "
        "leakage-resistant splits, controlled baseline comparisons with confidence intervals, and stronger multilingual "
        "rule coverage."
    )

    # ---------- 9. Conclusion ----------
    add_heading_styled(doc, "IX. CONCLUSION", level=1)
    add_body(doc,
        "This paper presented PreserveLex, a JUDGEBERT-inspired multilingual system for assessing whether a simplified "
        "legal clause preserves the meaning of its source. It couples an XLM-RoBERTa regression model with transparent "
        "advisory checks over five legal review dimensions. The separation of a scalar estimate from rule-based "
        "warnings is a design choice intended to support reviewer inspection. The central claim of superiority over "
        "original JUDGEBERT or over strong lexical and semantic baselines remains unproven and requires independently "
        "annotated, same-test-set evidence."
    )

    # ---------- References ----------
    add_heading_styled(doc, "REFERENCES", level=1)
    refs = [
        '[1] D. Beauchemin, M.-A. Albert-Rochette, R. Khoury, and P. Déziel, "JUDGEBERT: Assessing legal meaning preservation between sentences," in Proc. EMNLP, 2025.',
        '[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in Proc. EMNLP-IJCNLP, 2019.',
        '[3] T. Zhang et al., "BERTScore: Evaluating text generation with BERT," in Proc. ICLR, 2020.',
        '[4] A. Conneau et al., "Unsupervised cross-lingual representation learning at scale," in Proc. ACL, 2020.',
        '[5] I. Chalkidis et al., "LEGAL-BERT: The muppets straight out of law school," in Findings of EMNLP, 2020.',
        '[6] D. Beauchemin, H. Saggion, and R. Khoury, "MeaningBERT: Assessing meaning preservation between sentences," Front. Artif. Intell., 2023.',
    ]
    for r in refs:
        p = doc.add_paragraph()
        run = p.add_run(r)
        set_run_font(run, size=9)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc.save(output_path)
    print(f"IEEE-style paper saved to {output_path}")
    return output_path


if __name__ == "__main__":
    create_paper()
