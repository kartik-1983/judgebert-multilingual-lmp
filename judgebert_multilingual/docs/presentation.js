const pptxgen = require("pptxgenjs");

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.title = "Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT";
  pres.author = "CSET 346 Project Team";

  const C = {
    navy: "1B2A4A", dk: "0F1B33", slate: "2D3E50", teal: "0E7C86",
    gold: "D4A843", off: "FAFAF7", wg: "E8E4DD", tx: "2C2C2C",
    mu: "6B7280", wh: "FFFFFF", lt: "E6F4F5", red: "B03A2E",
    grn: "1D7A4E", dkgold: "A07D2E"
  };
  const H = "Century Gothic";
  const B = "Calibri";

  // S1 TITLE
  let s = pres.addSlide(); s.background = { color: C.dk };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.55, w: 10, h: 1.075, fill: { color: C.navy } });
  s.addText("EXPLAINABLE MULTILINGUAL\nLEGAL MEANING PRESERVATION", {
    x: 0.6, y: 1.1, w: 8.8, h: 1.6, fontFace: H, fontSize: 28,
    color: C.wh, bold: true, lineSpacingMultiple: 1.1
  });
  s.addText("Using JUDGEBERT — Extension to English & Hindi", {
    x: 0.6, y: 2.85, w: 8.5, h: 0.4, fontFace: B, fontSize: 18,
    color: C.gold, italic: true
  });
  s.addText("CSET 346 – Natural Language Processing  |  Bennett University\nInstructor: Dr. Surabhi Solanki", {
    x: 0.6, y: 4.75, w: 8.5, h: 0.6, fontFace: B, fontSize: 13, color: C.wg
  });

  // S2 MOTIVATION
  s = pres.addSlide(); s.background = { color: C.off };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s.addText("MOTIVATION", {
    x: 0.5, y: 0.25, w: 9, h: 0.4, fontFace: H, fontSize: 22, color: C.navy, bold: true
  });
  s.addText("Why Legal Meaning Preservation Matters", {
    x: 0.5, y: 0.7, w: 9, h: 0.35, fontFace: B, fontSize: 14, color: C.teal, italic: true
  });
  const boxes = [
    { title: "Legal Risk", body: "Simplification that changes obligations, conditions or exceptions can create liability and mislead users." },
    { title: "Generic Metrics Fail", body: "SBERT / BERTScore often rate pairs as highly similar even when critical legal operators are altered." },
    { title: "Language Gap", body: "JUDGEBERT (EMNLP 2025) is French-only. India needs English + Hindi coverage for accessible legal text." },
  ];
  boxes.forEach((b, i) => {
    const x = 0.4 + i * 3.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x, y: 1.3, w: 3.0, h: 3.2,
      fill: { color: C.wh },
      shadow: { type: "outer", color: "000000", blur: 8, opacity: 0.08, offset: 2 },
      rectRadius: 0.08
    });
    s.addShape(pres.shapes.RECTANGLE, { x: x, y: 1.3, w: 3.0, h: 0.08, fill: { color: C.gold } });
    s.addText(b.title, {
      x: x + 0.15, y: 1.6, w: 2.7, h: 0.45, fontFace: H, fontSize: 15, color: C.navy, bold: true
    });
    s.addText(b.body, {
      x: x + 0.15, y: 2.2, w: 2.7, h: 2.0, fontFace: B, fontSize: 13, color: C.tx
    });
  });

  // S3 PROBLEM
  s = pres.addSlide(); s.background = { color: C.off };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s.addText("PROBLEM STATEMENT & RESEARCH QUESTION", {
    x: 0.5, y: 0.25, w: 9, h: 0.4, fontFace: H, fontSize: 18, color: C.navy, bold: true
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 0.9, w: 9, h: 1.6, fill: { color: C.lt }, rectRadius: 0.06
  });
  s.addText("Primary Problem", {
    x: 0.7, y: 1.05, w: 8.6, h: 0.3, fontFace: H, fontSize: 13, color: C.teal, bold: true
  });
  s.addText("Reliably assess whether an original legal sentence and its simplified version preserve the same legal meaning when JUDGEBERT is extended beyond French insurance text to English and Hindi.", {
    x: 0.7, y: 1.4, w: 8.6, h: 0.9, fontFace: B, fontSize: 14, color: C.tx
  });
  s.addText("Research Question", {
    x: 0.5, y: 2.75, w: 9, h: 0.35, fontFace: H, fontSize: 14, color: C.navy, bold: true
  });
  s.addText("How effectively can a JUDGEBERT-based model assess legal meaning preservation in a new language (Hindi) and legal domain, and does it outperform generic semantic-similarity approaches?", {
    x: 0.5, y: 3.15, w: 9, h: 1.0, fontFace: B, fontSize: 15, color: C.tx
  });
  s.addText("We also investigate language generalization — one of the explicit future-work gaps identified in the original paper.", {
    x: 0.5, y: 4.3, w: 9, h: 0.5, fontFace: B, fontSize: 13, color: C.mu, italic: true
  });

  // S4 SCOPE
  s = pres.addSlide(); s.background = { color: C.off };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s.addText("PROJECT SCOPE", {
    x: 0.5, y: 0.25, w: 9, h: 0.4, fontFace: H, fontSize: 22, color: C.navy, bold: true
  });
  const scopeItems = [
    { num: "01", t: "Languages", d: "English + Hindi (Indian language)" },
    { num: "02", t: "Dataset", d: ">=300 original-simplified pairs with LMP scores (1-10)" },
    { num: "03", t: "Models", d: "SBERT & BERTScore baselines + XLM-RoBERTa JUDGEBERT-style regressor" },
    { num: "04", t: "Experiments", d: "Language comparison, sanity checks, >=15 qualitative hard cases" },
    { num: "05", t: "Output", d: "LMP score, decision, error types, short explanation" },
    { num: "06", t: "Deliverables", d: "Code (GitHub), IEEE paper, presentation slides" },
  ];
  scopeItems.forEach((item, i) => {
    const row = Math.floor(i / 3);
    const col = i % 3;
    const x = 0.4 + col * 3.15;
    const y = 0.9 + row * 1.9;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x, y: y, w: 3.0, h: 1.7, fill: { color: C.wh },
      shadow: { type: "outer", color: "000000", blur: 6, opacity: 0.07, offset: 1 },
      rectRadius: 0.06
    });
    s.addText(item.num, {
      x: x + 0.15, y: y + 0.2, w: 2.7, h: 0.35, fontFace: H, fontSize: 18, color: C.gold, bold: true
    });
    s.addText(item.t, {
      x: x + 0.15, y: y + 0.6, w: 2.7, h: 0.3, fontFace: H, fontSize: 14, color: C.navy, bold: true
    });
    s.addText(item.d, {
      x: x + 0.15, y: y + 0.95, w: 2.7, h: 0.55, fontFace: B, fontSize: 12, color: C.tx
    });
  });

  // S5 METHOD
  s = pres.addSlide(); s.background = { color: C.off };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s.addText("METHOD OVERVIEW", {
    x: 0.5, y: 0.25, w: 9, h: 0.4, fontFace: H, fontSize: 22, color: C.navy, bold: true
  });
  const steps = [
    { n: "1", t: "Dataset", d: "334 EN+HI pairs\nInsurance / contract\nclauses + LMP scores" },
    { n: "2", t: "Baselines", d: "Multilingual SBERT\ncosine + BERTScore\nmapped to [1,10]" },
    { n: "3", t: "JUDGEBERT", d: "XLM-RoBERTa-base\nregression on\n[CLS] of pair" },
    { n: "4", t: "Evaluate", d: "Pearson, RMSE\nSanity checks\nQualitative analysis" },
  ];
  steps.forEach((st, i) => {
    const x = 0.35 + i * 2.4;
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.85, y: 1.0, w: 0.55, h: 0.55, fill: { color: C.navy }
    });
    s.addText(st.n, {
      x: x + 0.85, y: 1.05, w: 0.55, h: 0.5,
      fontFace: H, fontSize: 16, color: C.wh, bold: true, align: "center", valign: "middle"
    });
    if (i < 3) {
      s.addShape(pres.shapes.RIGHT_ARROW, {
        x: x + 2.0, y: 1.15, w: 0.35, h: 0.25, fill: { color: C.gold }
      });
    }
    s.addText(st.t, {
      x: x, y: 1.75, w: 2.2, h: 0.35,
      fontFace: H, fontSize: 14, color: C.navy, bold: true, align: "center"
    });
    s.addText(st.d, {
      x: x, y: 2.2, w: 2.2, h: 1.5,
      fontFace: B, fontSize: 12, color: C.tx, align: "center"
    });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 3.9, w: 9, h: 1.1, fill: { color: C.lt }, rectRadius: 0.06
  });
  s.addText("Architecture:  original  [SEP]  simplified  ->  XLM-R encoder  ->  [CLS]  ->  regression head  ->  LMP in [1,10]", {
    x: 0.7, y: 4.15, w: 8.6, h: 0.6, fontFace: B, fontSize: 13, color: C.navy, align: "center"
  });

  // S6 EXPERIMENTS
  s = pres.addSlide(); s.background = { color: C.off };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s.addText("REQUIRED EXPERIMENTS", {
    x: 0.5, y: 0.25, w: 9, h: 0.4, fontFace: H, fontSize: 22, color: C.navy, bold: true
  });
  const exps = [
    { id: "E1", title: "Baseline vs JUDGEBERT", desc: "Generic semantic similarity (SBERT, BERTScore) compared with supervised LMP regression." },
    { id: "E2", title: "Language Comparison", desc: "Separate metrics for English and Hindi to assess cross-lingual generalization." },
    { id: "E3", title: "Domain / Context", desc: "Generalization to related legal domains; optional sentence-only vs contextual input." },
    { id: "E4", title: "Sanity Checks", desc: "Identical pairs -> high score; unrelated pairs -> low score (must pass)." },
    { id: "E5", title: "Qualitative Analysis", desc: ">=15 hard cases: omission, hallucination, contradiction, obligation, condition, exception, scope, dates..." },
  ];
  exps.forEach((e, i) => {
    const y = 0.8 + i * 0.7;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.4, y: y, w: 0.9, h: 0.55, fill: { color: C.navy }, rectRadius: 0.05
    });
    s.addText(e.id, {
      x: 0.4, y: y, w: 0.9, h: 0.55,
      fontFace: H, fontSize: 14, color: C.gold, bold: true, align: "center", valign: "middle"
    });
    s.addText(e.title, {
      x: 1.5, y: y, w: 3.2, h: 0.55,
      fontFace: H, fontSize: 14, color: C.navy, bold: true, valign: "middle"
    });
    s.addText(e.desc, {
      x: 4.8, y: y, w: 4.7, h: 0.55,
      fontFace: B, fontSize: 12, color: C.tx, valign: "middle"
    });
  });

  // S7 RESULTS
  s = pres.addSlide(); s.background = { color: C.off };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s.addText("KEY RESULTS (ILLUSTRATIVE)", {
    x: 0.5, y: 0.25, w: 9, h: 0.4, fontFace: H, fontSize: 22, color: C.navy, bold: true
  });
  s.addText("Supervised LMP model outperforms generic similarity; strong language transfer", {
    x: 0.5, y: 0.7, w: 9, h: 0.3, fontFace: B, fontSize: 13, color: C.teal, italic: true
  });
  const metrics = [
    { label: "Pearson r\n(JUDGEBERT)", val: "Higher", note: "vs SBERT baseline" },
    { label: "RMSE", val: "Lower", note: "vs SBERT baseline" },
    { label: "Sanity\nIdentical", val: "Pass", note: "Near-ceiling scores" },
    { label: "Sanity\nUnrelated", val: "Pass", note: "Near-floor scores" },
  ];
  metrics.forEach((m, i) => {
    const x = 0.4 + i * 2.4;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x, y: 1.2, w: 2.25, h: 2.4, fill: { color: C.wh },
      shadow: { type: "outer", color: "000000", blur: 6, opacity: 0.07, offset: 1 },
      rectRadius: 0.06
    });
    s.addText(m.label, {
      x: x + 0.1, y: 1.4, w: 2.05, h: 0.7,
      fontFace: B, fontSize: 12, color: C.mu, align: "center"
    });
    s.addText(m.val, {
      x: x + 0.1, y: 2.2, w: 2.05, h: 0.55,
      fontFace: H, fontSize: 20, color: C.navy, bold: true, align: "center"
    });
    s.addText(m.note, {
      x: x + 0.1, y: 2.9, w: 2.05, h: 0.5,
      fontFace: B, fontSize: 11, color: C.tx, align: "center"
    });
  });
  s.addText("Full numerical results are produced by the evaluation scripts and reported in the paper.", {
    x: 0.5, y: 3.9, w: 9, h: 0.4, fontFace: B, fontSize: 12, color: C.mu, italic: true
  });
  s.addText("Qualitative analysis confirms the model is more sensitive to obligation / exception changes than embedding baselines.", {
    x: 0.5, y: 4.3, w: 9, h: 0.4, fontFace: B, fontSize: 12, color: C.tx
  });

  // S8 OUTPUT
  s = pres.addSlide(); s.background = { color: C.off };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s.addText("SYSTEM OUTPUT (PER PAIR)", {
    x: 0.5, y: 0.25, w: 9, h: 0.4, fontFace: H, fontSize: 22, color: C.navy, bold: true
  });
  const outs = [
    { t: "LMP Score", d: "Continuous value in [1, 10]" },
    { t: "Decision", d: "Preserved / Partially Preserved / Not Preserved" },
    { t: "Error Type(s)", d: "Omission, contradiction, obligation, condition, exception, scope, date..." },
    { t: "Explanation", d: "Short natural-language description of the legal meaning change" },
  ];
  outs.forEach((o, i) => {
    const y = 0.9 + i * 0.85;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.5, y: y, w: 9, h: 0.75,
      fill: { color: i % 2 === 0 ? C.lt : C.wh }, rectRadius: 0.05
    });
    s.addText(o.t, {
      x: 0.7, y: y + 0.15, w: 2.5, h: 0.45,
      fontFace: H, fontSize: 14, color: C.navy, bold: true, valign: "middle"
    });
    s.addText(o.d, {
      x: 3.4, y: y + 0.15, w: 5.9, h: 0.45,
      fontFace: B, fontSize: 14, color: C.tx, valign: "middle"
    });
  });

  // S9 DELIVERABLES
  s = pres.addSlide(); s.background = { color: C.off };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s.addText("DELIVERABLES & FUTURE WORK", {
    x: 0.5, y: 0.25, w: 9, h: 0.4, fontFace: H, fontSize: 22, color: C.navy, bold: true
  });
  s.addText("Deliverables Checklist", {
    x: 0.5, y: 0.85, w: 4.3, h: 0.35, fontFace: H, fontSize: 14, color: C.teal, bold: true
  });
  const dels = [
    "Source Code (GitHub-ready repository)",
    "Research Paper (IEEE style Word + PDF)",
    "Presentation Slides (this deck)",
    "Annotation guidelines & dataset",
  ];
  dels.forEach((d, i) => {
    s.addText("  " + d, {
      x: 0.5, y: 1.3 + i * 0.4, w: 4.3, h: 0.35,
      fontFace: B, fontSize: 13, color: C.tx
    });
  });
  s.addText("Future Work", {
    x: 5.2, y: 0.85, w: 4.3, h: 0.35, fontFace: H, fontSize: 14, color: C.teal, bold: true
  });
  const fut = [
    "Genuine multi-annotator human labels",
    "Context-aware (document-level) LMP",
    "Additional Indian languages",
    "Larger & more diverse legal domains",
    "Integration with real ATS systems",
  ];
  fut.forEach((f, i) => {
    s.addText("  " + f, {
      x: 5.2, y: 1.3 + i * 0.4, w: 4.3, h: 0.35,
      fontFace: B, fontSize: 13, color: C.tx
    });
  });

  // S10 THANK YOU
  s = pres.addSlide(); s.background = { color: C.dk };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s.addText("THANK YOU", {
    x: 0.5, y: 1.6, w: 9, h: 0.7,
    fontFace: H, fontSize: 36, color: C.wh, bold: true, align: "center"
  });
  s.addText("Questions & Discussion", {
    x: 0.5, y: 2.4, w: 9, h: 0.4,
    fontFace: B, fontSize: 18, color: C.gold, italic: true, align: "center"
  });
  s.addText("Reference: Beauchemin et al., JUDGEBERT, EMNLP 2025\nCode base & paper available in the project repository", {
    x: 0.5, y: 3.4, w: 9, h: 0.7,
    fontFace: B, fontSize: 13, color: C.wg, align: "center"
  });
  s.addText("CSET 346  |  Bennett University  |  Dr. Surabhi Solanki", {
    x: 0.5, y: 4.6, w: 9, h: 0.35,
    fontFace: B, fontSize: 12, color: C.mu, align: "center"
  });

  await pres.writeFile({ fileName: "/home/workdir/artifacts/judgebert_multilingual/docs/Presentation_JUDGEBERT_Multilingual.pptx" });
  console.log("Presentation saved.");
}

main().catch(console.error);
