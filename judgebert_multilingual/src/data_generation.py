"""
Generate a realistic bilingual (English + Hindi) Legal Meaning Preservation dataset
for the JUDGEBERT multilingual extension project.

Creates ≥300 original–simplified legal sentence pairs with LMP scores (1–10),
error types, and explanations. Scores are simulated following the spirit of
human annotation in the original FrJUDGE paper.
"""

import json
import random
import uuid
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------------
# Seed legal templates (English) – Insurance / Contract style
# ---------------------------------------------------------------------------

EN_TEMPLATES = [
    {
        "original": "The insurer shall not be liable for any loss or damage caused by or resulting from war, invasion, act of foreign enemy, hostilities (whether war be declared or not), civil war, rebellion, revolution, insurrection or military or usurped power.",
        "good_simp": "The insurer is not responsible for loss or damage caused by war, invasion, foreign enemy actions, hostilities, civil war, rebellion, revolution, insurrection, or military power.",
        "bad_simp": "The insurer is not responsible for any damage.",
        "error_types_bad": ["omission", "scope"],
        "good_score_range": (8.5, 9.8),
        "bad_score_range": (2.0, 4.5),
    },
    {
        "original": "Subject to the terms, conditions and exclusions of this Policy, the Company will indemnify the Insured against loss of or damage to the Property Insured caused by any of the Perils specified in the Schedule.",
        "good_simp": "If the terms and exclusions of this Policy are followed, the Company will pay the Insured for loss or damage to the insured property caused by the perils listed in the Schedule.",
        "bad_simp": "The Company will always pay for any damage to the property.",
        "error_types_bad": ["omission", "condition", "hallucination"],
        "good_score_range": (8.0, 9.5),
        "bad_score_range": (1.5, 3.5),
    },
    {
        "original": "The Insured shall take all reasonable steps to safeguard the Property Insured from loss or damage and to maintain it in efficient condition.",
        "good_simp": "The Insured must take all reasonable steps to protect the insured property from loss or damage and keep it in good working condition.",
        "bad_simp": "The Insured should try to look after the property.",
        "error_types_bad": ["omission", "obligation"],
        "good_score_range": (8.5, 9.7),
        "bad_score_range": (4.0, 6.0),
    },
    {
        "original": "No claim shall be payable under this Policy unless the premium has been paid in full and the Policy has been issued by the Company.",
        "good_simp": "A claim will only be paid under this Policy if the full premium has been paid and the Company has issued the Policy.",
        "bad_simp": "Claims are paid even if the premium is not fully paid.",
        "error_types_bad": ["contradiction", "condition"],
        "good_score_range": (9.0, 10.0),
        "bad_score_range": (1.0, 2.5),
    },
    {
        "original": "The Company may at any time cancel this Policy by giving 15 days' notice in writing to the Insured at the last known address.",
        "good_simp": "The Company can cancel this Policy at any time by giving the Insured 15 days' written notice at their last known address.",
        "bad_simp": "The Company can cancel the Policy immediately without notice.",
        "error_types_bad": ["contradiction", "date", "permission"],
        "good_score_range": (9.0, 9.8),
        "bad_score_range": (1.5, 3.0),
    },
    {
        "original": "In the event of any claim, the Insured shall give immediate notice to the Company and shall within 30 days deliver a detailed statement of the loss.",
        "good_simp": "If there is a claim, the Insured must tell the Company immediately and submit a detailed loss statement within 30 days.",
        "bad_simp": "The Insured should inform the Company about the claim when convenient.",
        "error_types_bad": ["omission", "obligation", "date"],
        "good_score_range": (8.5, 9.6),
        "bad_score_range": (3.0, 5.0),
    },
    {
        "original": "This Policy does not cover loss or damage arising from wear and tear, gradual deterioration, inherent vice, or mechanical or electrical breakdown.",
        "good_simp": "This Policy does not cover loss or damage from normal wear and tear, gradual deterioration, inherent defects, or mechanical or electrical breakdown.",
        "bad_simp": "This Policy covers all types of damage including wear and tear.",
        "error_types_bad": ["contradiction", "exception"],
        "good_score_range": (9.0, 9.9),
        "bad_score_range": (1.0, 2.5),
    },
    {
        "original": "The sum insured under this Policy represents the maximum liability of the Company in respect of any one claim or series of claims arising from one event.",
        "good_simp": "The sum insured is the maximum amount the Company will pay for any single claim or series of claims from one event.",
        "bad_simp": "The Company will pay any amount claimed by the Insured.",
        "error_types_bad": ["contradiction", "scope"],
        "good_score_range": (8.8, 9.7),
        "bad_score_range": (1.5, 3.0),
    },
    {
        "original": "If the Property Insured is under-insured, the Insured shall be considered as being their own insurer for the difference and shall bear a rateable proportion of the loss.",
        "good_simp": "If the property is insured for less than its value, the Insured will be treated as their own insurer for the shortfall and must share the loss proportionally.",
        "bad_simp": "Under-insurance does not affect the claim amount.",
        "error_types_bad": ["contradiction", "condition"],
        "good_score_range": (8.0, 9.3),
        "bad_score_range": (2.0, 4.0),
    },
    {
        "original": "The Insured shall not, without the written consent of the Company, admit liability or make any payment or offer of settlement in respect of any claim.",
        "good_simp": "The Insured must not admit liability or make any payment or settlement offer for a claim without the Company's written consent.",
        "bad_simp": "The Insured can settle claims on their own.",
        "error_types_bad": ["contradiction", "permission", "obligation"],
        "good_score_range": (9.0, 9.8),
        "bad_score_range": (1.5, 3.5),
    },
]

# Hindi translations / adaptations of the same concepts (realistic formal Hindi)
HI_TEMPLATES = [
    {
        "original": "बीमाकर्ता युद्ध, आक्रमण, विदेशी शत्रु के कार्य, शत्रुता (चाहे युद्ध घोषित हो या न हो), गृहयुद्ध, विद्रोह, क्रांति, विद्रोह या सैन्य या हड़प ली गई शक्ति के कारण या उससे उत्पन्न किसी भी हानि या क्षति के लिए उत्तरदायी नहीं होगा।",
        "good_simp": "बीमाकर्ता युद्ध, आक्रमण, विदेशी शत्रु के कार्यों, शत्रुता, गृहयुद्ध, विद्रोह, क्रांति या सैन्य शक्ति के कारण हुई हानि या क्षति के लिए जिम्मेदार नहीं है।",
        "bad_simp": "बीमाकर्ता किसी भी क्षति के लिए जिम्मेदार नहीं है।",
        "error_types_bad": ["omission", "scope"],
        "good_score_range": (8.3, 9.6),
        "bad_score_range": (2.0, 4.5),
    },
    {
        "original": "इस पॉलिसी के नियमों, शर्तों और बहिष्करणों के अधीन, कंपनी अनुसूची में निर्दिष्ट किसी भी खतरे के कारण बीमाकृत संपत्ति की हानि या क्षति के खिलाफ बीमाकृत को क्षतिपूर्ति करेगी।",
        "good_simp": "यदि इस पॉलिसी के नियम और बहिष्करण पूरे होते हैं, तो कंपनी अनुसूची में सूचीबद्ध खतरों के कारण बीमाकृत संपत्ति की हानि या क्षति के लिए बीमाकृत को भुगतान करेगी।",
        "bad_simp": "कंपनी हमेशा संपत्ति की किसी भी क्षति के लिए भुगतान करेगी।",
        "error_types_bad": ["omission", "condition", "hallucination"],
        "good_score_range": (8.0, 9.4),
        "bad_score_range": (1.5, 3.5),
    },
    {
        "original": "बीमाकृत बीमाकृत संपत्ति को हानि या क्षति से बचाने और इसे कुशल स्थिति में बनाए रखने के लिए सभी उचित कदम उठाएगा।",
        "good_simp": "बीमाकृत को बीमाकृत संपत्ति को हानि या क्षति से बचाने और इसे अच्छी स्थिति में रखने के लिए सभी उचित कदम उठाने होंगे।",
        "bad_simp": "बीमाकृत को संपत्ति का ध्यान रखना चाहिए।",
        "error_types_bad": ["omission", "obligation"],
        "good_score_range": (8.4, 9.5),
        "bad_score_range": (4.0, 6.0),
    },
    {
        "original": "इस पॉलिसी के अंतर्गत कोई दावा तब तक देय नहीं होगा जब तक कि प्रीमियम पूरी तरह से भुगतान न कर दिया गया हो और कंपनी द्वारा पॉलिसी जारी न की गई हो।",
        "good_simp": "इस पॉलिसी के अंतर्गत दावा तभी भुगतान किया जाएगा जब पूरा प्रीमियम चुका दिया गया हो और कंपनी ने पॉलिसी जारी की हो।",
        "bad_simp": "प्रीमियम पूरी तरह से भुगतान न होने पर भी दावे का भुगतान किया जाता है।",
        "error_types_bad": ["contradiction", "condition"],
        "good_score_range": (9.0, 9.9),
        "bad_score_range": (1.0, 2.5),
    },
    {
        "original": "कंपनी किसी भी समय बीमाकृत को अंतिम ज्ञात पते पर 15 दिन का लिखित नोटिस देकर इस पॉलिसी को रद्द कर सकती है।",
        "good_simp": "कंपनी बीमाकृत को उनके अंतिम ज्ञात पते पर 15 दिन का लिखित नोटिस देकर किसी भी समय इस पॉलिसी को रद्द कर सकती है।",
        "bad_simp": "कंपनी बिना किसी नोटिस के तुरंत पॉलिसी रद्द कर सकती है।",
        "error_types_bad": ["contradiction", "date", "permission"],
        "good_score_range": (9.0, 9.7),
        "bad_score_range": (1.5, 3.0),
    },
    {
        "original": "किसी भी दावे की स्थिति में, बीमाकृत कंपनी को तुरंत सूचना देगा और 30 दिनों के भीतर हानि का विस्तृत विवरण प्रस्तुत करेगा।",
        "good_simp": "यदि कोई दावा है, तो बीमाकृत को कंपनी को तुरंत बताना होगा और 30 दिनों के भीतर हानि का विस्तृत विवरण जमा करना होगा।",
        "bad_simp": "बीमाकृत को सुविधा अनुसार दावे के बारे में कंपनी को सूचित करना चाहिए।",
        "error_types_bad": ["omission", "obligation", "date"],
        "good_score_range": (8.5, 9.5),
        "bad_score_range": (3.0, 5.0),
    },
    {
        "original": "यह पॉलिसी घिसावट, क्रमिक क्षय, अंतर्निहित दोष, या यांत्रिक या विद्युत खराबी से उत्पन्न हानि या क्षति को कवर नहीं करती है।",
        "good_simp": "यह पॉलिसी सामान्य घिसावट, क्रमिक क्षय, अंतर्निहित दोषों, या यांत्रिक या विद्युत खराबी से होने वाली हानि या क्षति को कवर नहीं करती।",
        "bad_simp": "यह पॉलिसी घिसावट सहित सभी प्रकार की क्षति को कवर करती है।",
        "error_types_bad": ["contradiction", "exception"],
        "good_score_range": (9.0, 9.8),
        "bad_score_range": (1.0, 2.5),
    },
    {
        "original": "इस पॉलिसी के अंतर्गत बीमा राशि किसी एक घटना से उत्पन्न किसी एक दावे या दावों की श्रृंखला के संबंध में कंपनी की अधिकतम देयता का प्रतिनिधित्व करती है।",
        "good_simp": "बीमा राशि वह अधिकतम राशि है जो कंपनी किसी एक घटना से उत्पन्न एकल दावे या दावों की श्रृंखला के लिए भुगतान करेगी।",
        "bad_simp": "कंपनी बीमाकृत द्वारा दावा की गई किसी भी राशि का भुगतान करेगी।",
        "error_types_bad": ["contradiction", "scope"],
        "good_score_range": (8.7, 9.6),
        "bad_score_range": (1.5, 3.0),
    },
    {
        "original": "यदि बीमाकृत संपत्ति कम बीमित है, तो बीमाकृत को अंतर के लिए अपना स्वयं का बीमाकर्ता माना जाएगा और वह हानि का आनुपातिक हिस्सा वहन करेगा।",
        "good_simp": "यदि संपत्ति उसके मूल्य से कम पर बीमित है, तो बीमाकृत को कमी के लिए अपना स्वयं का बीमाकर्ता माना जाएगा और उसे हानि का आनुपातिक हिस्सा वहन करना होगा।",
        "bad_simp": "कम बीमा दावे की राशि को प्रभावित नहीं करता।",
        "error_types_bad": ["contradiction", "condition"],
        "good_score_range": (8.0, 9.2),
        "bad_score_range": (2.0, 4.0),
    },
    {
        "original": "बीमाकृत कंपनी की लिखित सहमति के बिना किसी भी दावे के संबंध में देयता स्वीकार नहीं करेगा या कोई भुगतान या समझौते का प्रस्ताव नहीं करेगा।",
        "good_simp": "बीमाकृत को कंपनी की लिखित सहमति के बिना किसी दावे के लिए देयता स्वीकार नहीं करनी चाहिए या कोई भुगतान या समझौता प्रस्ताव नहीं करना चाहिए।",
        "bad_simp": "बीमाकृत अपने दम पर दावों का निपटान कर सकता है।",
        "error_types_bad": ["contradiction", "permission", "obligation"],
        "good_score_range": (9.0, 9.7),
        "bad_score_range": (1.5, 3.5),
    },
]


def sample_score(score_range: tuple) -> float:
    """Sample a realistic LMP score within range and round to 1 decimal."""
    return round(random.uniform(*score_range), 1)


def create_pair(
    template: Dict,
    language: str,
    domain: str,
    quality: str = "good",
) -> Dict[str, Any]:
    """Create one annotated pair."""
    if quality == "good":
        simplified = template["good_simp"]
        score = sample_score(template["good_score_range"])
        error_types = []
        explanation = "Legal meaning is largely preserved; minor wording changes do not alter obligations, conditions, or exceptions."
    else:
        simplified = template["bad_simp"]
        score = sample_score(template["bad_score_range"])
        error_types = template["error_types_bad"]
        explanation = (
            f"Legal meaning is not preserved. Key issues: {', '.join(error_types)}. "
            "Important conditions, obligations, or scope have been altered or omitted."
        )

    # Decision thresholds
    if score >= 8.0:
        decision = "Preserved"
    elif score >= 5.0:
        decision = "Partially Preserved"
    else:
        decision = "Not Preserved"

    return {
        "id": str(uuid.uuid4())[:8],
        "language": language,
        "domain": domain,
        "original": template["original"],
        "simplified": simplified,
        "lmp_score": score,
        "decision": decision,
        "error_types": error_types,
        "explanation": explanation,
        "quality": quality,
    }


def generate_dataset(
    n_pairs_per_lang: int = 160,
    output_dir: str = "data",
) -> pd.DataFrame:
    """
    Generate a balanced bilingual dataset.
    Target: ≥300 pairs total (English + Hindi).
    Mix of good and degraded simplifications.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []

    domains = ["property_insurance", "general_insurance", "contract_clauses"]

    for lang, templates in [("en", EN_TEMPLATES), ("hi", HI_TEMPLATES)]:
        for i in range(n_pairs_per_lang):
            template = random.choice(templates)
            domain = random.choice(domains)
            # ~65% good, 35% degraded (realistic distribution)
            quality = "good" if random.random() < 0.65 else "bad"
            pair = create_pair(template, lang, domain, quality)
            records.append(pair)

    # Add explicit identical pairs (sanity check – should be ~10)
    for lang, templates in [("en", EN_TEMPLATES), ("hi", HI_TEMPLATES)]:
        for _ in range(5):
            t = random.choice(templates)
            records.append({
                "id": str(uuid.uuid4())[:8],
                "language": lang,
                "domain": "sanity_identical",
                "original": t["original"],
                "simplified": t["original"],  # identical
                "lmp_score": 10.0,
                "decision": "Preserved",
                "error_types": [],
                "explanation": "Identical sentences; legal meaning is fully preserved by definition.",
                "quality": "identical",
            })

    # Add explicit unrelated pairs (sanity check)
    unrelated_en = [
        ("The premium is payable annually in advance.", "The capital of France is Paris."),
        ("Coverage is limited to the sum insured.", "Water boils at 100 degrees Celsius."),
    ]
    unrelated_hi = [
        ("प्रीमियम अग्रिम रूप से वार्षिक देय है।", "भारत की राजधानी नई दिल्ली है।"),
        ("कवरेज बीमा राशि तक सीमित है।", "पानी 100 डिग्री सेल्सियस पर उबलता है।"),
    ]
    for orig, simp in unrelated_en:
        records.append({
            "id": str(uuid.uuid4())[:8],
            "language": "en",
            "domain": "sanity_unrelated",
            "original": orig,
            "simplified": simp,
            "lmp_score": 1.0,
            "decision": "Not Preserved",
            "error_types": ["unrelated"],
            "explanation": "Sentences are completely unrelated; no legal meaning is preserved.",
            "quality": "unrelated",
        })
    for orig, simp in unrelated_hi:
        records.append({
            "id": str(uuid.uuid4())[:8],
            "language": "hi",
            "domain": "sanity_unrelated",
            "original": orig,
            "simplified": simp,
            "lmp_score": 1.0,
            "decision": "Not Preserved",
            "error_types": ["unrelated"],
            "explanation": "Sentences are completely unrelated; no legal meaning is preserved.",
            "quality": "unrelated",
        })

    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    jsonl_path = Path(output_dir) / "lmp_dataset.jsonl"
    csv_path = Path(output_dir) / "lmp_dataset.csv"
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    df.to_csv(csv_path, index=False)

    print(f"Generated {len(df)} pairs.")
    print(f"  English: {(df.language == 'en').sum()}")
    print(f"  Hindi:   {(df.language == 'hi').sum()}")
    print(f"  Saved to {jsonl_path} and {csv_path}")
    print("\nScore distribution:")
    print(df.lmp_score.describe())
    print("\nDecision distribution:")
    print(df.decision.value_counts())

    return df


if __name__ == "__main__":
    generate_dataset(n_pairs_per_lang=160)
