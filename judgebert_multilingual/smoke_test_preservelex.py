"""
Non-destructive PreserveLex inference smoke test.
Run from the judgebert_multilingual project folder:
    python smoke_test_preservelex.py

Loads the existing checkpoint; does not train or modify project data/model files.
Writes actual predictions to results/demo_smoke_test_results.csv.
"""
from pathlib import Path
import sys
import csv
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import JudgeBERTRegressor, get_tokenizer, prepare_pair
from src.utils import decision_from_score

CKPT = ROOT / "models" / "judgebert_xlmr" / "best_model.pt"
OUT = ROOT / "results" / "demo_smoke_test_results.csv"

CASES = [
    {
        "case": "English identical sanity check",
        "language": "English",
        "original": "The insured shall notify the insurer immediately after any loss or damage.",
        "simplified": "The insured shall notify the insurer immediately after any loss or damage.",
    },
    {
        "case": "English meaning-preserving paraphrase",
        "language": "English",
        "original": "The insured shall notify the insurer immediately after any loss or damage.",
        "simplified": "The insured must inform the insurer immediately after any loss or damage occurs.",
    },
    {
        "case": "English obligation and timing changed",
        "language": "English",
        "original": "The insured shall notify the insurer immediately after any loss or damage.",
        "simplified": "The insured may notify the insurer whenever convenient.",
    },
    {
        "case": "Hindi identical sanity check",
        "language": "Hindi",
        "original": "बीमित व्यक्ति किसी भी हानि या क्षति के तुरंत बाद बीमाकर्ता को सूचित करेगा।",
        "simplified": "बीमित व्यक्ति किसी भी हानि या क्षति के तुरंत बाद बीमाकर्ता को सूचित करेगा।",
    },
    {
        "case": "Hindi obligation and timing changed",
        "language": "Hindi",
        "original": "बीमित व्यक्ति किसी भी हानि या क्षति के तुरंत बाद बीमाकर्ता को सूचित करेगा।",
        "simplified": "बीमित व्यक्ति अपनी सुविधा के अनुसार बीमाकर्ता को सूचित कर सकता है।",
    },
]

def main():
    if not CKPT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    print("Loading checkpoint:", CKPT)
    checkpoint = torch.load(CKPT, map_location="cpu")
    model_name = checkpoint.get("model_name", "xlm-roberta-base")
    args = checkpoint.get("args", {})
    dropout = float(args.get("dropout", 0.1))

    tokenizer = get_tokenizer(model_name)
    model = JudgeBERTRegressor(model_name=model_name, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    rows = []
    print("\nCheckpoint loaded. Running inference (these are actual model outputs):\n")
    for item in CASES:
        encoded = prepare_pair(
            tokenizer, item["original"], item["simplified"], max_length=256
        )
        batch = {k: v.unsqueeze(0) for k, v in encoded.items()}
        with torch.inference_mode():
            score = float(model(**batch)["predictions"].item())
        decision = decision_from_score(score)
        rows.append({
            "case": item["case"],
            "language": item["language"],
            "score": round(score, 4),
            "decision": decision,
            "original": item["original"],
            "simplified": item["simplified"],
        })
        print(f"{item['case']}: {score:.2f}/10 — {decision}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nPASS: checkpoint loaded and all 5 inference cases completed.")
    print("Actual outputs saved to:", OUT)
    print("Reminder: smoke tests check execution only; they do not establish model accuracy.")

if __name__ == "__main__":
    main()
