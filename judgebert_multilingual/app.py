"""
PreserveLex – Professional Web Demo (polished UI)
"""

import hashlib
import hmac
import html as _html
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from src.legal_data import ALL, CATEGORIES, COUNTRIES, REVIEWED
from src.legal_checks import analyze_changes

MODEL = None
TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_TRAINED_MODEL = False
SBERT_MODEL = None
LOAD_ERROR = ""      # why the trained model was not loaded (shown in the UI)
LOADED_PATH = None


# ----------------------------------------------------------------------------
# Input-language validation
# ----------------------------------------------------------------------------
LANGUAGE_CODE_MAP = {"en": "English", "hi": "Hindi", "de": "German", "es": "Spanish", "fr": "French"}

def detect_input_language(text):
    """Return (language label, confidence, error)."""
    text = (text or "").strip()
    if len(text) < 12:
        return None, 0.0, "Please enter a longer sentence so the language can be identified reliably."

    devanagari = sum("\u0900" <= ch <= "\u097f" for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    if letters and devanagari / letters >= 0.20:
        return "Hindi", 1.0, ""

    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 0
        predictions = detect_langs(text)
    except ImportError:
        return None, 0.0, "Language detection dependency is missing. Run: pip install langdetect"
    except Exception as exc:
        return None, 0.0, f"Could not identify the language ({type(exc).__name__}). Please enter more text."

    if not predictions:
        return None, 0.0, "Could not identify the language. Please enter more text."

    best = predictions[0]
    confidence = float(best.prob)
    if confidence < 0.70:
        return None, confidence, "Language detection is uncertain. Please provide a longer, clearer text."

    return LANGUAGE_CODE_MAP.get(best.lang, "Other"), confidence, ""


def validate_language_pair(original, simplified, selected_language):
    """Validate both actual input languages against each other and the selection."""
    original_lang, _, original_error = detect_input_language(original)
    simplified_lang, _, simplified_error = detect_input_language(simplified)

    if original_error:
        return original_error
    if simplified_error:
        return simplified_error
    if original_lang != simplified_lang:
        return (f"The two texts appear to use different languages "
                f"({original_lang} and {simplified_lang}). Please select matching texts.")

    selected_language = selected_language or "English"
    if selected_language != original_lang:
        return (f"Language mismatch: you selected {selected_language}, but both text fields "
                f"appear to be {original_lang}. Change the language selection or correct the text.")
    return ""


# ----------------------------------------------------------------------------
# Model loading / prediction (unchanged logic)
# ----------------------------------------------------------------------------
def try_load_trained_model():
    global MODEL, TOKENIZER, USE_TRAINED_MODEL, LOAD_ERROR, LOADED_PATH

    ckpt_paths = [
        Path("models/judgebert_xlmr/best_model.pt"),
    ]

    try:
        from src.model import JudgeBERTRegressor, get_tokenizer

        ckpt_path = next((p for p in ckpt_paths if p.is_file()), None)
        if ckpt_path is None:
            LOAD_ERROR = (
                "Checkpoint not found. Expected "
                "models/judgebert_xlmr/best_model.pt"
            )
            return False

        try:
            ckpt = torch.load(
                ckpt_path, map_location=DEVICE, weights_only=True
            )
        except Exception:
            ckpt = torch.load(
                ckpt_path, map_location=DEVICE, weights_only=False
            )

        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(
                "Checkpoint does not contain a 'model_state_dict' entry."
            )

        model_name = ckpt.get("model_name", "xlm-roberta-base")
        TOKENIZER = get_tokenizer(model_name)
        MODEL = JudgeBERTRegressor(model_name=model_name).to(DEVICE)
        MODEL.load_state_dict(ckpt["model_state_dict"], strict=True)
        MODEL.eval()

        USE_TRAINED_MODEL = True
        LOADED_PATH = str(ckpt_path)
        LOAD_ERROR = ""
        print(f"[PreserveLex] Loaded trained model from {ckpt_path}")
        return True

    except Exception as e:
        MODEL = None
        TOKENIZER = None
        USE_TRAINED_MODEL = False
        LOADED_PATH = None
        LOAD_ERROR = f"{type(e).__name__}: {e}"
        print(f"[PreserveLex] Trained model loading failed: {LOAD_ERROR}")
        return False


def model_status_html():
    if USE_TRAINED_MODEL:
        return ('<div class="plx-model plx-model-ok"><span class="plx-dot"></span>'
                f'Model: fine-tuned XLM-RoBERTa ({_html.escape(Path(LOADED_PATH).name)})</div>')
    reason = _html.escape(LOAD_ERROR or "trained model not loaded")
    return ('<div class="plx-model plx-model-warn"><span class="plx-dot"></span>'
            f'Fallback: SBERT similarity only. Trained model not loaded: {reason}</div>')


def load_sbert():
    global SBERT_MODEL
    if SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        SBERT_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return SBERT_MODEL


def predict_sbert(original, simplified):
    model = load_sbert()
    emb = model.encode([original, simplified], convert_to_numpy=True)
    cos = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-8))
    score = 1.0 + 9.0 * max(0.0, min(1.0, (cos - 0.3) / 0.7))
    return round(score, 2)


def predict_trained(original, simplified):
    from src.model import prepare_pair
    enc = prepare_pair(TOKENIZER, original, simplified, max_length=256)
    input_ids = enc["input_ids"].unsqueeze(0).to(DEVICE)
    attention_mask = enc["attention_mask"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = MODEL(input_ids=input_ids, attention_mask=attention_mask)
        score = float(out["predictions"].cpu().item())
    return round(score, 2)


# ----------------------------------------------------------------------------
# Authentication (accounts stored as salted PBKDF2 hashes in users.json)
# ----------------------------------------------------------------------------
USERS_FILE = Path(os.environ.get("PLX_USERS_FILE", "users.json"))
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD = 8
MAX_FAILS = 5        # failed sign-ins allowed ...
LOCK_SECONDS = 60    # ... before the username is locked for this long

_USERS_LOCK = threading.Lock()
_FAILS = {}          # username -> {"count": int, "until": float}


def _hash_pw(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex(), digest.hex()


def _load_users():
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_users(users):
    tmp = USERS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, indent=2), encoding="utf-8")
    tmp.replace(USERS_FILE)


def _add_user(username, password):
    salt, digest = _hash_pw(password)
    users = _load_users()
    users[username.lower()] = {"name": username, "salt": salt, "hash": digest}
    _save_users(users)


def init_users():
    """Create a first account when no users file exists yet."""
    with _USERS_LOCK:
        if USERS_FILE.exists():
            return
        user = os.environ.get("PLX_ADMIN_USER", "demo")
        pw = os.environ.get("PLX_ADMIN_PASS", "preservelex")
        _add_user(user, pw)
        print(f"[PreserveLex] Created {USERS_FILE} with account '{user}'. "
              f"Change the password or set PLX_ADMIN_USER / PLX_ADMIN_PASS.")


def verify_login(username, password):
    """Return (ok, error_message)."""
    key = (username or "").strip().lower()
    now = time.time()
    fail = _FAILS.get(key)
    if fail and fail["until"] > now:
        return False, f"Too many attempts. Try again in {int(fail['until'] - now) + 1} seconds."

    user = _load_users().get(key)
    # Always run the hash so response time does not reveal whether a user exists.
    salt = bytes.fromhex(user["salt"]) if user else b"\x00" * 16
    _, digest = _hash_pw(password or "", salt)
    ok = bool(user) and hmac.compare_digest(digest, user["hash"])

    if ok:
        _FAILS.pop(key, None)
        return True, user["name"]

    entry = _FAILS.setdefault(key, {"count": 0, "until": 0.0})
    entry["count"] += 1
    if entry["count"] >= MAX_FAILS:
        entry.update(count=0, until=now + LOCK_SECONDS)
    return False, "Incorrect username or password."


def _msg(text, kind="err"):
    role = "alert" if kind == "err" else "status"
    return f'<div class="plx-msg plx-msg-{kind}" role="{role}">{_html.escape(text)}</div>'


def _user_badge(name):
    initial = _html.escape(name[:1].upper())
    return (f'<div class="plx-user"><span class="plx-avatar">{initial}</span>'
            f'<span>Signed in as <b>{_html.escape(name)}</b></span></div>')


def _enter(name, msg_html):
    """Outputs: session, login_view, app_view, message, user badge."""
    return ({"user": name}, gr.update(visible=False), gr.update(visible=True),
            msg_html, _user_badge(name))


def _stay(msg_html):
    return ({"user": None}, gr.update(visible=True), gr.update(visible=False), msg_html, "")


def do_login(username, password):
    ok, info = verify_login(username, password)
    if not ok:
        return _stay(_msg(info)) + ("",)          # also clears the password box
    return _enter(info, "") + ("",)


def do_signup(username, password, confirm):
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        return _stay(_msg("Username must be 3-32 characters: letters, numbers, . _ -")) + (password, confirm)
    if len(password or "") < MIN_PASSWORD:
        return _stay(_msg(f"Password must be at least {MIN_PASSWORD} characters.")) + (password, confirm)
    if password != confirm:
        return _stay(_msg("The two passwords do not match.")) + (password, confirm)
    with _USERS_LOCK:
        if username.lower() in _load_users():
            return _stay(_msg("That username is taken. Choose another.")) + (password, confirm)
        _add_user(username, password)
    return _enter(username, "") + ("", "")


def do_logout():
    """Outputs: session, login_view, app_view, login message, result, original, simplified, badge."""
    return ({"user": None}, gr.update(visible=True), gr.update(visible=False),
            _msg("You have been signed out.", "ok"), EMPTY_HTML, "", "", "")


# ----------------------------------------------------------------------------
# Result cards (HTML). All styling lives in CUSTOM_CSS via .plx-* classes.
# ----------------------------------------------------------------------------
EMPTY_HTML = """
<div class="plx-card plx-empty">
    <svg viewBox="0 0 24 24" width="38" height="38" fill="none" stroke="currentColor" stroke-width="1.5"
         stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M12 3v18M5 7h14M5 7l-2.5 6a3 3 0 0 0 5 0L5 7zM19 7l-2.5 6a3 3 0 0 0 5 0L19 7z"/>
    </svg>
    <p class="plx-empty-title">No evaluation yet</p>
    <p class="plx-empty-sub">Paste a clause and its simplified version, then choose Evaluate.</p>
</div>
"""

LOADING_HTML = """
<div class="plx-card plx-loading" aria-live="polite">
    <div class="plx-skel plx-skel-ring"></div>
    <div class="plx-skel-col">
        <div class="plx-skel plx-skel-line" style="width:70%"></div>
        <div class="plx-skel plx-skel-line" style="width:45%"></div>
        <div class="plx-skel plx-skel-line" style="width:85%"></div>
    </div>
    <p class="plx-loading-text">Comparing meaning<span class="plx-dots"><i>.</i><i>.</i><i>.</i></span></p>
</div>
"""

WARN_HTML = """
<div class="plx-card plx-empty">
    <p class="plx-empty-title">Both fields are required</p>
    <p class="plx-empty-sub">Enter the original text and the simplified text, then evaluate again.</p>
</div>
"""

CIRC = 339.29  # circumference of the gauge ring (r = 54)


def findings_html(findings, score):
    esc = _html.escape
    if findings:
        kinds = []
        for f in findings:
            if f["kind"] not in [k for k, _ in kinds]:
                kinds.append((f["kind"], f["severity"]))
        chips = "".join(f'<span class="plx-flag plx-sev-{sev}">{esc(k)}</span>' for k, sev in kinds)
        notes = "".join(f"<li>{esc(f['detail'])}</li>" for f in findings[:4])
        review = ""
        if score >= 8.0 and any(f["severity"] == "high" for f in findings):
            review = ('<p class="plx-review">The model scored this as preserved, but the checks found a high-risk change. '
                      'Read the flagged parts before trusting the score.</p>')
        body = f'<div class="plx-flags">{chips}</div><ul class="plx-notes">{notes}</ul>{review}'
    else:
        body = ('<div class="plx-flags"><span class="plx-flag plx-sev-none">None detected</span></div>'
                '<ul class="plx-notes"><li>No change to negation, obligations, numbers, conditions, exceptions or scope was found.</li></ul>')
    return ('<div class="plx-find"><span class="plx-find-title">Legal error type and what changed</span>' + body +
            '<p class="plx-fine">Rule-based check for English, German, Spanish and Hindi. It can miss or over-flag changes.</p></div>')


def result_html(score, decision, tone, blurb, language, method, findings=None):
    whole, frac = divmod(round(score * 10), 10)
    if score >= 10:
        whole, frac = 10, 0
    offset = CIRC * (1 - max(0.0, min(10.0, score)) / 10)
    pct = max(0.0, min(100.0, score * 10))
    language = _html.escape(str(language))
    method = _html.escape(method)
    find_block = findings_html(findings or [], score)
    return f"""
<div class="plx-card plx-result plx-{tone}" role="status">
    <div class="plx-top">
        <div class="plx-gauge" style="--off:{offset:.2f}">
            <svg viewBox="0 0 120 120" aria-hidden="true">
                <circle class="plx-ring-bg" cx="60" cy="60" r="54"/>
                <circle class="plx-ring" cx="60" cy="60" r="54" transform="rotate(-90 60 60)"/>
            </svg>
            <div class="plx-score" aria-label="Score {score:.1f} out of 10">
                <span class="plx-num" style="--i:{whole};--d:{frac}"></span>
                <span class="plx-out">out of 10</span>
            </div>
        </div>
        <div class="plx-verdict">
            <span class="plx-badge">{decision}</span>
            <p class="plx-blurb">{blurb}</p>
        </div>
    </div>

    <div class="plx-meter" style="--w:{pct:.1f}%">
        <div class="plx-track"><div class="plx-fill"></div></div>
        <div class="plx-ticks">
            <span style="left:50%">5</span>
            <span style="left:80%">8</span>
        </div>
    </div>

    {find_block}

    <div class="plx-meta">
        <div class="plx-tile"><span>Language</span><b>{language}</b></div>
        <div class="plx-tile"><span>Model</span><b>{method}</b></div>
    </div>
</div>
"""


LOCKED_HTML = """
<div class="plx-card plx-empty">
    <p class="plx-empty-title">Sign in required</p>
    <p class="plx-empty-sub">Your session has ended. Reload the page and sign in again.</p>
</div>
"""


def predict_lmp(original, simplified, language, session):
    # Authentication
    if not session or not session.get("user"):
        yield LOCKED_HTML, None, None
        return

    # Required fields
    if not original or not original.strip() or not simplified or not simplified.strip():
        yield WARN_HTML, None, None
        return

    # Validate both actual input languages BEFORE running either model.
    language_error = validate_language_pair(original, simplified, language)
    if language_error:
        error_html = (
            '<div class="plx-card plx-empty">'
            '<p class="plx-empty-title">Language validation failed</p>'
            f'<p class="plx-empty-sub">{_html.escape(language_error)}</p>'
            '</div>'
        )
        yield error_html, None, None
        return

    yield LOADING_HTML, None, None
    started = time.time()

    if USE_TRAINED_MODEL:
        score = predict_trained(original, simplified)
        method = "Fine-tuned XLM-RoBERTa"
    else:
        score = predict_sbert(original, simplified)
        method = "Multilingual SBERT (fallback)"

    elapsed = time.time() - started
    if elapsed < 0.6:
        time.sleep(0.6 - elapsed)

    if score >= 8.0:
        decision, tone = "Preserved", "ok"
        blurb = "The simplified text stays close to the meaning of the original."
    elif score >= 5.0:
        decision, tone = "Partially preserved", "mid"
        blurb = "Some meaning is kept. Check conditions, exceptions and scope before using it."
    else:
        decision, tone = "Not preserved", "bad"
        blurb = "The simplified text likely changes or drops legal meaning. Review it before use."

    findings = analyze_changes(original, simplified)
    yield result_html(score, decision, tone, blurb, language, method, findings), score, decision


def clear_all():
    return "", "", "English", EMPTY_HTML, None, None



def clear_result():
    """Clear stale evaluation output when an evaluation input changes."""
    return EMPTY_HTML, None, None


# ----------------------------------------------------------------------------
# Legal library (data lives in legal_data.py)
# ----------------------------------------------------------------------------
ALL_CATEGORIES = "All categories"
COUNTRY_NAMES = [c["name"] for c in COUNTRIES]


def _slug(text):
    return re.sub(r"[^a-z]+", "-", text.lower()).strip("-")


def _rule_card(rule, idx, country=None):
    esc = _html.escape
    tag = f'<span class="plx-where">{esc(country)}</span>' if country else ""
    year = f'<span class="plx-year">{esc(rule["year"])}</span>' if rule["year"] not in ("", "-") else ""
    source_url = rule.get("source_url", "")
    source_name = rule.get("source", "Official source")
    source = (f'<a class="plx-link" href="{esc(source_url)}" target="_blank" rel="noopener noreferrer">'
              f'Source: {esc(source_name)} ↗</a>') if source_url else '<span class="plx-year">Source link pending verification</span>'
    return (
        f'<article class="plx-rule" style="--n:{idx}">'
        f'<div class="plx-rule-top"><span class="plx-cat plx-cat-{_slug(rule["cat"])}">{esc(rule["cat"])}</span>{tag}{year}</div>'
        f'<h4>{esc(rule["title"])}</h4><p>{esc(rule["summary"])}</p><div class="plx-rule-source">{source}</div></article>'
    )


def render_library(country, query, category):
    q = (query or "").strip().lower()
    cat = category or ALL_CATEGORIES
    esc = _html.escape
    parts, total, idx = [], 0, 0

    for c in COUNTRIES:
        if country not in (ALL, c["name"]):
            continue
        rules = [
            r for r in c["rules"]
            if (cat == ALL_CATEGORIES or r["cat"] == cat)
            and (not q or q in f'{r["title"]} {r["summary"]} {r["cat"]}'.lower())
        ]
        if not rules:
            continue
        rules.sort(key=lambda r: CATEGORIES.index(r["cat"]))
        total += len(rules)
        if country == ALL:
            parts.append(f'<h3 class="plx-country">{esc(c["name"])}</h3>')
        else:
            parts.append(
                '<div class="plx-cinfo">'
                f'<div><span>Legal system</span><p>{esc(c["system"])}</p></div>'
                f'<div><span>Key regulators</span><p>{esc(c["regulators"])}</p></div>'
                f'<a class="plx-link" href="{esc(c["portal"])}" target="_blank" rel="noopener noreferrer">'
                f'Official source: {esc(c["portal_label"])}</a></div>'
            )
        cards = []
        for r in rules:
            cards.append(_rule_card(r, idx))
            idx += 1
        parts.append('<div class="plx-rules">' + "".join(cards) + "</div>")

    if not total:
        return ('<div class="plx-card plx-empty"><p class="plx-empty-title">No matching laws</p>'
                '<p class="plx-empty-sub">Try a shorter search or set the category to All categories.</p></div>')

    count = f'<p class="plx-count">{total} {"law" if total == 1 else "laws"} shown</p>'
    note = (f'<p class="plx-disclaimer">Summaries for general information, not legal advice. This is a curated starter collection, '
            f'not a complete or exhaustive statement of the law. Each card links to its identified official source where available. '
            f'Last reviewed {esc(REVIEWED)}. Verify amendments, commencement, rules and jurisdiction on the official source before relying on it.</p>')
    return count + "".join(parts) + note


# ----------------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------------
EXAMPLES = [
    # Preserved examples — English
    [
        "The tenant shall pay the monthly rent to the landlord on or before the fifth day of each calendar month.",
        "The tenant must pay the landlord the monthly rent by the fifth day of every month.",
        "English",
    ],
    [
        "The buyer shall remit the outstanding purchase amount within thirty days from the date of receipt of the invoice.",
        "The buyer must pay the remaining purchase amount within 30 days after receiving the invoice.",
        "English",
    ],
    [
        "The receiving party shall not disclose any confidential information to third parties without obtaining prior written consent from the disclosing party.",
        "The receiving party must not share confidential information with others unless the disclosing party gives written permission first.",
        "English",
    ],
    [
        "Either party may terminate this agreement by providing thirty days' written notice to the other party.",
        "Either party can end this agreement by giving the other party 30 days' written notice.",
        "English",
    ],
    [
        "The consumer shall be entitled to request a refund within fourteen days of receiving the defective product, subject to the applicable terms and conditions.",
        "The consumer may request a refund within 14 days after receiving the defective product, subject to the same terms and conditions.",
        "English",
    ],
    # English: meaning kept
    [
        "The insurer shall not be liable for any loss or damage caused by or resulting from war, invasion, act of foreign enemy, hostilities, civil war, rebellion or military power.",
        "The insurer is not responsible for loss or damage caused by war, invasion, foreign enemy actions, hostilities, civil war, rebellion or military power.",
        "English",
    ],
    # English: condition dropped and meaning reversed
    [
        "No claim shall be payable under this Policy unless the premium has been paid in full and the Policy has been issued by the Company.",
        "Claims are paid even if the premium is not fully paid.",
        "English",
    ],
    # English: obligation became permission, 14 days became 30
    [
        "The tenant must pay the rent within 14 days of receiving the invoice.",
        "The tenant can pay the rent within 30 days.",
        "English",
    ],
    # Preserved examples — Hindi
    [
        "किरायेदार प्रत्येक महीने की पाँच तारीख तक मकान मालिक को मासिक किराया देगा।",
        "किरायेदार को हर महीने की पाँच तारीख तक मकान मालिक को किराया देना होगा।",
        "Hindi",
    ],
    [
        "खरीदार चालान प्राप्त होने की तारीख से तीस दिनों के भीतर शेष खरीद राशि का भुगतान करेगा।",
        "खरीदार को चालान मिलने के 30 दिनों के अंदर बाकी खरीद राशि चुकानी होगी।",
        "Hindi",
    ],
    # Hindi: meaning kept
    [
        "बीमाकर्ता किसी भी युद्ध, आक्रमण, गृहयुद्ध या सैन्य शक्ति के कारण हुई किसी भी हानि या क्षति के लिए उत्तरदायी नहीं होगा।",
        "युद्ध, आक्रमण, गृहयुद्ध या सैन्य शक्ति से हुए नुकसान के लिए बीमाकर्ता जिम्मेदार नहीं है।",
        "Hindi",
    ],
    # Hindi: condition dropped and meaning reversed
    [
        "पॉलिसी के तहत कोई भी दावा तभी देय होगा जब प्रीमियम का पूरा भुगतान कर दिया गया हो।",
        "प्रीमियम पूरा न चुकाने पर भी दावा दिया जाएगा।",
        "Hindi",
    ],
    # German
    [
        "Der Versicherer haftet nicht für Verluste oder Schäden, die durch Krieg, Invasion, Bürgerkrieg oder militärische Macht verursacht werden.",
        "Der Versicherer ist nicht verantwortlich für Verluste durch Krieg, Invasion, Bürgerkrieg oder militärische Macht.",
        "German",
    ],
    # Spanish
    [
        "El asegurador no será responsable de ninguna pérdida o daño causado por guerra, invasión, hostilidades o poder militar.",
        "El asegurador no es responsable de la pérdida causada por guerra, invasión o poder militar.",
        "Spanish",
    ],
    # Sanity check 1: identical pair, expect a score near 10
    [
        "The insurer shall not be liable for any loss or damage caused by war.",
        "The insurer shall not be liable for any loss or damage caused by war.",
        "English",
    ],
    # Sanity check 2: unrelated pair, expect a score near 1
    [
        "The insurer shall not be liable for any loss or damage caused by war, invasion or military power.",
        "The weather in Delhi is usually hot and humid during June.",
        "English",
    ],
]


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

:root {
    --ink: #0f1b2d;
    --ink-2: #1b2b45;
    --paper: #eef1f6;
    --surface: #ffffff;
    --line: #d5dbe6;
    --muted: #5b667a;
    --seal: #2f4bff;
    --ok: #28c27a;
    --mid: #e8a912;
    --bad: #ef5350;
    --ease: cubic-bezier(.2, .7, .2, 1);
}

@property --i { syntax: '<integer>'; inherits: false; initial-value: 0; }
@property --d { syntax: '<integer>'; inherits: false; initial-value: 0; }

body, .gradio-container { background: var(--paper) !important; }
.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
footer { display: none !important; }


/* ---------- PreserveLex side guidance panels ---------- */
.plx-workspace { align-items: flex-start !important; gap: 1.1rem !important; }
.plx-side-card {
    background: rgba(255,255,255,.82);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1rem;
    color: var(--ink);
    box-shadow: 0 8px 24px rgba(25, 42, 80, .045);
}
.plx-side-card h3 {
    margin: 0 0 .7rem;
    color: var(--ink);
    font-size: .98rem;
    font-weight: 750;
}
.plx-side-card p, .plx-side-card li {
    color: var(--muted);
    font-size: .79rem;
    line-height: 1.5;
}
.plx-side-card ul { padding-left: 1.05rem; margin: .45rem 0 .8rem; }
.plx-side-card li { margin: .35rem 0; }
.plx-side-item {
    padding: .55rem .6rem;
    border-radius: 10px;
    background: #f4f6fb;
    margin: .45rem 0;
}
.plx-side-item strong { display: block; font-size: .8rem; color: var(--ink); }
.plx-side-item span { display: block; color: var(--muted); font-size: .72rem; line-height: 1.4; margin-top: .15rem; }
.plx-side-note {
    border-left: 3px solid #7c83ff;
    padding: .55rem .65rem;
    background: #f3f4ff;
    border-radius: 0 9px 9px 0;
    color: #4d5680 !important;
    font-size: .72rem !important;
}
@media (max-width: 1050px) {
    .plx-workspace { flex-wrap: wrap !important; }
    .plx-workspace > div { min-width: 100% !important; }
}

/* ---------- Header: the one page-load moment ---------- */
.plx-header { text-align: center; padding: 2rem 0 1.2rem; }
.plx-title {
    margin: 0; font-family: 'Newsreader', Georgia, serif; font-weight: 600;
    font-size: clamp(2.4rem, 6vw, 3.4rem); letter-spacing: -0.02em; color: var(--ink);
    animation: plx-rise .8s var(--ease) both;
}
.plx-sub {
    margin: .35rem 0 0; font-size: 1.05rem; color: var(--muted);
    animation: plx-rise .8s .12s var(--ease) both;
}
.plx-chips { display: flex; gap: .5rem; justify-content: center; margin-top: 1rem; flex-wrap: wrap; }
.plx-chip {
    padding: .3rem .8rem; border-radius: 999px; font-size: .82rem; color: var(--ink-2);
    background: var(--surface); border: 1px solid var(--line);
    animation: plx-rise .7s var(--ease) both;
    transition: transform .2s var(--ease), border-color .2s, box-shadow .2s;
}
.plx-chip:nth-child(1) { animation-delay: .22s; }
.plx-chip:nth-child(2) { animation-delay: .29s; }
.plx-chip:nth-child(3) { animation-delay: .36s; }
.plx-chip:nth-child(4) { animation-delay: .43s; }
.plx-chip:hover { transform: translateY(-2px); border-color: var(--seal); box-shadow: 0 6px 14px -8px var(--seal); }

@keyframes plx-rise { from { opacity: 0; transform: translateY(14px); } }

/* ---------- Inputs ---------- */
.plx-input textarea, .plx-input input {
    border-radius: 12px !important;
    border: 1.5px solid var(--line) !important;
    background: var(--surface) !important;
    transition: border-color .2s, box-shadow .25s var(--ease), transform .25s var(--ease) !important;
}
.plx-input textarea:hover { border-color: #b4bfd3 !important; }
.plx-input textarea:focus, .plx-input input:focus {
    border-color: var(--seal) !important;
    box-shadow: 0 0 0 4px rgba(47, 75, 255, .14) !important;
    outline: none !important;
}
.plx-input label span { font-weight: 600 !important; color: var(--ink-2) !important; }

/* ---------- Buttons: press + ripple ---------- */
button.plx-btn {
    position: relative; overflow: hidden; border-radius: 12px !important; font-weight: 600 !important;
    transition: transform .15s var(--ease), box-shadow .25s var(--ease), background .25s, filter .25s !important;
}
button.plx-primary {
    background: linear-gradient(135deg, var(--seal), #5a3bff) !important; color: #fff !important; border: none !important;
    box-shadow: 0 10px 22px -10px rgba(47, 75, 255, .8) !important;
}
button.plx-primary:hover { transform: translateY(-2px); box-shadow: 0 16px 28px -12px rgba(47, 75, 255, .9) !important; filter: brightness(1.06); }
button.plx-ghost { background: var(--surface) !important; color: var(--ink-2) !important; border: 1.5px solid var(--line) !important; }
button.plx-ghost:hover { border-color: var(--ink-2) !important; transform: translateY(-1px); }
button.plx-btn:active { transform: translateY(0) scale(.96) !important; box-shadow: none !important; }
button.plx-btn:focus-visible { outline: 3px solid rgba(47, 75, 255, .4); outline-offset: 2px; }

.plx-ripple {
    position: absolute; border-radius: 50%; pointer-events: none; transform: scale(0);
    background: rgba(255, 255, 255, .55); animation: plx-ripple .65s ease-out forwards;
}
button.plx-ghost .plx-ripple { background: rgba(47, 75, 255, .2); }
@keyframes plx-ripple { to { transform: scale(1); opacity: 0; } }

/* Example rows react to clicks too */
.gradio-container table tbody tr { transition: background .2s, transform .15s var(--ease); cursor: pointer; }
.gradio-container table tbody tr:hover { background: rgba(47, 75, 255, .06) !important; }
.gradio-container table tbody tr:active { transform: scale(.995); background: rgba(47, 75, 255, .12) !important; }

/* ---------- Result cards ---------- */
.plx-card {
    font-family: inherit; border-radius: 20px; padding: 1.6rem 1.7rem; color: #f1f5f9;
    background: linear-gradient(165deg, #0f1b2d 0%, #1b2b45 100%);
    border: 1px solid #2c3d5c; box-shadow: 0 24px 48px -24px rgba(15, 27, 45, .6);
    animation: plx-pop .55s var(--ease) both;
}
@keyframes plx-pop { from { opacity: 0; transform: translateY(16px) scale(.97); } }

.plx-empty {
    min-height: 260px; display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; gap: .35rem; background: transparent; color: var(--muted);
    border: 1.5px dashed #b4bfd3; box-shadow: none;
}
.plx-empty-title { margin: .5rem 0 0; font-weight: 600; color: var(--ink-2); font-size: 1.02rem; }
.plx-empty-sub { margin: 0; font-size: .9rem; max-width: 24ch; }

/* loading */
.plx-loading { display: grid; grid-template-columns: 112px 1fr; gap: 1.2rem; align-items: center; min-height: 260px; }
.plx-skel {
    border-radius: 12px;
    background: linear-gradient(90deg, #1b2b45 25%, #2e4266 50%, #1b2b45 75%); background-size: 200% 100%;
    animation: plx-shimmer 1.3s linear infinite;
}
.plx-skel-ring { width: 104px; height: 104px; border-radius: 50%; }
.plx-skel-col { display: grid; gap: .7rem; }
.plx-skel-line { height: 14px; }
.plx-loading-text { grid-column: 1 / -1; margin: 0; text-align: center; color: #94a3b8; font-size: .92rem; }
.plx-dots i { font-style: normal; animation: plx-blink 1.2s infinite both; }
.plx-dots i:nth-child(2) { animation-delay: .2s; }
.plx-dots i:nth-child(3) { animation-delay: .4s; }
@keyframes plx-shimmer { to { background-position: -200% 0; } }
@keyframes plx-blink { 0%, 80%, 100% { opacity: .2; } 40% { opacity: 1; } }

/* result tones */
.plx-ok  { --accent: var(--ok); }
.plx-mid { --accent: var(--mid); }
.plx-bad { --accent: var(--bad); }
.plx-result { border-color: color-mix(in srgb, var(--accent) 35%, #2c3d5c); }

.plx-top { display: grid; grid-template-columns: 132px 1fr; gap: 1.3rem; align-items: center; margin-bottom: 1.5rem; }
.plx-gauge { position: relative; width: 132px; height: 132px; }
.plx-gauge svg { width: 100%; height: 100%; }
.plx-ring-bg { fill: none; stroke: #263756; stroke-width: 9; }
.plx-ring {
    fill: none; stroke: var(--accent); stroke-width: 9; stroke-linecap: round;
    stroke-dasharray: 339.29; stroke-dashoffset: var(--off);
    filter: drop-shadow(0 0 6px color-mix(in srgb, var(--accent) 60%, transparent));
    animation: plx-ring 1.3s .1s var(--ease) both;
}
@keyframes plx-ring { from { stroke-dashoffset: 339.29; } }

.plx-score { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.plx-num {
    font-family: 'Newsreader', Georgia, serif; font-size: 2.5rem; font-weight: 600; line-height: 1; color: var(--accent);
    animation: plx-count-i 1.3s .1s var(--ease) both, plx-count-d 1.3s .1s linear both;
    counter-reset: i var(--i) d var(--d);
}
.plx-num::before { content: counter(i) "." counter(d); }
.plx-out { margin-top: .25rem; font-size: .72rem; color: #94a3b8; }
@keyframes plx-count-i { from { --i: 0; } }
@keyframes plx-count-d { from { --d: 0; } }

.plx-badge {
    display: inline-block; padding: .4rem .95rem; border-radius: 999px; font-size: .86rem; font-weight: 600;
    color: var(--accent); background: color-mix(in srgb, var(--accent) 16%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent);
    animation: plx-badge .6s .9s var(--ease) both;
}
@keyframes plx-badge { from { opacity: 0; transform: scale(.7); } 60% { transform: scale(1.08); } }
.plx-blurb { margin: .8rem 0 0; color: #cbd5e1; font-size: .95rem; line-height: 1.5; max-width: 44ch; animation: plx-rise .6s 1s var(--ease) both; }

/* meter with thresholds */
.plx-meter { margin-bottom: 1.5rem; }
.plx-track { height: 8px; border-radius: 99px; background: #263756; overflow: hidden; }
.plx-fill {
    height: 100%; width: var(--w); border-radius: 99px;
    background: linear-gradient(90deg, color-mix(in srgb, var(--accent) 70%, transparent), var(--accent));
    animation: plx-fill 1.3s .1s var(--ease) both;
}
@keyframes plx-fill { from { width: 0; } }
.plx-ticks { position: relative; height: 1rem; margin-top: .3rem; }
.plx-ticks span { position: absolute; transform: translateX(-50%); font-size: .7rem; color: #64748b; }
.plx-ticks span::before { content: ""; display: block; width: 1px; height: 5px; margin: 0 auto 2px; background: #475a7c; }

.plx-meta { display: grid; grid-template-columns: 1fr 1fr; gap: .9rem; }
.plx-tile {
    background: #0f1b2d; border: 1px solid #263756; border-radius: 12px; padding: .8rem 1rem;
    animation: plx-rise .6s 1.1s var(--ease) both;
    transition: border-color .2s, transform .2s var(--ease);
}
.plx-tile:hover { border-color: var(--accent); transform: translateY(-2px); }
.plx-tile span { display: block; font-size: .74rem; color: #7c8aa5; margin-bottom: .2rem; }
.plx-tile b { font-weight: 500; font-size: .92rem; color: #e2e8f0; }

/* ---------- Auth ---------- */
.plx-auth {
    max-width: 440px; margin: 0 auto !important; padding: 1.4rem 1.5rem 1.6rem !important;
    background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 20px !important;
    box-shadow: 0 24px 48px -28px rgba(15, 27, 45, .35);
    animation: plx-pop .55s var(--ease) both;
}
.plx-app { animation: plx-fade .5s var(--ease) both; }
@keyframes plx-fade { from { opacity: 0; transform: translateY(10px); } }

.plx-msg { margin: .5rem 0 0; padding: .65rem .9rem; border-radius: 10px; font-size: .9rem; }
.plx-msg-err { background: #fdecec; color: #b42318; border: 1px solid #f5c2c0; animation: plx-shake .45s; }
.plx-msg-ok  { background: #e8f8ef; color: #17734a; border: 1px solid #b7e6cd; animation: plx-rise .4s var(--ease) both; }
@keyframes plx-shake {
    0%, 100% { transform: translateX(0); } 20% { transform: translateX(-8px); }
    40% { transform: translateX(8px); } 60% { transform: translateX(-5px); } 80% { transform: translateX(5px); }
}

.plx-user {
    display: inline-flex; align-items: center; gap: .6rem; padding: .35rem .9rem .35rem .4rem;
    background: var(--surface); border: 1px solid var(--line); border-radius: 999px;
    font-size: .88rem; color: var(--ink-2); animation: plx-rise .5s var(--ease) both;
}
.plx-avatar {
    width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center;
    background: linear-gradient(135deg, var(--seal), #5a3bff); color: #fff; font-weight: 600; font-size: .85rem;
}

/* ---------- Side rails (fill the empty margins on wide screens) ---------- */
:root {
    --glyphs: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='150' height='150'%3E%3Cg fill='%232f4bff' fill-opacity='.12' font-family='Georgia,serif'%3E%3Ctext x='14' y='46' font-size='36'%3E%C2%A7%3C/text%3E%3Ctext x='92' y='70' font-size='22'%3E%C2%B6%3C/text%3E%3Ctext x='40' y='124' font-size='28'%3E%C2%A7%3C/text%3E%3Ctext x='112' y='136' font-size='16'%3E%C2%A9%3C/text%3E%3C/g%3E%3C/svg%3E");
}
body::before, body::after {
    content: ""; display: none; position: fixed; top: 0; bottom: 0; z-index: 0; pointer-events: none;
    width: calc((100vw - 1000px) / 2 - 14px);
    background-image: var(--glyphs); background-size: 150px 150px;
    -webkit-mask-image: linear-gradient(to bottom, transparent, #000 14%, #000 86%, transparent);
    mask-image: linear-gradient(to bottom, transparent, #000 14%, #000 86%, transparent);
    animation: plx-drift 90s linear infinite;
}
body::before { left: 0; border-right: 1px solid; border-image: linear-gradient(to bottom, transparent, rgba(47, 75, 255, .35), transparent) 1; }
body::after  { right: 0; border-left: 1px solid;  border-image: linear-gradient(to bottom, transparent, rgba(47, 75, 255, .35), transparent) 1; animation-direction: reverse; }
@keyframes plx-drift { to { background-position: 0 600px; } }
@media (min-width: 1200px) { body::before, body::after { display: block; } }

/* ---------- Tabs ---------- */
.plx-tabs button[role="tab"] { font-weight: 600 !important; transition: color .2s, background .2s; }
.plx-tabs .tab-wrapper, .plx-tabs [role="tablist"] { border-color: var(--line) !important; }
.plx-tabs [role="tabpanel"], .plx-tabs .tabitem { animation: plx-fade .4s var(--ease) both; }

/* ---------- Legal library ---------- */
.plx-rule-source {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    width: 100%;
    min-width: 0;
    margin-top: auto;
    padding-top: .9rem;
    font-size: .82rem;
}
.plx-rule-source a { color: var(--seal); text-decoration: none; font-weight: 650; }

.plx-pills .wrap { gap: .5rem !important; }
.plx-pills label {
    border-radius: 999px !important; padding: .4rem 1rem !important; background: var(--surface) !important;
    border: 1.5px solid var(--line) !important; cursor: pointer;
    transition: transform .15s var(--ease), background .2s, border-color .2s, color .2s, box-shadow .2s !important;
}
.plx-pills label:hover { border-color: var(--seal) !important; transform: translateY(-1px); }
.plx-pills label:active { transform: scale(.95); }
.plx-pills label.selected, .plx-pills label:has(input:checked) {
    background: var(--ink) !important; border-color: var(--ink) !important; color: #fff !important;
    box-shadow: 0 8px 16px -10px rgba(15, 27, 45, .8);
}
.plx-pills input[type="radio"] { position: absolute; opacity: 0; pointer-events: none; }
.plx-pills label span { color: inherit !important; }

.plx-count { margin: .4rem 0 .8rem; font-size: .88rem; color: var(--muted); }
.plx-country { margin: 1.4rem 0 .7rem; font-family: 'Newsreader', Georgia, serif; font-size: 1.5rem; color: var(--ink); }
.plx-cinfo {
    display: grid; grid-template-columns: 1fr 1fr auto; gap: 1rem 1.5rem; align-items: start; padding: 1rem 1.2rem;
    background: var(--surface); border: 1px solid var(--line); border-radius: 14px; margin-bottom: 1rem;
    animation: plx-rise .5s var(--ease) both;
}
.plx-cinfo span { font-size: .78rem; color: var(--muted); }
.plx-cinfo p { margin: .15rem 0 0; font-size: .92rem; color: var(--ink-2); line-height: 1.45; }
.plx-link {
    display: inline-flex !important;
    align-items: center;
    justify-content: flex-start;
    align-self: flex-start;
    box-sizing: border-box;
    width: fit-content;
    max-width: 100%;
    min-width: 0;
    height: auto !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: .55rem .85rem !important;
    border-radius: 10px;
    font-size: .86rem;
    font-weight: 600;
    line-height: 1.35;
    text-align: left;
    white-space: normal !important;
    overflow-wrap: anywhere;
    word-break: normal;
    text-decoration: none !important;
    color: var(--seal) !important;
    border: 1.5px solid var(--seal);
    background: transparent;
    box-shadow: none !important;
    transition: background .2s, color .2s, transform .15s var(--ease);
}
.plx-link:hover { background: var(--seal); color: #fff !important; text-decoration: none !important; }
.plx-link:active { transform: translateY(1px); }

.plx-rules { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(290px, 100%), 1fr)); gap: .9rem; align-items: stretch; }
.plx-rule {
    display: flex;
    flex-direction: column;
    min-width: 0;
    box-sizing: border-box;
    background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem 1.1rem;
    animation: plx-rise .5s calc(min(var(--n), 12) * 40ms) var(--ease) both;
    transition: transform .2s var(--ease), box-shadow .25s var(--ease), border-color .2s;
}
.plx-rule:hover { transform: translateY(-3px); border-color: var(--seal); box-shadow: 0 14px 26px -18px rgba(47, 75, 255, .7); }
.plx-rule:active { transform: translateY(-1px) scale(.99); }
.plx-rule h4 { margin: .6rem 0 .35rem; font-size: 1rem; line-height: 1.3; color: var(--ink); }
.plx-rule p { margin: 0; font-size: .88rem; line-height: 1.55; color: var(--muted); }
.plx-rule-top { display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; }
.plx-cat, .plx-where, .plx-year { font-size: .74rem; padding: .15rem .55rem; border-radius: 999px; font-weight: 600; }
.plx-where { background: var(--ink); color: #fff; }
.plx-year { margin-left: auto; padding: 0; color: var(--muted); font-weight: 500; }
.plx-cat-core-framework { background: #e7ecf6; color: #2b3d63; }
.plx-cat-contracts { background: #e8e5fb; color: #43309a; }
.plx-cat-insurance { background: #dff3f4; color: #0f6b72; }
.plx-cat-consumer-protection { background: #fdeccf; color: #8a5200; }
.plx-cat-plain-language { background: #dff6e8; color: #17734a; }
.plx-cat-data-protection { background: #fde4ea; color: #a1243f; }
.plx-cat-ai-technology { background: #e3e9ff; color: #2f4bff; }
.plx-cat-legal-practice { background: #efe3f3; color: #6b2a80; }
.plx-cat-courts-language { background: #f1e7da; color: #7a4b1d; }
.plx-disclaimer { margin: 1.4rem 0 0; padding: .8rem 1rem; font-size: .82rem; line-height: 1.5; color: var(--muted); border-left: 3px solid var(--seal); background: rgba(47, 75, 255, .05); border-radius: 0 10px 10px 0; }
@media (max-width: 760px) { .plx-cinfo { grid-template-columns: 1fr; } }

/* ---------- Findings inside the result card ---------- */
.plx-find { margin-bottom: 1.3rem; animation: plx-rise .6s 1.05s var(--ease) both; }
.plx-find-title { display: block; font-size: .78rem; color: #7c8aa5; margin-bottom: .5rem; }
.plx-flags { display: flex; flex-wrap: wrap; gap: .4rem; }
.plx-flag { padding: .25rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 600; border: 1px solid; }
.plx-sev-high   { color: #ff8a86; background: rgba(239, 83, 80, .14); border-color: rgba(239, 83, 80, .4); }
.plx-sev-medium { color: #f4c14d; background: rgba(232, 169, 18, .14); border-color: rgba(232, 169, 18, .4); }
.plx-sev-none   { color: #5fdc9f; background: rgba(40, 194, 122, .14); border-color: rgba(40, 194, 122, .4); }
.plx-notes { margin: .7rem 0 0; padding-left: 1.1rem; font-size: .88rem; line-height: 1.5; color: #cbd5e1; }
.plx-notes li { margin-bottom: .25rem; }
.plx-review { margin: .7rem 0 0; padding: .55rem .8rem; font-size: .85rem; line-height: 1.45; color: #f4c14d; background: rgba(232, 169, 18, .1); border-left: 3px solid #e8a912; border-radius: 0 8px 8px 0; }
.plx-fine { margin: .6rem 0 0; font-size: .74rem; color: #64748b; }

/* ---------- Model status pill ---------- */
.plx-model { display: inline-flex; align-items: center; gap: .5rem; padding: .35rem .9rem; border-radius: 999px; font-size: .82rem; background: var(--surface); border: 1px solid var(--line); color: var(--ink-2); animation: plx-rise .5s .1s var(--ease) both; }
.plx-dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.plx-model-ok .plx-dot { background: var(--ok); box-shadow: 0 0 0 3px rgba(40, 194, 122, .2); }
.plx-model-warn { border-color: #f0cf84; background: #fff8e6; color: #7a5200; }
.plx-model-warn .plx-dot { background: var(--mid); box-shadow: 0 0 0 3px rgba(232, 169, 18, .25); }

.plx-footer { text-align: center; padding: 1.6rem 0 .5rem; font-size: .8rem; color: #8592a8; }

@media (max-width: 640px) {
    .plx-top { grid-template-columns: 1fr; justify-items: center; text-align: center; }
    .plx-meta { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .01ms !important; animation-delay: 0s !important; transition-duration: .01ms !important; }
}
"""

# Click ripple + force light mode (the palette is designed for it)
CUSTOM_JS = """
() => {
    document.body.classList.remove('dark');
    document.addEventListener('pointerdown', (e) => {
        const b = e.target.closest('button.plx-btn');
        if (!b) return;
        const r = b.getBoundingClientRect();
        const s = Math.max(r.width, r.height) * 2;
        const el = document.createElement('span');
        el.className = 'plx-ripple';
        el.style.cssText = `width:${s}px;height:${s}px;left:${e.clientX - r.left - s / 2}px;top:${e.clientY - r.top - s / 2}px`;
        b.appendChild(el);
        setTimeout(() => el.remove(), 650);
    });
}
"""


def build_demo():
    print("[PreserveLex] LANGUAGE_VALIDATION_BUILD=2026-11-26-ACTIVE")
    try_load_trained_model()
    print("[PreserveLex] " + (f"Using fine-tuned model: {LOADED_PATH}" if USE_TRAINED_MODEL
                              else f"Using SBERT fallback. Reason: {LOAD_ERROR}"))
    init_users()

    with gr.Blocks(
        title="PreserveLex — Multilingual Legal Meaning Preservation",
        css=CUSTOM_CSS,
        js=CUSTOM_JS,
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
    ) as demo:
        session = gr.State({"user": None})   # per-browser-session, kept on the server

        gr.HTML("""
        <div class="plx-header">
            <h1 class="plx-title">PreserveLex</h1>
            <p class="plx-sub">Check whether a simplified legal text keeps the meaning of the original.</p>
            <div class="plx-chips">
                <span class="plx-chip">English</span>
                <span class="plx-chip">Hindi</span>
                <span class="plx-chip">German</span>
                <span class="plx-chip">Spanish</span>
                <span class="plx-chip">French</span>
            </div>
        </div>
        """)

        # ------------------------------------------------------------ login view
        with gr.Column(visible=True, elem_classes="plx-auth") as login_view:
            with gr.Tabs():
                with gr.Tab("Sign in"):
                    li_user = gr.Textbox(label="Username", placeholder="Your username",
                                         max_lines=1, elem_classes="plx-input")
                    li_pass = gr.Textbox(label="Password", placeholder="Your password",
                                         type="password", max_lines=1, elem_classes="plx-input")
                    li_btn = gr.Button("Sign in", variant="primary", size="lg",
                                       elem_classes=["plx-btn", "plx-primary"])
                    li_msg = gr.HTML()
                with gr.Tab("Create account"):
                    su_user = gr.Textbox(label="Username", placeholder="3-32 letters, numbers, . _ -",
                                         max_lines=1, elem_classes="plx-input")
                    su_pass = gr.Textbox(label="Password", placeholder=f"At least {MIN_PASSWORD} characters",
                                         type="password", max_lines=1, elem_classes="plx-input")
                    su_pass2 = gr.Textbox(label="Confirm password", placeholder="Repeat the password",
                                          type="password", max_lines=1, elem_classes="plx-input")
                    su_btn = gr.Button("Create account", variant="primary", size="lg",
                                       elem_classes=["plx-btn", "plx-primary"])
                    su_msg = gr.HTML()

        # -------------------------------------------------------------- app view
        with gr.Column(visible=False, elem_classes="plx-app") as app_view:
            with gr.Row():
                user_badge = gr.HTML()
                #gr.HTML(model_status_html())
                logout_btn = gr.Button("Sign out", size="sm", scale=0, min_width=110,
                                       elem_classes=["plx-btn", "plx-ghost"])

            with gr.Row(elem_classes="plx-workspace", equal_height=False):
                with gr.Column(scale=2, min_width=205):
                    gr.HTML("""
                    <section class="plx-side-card">
                      <h3>What does preservation mean?</h3>
                      <p>A simplified clause is preserved when it communicates the same legal effect as the original.</p>
                      <div class="plx-side-item">
                        <strong>Obligations</strong>
                        <span>Who must do something, and what they must do.</span>
                      </div>
                      <div class="plx-side-item">
                        <strong>Rights & permissions</strong>
                        <span>Who may act, claim, refuse, or receive a benefit.</span>
                      </div>
                      <div class="plx-side-item">
                        <strong>Deadlines & amounts</strong>
                        <span>Dates, time limits, payment values, and quantities.</span>
                      </div>
                      <div class="plx-side-item">
                        <strong>Conditions & exceptions</strong>
                        <span>Requirements, exclusions, and circumstances that change a rule.</span>
                      </div>
                      <div class="plx-side-item">
                        <strong>Scope & parties</strong>
                        <span>Who the clause applies to and what it covers.</span>
                      </div>
                    </section>
                    """)
                with gr.Column(scale=8, min_width=480):
                    with gr.Tabs(elem_classes="plx-tabs"):
                        with gr.Tab("Evaluator"):
                            with gr.Row(equal_height=False):
                                with gr.Column(scale=5):
                                    original = gr.Textbox(
                                        label="Original legal text",
                                        placeholder="Paste the original legal clause or sentence…",
                                        lines=5, max_lines=8, elem_classes="plx-input",
                                    )
                                    simplified = gr.Textbox(
                                        label="Simplified text",
                                        placeholder="Paste the simplified version…",
                                        lines=5, max_lines=8, elem_classes="plx-input",
                                    )
                                    language = gr.Dropdown(
                                        choices=["English", "Hindi", "German", "Spanish", "French", "Other"],
                                        value="English", label="Language", elem_classes="plx-input",
                                    )
                                    with gr.Row():
                                        btn = gr.Button("Evaluate preservation", variant="primary", size="lg",
                                                        scale=3, elem_classes=["plx-btn", "plx-primary"])
                                        clear_btn = gr.Button("Clear", size="lg", scale=1,
                                                              elem_classes=["plx-btn", "plx-ghost"])

                                with gr.Column(scale=4):
                                    result = gr.HTML(value=EMPTY_HTML)
                                    score_num = gr.Number(visible=False)
                                    decision = gr.Textbox(visible=False)

                            gr.Examples(
                                examples=EXAMPLES,
                                inputs=[original, simplified, language],
                                label="Try an example",
                                examples_per_page=5,
                            )

                        with gr.Tab("Legal library"):
                            gr.HTML('<p class="plx-count" style="margin:.2rem 0 .6rem">Key laws by country that affect legal texts, contracts, insurance, consumer rights, data and AI.</p>')
                            lib_country = gr.Radio(
                                choices=[ALL] + COUNTRY_NAMES, value="India",
                                label="Country or region", elem_classes="plx-pills",
                            )
                            with gr.Row():
                                lib_query = gr.Textbox(
                                    label="Search", placeholder="Search, e.g. unfair terms, consent, insurance…",
                                    max_lines=1, scale=3, elem_classes="plx-input",
                                )
                                lib_cat = gr.Dropdown(
                                    choices=[ALL_CATEGORIES] + CATEGORIES, value=ALL_CATEGORIES,
                                    label="Category", scale=2, elem_classes="plx-input",
                                )
                            lib_html = gr.HTML(value=render_library("India", "", ALL_CATEGORIES))

                with gr.Column(scale=2, min_width=205):
                    gr.HTML("""
                    <section class="plx-side-card">
                      <h3>How to read the score</h3>
                      <div class="plx-side-item">
                        <strong>8.0–10 · Preserved</strong>
                        <span>Meaning appears substantially unchanged.</span>
                      </div>
                      <div class="plx-side-item">
                        <strong>5.0–7.9 · Partially preserved</strong>
                        <span>Some meaning may have shifted or become unclear.</span>
                      </div>
                      <div class="plx-side-item">
                        <strong>1.0–4.9 · Not preserved</strong>
                        <span>A material difference may change the legal effect.</span>
                      </div>
                      <h3 style="margin-top:1rem">Review these carefully</h3>
                      <ul>
                        <li>Negation: must not vs. must</li>
                        <li>Permission: may vs. shall</li>
                        <li>Deadlines: 5 days vs. 15 days</li>
                        <li>Exceptions and conditions</li>
                        <li>Party responsible for an action</li>
                      </ul>
                      <p class="plx-side-note">The score is an automated estimate, not a legal determination. Always review the original clause and flagged changes.</p>
                    </section>
                    """)

        gr.HTML("""
        <div class="plx-footer">
            PreserveLex · CSET 346 Natural Language Processing · Bennett University · Built with XLM-RoBERTa
        </div>
        """)

        # ---------------------------------------------------------------- events
        auth_outputs = [session, login_view, app_view]

        login_args = dict(fn=do_login, inputs=[li_user, li_pass],
                          outputs=auth_outputs + [li_msg, user_badge, li_pass], show_progress="hidden")
        li_btn.click(**login_args)
        li_pass.submit(**login_args)     # Enter key
        li_user.submit(**login_args)

        signup_args = dict(fn=do_signup, inputs=[su_user, su_pass, su_pass2],
                           outputs=auth_outputs + [su_msg, user_badge, su_pass, su_pass2], show_progress="hidden")
        su_btn.click(**signup_args)
        su_pass2.submit(**signup_args)

        logout_btn.click(
            fn=do_logout, inputs=None,
            outputs=auth_outputs + [li_msg, result, original, simplified, user_badge],
            show_progress="hidden",
        )

        btn.click(
            fn=predict_lmp,
            inputs=[original, simplified, language, session],
            outputs=[result, score_num, decision],
            show_progress="hidden",  # we render our own loading card
        )
        # Clear stale results when either text or the selected language changes.
        for component in (original, simplified, language):
            component.change(
                fn=clear_result,
                inputs=None,
                outputs=[result, score_num, decision],
                show_progress="hidden",
            )
        lib_inputs = [lib_country, lib_query, lib_cat]
        for comp in lib_inputs:
            comp.change(fn=render_library, inputs=lib_inputs, outputs=lib_html, show_progress="hidden")

        clear_btn.click(
            fn=clear_all, inputs=None,
            outputs=[original, simplified, language, result, score_num, decision],
            show_progress="hidden",
        )

    return demo



if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
