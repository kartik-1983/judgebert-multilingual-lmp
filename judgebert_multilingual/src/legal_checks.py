"""Transparent heuristic checks for possible legal meaning changes.

These checks are advisory indicators, not legal reasoning or validated legal determinations.
"""
import re


MODALS = {
    "mandatory": re.compile(r"\\b(?:must|shall|required to|is required to|is obligated to|deberá|debe|muss|doit)\\b", re.I),
    "discretionary": re.compile(r"\\b(?:may|might|can|could|is permitted to|is allowed to|podrá|puede|kann|peut)\\b", re.I),
    "prohibitive": re.compile(r"\\b(?:must not|shall not|may not|cannot|prohibited|forbidden|no deberá|no debe|darf nicht|ne doit pas)\\b", re.I),
}
CONDITION = re.compile(r"\\b(?:if|unless|except|provided that|subject to|only if|when|where|salvo|excepto|wenn|sauf|siempre que)\\b", re.I)
SCOPE = re.compile(r"\\b(?:all|any|only|solely|each|every|none|each party|either party|both parties|todos?|únicamente|nur|tous|chaque)\\b", re.I)
PARTIES = re.compile(r"\\b(?:tenant|landlord|buyer|seller|consumer|supplier|employee|employer|company|party|parties|applicant|authority|lessee|lessor)\\b", re.I)
NUMERIC = re.compile(r"(?:₹|\\$|€|£|\\b(?:rs\\.?|inr|usd|eur|gbp)\\b)?\\s*\\d[\\d,]*(?:\\.\\d+)?\\s*(?:%|days?|weeks?|months?|years?|hours?|rupees?)?", re.I)
DATE_WORDS = re.compile(r"\\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\\b", re.I)


def _add(findings, kind, severity, detail):
    findings.append({"kind": kind, "severity": severity, "detail": detail})


def analyze_changes(original, simplified):
    """Return transparent flags for possible changes across five review dimensions."""
    original = original or ""
    simplified = simplified or ""
    findings = []

    # Obligations: detect both loss of mandatory force and changes in modal class.
    o_mandatory = bool(MODALS["mandatory"].search(original))
    s_mandatory = bool(MODALS["mandatory"].search(simplified))
    o_discretionary = bool(MODALS["discretionary"].search(original))
    s_discretionary = bool(MODALS["discretionary"].search(simplified))
    if (o_mandatory and s_discretionary and not s_mandatory) or (o_discretionary and s_mandatory and not s_discretionary):
        _add(findings, "Obligation changed", "high",
             "The modal wording changed between mandatory and discretionary language (for example, must versus should/may). Check whether the duty has been weakened or strengthened.")

    # Rights and permissions: modal/permission wording and prohibitions.
    o_prohibited = bool(MODALS["prohibitive"].search(original))
    s_prohibited = bool(MODALS["prohibitive"].search(simplified))
    if o_prohibited != s_prohibited:
        _add(findings, "Rights or permissions changed", "high",
             "Prohibitive wording appears in only one version. Check whether an action has become allowed or forbidden.")
    elif o_discretionary != s_discretionary and not (o_mandatory and s_discretionary and not s_mandatory):
        _add(findings, "Rights or permissions changed", "medium",
             "Permission-related modal wording differs. Check who is allowed or entitled to act.")

    # Negation.
    neg = re.compile(r"\\b(?:not|no|never|neither|without|nunca|nicht|kein|ne|pas|aucun)\\b", re.I)
    if bool(neg.search(original)) != bool(neg.search(simplified)):
        _add(findings, "Negation changed", "high",
             "Negation-related wording differs; check whether a prohibition or exclusion was reversed or removed.")

    # Deadlines and amounts: compare extracted numeric tokens and named date words.
    nums_o, nums_s = NUMERIC.findall(original), NUMERIC.findall(simplified)
    dates_o, dates_s = DATE_WORDS.findall(original), DATE_WORDS.findall(simplified)
    if nums_o != nums_s or [x.lower() for x in dates_o] != [x.lower() for x in dates_s]:
        _add(findings, "Deadlines or amounts changed", "high",
             f"Numeric or date details differ (original: {', '.join(nums_o + dates_o) or 'none'}; simplified: {', '.join(nums_s + dates_s) or 'none'}).")

    # Conditions and exceptions.
    if bool(CONDITION.search(original)) != bool(CONDITION.search(simplified)):
        _add(findings, "Conditions or exceptions changed", "medium",
             "A condition or exception cue appears in only one version; verify whether a qualification was added or removed.")

    # Scope and parties.
    scope_o, scope_s = SCOPE.findall(original), SCOPE.findall(simplified)
    parties_o, parties_s = PARTIES.findall(original), PARTIES.findall(simplified)
    if [x.lower() for x in scope_o] != [x.lower() for x in scope_s] or [x.lower() for x in parties_o] != [x.lower() for x in parties_s]:
        _add(findings, "Scope or parties changed", "medium",
             "Scope terms or named party references differ. Check who the clause applies to and who is responsible.")

    return findings


def review_dimensions(findings):
    """Build a five-category review checklist from heuristic findings."""
    mapping = [
        ("Obligations", ("Obligation changed",)),
        ("Rights & permissions", ("Rights or permissions changed", "Negation changed")),
        ("Deadlines & amounts", ("Deadlines or amounts changed",)),
        ("Conditions & exceptions", ("Conditions or exceptions changed",)),
        ("Scope & parties", ("Scope or parties changed",)),
    ]
    kinds = {item["kind"] for item in findings}
    result = []
    for label, related in mapping:
        matched = [kind for kind in related if kind in kinds]
        result.append({
            "label": label,
            "status": "Potential change" if matched else "No change detected",
            "flagged": bool(matched),
        })
    return result
