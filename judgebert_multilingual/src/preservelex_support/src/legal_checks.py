"""Transparent, lightweight text-change flags for PreserveLex.

These are heuristic indicators only, not legal reasoning or validated detection.
"""
import re


def _add(findings, kind, severity, detail):
    findings.append({"kind": kind, "severity": severity, "detail": detail})


def analyze_changes(original, simplified):
    original = original or ""
    simplified = simplified or ""
    findings = []
    o = original.lower()
    s = simplified.lower()

    # Modal/obligation language changes.
    mandatory = r"\b(shall|must|required to|is obligated to|deberá|debe|muss|doit)\b"
    optional = r"\b(may|might|can|permitted to|podrá|puede|kann|peut)\b"
    if re.search(mandatory, o) and re.search(optional, s) and not re.search(mandatory, s):
        _add(findings, "Obligation changed", "high",
             "Mandatory wording in the original may have become optional wording in the simplified text.")

    # Negation differences.
    neg = r"\b(not|no|never|neither|without|nunca|no|nicht|kein|ne|pas|aucun)\b"
    if bool(re.search(neg, o)) != bool(re.search(neg, s)):
        _add(findings, "Negation changed", "high",
             "Negation-related wording differs; check whether a prohibition or exclusion was reversed or removed.")

    # Numbers, dates, percentages and currency-like amounts.
    number = r"\b\d+(?:[.,]\d+)?\s*(?:%|days?|weeks?|months?|years?|rupees?|rs\.?|eur|usd)?\b"
    nums_o = re.findall(number, o)
    nums_s = re.findall(number, s)
    if nums_o != nums_s and (nums_o or nums_s):
        _add(findings, "Numbers or deadlines changed", "high",
             f"Numeric details differ (original: {', '.join(nums_o) or 'none'}; simplified: {', '.join(nums_s) or 'none'}).")

    # Exceptions/conditions cues.
    condition = r"\b(except|unless|provided that|subject to|only if|excepto|salvo|außer|sauf|unless)\b"
    if bool(re.search(condition, o)) != bool(re.search(condition, s)):
        _add(findings, "Condition or exception changed", "medium",
             "Condition or exception wording appears in only one version; verify the full clause.")

    # Scope cues.
    scope = r"\b(all|any|only|solely|each|every|none|todos?|únicamente|nur|tous|chaque)\b"
    if bool(re.search(scope, o)) != bool(re.search(scope, s)):
        _add(findings, "Scope wording changed", "medium",
             "A scope-related term appears in only one version; check who or what the clause covers.")

    return findings
