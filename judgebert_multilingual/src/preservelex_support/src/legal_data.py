"""Minimal compatibility data for the PreserveLex legal-library UI.

The original curated legal-library dataset was not present in the project folder.
This module intentionally does not invent legal entries. Restore the original
COUNTRIES data here when it becomes available.
"""
ALL = "All countries"
CATEGORIES = [
    "Core framework",
    "Contracts",
    "Insurance",
    "Consumer protection",
    "Plain language",
    "Data protection",
    "AI & technology",
    "Legal practice",
    "Courts & language",
]
REVIEWED = "Not yet verified"
COUNTRIES = [
    {
        "name": "India",
        "system": "Common-law legal system",
        "regulators": "Refer to the relevant official regulator for the subject matter",
        "portal": "https://www.indiacode.nic.in/",
        "portal_label": "India Code",
        "rules": [],
    }
]
