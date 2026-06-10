"""
Module: section_weighter.py
Adjusts match confidence based on which document section the match appears in.
Risk Description / Property Details = high weight (promote confidence)
General Conditions / Exclusions / Boilerplate = low weight (demote to informational)
"""

from typing import Union


SECTION_WEIGHTS = {
    # High weight — adverse language here is genuinely significant
    "risk description": 1.0,
    "property description": 1.0,
    "risk details": 1.0,
    "insured premises": 1.0,
    "prior claims": 1.0,
    "claims history": 1.0,
    "loss history": 1.0,
    "operations": 0.9,
    "business operations": 0.9,
    "occupancy": 0.9,
    "body": 0.8,            # unclassified body text — neutral
    "table": 0.8,
    "acord_table": 0.9,
    "email_body": 0.8,

    # Low weight — standard boilerplate, not submission-specific language
    "general conditions": 0.3,
    "exclusions": 0.3,
    "limitations": 0.3,
    "endorsements": 0.4,
    "signature": 0.1,
}

CONFIDENCE_PROMOTION = {
    # weight >= 0.9 + High -> stays High
    # weight >= 0.9 + Medium -> promoted to High
    # weight < 0.4 + any -> demoted to Informational
}


def get_section_weight(section: str) -> float:
    if not section:
        return 0.8
    return SECTION_WEIGHTS.get(section.lower(), 0.8)


def apply_section_weighting(matches: list) -> list:
    """
    Adjusts confidence label on each match based on section weight.
    Adds .section_weight and may override .confidence.
    """
    for match in matches:
        section = getattr(match, "section", "") or ""
        weight = get_section_weight(section)
        match.section_weight = weight

        if weight < 0.4:
            match.confidence = "Informational"
        elif weight >= 0.9 and match.confidence == "Medium":
            match.confidence = "High"
        # Low confidence semantic matches in high-weight sections -> Medium
        elif weight >= 0.9 and match.confidence == "Low":
            match.confidence = "Medium"

    return matches
