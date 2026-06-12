"""
Module: section_weighter.py
Adjusts match confidence based on section AND source_quality.
"""

SOURCE_QUALITY_MULTIPLIER = {
    "rich":        1.0,
    "table_row":   0.80,
    "header":      0.15,
    "boilerplate": 0.05,
}

SECTION_WEIGHTS = {
    "risk description":    1.0,
    "property description":1.0,
    "risk details":        1.0,
    "insured premises":    1.0,
    "prior claims":        1.0,
    "claims history":      1.0,
    "loss history":        1.0,
    "financial":           0.95,
    "safety":              0.95,
    "operations":          0.9,
    "business operations": 0.9,
    "occupancy":           0.9,
    "body":                0.8,
    "table":               0.75,
    "acord_table":         0.85,
    "email_body":          0.85,
    "general conditions":  0.3,
    "exclusions":          0.3,
    "limitations":         0.3,
    "endorsements":        0.4,
    "signature":           0.1,
}


def get_section_weight(section: str) -> float:
    if not section:
        return 0.8
    s = section.lower()
    if s in SECTION_WEIGHTS:
        return SECTION_WEIGHTS[s]
    for key, w in SECTION_WEIGHTS.items():
        if s.startswith(key):
            return w
    return 0.8


def apply_section_weighting(matches: list) -> list:
    for match in matches:
        section = getattr(match, "section", "") or ""
        quality = getattr(match, "source_quality", "rich") or "rich"
        sec_w   = get_section_weight(section)
        qual_m  = SOURCE_QUALITY_MULTIPLIER.get(quality, 1.0)
        weight  = sec_w * qual_m
        match.section_weight = round(weight, 3)

        if quality in ("header", "boilerplate") or weight < 0.25:
            match.confidence = "Informational"
        elif weight < 0.4:
            if match.confidence in ("High", "Medium"):
                match.confidence = "Low"
        elif weight >= 0.9 and match.confidence == "Medium":
            match.confidence = "High"
        elif weight >= 0.9 and match.confidence == "Low":
            match.confidence = "Medium"

    return matches
