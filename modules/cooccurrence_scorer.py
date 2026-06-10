"""
Module: cooccurrence_scorer.py

Fully dynamic co-occurrence risk scorer.

The system classifies every category in the supplied keyword dictionary
into one or more risk domains by scanning its keywords against domain
signal word lists. Co-occurrence weights are assigned at the domain-pair
level, not the category-name level — so this works correctly regardless
of what your categories are called or how many categories you have.

Domain signals have been expanded to cover business-operations vocabulary
(hospitality, trades, liquor liability, premises liability, etc.)
in addition to the standard commercial property insurance vocabulary.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, FrozenSet
from config.settings import settings


# ── Risk domains ───────────────────────────────────────────────────────────────
# Semantic domains that represent underwriting risk categories.
# Stable — new category names route through these.

RISK_DOMAINS = [
    "legal",           # litigation, violations, prior claims
    "environmental",   # contamination, hazmat, pollution
    "financial",       # bankruptcy, insolvency, distress
    "structural",      # foundation, walls, maintenance
    "fire",            # fire risk, suppression, ignition
    "flood",           # water damage, drainage, flood zone
    "regulatory",      # OSHA, permits, code violations, licensing
    "occupancy",       # vacant, illegal use, special venues
    "liquor",          # alcohol service, licensing, promotions
    "safety",          # PPE, training, programs, documentation
    "labor",           # subcontractors, contracts, classification
    "crime",           # violence, theft, security
    "trades",          # construction, roofing, high-risk work
    "liability",       # premises, general liability exposure
]


# ── Domain signal words ────────────────────────────────────────────────────────
# For each domain, list phrases/words whose presence in a category's keywords
# or name indicates membership in that domain.
# Multi-word phrases are matched as substrings (case-insensitive).

DOMAIN_SIGNALS: Dict[str, List[str]] = {
    "legal": [
        "lawsuit", "litigation", "litigate", "sued", "plaintiff", "defendant",
        "court", "judgment", "arbitration", "settlement", "claim", "claims",
        "class action", "legal action", "attorney", "verdict", "injunction",
        "negligence", "alleged", "damages", "compensation", "indemnity",
        "prior violence", "assault", "battery", "shooting", "fatality",
        "violation", "violations", "revoked", "suspension", "license issues",
        "prior liquor", "prior loss",
    ],
    "environmental": [
        "asbestos", "mold", "contamination", "contaminate", "pollution",
        "pollutant", "hazardous", "toxic", "remediation", "remediate",
        "underground storage", "groundwater", "soil contamination",
        "brownfield", "superfund", "lead paint", "environmental",
        "epa", "discharge", "waste", "spill", "chemical",
        "mold/rot", "water intrusion", "grease buildup",
    ],
    "financial": [
        "bankruptcy", "bankrupt", "insolvency", "insolvent", "foreclosure",
        "receivership", "default", "liquidation", "chapter 11", "chapter 7",
        "creditor", "debt restructuring", "lien", "financial distress",
        "thin balance sheet", "bonding issues", "financial strain",
        "cash labor", "ghost payroll", "labor misclassification",
    ],
    "structural": [
        "foundation", "structural", "subsidence", "sinkhole", "collapse",
        "load bearing", "cracked", "settling", "heaving", "earth movement",
        "underpinning", "retaining wall", "slope failure", "landslide",
        "demolition", "condemned", "foundation settling", "bulging walls",
        "leaning walls", "sagging floors", "spalling", "cracking",
        "grandfathered systems", "no major rehab", "deferred maintenance",
        "structural steel", "structural defect",
    ],
    "fire": [
        "fire", "explosion", "combustible", "flammable", "arson",
        "sprinkler", "fire suppression", "smoke", "flash point",
        "accelerant", "ignition", "burn", "pyrotechnic",
        "no sprinklers", "no fire alarm", "no smoke detectors",
        "hood fire suppression", "no regular hood cleaning",
        "no regular duct cleaning", "grease buildup",
        "knob-and-tube", "aluminum wiring", "federal pacific",
        "stab-lok", "zinsco", "hot work", "torch", "welding", "brazing",
        "hazardous processes",
    ],
    "flood": [
        "flood", "flooding", "flooded", "water damage", "water intrusion",
        "storm surge", "hurricane", "inundation", "drainage", "waterlogged",
        "flood zone", "flood plain", "flash flood", "stormwater",
        "runoff", "overflow", "leaks",
    ],
    "regulatory": [
        "violation", "citation", "osha", "epa", "cease and desist",
        "enforcement", "non-compliant", "noncompliant", "permit",
        "code violation", "regulatory", "license revoked", "fine",
        "penalty", "sanction", "inspection", "compliance",
        "no osha training", "no osha certifications", "osha violations",
        "license in process", "missing logs", "outdated documentation",
        "old safety manual", "no safety program", "no safety manual",
        "prior liquor violations", "revoked license", "license issues",
        "limited liquor controls", "no tips", "no alcohol training",
        "inconsistent ppe", "ppe optional",
    ],
    "occupancy": [
        "vacant", "abandoned", "unauthorized", "illegal use", "change of use",
        "nightclub", "cannabis", "dispensary", "explosive storage",
        "chemical storage", "high risk", "occupancy", "habitability",
        "condemned", "uninhabitable", "gun range", "pyrotechnics",
        "special hazard", "amusement devices", "pinball", "arcade",
        "pool tables", "bowling", "shuffleboard",
        "live entertainment", "dancing", "dance floor", "dj",
        "live band", "karaoke", "live music", "solo-musician",
    ],
    "liquor": [
        "liquor", "alcohol", "bar", "bartender", "drunk", "intoxicated",
        "full bar", "cocktail", "spirits", "bottle service", "byob",
        "bring your own bottle", "corkage", "happy hour", "drink specials",
        "all you can drink", "ladies night", "college bar", "nightlife",
        "set-ups", "high proof", "bar-heavy", "heavy liquor",
        "aggressive drink", "late-night bar", "nightclub", "open after midnight",
        "open after 2am", "late hours", "after hours", "new venture bars",
        "no tips", "no alcohol training", "prior liquor",
    ],
    "safety": [
        "safety", "osha", "ppe", "harness", "training", "certification",
        "no safety manual", "no safety program", "informal safety",
        "missing logs", "old safety manual", "inconsistent ppe",
        "ppe optional", "inconsistent harness", "no osha",
        "adverse loss history", "worker injury", "struck-by",
        "frequent falls", "work at heights", "over 3 stories",
        "scaffolding", "cranes", "limited liquor controls",
    ],
    "labor": [
        "subcontractor", "contractor", "1099", "employee", "payroll",
        "uninsured subs", "no cois", "uninsured subcontractors",
        "cash labor", "ghost payroll", "labor misclassification",
        "1099 only", "1099 employees", "weak contracts",
        "no written contracts", "no additional insured",
        "no hold harmless", "no ai/hh",
    ],
    "crime": [
        "crime", "assault", "battery", "fight", "shooting", "fatality",
        "theft", "robbery", "vandalism", "gang", "violence",
        "prior violence", "bouncers", "no id scanner", "lax security",
        "untrained door staff", "no surveillance", "door person",
        "high-crime area", "high-crime mercantile", "poor security",
        "late-night", "cash business", "24 hour",
    ],
    "trades": [
        "roofing", "demolition", "structural steel", "tree removal",
        "landscaping", "swimming pool", "plumbing", "remodel",
        "cranes", "scaffolding", "snow removal", "ice removal",
        "snow treatment", "ice treatment", "hot work", "welding",
        "brazing", "torch", "high-risk trades", "residential new construction",
        "new homes", "condos", "ground-up construction", "custom homes",
        "work at heights", "exterior", "over 3 stories",
        "experience gaps", "new to trade", "expanding into new work",
        "wide radius operations", "multi-state", "nationwide",
    ],
    "liability": [
        "premises", "slip", "fall", "trip", "injury", "bodily injury",
        "property damage", "negligence", "patron", "guest",
        "frequent falls", "worker injury", "major property damage",
        "struck-by", "adverse loss history", "poor property condition",
        "deferred maintenance", "lack of lighting", "uneven sidewalks",
        "uncovered exterior stairwells", "poor condition",
        "submission inconsistencies", "conflicting answers",
        "website vs app mismatch",
    ],
}


# ── Domain-pair weights ────────────────────────────────────────────────────────
# Weights for combinations of two domains appearing in the same document section.
# Encodes underwriting logic — not category names.
# Scale: 0.0 = no meaningful interaction, 1.0 = almost always declined.
# Unlisted pairs default to DEFAULT_PAIR_WEIGHT.

DOMAIN_PAIR_WEIGHTS: Dict[FrozenSet[str], float] = {
    # Legal / prior losses compounds everything
    frozenset({"legal", "environmental"}):  0.95,
    frozenset({"legal", "financial"}):      0.92,
    frozenset({"legal", "structural"}):     0.88,
    frozenset({"legal", "fire"}):           0.88,
    frozenset({"legal", "regulatory"}):     0.90,
    frozenset({"legal", "occupancy"}):      0.82,
    frozenset({"legal", "liquor"}):         0.92,
    frozenset({"legal", "safety"}):         0.85,
    frozenset({"legal", "labor"}):          0.80,
    frozenset({"legal", "crime"}):          0.95,
    frozenset({"legal", "trades"}):         0.85,
    frozenset({"legal", "liability"}):      0.90,

    # Liquor + crime is the highest-risk hospitality combination
    frozenset({"liquor", "crime"}):         0.95,
    frozenset({"liquor", "safety"}):        0.90,
    frozenset({"liquor", "regulatory"}):    0.92,
    frozenset({"liquor", "liability"}):     0.88,
    frozenset({"liquor", "occupancy"}):     0.85,
    frozenset({"liquor", "fire"}):          0.85,

    # Crime combinations
    frozenset({"crime", "regulatory"}):     0.88,
    frozenset({"crime", "safety"}):         0.85,
    frozenset({"crime", "liability"}):      0.90,
    frozenset({"crime", "occupancy"}):      0.85,
    frozenset({"crime", "financial"}):      0.82,

    # Environmental and structural
    frozenset({"environmental", "regulatory"}): 0.88,
    frozenset({"environmental", "structural"}): 0.85,
    frozenset({"environmental", "financial"}):  0.80,
    frozenset({"environmental", "fire"}):       0.82,
    frozenset({"environmental", "liability"}):  0.82,

    # Fire + safety/regulatory = serious risk
    frozenset({"fire", "safety"}):          0.90,
    frozenset({"fire", "regulatory"}):      0.88,
    frozenset({"fire", "occupancy"}):       0.82,
    frozenset({"fire", "trades"}):          0.85,
    frozenset({"fire", "liability"}):       0.85,

    # Trades + safety/labor
    frozenset({"trades", "safety"}):        0.88,
    frozenset({"trades", "labor"}):         0.85,
    frozenset({"trades", "regulatory"}):    0.82,
    frozenset({"trades", "liability"}):     0.85,
    frozenset({"trades", "financial"}):     0.78,

    # Financial distress + physical issues
    frozenset({"financial", "structural"}): 0.82,
    frozenset({"financial", "regulatory"}): 0.80,
    frozenset({"financial", "safety"}):     0.80,
    frozenset({"financial", "trades"}):     0.78,
    frozenset({"financial", "liability"}):  0.78,

    # Structural + safety/regulatory
    frozenset({"structural", "safety"}):    0.80,
    frozenset({"structural", "regulatory"}): 0.78,
    frozenset({"structural", "liability"}): 0.82,
    frozenset({"structural", "flood"}):     0.82,

    # Labor violations
    frozenset({"labor", "safety"}):         0.85,
    frozenset({"labor", "regulatory"}):     0.82,
    frozenset({"labor", "liability"}):      0.80,
    frozenset({"labor", "trades"}):         0.82,

    # Flood
    frozenset({"flood", "structural"}):     0.82,
    frozenset({"flood", "regulatory"}):     0.75,
    frozenset({"flood", "legal"}):          0.85,
    frozenset({"flood", "financial"}):      0.78,
    frozenset({"flood", "environmental"}):  0.75,

    # Regulatory + safety
    frozenset({"regulatory", "safety"}):    0.85,
    frozenset({"regulatory", "occupancy"}): 0.82,
    frozenset({"regulatory", "liability"}): 0.80,
}

DEFAULT_PAIR_WEIGHT = 0.30
HIGH_RISK_THRESHOLD = 0.60   # pairs below this are not flagged


# ── Category classifier ────────────────────────────────────────────────────────

def classify_category(category_name: str, keywords: List[str],
                      threshold: int = 2) -> List[str]:
    """
    Assign a category to one or more risk domains.
    Scans category name + all keywords for domain signal words.
    threshold: minimum signal hits to claim domain membership.
    """
    text = " ".join([category_name] + keywords).lower()
    domains_found = []
    for domain, signals in DOMAIN_SIGNALS.items():
        hits = sum(1 for signal in signals if signal in text)
        if hits >= threshold:
            domains_found.append(domain)
    return domains_found


def build_category_domain_map(keywords_dict: dict) -> Dict[str, List[str]]:
    """Build {category: [domains]} map from the full keyword dictionary."""
    return {
        cat: classify_category(cat, kws)
        for cat, kws in keywords_dict.items()
    }


def get_pair_weight(domains_a: List[str], domains_b: List[str]) -> float:
    """Return highest applicable weight for two categories' domain sets."""
    if not domains_a or not domains_b:
        return DEFAULT_PAIR_WEIGHT
    best = DEFAULT_PAIR_WEIGHT
    for da in domains_a:
        for db in domains_b:
            if da == db:
                best = max(best, 0.60)
            else:
                best = max(best, DOMAIN_PAIR_WEIGHTS.get(frozenset({da, db}),
                                                          DEFAULT_PAIR_WEIGHT))
    return best


# ── Cache ──────────────────────────────────────────────────────────────────────
_CACHE: Dict[str, Dict[str, List[str]]] = {}

def _get_domain_map(keywords_dict: dict) -> Dict[str, List[str]]:
    import hashlib, json
    key = hashlib.md5(json.dumps(keywords_dict, sort_keys=True).encode()).hexdigest()
    if key not in _CACHE:
        _CACHE[key] = build_category_domain_map(keywords_dict)
    return _CACHE[key]


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class CooccurrenceResult:
    document_risk_score: str
    risk_score_value: float
    triggered_categories: List[str]
    high_risk_combos_found: List[Tuple[str, str, float]]
    category_frequency: Dict[str, int]
    category_domains: Dict[str, List[str]]
    notes: str = ""


# ── Main function ──────────────────────────────────────────────────────────────

def compute_cooccurrence(matches: list, ingested_doc,
                         keywords_dict: dict = None) -> CooccurrenceResult:
    """
    Compute cross-category co-occurrence risk score dynamically from matches.

    Steps:
      1. Build chunk → categories map from affirmed matches only
      2. Classify each triggered category into risk domains (from keyword dict)
      3. Within proximity windows, find category pairs and score via domain weights
      4. Compute composite document risk score
    """
    # Step 1: chunk → categories
    chunk_categories: Dict[int, Set[str]] = defaultdict(set)
    category_freq: Dict[str, int] = defaultdict(int)

    for match in matches:
        if not getattr(match, "affirmed", True):
            continue
        cat = getattr(match, "category", "")
        if cat:
            chunk_categories[match.chunk_index].add(cat)
            category_freq[cat] += 1

    triggered = list(category_freq.keys())

    # Step 2: domain classification
    domain_map = _get_domain_map(keywords_dict) if keywords_dict else {c: [] for c in triggered}

    # Step 3: proximity-window pair scoring
    window = settings.cooccurrence_window_paragraphs
    high_risk_found: List[Tuple[str, str, float]] = []
    seen_pairs: Set[FrozenSet[str]] = set()

    chunk_indices = sorted(chunk_categories.keys())
    for i, idx in enumerate(chunk_indices):
        window_cats: Set[str] = set()
        for j in range(i, min(i + window, len(chunk_indices))):
            window_cats.update(chunk_categories[chunk_indices[j]])

        if len(window_cats) < 2:
            continue

        cat_list = sorted(window_cats)
        for a in range(len(cat_list)):
            for b in range(a + 1, len(cat_list)):
                cat_a, cat_b = cat_list[a], cat_list[b]
                pair_key = frozenset({cat_a, cat_b})
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                weight = get_pair_weight(
                    domain_map.get(cat_a, []),
                    domain_map.get(cat_b, [])
                )
                if weight >= HIGH_RISK_THRESHOLD:
                    high_risk_found.append((cat_a, cat_b, weight))

    high_risk_found.sort(key=lambda x: x[2], reverse=True)

    # Step 4: composite score
    base_score   = min(len(triggered) * 0.08, 0.45)
    combo_boost  = (high_risk_found[0][2] * 0.50) if high_risk_found else 0.0
    all_domains  = set(d for c in triggered for d in domain_map.get(c, []))
    diversity_bonus = 0.10 if len(all_domains) >= 3 else 0.0
    raw_score    = min(base_score + combo_boost + diversity_bonus, 1.0)

    if raw_score >= 0.65 or (high_risk_found and high_risk_found[0][2] >= 0.85):
        label = "High"
    elif raw_score >= 0.35 or len(triggered) >= 2:
        label = "Medium"
    else:
        label = "Low"

    # Notes
    notes_parts = []
    for c1, c2, w in high_risk_found[:3]:
        d1 = "/".join(domain_map.get(c1, ["?"]))
        d2 = "/".join(domain_map.get(c2, ["?"]))
        notes_parts.append(f"{c1} [{d1}] + {c2} [{d2}] (weight {w:.2f})")
    if not notes_parts:
        domains_str = ", ".join(sorted(all_domains)) or "unclassified"
        notes_parts.append(
            f"No high-risk combinations detected. Domains active: {domains_str}"
            if triggered else "No adverse matches found"
        )

    return CooccurrenceResult(
        document_risk_score=label,
        risk_score_value=round(raw_score, 3),
        triggered_categories=triggered,
        high_risk_combos_found=high_risk_found,
        category_frequency=dict(category_freq),
        category_domains=domain_map,
        notes="; ".join(notes_parts),
    )


# ── Visibility utilities ───────────────────────────────────────────────────────

def explain_domain_classification(keywords_dict: dict) -> None:
    """
    Print a report showing how each category was classified into domains
    and what pair weights apply. Call after loading a new keyword file.
    """
    domain_map = build_category_domain_map(keywords_dict)
    print("=" * 65)
    print(f"Domain classification — {len(keywords_dict)} categories")
    print("=" * 65)
    unclassified = []
    for cat, domains in domain_map.items():
        kw_count = len(keywords_dict.get(cat, []))
        if domains:
            print(f"  {cat} ({kw_count} kws) → {', '.join(domains)}")
        else:
            unclassified.append(cat)
    if unclassified:
        print(f"\n  UNCLASSIFIED ({len(unclassified)} categories):")
        for cat in unclassified:
            print(f"    - {cat}  [{', '.join(keywords_dict.get(cat,[])[:3])}...]")

    print("\n" + "=" * 65)
    print("High-risk category pairs (weight >= 0.75):")
    print("=" * 65)
    cats = list(domain_map.keys())
    pairs = []
    for a in range(len(cats)):
        for b in range(a + 1, len(cats)):
            w = get_pair_weight(domain_map[cats[a]], domain_map[cats[b]])
            if w >= 0.75:
                pairs.append((cats[a], cats[b], w))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for c1, c2, w in pairs[:20]:
        print(f"  {w:.2f}  {c1}  +  {c2}")
    if not pairs:
        print("  None found above 0.75 threshold")
