"""
Confidence scoring — pure function, no I/O.

Takes Signal 1 and Signal 2 outputs and fuses them into a single score
using a weighted average followed by feature-level hard overrides.

Output
------
combined_score       float 0-1
category             "human" | "uncertain" | "ai_assisted" | "ai_generated"
contributing_features  list[str]  plain-English reasons (for labels + appeals)
"""

from dataclasses import dataclass

# Weights must sum to 1.0
WEIGHT_S1 = 0.6
WEIGHT_S2 = 0.4

# Category thresholds
THRESHOLDS = [
    (0.35, "human"),
    (0.55, "uncertain"),
    (0.75, "ai_assisted"),
    (1.01, "ai_generated"),
]

# Hard override rules: (condition_fn, floor_or_ceiling, direction, description)
# direction "floor" means push score UP to at least the value
# direction "ceiling" means cap score DOWN to at most the value
OVERRIDE_RULES = [
    # Very high transition phrase frequency + flat sentence rhythm
    (
        lambda s1, s2, f: f.get("transition_phrase_freq", 0) > 0.12
                          and f.get("sentence_len_variance", 99) < 2.0,
        0.70, "floor",
        "high transition phrase frequency + low sentence length variance",
    ),
    # Very low vocabulary richness
    (
        lambda s1, s2, f: f.get("ttr", 1) < 0.30,
        0.65, "floor",
        "very low type-token ratio (vocabulary repetition)",
    ),
    # Groq is highly confident it's AI
    (
        lambda s1, s2, f: s1 > 0.90,
        0.85, "floor",
        "Signal 1 (Groq) confidence > 90%",
    ),
    # Both signals say strongly human
    (
        lambda s1, s2, f: s1 < 0.20 and s2 < 0.25,
        0.20, "ceiling",
        "both signals indicate strong human authorship",
    ),
]


@dataclass
class ScoringResult:
    combined_score: float
    category: str
    contributing_features: list[str]


def score(signal1: dict, signal2: dict) -> ScoringResult:
    """
    signal1: { score, rationale, flags }
    signal2: { score, label, features }
    """
    s1 = float(signal1["score"])
    s2 = float(signal2["score"])
    features = signal2.get("features", {})

    # Stage 1 — weighted average
    combined = round(WEIGHT_S1 * s1 + WEIGHT_S2 * s2, 4)
    reasons: list[str] = []

    # Stage 2 — hard overrides
    for condition, value, direction, description in OVERRIDE_RULES:
        try:
            fired = condition(s1, s2, features)
        except Exception:
            fired = False
        if not fired:
            continue
        if direction == "floor" and combined < value:
            combined = value
            reasons.append(f"override (floor {value}): {description}")
        elif direction == "ceiling" and combined > value:
            combined = value
            reasons.append(f"override (ceiling {value}): {description}")

    # Stage 3 — bucket
    category = "ai_generated"
    for threshold, label in THRESHOLDS:
        if combined < threshold:
            category = label
            break

    # Collect human-readable contributing features
    if signal1.get("rationale"):
        reasons.append(f"Signal 1 rationale: {signal1['rationale']}")
    for flag in signal1.get("flags", []):
        reasons.append(f"Signal 1 flag: {flag}")

    notable = _notable_features(features)
    reasons.extend(notable)

    return ScoringResult(
        combined_score=combined,
        category=category,
        contributing_features=reasons,
    )


def _notable_features(features: dict) -> list[str]:
    """Surface the two or three most diagnostic feature values."""
    notes = []
    ttr = features.get("ttr")
    if ttr is not None:
        if ttr < 0.40:
            notes.append(f"low type-token ratio ({ttr:.2f}) — repetitive vocabulary")
        elif ttr > 0.70:
            notes.append(f"high type-token ratio ({ttr:.2f}) — rich vocabulary")

    slv = features.get("sentence_len_variance")
    if slv is not None:
        if slv < 3.0:
            notes.append(f"low sentence length variance ({slv:.1f}) — uniform rhythm")
        elif slv > 15.0:
            notes.append(f"high sentence length variance ({slv:.1f}) — varied rhythm")

    tpf = features.get("transition_phrase_freq")
    if tpf is not None and tpf > 0.05:
        notes.append(f"elevated transition phrase frequency ({tpf:.2f})")

    read = features.get("readability_score")
    if read is not None and read > 0.75:
        notes.append(f"high readability score ({read:.2f}) — very polished, consistent prose")

    return notes
