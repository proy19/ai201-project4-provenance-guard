"""
Transparency label — maps a scoring result to a human-readable string.

Designed to be i18n-ready: all copy lives here, not scattered through
the API layer.
"""

from confidence import ScoringResult

CATEGORY_LABELS = {
    "human":        "Written by a human",
    "uncertain":    "Authorship uncertain",
    "ai_assisted":  "Likely AI-assisted",
    "ai_generated": "Likely AI-generated",
}

CATEGORY_DESCRIPTIONS = {
    "human": (
        "Our analysis found strong indicators of human authorship: "
        "varied sentence rhythm, rich vocabulary, and idiosyncratic style."
    ),
    "uncertain": (
        "Our signals returned mixed results. The content shows some "
        "characteristics of both human and AI writing."
    ),
    "ai_assisted": (
        "Our analysis found elevated indicators of AI involvement: "
        "uniform prose rhythm, high readability, and generic structure."
    ),
    "ai_generated": (
        "Our analysis found strong indicators of AI authorship: "
        "very low vocabulary variance, flat sentence rhythm, and "
        "frequent use of AI-characteristic transition phrases."
    ),
}


def render(scoring: ScoringResult, content_id: str) -> str:
    """Return the short label string shown to users."""
    cat = scoring.category
    label = CATEGORY_LABELS.get(cat, "Unknown")
    pct = int(round(scoring.combined_score * 100))
    return f"{label} ({pct}% AI confidence)"


def render_full(scoring: ScoringResult, content_id: str) -> dict:
    """Return the full label payload including description and badge metadata."""
    cat = scoring.category
    return {
        "label": render(scoring, content_id),
        "short_category": cat,
        "description": CATEGORY_DESCRIPTIONS.get(cat, ""),
        "score": scoring.combined_score,
        "badge_color": _badge_color(cat),
    }


def _badge_color(category: str) -> str:
    return {
        "human":        "green",
        "uncertain":    "amber",
        "ai_assisted":  "orange",
        "ai_generated": "red",
    }.get(category, "gray")
