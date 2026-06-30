"""
Signal 2 — Stylometric heuristics classifier.

Extracts eight features from raw text and scores them with a logistic
regression trained on synthetic human/AI samples. In production you would
replace _train_model() with a persisted model loaded from disk.

Features
--------
ttr                    type-token ratio
sentence_len_variance  variance of sentence lengths (words)
func_word_freq         function-word rate
punctuation_entropy    Shannon entropy over punctuation chars
burstiness             coefficient of variation of inter-word-type gaps
readability_score      Flesch Reading Ease (normalised 0-1, inverted)
transition_phrase_freq rate of LLM-characteristic transition phrases
avg_sentence_len       mean words per sentence (normalised)
"""

import re
import math
import string
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUNCTION_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
}

TRANSITION_PHRASES = [
    "furthermore", "moreover", "in conclusion", "it is worth noting",
    "it should be noted", "in summary", "to summarize", "in addition",
    "additionally", "nevertheless", "nonetheless", "on the other hand",
    "as a result", "consequently", "therefore", "thus", "hence",
    "it is important to", "it is essential to", "one must consider",
    "in this context", "with that said", "having said that",
    "it goes without saying", "needless to say",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def _ttr(words: list[str]) -> float:
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _sentence_len_variance(sentences: list[str]) -> float:
    if len(sentences) < 2:
        return 0.0
    lengths = [len(_words(s)) for s in sentences]
    return float(np.var(lengths))


def _func_word_freq(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(1 for w in words if w in FUNCTION_WORDS) / len(words)


def _punctuation_entropy(text: str) -> float:
    puncs = [c for c in text if c in string.punctuation]
    if not puncs:
        return 0.0
    counts = Counter(puncs)
    total = len(puncs)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _burstiness(words: list[str]) -> float:
    """
    Burstiness of word-type occurrence gaps.
    High burstiness = human (words cluster in topics).
    Near 0 = LLM (even distribution).
    Returns coefficient of variation of gap sizes, capped at 1.
    """
    if len(words) < 10:
        return 0.5
    type_positions: dict[str, list[int]] = {}
    for i, w in enumerate(words):
        type_positions.setdefault(w, []).append(i)

    cvs = []
    for positions in type_positions.values():
        if len(positions) < 2:
            continue
        gaps = np.diff(positions).astype(float)
        mu = gaps.mean()
        if mu == 0:
            continue
        cvs.append(gaps.std() / mu)

    if not cvs:
        return 0.5
    return float(min(np.mean(cvs), 3.0) / 3.0)  # normalise to ~0-1


def _readability(text: str, sentences: list[str], words: list[str]) -> float:
    """
    Flesch Reading Ease, inverted and normalised to 0-1.
    High score = easy to read (LLM) → high value here.
    """
    if not sentences or not words:
        return 0.5
    syllables = sum(_count_syllables(w) for w in words)
    asl = len(words) / len(sentences)           # avg sentence length
    asw = syllables / len(words)                 # avg syllables per word
    fre = 206.835 - 1.015 * asl - 84.6 * asw   # Flesch Reading Ease
    fre = max(0.0, min(100.0, fre))
    return fre / 100.0                           # 0 = hard, 1 = easy (LLM-like)


def _count_syllables(word: str) -> int:
    word = word.lower().strip("'")
    if not word:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _transition_phrase_freq(text: str, words: list[str]) -> float:
    if not words:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for p in TRANSITION_PHRASES if p in text_lower)
    # normalise by text length (per 100 words)
    return hits / max(len(words) / 100, 1)


def _avg_sentence_len(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    lengths = [len(_words(s)) for s in sentences]
    raw = float(np.mean(lengths))
    return raw / 50.0  # normalise: 50 words/sentence ≈ ceiling


def extract_features(text: str) -> dict:
    words = _words(text)
    sentences = _sentences(text)
    return {
        "ttr": _ttr(words),
        "sentence_len_variance": _sentence_len_variance(sentences),
        "func_word_freq": _func_word_freq(words),
        "punctuation_entropy": _punctuation_entropy(text),
        "burstiness": _burstiness(words),
        "readability_score": _readability(text, sentences, words),
        "transition_phrase_freq": _transition_phrase_freq(text, words),
        "avg_sentence_len": _avg_sentence_len(sentences),
    }


FEATURE_ORDER = [
    "ttr",
    "sentence_len_variance",
    "func_word_freq",
    "punctuation_entropy",
    "burstiness",
    "readability_score",
    "transition_phrase_freq",
    "avg_sentence_len",
]


# ---------------------------------------------------------------------------
# Model — logistic regression on stylometric features
# In production: load from a pickle trained on real data.
# Here we use synthetic weights that encode the domain knowledge above.
# ---------------------------------------------------------------------------

def _build_model() -> Pipeline:
    """
    Build a logistic regression with hand-calibrated coefficients.

    Positive coefficient → feature pushes toward AI (label=1).
    Negative coefficient → feature pushes toward human (label=0).

    Intercept is set so that a text with all features at their
    "neutral" midpoint returns a score near 0.50.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])

    # Synthetic training data that encodes domain knowledge.
    # Rows: [ttr, slv, fwf, pe, burst, read, tpf, asl]
    # label 0 = human, 1 = ai
    X_train = np.array([
        # --- human examples ---
        [0.72, 18.0, 0.44, 2.1, 0.70, 0.45, 0.00, 0.22],   # rich vocab, varied rhythm
        [0.65, 25.0, 0.50, 2.5, 0.65, 0.50, 0.00, 0.18],
        [0.80, 30.0, 0.38, 2.8, 0.80, 0.40, 0.01, 0.15],
        [0.60, 12.0, 0.55, 1.8, 0.60, 0.55, 0.00, 0.25],
        [0.75, 20.0, 0.42, 2.3, 0.75, 0.48, 0.01, 0.20],
        [0.68, 22.0, 0.47, 2.4, 0.72, 0.46, 0.00, 0.19],
        [0.78, 28.0, 0.40, 2.6, 0.78, 0.42, 0.00, 0.16],
        # --- ai examples ---
        [0.35, 2.0,  0.41, 0.9, 0.25, 0.82, 0.08, 0.38],   # flat, easy, transition-heavy
        [0.42, 3.0,  0.40, 1.0, 0.28, 0.78, 0.10, 0.36],
        [0.38, 1.5,  0.43, 0.8, 0.22, 0.85, 0.12, 0.40],
        [0.40, 2.5,  0.42, 1.1, 0.30, 0.80, 0.09, 0.37],
        [0.36, 2.0,  0.41, 0.9, 0.24, 0.83, 0.11, 0.39],
        [0.44, 3.5,  0.39, 1.2, 0.32, 0.76, 0.07, 0.35],
        [0.33, 1.0,  0.44, 0.7, 0.20, 0.88, 0.13, 0.42],
    ])
    y_train = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    model.fit(X_train, y_train)
    return model


_MODEL: Pipeline | None = None


def _get_model() -> Pipeline:
    global _MODEL
    if _MODEL is None:
        _MODEL = _build_model()
    return _MODEL


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify(text: str) -> dict:
    """
    Returns:
        score  — float 0-1, probability of AI authorship
        label  — "human" | "ai_generated"
        features — raw feature dict (for audit log + appeal context)
    """
    features = extract_features(text)
    x = np.array([[features[f] for f in FEATURE_ORDER]])
    model = _get_model()
    score = float(model.predict_proba(x)[0][1])
    label = "ai_generated" if score >= 0.5 else "human"
    return {
        "score": round(score, 4),
        "label": label,
        "features": {k: round(v, 4) for k, v in features.items()},
    }
