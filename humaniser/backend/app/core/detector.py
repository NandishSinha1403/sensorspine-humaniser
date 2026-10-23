"""
detector.py — State-of-the-art multi-signal AI text detector.

Architecture: 9 orthogonal signal channels → weighted fusion → sigmoid calibration.

Channels:
  1. Sentence-length burstiness (std dev of word counts)
  2. Syntactic tree depth variance (dependency parse depth distribution)
  3. AI signature phrase density (known AI-marker vocabulary)
  4. Perplexity proxy (Brown corpus bigram conditional probability)
  5. MATTR — Moving Average Type-Token Ratio (lexical diversity)
  6. Punctuation uniformity (comma/semicolon variance)
  7. Sentence-opener entropy (how varied the first POS tag of each sentence is)
  8. Clause-to-sentence ratio (subordinate clause density)
  9. Discourse connector density (how heavily connectors are used)

This is a heuristic detector — no ML models or API calls. Designed to be
fast, synchronous, and deterministic for use in the self-audit loop.
"""

import numpy as np
import re
import math
import logging
import json
import os
from typing import List, Dict, Any, Tuple
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

logger = logging.getLogger("humaniser.detector")

# ---------------------------------------------------------------------------
# spaCy loader (shared)
# ---------------------------------------------------------------------------
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_md")
        except Exception:
            try:
                import spacy
                _nlp = spacy.load("en_core_web_sm")
            except Exception:
                _nlp = False
    return _nlp if _nlp is not False else None

# ---------------------------------------------------------------------------
# Brown corpus frequency data (for perplexity proxy)
# ---------------------------------------------------------------------------
WORD_FREQ: Dict = {}
BIGRAM_FREQ: Dict = {}
TOTAL_BROWN_WORDS: int = 0

def load_brown_corpus_data():
    """Load pre-computed Brown corpus word/bigram frequencies."""
    global WORD_FREQ, BIGRAM_FREQ, TOTAL_BROWN_WORDS
    try:
        data_path = os.path.join(os.path.dirname(__file__), "brown_frequencies.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                data = json.load(f)
                WORD_FREQ = data["unigrams"]
                BIGRAM_FREQ = data["bigrams"]
                TOTAL_BROWN_WORDS = data["total_words"]
            logger.info("[Detector] Loaded frequencies: %d unigrams, %d bigrams", len(WORD_FREQ), len(BIGRAM_FREQ))
            return True
        else:
            from nltk.corpus import brown
            words = [w.lower() for w in brown.words() if w.isalnum()]
            WORD_FREQ = dict(FreqDist(words))
            TOTAL_BROWN_WORDS = len(words)
            BIGRAM_FREQ = {}
            for i in range(len(words) - 1):
                pair = f"{words[i]}|{words[i + 1]}"
                BIGRAM_FREQ[pair] = BIGRAM_FREQ.get(pair, 0) + 1
            return True
    except Exception as e:
        logger.warning(f"[Detector] Could not load Brown corpus: {e}")
        return False

load_brown_corpus_data()

# ---------------------------------------------------------------------------
# AI Signature Phrases (expanded — 60+ markers)
# ---------------------------------------------------------------------------
AI_SIGNATURE_PHRASES = [
    # Classic transition markers overused by LLMs
    "furthermore", "additionally", "moreover", "consequently",
    "it is worth noting", "it is important to note", "it should be noted",
    "in conclusion", "to summarize", "to conclude",
    # Hedging/meta phrases
    "plays a crucial role", "has been shown", "research has shown",
    "it is essential", "it is crucial", "it is imperative",
    "delve into", "shed light on", "it is clear that", "it is undeniable",
    # Enumerators
    "firstly", "secondly", "thirdly",
    # Temporal padding
    "in today's world", "in recent years", "has become increasingly",
    "in the modern era", "in contemporary society",
    # Filler phrases
    "needless to say", "it goes without saying", "at the end of the day",
    "all things considered", "taking everything into consideration",
    "there is no doubt", "without a doubt", "it can be argued",
    "one must consider",
    # Demonstration/emphasis
    "this demonstrates", "this highlights", "this underscores",
    "this emphasizes", "this suggests that", "this indicates that",
    # Range/quantity
    "a wide range of", "a variety of", "in the realm of",
    "a myriad of", "a plethora of",
    # Connector padding
    "the fact that", "in terms of", "with regard to", "when it comes to",
    "in order to", "due to the fact", "in light of",
    "is the process of", "play an important role", "has become essential",
    "as well as", "helps reduce", "as mentioned earlier",
    "not only", "but also", "in a way that", "by contrast",
    # Verb-heavy AI patterns
    "serves as a testament", "poses significant", "raises significant",
    "underscores the importance", "highlights the need",
]

# Weighted phrase categories (some phrases are stronger AI signals)
AI_STRONG_SIGNALS = {
    "delve into", "it is worth noting", "it is important to note",
    "in today's world", "in the modern era", "plays a crucial role",
    "a myriad of", "a plethora of", "needless to say",
    "it goes without saying", "serves as a testament",
    "underscores the importance", "highlights the need",
    "in contemporary society", "this demonstrates",
}


# ---------------------------------------------------------------------------
# Signal Channel 1: Sentence-Length Burstiness
# ---------------------------------------------------------------------------
def calculate_burstiness(sentences: List[str]) -> float:
    """
    Score 0–100.  100 = low variance (AI-like).  0 = high variance (human).
    AI generators produce near-uniform sentence lengths (std ≈ 2–5).
    Human writing varies widely (std ≈ 8–18).
    """
    if len(sentences) < 2:
        return 0.0
    lengths = [len(word_tokenize(s)) for s in sentences]
    std_dev = float(np.std(lengths))
    mean_len = float(np.mean(lengths))

    # Coefficient of variation is more robust than raw std
    cv = std_dev / mean_len if mean_len > 0 else 0

    if cv >= 0.60:   # Very high variance → human
        return 0.0
    if cv <= 0.10:   # Near-uniform → strong AI signal
        return 100.0
    return 100.0 * (1 - (cv - 0.10) / (0.60 - 0.10))


# ---------------------------------------------------------------------------
# Signal Channel 2: Syntactic Tree Depth Variance
# ---------------------------------------------------------------------------
def get_tree_depth(token) -> int:
    """Recursively compute maximum depth of a dependency subtree."""
    if token is None:
        return 0
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(get_tree_depth(c) for c in children)


def calculate_syntactic_variance(text: str) -> float:
    """
    Score 0–100. AI produces uniform medium-depth trees (std ≈ 0.5–1.0).
    Humans shift between flat (depth 2–3) and deeply nested (depth 10+).
    """
    sp = _get_nlp()
    if not sp:
        return 50.0

    doc = sp(text)
    depths = []
    for sent in doc.sents:
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        if root:
            depths.append(get_tree_depth(root))

    if len(depths) < 3:
        return 30.0

    std_dev = float(np.std(depths))
    depth_range = max(depths) - min(depths)

    # Both variance and range matter
    variance_score = 0.0
    if std_dev <= 0.8:
        variance_score = 100.0
    elif std_dev >= 3.5:
        variance_score = 0.0
    else:
        variance_score = 100.0 * (1 - (std_dev - 0.8) / (3.5 - 0.8))

    range_score = 0.0
    if depth_range <= 2:
        range_score = 100.0
    elif depth_range >= 8:
        range_score = 0.0
    else:
        range_score = 100.0 * (1 - (depth_range - 2) / (8 - 2))

    return 0.6 * variance_score + 0.4 * range_score


# ---------------------------------------------------------------------------
# Signal Channel 3: AI Signature Phrase Density
# ---------------------------------------------------------------------------
def calculate_phrase_score(text: str) -> float:
    """Score 0–100 based on density and strength of AI-marker phrases."""
    text_lower = text.lower()
    word_count = max(len(word_tokenize(text)), 1)

    weighted_matches = 0
    for phrase in AI_SIGNATURE_PHRASES:
        count = text_lower.count(phrase)
        if count > 0:
            weight = 2.0 if phrase in AI_STRONG_SIGNALS else 1.0
            weighted_matches += count * weight

    # Normalize by text length, then scale
    density = (weighted_matches / word_count) * 100
    return min(density * 15, 100.0)


# ---------------------------------------------------------------------------
# Signal Channel 4: Perplexity Proxy (Brown corpus bigram model)
# ---------------------------------------------------------------------------
def calculate_perplexity_proxy(words: List[str]) -> float:
    """
    Higher score = more predictable text (AI-like).
    Uses conditional probability P(w_i | w_{i-1}) from Brown corpus bigrams.
    AI text selects high-probability next-tokens → low perplexity → high score.
    """
    if not words or not WORD_FREQ:
        return 50.0

    total_vocab = len(WORD_FREQ)
    log_probs = []

    for i in range(1, len(words)):
        w_prev = words[i - 1].lower()
        w_curr = words[i].lower()
        bigram_key = f"{w_prev}|{w_curr}"

        bigram_count = BIGRAM_FREQ.get(bigram_key, 0)
        unigram_count = WORD_FREQ.get(w_prev, 0)

        if bigram_count > 0 and unigram_count > 0:
            cond_prob = bigram_count / unigram_count
        else:
            freq = WORD_FREQ.get(w_curr, 0)
            cond_prob = (freq + 1) / (TOTAL_BROWN_WORDS + total_vocab)

        log_probs.append(np.log(cond_prob + 1e-10))

    if not log_probs:
        return 50.0

    avg_log_prob = float(np.mean(log_probs))

    # Map avg_log_prob to 0-100 score
    # More negative = more surprising = more human-like
    # Less negative = more predictable = more AI-like
    score = 100.0 * (1 - (avg_log_prob - (-4)) / ((-10) - (-4)))
    return float(max(0.0, min(100.0, score)))


# ---------------------------------------------------------------------------
# Signal Channel 5: MATTR (Moving Average Type-Token Ratio)
# ---------------------------------------------------------------------------
def calculate_mattr(words: List[str], window: int = 50) -> float:
    """
    Score 0–100.  AI clusters at MATTR 0.65–0.75 (medium diversity).
    Human writing is either very repetitive (narrative) or very diverse (technical).
    """
    if len(words) < window:
        if not words:
            return 0.0
        ttr = len(set(words)) / len(words)
        return 100.0 if 0.65 <= ttr <= 0.75 else 30.0

    ttrs = []
    for i in range(len(words) - window + 1):
        chunk = words[i:i + window]
        ttrs.append(len(set(chunk)) / window)

    avg_mattr = float(np.mean(ttrs))

    if 0.65 <= avg_mattr <= 0.75:
        return 100.0
    elif avg_mattr < 0.65:
        return max(0.0, 70.0 - (0.65 - avg_mattr) * 200)
    else:
        return max(0.0, 100.0 - (avg_mattr - 0.75) * 200)


# ---------------------------------------------------------------------------
# Signal Channel 6: Punctuation Uniformity
# ---------------------------------------------------------------------------
def calculate_punctuation_uniformity(sentences: List[str]) -> float:
    """
    Score 0–100.  AI uses commas/semicolons with low variance across sentences.
    Humans are erratic — some sentences have 0 commas, others have 5+.
    """
    if len(sentences) < 3:
        return 0.0

    # Comma variance
    comma_counts = [s.count(",") for s in sentences]
    comma_std = float(np.std(comma_counts))

    # Semicolon presence (AI rarely uses semicolons)
    semi_count = sum(s.count(";") for s in sentences)
    dash_count = sum(s.count("—") for s in sentences)

    # Low comma variance + no semicolons/dashes = AI
    punct_variety_penalty = 0
    if semi_count > 0:
        punct_variety_penalty += 15
    if dash_count > 0:
        punct_variety_penalty += 15

    if comma_std <= 0.3:
        base = 100.0
    elif comma_std >= 2.0:
        base = 0.0
    else:
        base = 100.0 * (1 - (comma_std - 0.3) / (2.0 - 0.3))

    return max(0.0, base - punct_variety_penalty)


# ---------------------------------------------------------------------------
# Signal Channel 7: Sentence-Opener Entropy
# ---------------------------------------------------------------------------
def calculate_opener_entropy(sentences: List[str]) -> float:
    """
    Score 0–100.  AI starts sentences with the same patterns (The, This, It, In).
    Human writing has higher opener diversity → lower score.
    """
    if len(sentences) < 4:
        return 30.0

    openers = []
    for s in sentences:
        words = s.strip().split()
        if words:
            # Use first word (lowered) as opener
            openers.append(words[0].lower().rstrip(",:;"))

    if not openers:
        return 30.0

    freq = Counter(openers)
    total = len(openers)

    # Shannon entropy
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Max entropy = log2(n) for n unique openers
    max_entropy = math.log2(total) if total > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Low entropy = repetitive openers = AI-like
    if normalized_entropy >= 0.85:
        return 0.0
    if normalized_entropy <= 0.30:
        return 100.0
    return 100.0 * (1 - (normalized_entropy - 0.30) / (0.85 - 0.30))


# ---------------------------------------------------------------------------
# Signal Channel 8: Clause-to-Sentence Ratio
# ---------------------------------------------------------------------------
def calculate_clause_ratio(text: str) -> float:
    """
    Score 0–100.  AI generates uniform clause density per sentence.
    Humans vary wildly — some simple sentences, some heavily subordinated.
    """
    sp = _get_nlp()
    if not sp:
        return 50.0

    doc = sp(text)
    clause_counts = []

    for sent in doc.sents:
        subordinate_markers = sum(
            1 for t in sent
            if t.dep_ in ("advcl", "relcl", "ccomp", "xcomp", "acl")
        )
        clause_counts.append(subordinate_markers)

    if len(clause_counts) < 3:
        return 30.0

    std_dev = float(np.std(clause_counts))
    mean_val = float(np.mean(clause_counts))

    # Very uniform clause density = AI
    if std_dev <= 0.3:
        return 90.0
    elif std_dev >= 2.0:
        return 10.0
    else:
        return 90.0 * (1 - (std_dev - 0.3) / (2.0 - 0.3))


# ---------------------------------------------------------------------------
# Signal Channel 9: Discourse Connector Density
# ---------------------------------------------------------------------------
def calculate_connector_density(text: str) -> float:
    """
    Score 0–100.  AI heavily over-uses discourse connectors.
    Humans use them sparingly and inconsistently.
    """
    connectors = {
        "however", "moreover", "furthermore", "additionally",
        "nevertheless", "consequently", "therefore", "thus",
        "hence", "accordingly", "meanwhile", "subsequently",
        "nonetheless", "conversely", "similarly", "likewise",
        "indeed", "certainly", "undoubtedly", "notably",
    }
    words = word_tokenize(text.lower())
    word_count = max(len(words), 1)
    connector_count = sum(1 for w in words if w in connectors)

    density = connector_count / word_count

    # AI typically has density > 0.02 (1 connector per 50 words)
    # Humans typically < 0.01 (1 per 100 words)
    if density >= 0.03:
        return 100.0
    if density <= 0.005:
        return 0.0
    return 100.0 * (density - 0.005) / (0.03 - 0.005)


# ===========================================================================
# Sigmoid Calibration (enterprise-grade curve)
# ===========================================================================
def _calibrate(raw_score: float) -> float:
    """
    Apply steep sigmoid logistic curve to convert raw weighted score
    to a 0–100 confidence score. Mimics enterprise detector thresholds.

    The curve ensures:
    - Unambiguous AI text → spikes to 95–100%
    - Clean human text → drops below 15%
    - Ambiguous middle → steep transition zone around 40%
    """
    k = 0.18    # Steepness of transition
    x0 = 40.0   # Midpoint of the sigmoid

    if raw_score <= 0:
        return 0.0
    if raw_score >= 100:
        return 100.0

    try:
        sigmoid = 100.0 / (1.0 + math.exp(-k * (raw_score - x0)))
    except OverflowError:
        return 0.0

    if sigmoid > 96.0:
        return 100.0
    if sigmoid < 12.0:
        return 0.0

    return round(sigmoid, 1)


# ===========================================================================
# Segment-based scoring (250-word chunks, matching Turnitin)
# ===========================================================================
def _split_into_segments(text: str, target_words: int = 250) -> List[str]:
    """Split text into ~250-word segments on sentence boundaries."""
    sentences = sent_tokenize(text)
    segments: List[str] = []
    current_segment: List[str] = []
    current_word_count = 0

    for sent in sentences:
        sent_words = len(word_tokenize(sent))
        if current_word_count + sent_words > target_words and current_segment:
            segments.append(" ".join(current_segment))
            current_segment = [sent]
            current_word_count = sent_words
        else:
            current_segment.append(sent)
            current_word_count += sent_words

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments


def score_segment(text: str) -> float:
    """
    Score a single ~250-word segment using all 9 signal channels.
    Returns a calibrated 0–100 score.
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    if not sentences or len(words) < 10:
        return 0.0

    # Compute all 9 signal channels
    s1 = calculate_burstiness(sentences)
    s2 = calculate_syntactic_variance(text)
    s3 = calculate_phrase_score(text)
    s4 = calculate_perplexity_proxy(words)
    s5 = calculate_mattr(words)
    s6 = calculate_punctuation_uniformity(sentences)
    s7 = calculate_opener_entropy(sentences)
    s8 = calculate_clause_ratio(text)
    s9 = calculate_connector_density(text)

    # Weighted fusion — phrase score and syntactic variance are strongest signals
    raw = (
        s1 * 0.08 +    # Burstiness
        s2 * 0.22 +    # Syntactic variance (strong signal)
        s3 * 0.22 +    # Phrase density (strong signal)
        s4 * 0.18 +    # Perplexity proxy
        s5 * 0.05 +    # MATTR
        s6 * 0.05 +    # Punctuation uniformity
        s7 * 0.08 +    # Opener entropy
        s8 * 0.06 +    # Clause ratio
        s9 * 0.06      # Connector density
    )

    return _calibrate(raw)


# ===========================================================================
# Per-sentence scoring (for UI heatmaps)
# ===========================================================================
def score_sentences(text: str) -> List[Dict[str, Any]]:
    """Compute per-sentence AI probability for heatmap visualization."""
    sentences = sent_tokenize(text)
    scored = []

    for i, sent in enumerate(sentences):
        words = word_tokenize(sent.lower())
        if len(words) < 4:
            score = 0.0
        else:
            p_score = calculate_phrase_score(sent) * 0.40
            perplex = calculate_perplexity_proxy(words) * 0.35
            conn = calculate_connector_density(sent) * 0.25

            raw = p_score + perplex + conn

            k = 0.22
            x0 = 42.0
            try:
                score = 100.0 / (1.0 + math.exp(-k * (raw - x0)))
            except OverflowError:
                score = 0.0

        scored.append({
            "index": i,
            "text": sent,
            "score": float(min(100.0, max(0.0, score))),
        })

    return scored


# ===========================================================================
# Main detection aggregator
# ===========================================================================
def detect_ai_score(text: str) -> float:
    """
    Main detection entry point. Uses Turnitin-style Fragment Toxicity:

    1. Split into ~250-word segments
    2. Score each segment independently
    3. Aggregate with a toxicity multiplier that penalizes
       documents where even a portion is clearly AI-generated

    This catches "sandwich" attacks where human text wraps AI paragraphs.
    """
    words = word_tokenize(text)
    if len(words) < 20:
        return score_segment(text)

    segments = _split_into_segments(text, target_words=250)
    if not segments:
        return 0.0

    scores_by_length = []
    for seg in segments:
        seg_words = len(word_tokenize(seg))
        seg_score = score_segment(seg)
        scores_by_length.append((seg_score, seg_words))

    total_words = sum(w for _, w in scores_by_length)
    if total_words == 0:
        return 0.0

    # Sort by score descending (most AI-like first)
    scores_by_length.sort(key=lambda x: x[0], reverse=True)

    flagged_ai_words = 0
    raw_weighted_sum = 0.0

    for s_score, s_words in scores_by_length:
        raw_weighted_sum += s_score * s_words
        if s_score > 70.0:
            flagged_ai_words += s_words

    base_average = raw_weighted_sum / total_words

    # Fragment Toxicity: even a small portion of clearly-AI text
    # disproportionately raises the overall score
    toxicity_ratio = flagged_ai_words / total_words

    if toxicity_ratio > 0.05:
        boost = 100.0 * (toxicity_ratio ** 0.5)
        final_score = base_average + (boost * 0.5)
    else:
        final_score = base_average

    return float(min(100.0, max(0.0, final_score)))
