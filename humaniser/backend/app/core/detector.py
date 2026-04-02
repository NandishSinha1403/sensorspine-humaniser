import numpy as np
import re
import logging
from typing import List, Dict, Any, Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import brown

logger = logging.getLogger("humaniser.detector")

# ---------------------------------------------------------------------------
# Load Brown corpus frequency distribution for the perplexity proxy
# ---------------------------------------------------------------------------
WORD_FREQ: Dict = {}
BIGRAM_FREQ: Dict = {}

def _load_brown_corpus():
    """Load Brown corpus word and bigram frequencies at module level."""
    global WORD_FREQ, BIGRAM_FREQ
    try:
        words = [w.lower() for w in brown.words() if w.isalnum()]
        WORD_FREQ = FreqDist(words)
        # Build bigram frequencies for improved perplexity estimation
        for i in range(len(words) - 1):
            pair = (words[i], words[i + 1])
            BIGRAM_FREQ[pair] = BIGRAM_FREQ.get(pair, 0) + 1
        logger.info("[Detector] Brown corpus loaded: %d unigrams, %d bigrams", len(WORD_FREQ), len(BIGRAM_FREQ))
    except LookupError:
        logger.warning(
            "[Detector] Brown corpus not available. Perplexity proxy will be degraded. "
            "Run: python -c \"import nltk; nltk.download('brown')\""
        )

_load_brown_corpus()

# ---------------------------------------------------------------------------
# AI Signature Phrase list
# ---------------------------------------------------------------------------
AI_SIGNATURE_PHRASES = [
    "furthermore", "additionally", "it is worth noting", "it is important to note",
    "in conclusion", "to summarize", "notably", "this demonstrates",
    "plays a crucial role", "has been shown", "research has shown",
    "it is essential", "in order to", "due to the fact", "in light of",
    "it should be noted", "delve into", "firstly", "secondly", "thirdly",
    "in today's world", "in recent years", "has become increasingly",
    "needless to say", "it goes without saying", "at the end of the day",
    "all things considered", "as mentioned earlier", "it can be argued",
    "there is no doubt", "without a doubt", "taking everything into consideration",
    "one must consider", "it is clear that", "it is undeniable", "this highlights",
    "this underscores", "this emphasizes", "shed light on", "it is imperative",
    "a wide range of", "a variety of", "in the realm of", "the fact that",
    "in terms of", "with regard to", "when it comes to",
]

# ---------------------------------------------------------------------------
# Individual signal calculators
# ---------------------------------------------------------------------------

def calculate_burstiness(sentences: List[str]) -> float:
    """Score 0-100 where 100 = low variance (AI-like), 0 = high variance (human-like)."""
    if len(sentences) < 2:
        return 0.0
    lengths = [len(word_tokenize(s)) for s in sentences]
    std_dev = float(np.std(lengths))

    # Human benchmark: Std Dev 8-15.  AI benchmark: 2-5.
    if std_dev >= 15:
        return 0.0
    if std_dev <= 2:
        return 100.0
    # Linear mapping: 2 → 100, 15 → 0
    return 100.0 * (1 - (std_dev - 2) / (15 - 2))


def calculate_phrase_score(text: str) -> float:
    """Score 0-100 based on density of AI-signature phrases."""
    text_lower = text.lower()
    matches = 0
    for phrase in AI_SIGNATURE_PHRASES:
        if phrase in text_lower:
            matches += text_lower.count(phrase)

    words = word_tokenize(text)
    word_count = len(words) if words else 1

    # Heuristic: 1 phrase per 100 words is significant
    phrase_density = (matches / word_count) * 100
    return min(phrase_density * 20, 100.0)


def calculate_mattr(words: List[str], window: int = 50) -> float:
    """Score 0-100.  AI clusters at MATTR 0.65-0.75 → score 100."""
    if len(words) < window:
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    ttrs = []
    for i in range(len(words) - window + 1):
        chunk = words[i : i + window]
        ttrs.append(len(set(chunk)) / window)

    avg_mattr = float(np.mean(ttrs))

    # AI clusters around 0.65-0.75.
    if 0.65 <= avg_mattr <= 0.75:
        return 100.0
    elif avg_mattr < 0.65:
        return 70.0  # Very repetitive
    else:
        return max(0.0, 100.0 - (avg_mattr - 0.75) * 200)


def calculate_perplexity_proxy(words: List[str]) -> float:
    """
    Improved perplexity proxy using unigram + bigram conditional probability.
    AI text uses the most predictable (highest probability) next words.
    High avg predictability → high score (AI-like).
    """
    if not words or not WORD_FREQ:
        return 50.0  # neutral fallback when corpus unavailable

    total_brown_words = WORD_FREQ.N() if hasattr(WORD_FREQ, 'N') else sum(WORD_FREQ.values())
    if total_brown_words == 0:
        return 50.0

    log_probs = []
    for i in range(1, len(words)):
        w_prev = words[i - 1].lower()
        w_curr = words[i].lower()
        bigram = (w_prev, w_curr)

        bigram_count = BIGRAM_FREQ.get(bigram, 0)
        unigram_count = WORD_FREQ.get(w_prev, 0)

        if bigram_count > 0 and unigram_count > 0:
            # Conditional probability: P(w_curr | w_prev)
            cond_prob = bigram_count / unigram_count
        else:
            # Fall back to unigram probability with Laplace smoothing
            freq = WORD_FREQ.get(w_curr, 0)
            cond_prob = (freq + 1) / (total_brown_words + len(WORD_FREQ))

        log_probs.append(np.log(cond_prob + 1e-10))

    if not log_probs:
        return 50.0

    # Average log probability. Higher (closer to 0) = more predictable = more AI-like
    avg_log_prob = float(np.mean(log_probs))

    # Map to 0-100. Typical ranges observed:
    #   Very predictable (AI): avg_log_prob ~ -4 to -6
    #   Less predictable (human): avg_log_prob ~ -7 to -10
    # Linear mapping: -4 → 100, -10 → 0
    score = 100.0 * (1 - (avg_log_prob - (-4)) / ((-10) - (-4)))
    return float(max(0.0, min(100.0, score)))


def calculate_punctuation_uniformity(sentences: List[str]) -> float:
    """Score 0-100. Low comma variance = AI-like (100)."""
    if len(sentences) < 2:
        return 0.0
    comma_counts = [s.count(",") for s in sentences]
    std_dev = float(np.std(comma_counts))

    if std_dev <= 0.2:
        return 100.0
    return max(0.0, 100.0 - (std_dev * 40))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _calibrate(raw_score: float) -> float:
    """
    Apply calibration curve to match aggressive detector scoring.
    High-Risk Boost: scores > 60 are multiplied by 1.3 (max 99).
    Low-Risk Reduction: scores < 25 are multiplied by 0.7.
    """
    if raw_score > 60:
        return min(99.0, raw_score * 1.3)
    elif raw_score < 25:
        return raw_score * 0.7
    return raw_score


# ---------------------------------------------------------------------------
# Segment-based scoring (matching Turnitin's 250-word approach)
# ---------------------------------------------------------------------------

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
    """Score a single text segment (ideally ~250 words) using 5 weighted signals."""
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    if not sentences:
        return 0.0

    s1 = calculate_burstiness(sentences) * 0.40
    s2 = calculate_phrase_score(text) * 0.35
    s3 = calculate_mattr(words) * 0.15
    s4 = calculate_perplexity_proxy(words) * 0.07
    s5 = calculate_punctuation_uniformity(sentences) * 0.03

    raw = s1 + s2 + s3 + s4 + s5
    return _calibrate(raw)


def score_sentences(text: str) -> List[Dict[str, Any]]:
    """Score individual sentences for heatmap display."""
    sentences = sent_tokenize(text)
    scored_sentences = []

    for i, sent in enumerate(sentences):
        words = word_tokenize(sent.lower())
        if len(words) < 3:
            score = 0
        else:
            s1 = calculate_phrase_score(sent) * 0.5
            s2 = calculate_mattr(words, window=len(words)) * 0.3
            word_lengths = [len(w) for w in words if w.isalnum()]
            length_var = float(np.std(word_lengths)) if word_lengths else 0
            s3 = max(0, 100 - (length_var * 40)) * 0.2
            score = s1 + s2 + s3

        scored_sentences.append({
            "index": i,
            "text": sent,
            "score": float(min(100, score)),
        })

    return scored_sentences


def detect_ai_score(text: str) -> float:
    """
    Main detection function. Uses 250-word segment scoring
    (matching how Turnitin actually processes documents).
    """
    words = word_tokenize(text)
    if len(words) < 20:
        # Very short text: score directly
        return score_segment(text)

    segments = _split_into_segments(text, target_words=250)
    if not segments:
        return 0.0

    # Weight each segment by its word count (longer segments contribute more)
    total_words = 0
    weighted_sum = 0.0
    for seg in segments:
        seg_words = len(word_tokenize(seg))
        seg_score = score_segment(seg)
        weighted_sum += seg_score * seg_words
        total_words += seg_words

    if total_words == 0:
        return 0.0

    return float(weighted_sum / total_words)
