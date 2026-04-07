import numpy as np
import re
import math
import logging
import json
import os
from typing import List, Dict, Any, Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

logger = logging.getLogger("humaniser.detector")

_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_md")
        except:
            try:
                _nlp = spacy.load("en_core_web_sm")
            except:
                _nlp = False
    return _nlp

# ---------------------------------------------------------------------------
# Load Brown corpus frequency distribution for the perplexity proxy
# ---------------------------------------------------------------------------
WORD_FREQ: Dict = {}
BIGRAM_FREQ: Dict = {}
TOTAL_BROWN_WORDS: int = 0

def load_brown_corpus_data():
    """Load Brown corpus word and bigram frequencies from pre-computed JSON."""
    global WORD_FREQ, BIGRAM_FREQ, TOTAL_BROWN_WORDS
    try:
        data_path = os.path.join(os.path.dirname(__file__), "brown_frequencies.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                data = json.load(f)
                WORD_FREQ = data["unigrams"]
                BIGRAM_FREQ = data["bigrams"]
                TOTAL_BROWN_WORDS = data["total_words"]
            logger.info("[Detector] Loaded pre-computed frequencies: %d unigrams, %d bigrams", len(WORD_FREQ), len(BIGRAM_FREQ))
            return True
        else:
            # Fallback if file doesn't exist
            from nltk.corpus import brown
            words = [w.lower() for w in brown.words() if w.isalnum()]
            WORD_FREQ = dict(FreqDist(words))
            TOTAL_BROWN_WORDS = len(words)
            BIGRAM_FREQ = {}
            for i in range(len(words) - 1):
                pair = f"{words[i]}|{words[i + 1]}"
                BIGRAM_FREQ[pair] = BIGRAM_FREQ.get(pair, 0) + 1
            logger.info("[Detector] Brown corpus computed: %d unigrams, %d bigrams", len(WORD_FREQ), len(BIGRAM_FREQ))
            return True
    except LookupError:
        logger.warning("[Detector] Brown corpus not yet available. Perplexity proxy will be degraded. ")
        return False
    except Exception as e:
        logger.error(f"[Detector] Unexpected error loading Brown corpus: {e}")
        return False

# Initial attempt (might fail if data not downloaded yet)
load_brown_corpus_data()

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
    "is the process of", "play an important role", "has become essential",
    "not only... but also", "in a way that", "minimizes harm",
    "crucial for development", "as well as", "helps reduce"
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
    return 100.0 * (1 - (std_dev - 2) / (15 - 2))


def get_tree_depth(token) -> int:
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(get_tree_depth(c) for c in children)

def calculate_syntactic_variance(text: str) -> float:
    """
    Score 0-100 based on Dependency Tree depth variance.
    AI generators output perfectly uniform, consistently medium-depth trees.
    Humans heavily shift between super-flat and hyper-nested clauses.
    """
    sp = _get_nlp()
    if not sp:
        return 50.0  # Fallback neutral
    
    doc = sp(text)
    depths = []
    
    for sent in doc.sents:
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        if root:
            depths.append(get_tree_depth(root))
            
    if len(depths) < 2:
        return 0.0
        
    std_dev = float(np.std(depths))
    
    # Typical Human variance: Std Dev > 2.5
    # Typical AI variance: Std Dev < 0.8
    if std_dev >= 3.0:
        return 0.0
    if std_dev <= 0.8:
        return 100.0
        
    return 100.0 * (1 - (std_dev - 0.8) / (3.0 - 0.8))


def calculate_phrase_score(text: str) -> float:
    """Score 0-100 based on density of AI-signature phrases."""
    text_lower = text.lower()
    matches = 0
    for phrase in AI_SIGNATURE_PHRASES:
        if phrase in text_lower:
            matches += text_lower.count(phrase)

    words = word_tokenize(text)
    word_count = len(words) if words else 1

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

    if 0.65 <= avg_mattr <= 0.75:
        return 100.0
    elif avg_mattr < 0.65:
        return 70.0  
    else:
        return max(0.0, 100.0 - (avg_mattr - 0.75) * 200)


def calculate_perplexity_proxy(words: List[str]) -> float:
    """
    Improved perplexity proxy using unigram + bigram conditional probability.
    AI text uses the most predictable (highest probability) next words.
    High avg predictability → high score (AI-like).
    """
    if not words or not WORD_FREQ:
        return 50.0  

    total_vocab_size = len(WORD_FREQ)

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
            # Smoothing: (count + 1) / (total words + vocabulary size)
            cond_prob = (freq + 1) / (TOTAL_BROWN_WORDS + total_vocab_size)

        log_probs.append(np.log(cond_prob + 1e-10))

    if not log_probs:
        return 50.0

    avg_log_prob = float(np.mean(log_probs))

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
# Turnitin Sigmoid Calibration
# ---------------------------------------------------------------------------

def _calibrate(raw_score: float) -> float:
    """
    Apply a steep Sigmoid logistic curve. This simulates Turnitin's "confidence"
    thresholds where unambiguous AI strings spike to 100%, and low threshold text
    flattens below 20%. Matches enterprise detectors mathematically.
    """
    k = 0.20  # Balanced steepness
    x0 = 38.0 # Balanced threshold — sensitive but not noisy
    
    if raw_score <= 0: return 0.0
    if raw_score >= 100: return 100.0
        
    try:
        sigmoid = 100.0 / (1.0 + math.exp(-k * (raw_score - x0)))
    except OverflowError:
        return 0.0
    
    if sigmoid > 96.0: return 100.0
    if sigmoid < 15.0: return 0.0
        
    return sigmoid


# ---------------------------------------------------------------------------
# Segment-based scoring (matching Turnitin's 250-word chunks)
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
    """Score a ~250-word segment using rigorous structural and perplexity NLP arrays."""
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    if not sentences:
        return 0.0

    s1 = calculate_burstiness(sentences) * 0.10          
    s2 = calculate_syntactic_variance(text) * 0.30       
    s3 = calculate_phrase_score(text) * 0.25             
    s4 = calculate_perplexity_proxy(words) * 0.30        
    s5 = calculate_mattr(words) * 0.03                   
    s6 = calculate_punctuation_uniformity(sentences) * 0.02

    raw = s1 + s2 + s3 + s4 + s5 + s6
    return _calibrate(raw)


def score_sentences(text: str) -> List[Dict[str, Any]]:
    """Strict per-sentence perplexity for accurate UI Heatmaps."""
    sentences = sent_tokenize(text)
    scored_sentences = []

    for i, sent in enumerate(sentences):
        words = word_tokenize(sent.lower())
        if len(words) < 4:
            score = 0.0
        else:
            p_score = calculate_phrase_score(sent) * 0.50
            perplex = calculate_perplexity_proxy(words) * 0.50
            
            raw = p_score + perplex
            
            k = 0.25
            x0 = 45.0
            try:
                score = 100.0 / (1.0 + math.exp(-k * (raw - x0)))
            except OverflowError:
                score = 0.0

        scored_sentences.append({
            "index": i,
            "text": sent,
            "score": float(min(100.0, max(0.0, score))),
        })

    return scored_sentences


def detect_ai_score(text: str) -> float:
    """
    Main detection aggregator.
    Mimics Turnitin's Fragment Toxicity system: Highly confident AI slices disproportionately
    increase the document's total cumulative score. Let's an essay written by humans but
    patched blindly with ChatGPT get correctly flagged.
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
        
    total_words = sum(w for s, w in scores_by_length)
    if total_words == 0:
        return 0.0

    scores_by_length.sort(key=lambda x: x[0], reverse=True)
    
    flagged_ai_words = 0
    raw_weighted_sum = 0.0
    
    for s_score, s_words in scores_by_length:
        raw_weighted_sum += s_score * s_words
        if s_score > 75.0:  
            flagged_ai_words += s_words
            
    base_average = raw_weighted_sum / total_words
    
    # Toxicity Multiplier calculations
    toxicity_ratio = flagged_ai_words / total_words
    
    if toxicity_ratio > 0.05: 
        boost = 100.0 * (toxicity_ratio ** 0.5) 
        final_score = base_average + (boost * 0.5)
    else:
        final_score = base_average
        
    return float(min(100.0, max(0.0, final_score)))
