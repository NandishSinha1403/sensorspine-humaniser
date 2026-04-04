import re
import random
import numpy as np
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Dict, Any, Optional, List

logger = logging.getLogger("humaniser.voice")

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

FORMAL_WORDS = [
    "furthermore", "nevertheless", "consequently", "subsequently", "analytical",
    "comprehensive", "methodology", "empirical", "theoretical", "jurisprudence",
    "fundamental", "significant", "substantial", "approximately", "necessitate",
]

SIMPLE_CONNECTORS_RE = re.compile(r'\b(and|but|so|also|yet)\b', re.IGNORECASE)
FORMAL_CONNECTORS_RE = re.compile(r'\b(however|therefore|consequently|furthermore|moreover|nevertheless|additionally)\b', re.IGNORECASE)


def extract_voice(text: str) -> Dict[str, Any]:
    """
    Extract a comprehensive 'voice fingerprint' from the original text.
    This is captured BEFORE humanization so we can correct drift afterwards.
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    if not sentences:
        return {}

    lengths = [len(word_tokenize(s)) for s in sentences]

    # Openers
    openers: List[str] = []
    for s in sentences:
        s_words = word_tokenize(s)
        if s_words:
            openers.append(s_words[0].lower())

    from collections import Counter
    fav_openers = [w for w, c in Counter(openers).most_common(5)]

    # Connector style
    simple_connectors = len(SIMPLE_CONNECTORS_RE.findall(text))
    formal_connectors = len(FORMAL_CONNECTORS_RE.findall(text))

    # Punctuation
    commas = text.count(",")
    semicolons = text.count(";")
    questions = text.count("?")
    parens = text.count("(")
    dashes = text.count("—") + text.count("–")
    total_chars = len(text) or 1

    # Formality: ratio of latinate/long words to total words
    formal_count = sum(1 for w in words if w in FORMAL_WORDS or len(w) > 10)

    # Person detection
    i_count = len(re.findall(r'\b(i|my|me)\b', text.lower()))
    we_count = len(re.findall(r'\b(we|our|us)\b', text.lower()))
    person = "third"
    if i_count > we_count and i_count > 0:
        person = "first"
    elif we_count > 0:
        person = "first_plural"

    return {
        "preferred_sentence_length": float(np.mean(lengths)),
        "length_variance": float(np.std(lengths)),
        "favorite_openers": fav_openers,
        "connector_ratio": formal_connectors / max(simple_connectors, 1),
        "punctuation_habits": {
            "comma_rate": commas / total_chars * 1000,
            "semicolon_rate": semicolons / total_chars * 1000,
            "question_rate": questions / total_chars * 1000,
            "paren_rate": parens / total_chars * 1000,
            "dash_rate": dashes / total_chars * 1000,
        },
        "formality_score": formal_count / max(len(words), 1),
        "person": person,
    }


def apply_voice(text: str, voice_profile: Dict[str, Any]) -> str:
    """
    Apply voice drift correction to ensure the humanized text still
    sounds like the original author and hasn't drifted too far.

    Corrections applied:
    1. Person preservation (I/we/third)
    2. Formality capping (if original was informal, trim formal replacements)
    3. Connector ratio re-balancing
    4. Sentence length drift correction
    """
    if not voice_profile:
        return text

    current_text = text

    # ── 1. Person preservation ──────────────────────────────────────────────
    if voice_profile["person"] == "first":
        current_text = re.sub(r'\bone must\b', 'I must', current_text, flags=re.IGNORECASE)
        current_text = re.sub(r'\bone should\b', 'I should', current_text, flags=re.IGNORECASE)
        current_text = re.sub(r'\bthe author\b', 'I', current_text, flags=re.IGNORECASE)
    elif voice_profile["person"] == "first_plural":
        current_text = re.sub(r'\bone must\b', 'we must', current_text, flags=re.IGNORECASE)
        current_text = re.sub(r'\bone should\b', 'we should', current_text, flags=re.IGNORECASE)
        current_text = re.sub(r'\bthe author\b', 'we', current_text, flags=re.IGNORECASE)

    # ── 2. Formality capping ────────────────────────────────────────────────
    # If the original was very informal (formality < 0.05), trim overly formal
    # words that the pipeline may have introduced.
    original_formality = voice_profile.get("formality_score", 0.1)

    # Re-measure formality on the current text
    current_words = word_tokenize(current_text.lower())
    current_formal = sum(1 for w in current_words if w in FORMAL_WORDS or len(w) > 10)
    current_formality = current_formal / max(len(current_words), 1)

    formality_drift = abs(current_formality - original_formality)
    if formality_drift > 0.20 * max(original_formality, 0.01) and original_formality < 0.08:
        # The pipeline made it too formal. Downgrade common formal words.
        formal_to_simple = {
            "Furthermore": "Also", "furthermore": "also",
            "Consequently": "So", "consequently": "so",
            "Nevertheless": "Still", "nevertheless": "still",
            "Moreover": "Also", "moreover": "also",
            "Additionally": "Also", "additionally": "also",
            "Subsequently": "Then", "subsequently": "then",
        }
        for formal, simple in formal_to_simple.items():
            current_text = current_text.replace(formal, simple)
        logger.debug("[Voice] Formality drift corrected (%.3f → capped)", formality_drift)

    # ── 3. Connector ratio re-balancing ─────────────────────────────────────
    original_ratio = voice_profile.get("connector_ratio", 0.5)
    current_simple = len(SIMPLE_CONNECTORS_RE.findall(current_text))
    current_formal_count = len(FORMAL_CONNECTORS_RE.findall(current_text))
    current_ratio = current_formal_count / max(current_simple, 1)

    # If connector style has shifted dramatically (ratio changed > 100%)
    if original_ratio < 0.5 and current_ratio > 1.0:
        # Original was simple-heavy, pipeline made it formal-heavy. Revert some.
        # Only revert up to half the excess formal connectors
        excess = int((current_ratio - original_ratio) * current_simple * 0.5)
        revert_map = [
            (r'\bHowever,', 'But'), (r'\bhowever,', 'but'),
            (r'\bTherefore,', 'So'), (r'\btherefore,', 'so'),
        ]
        reverted = 0
        for pattern, replacement in revert_map:
            if reverted >= excess:
                break
            current_text, n = re.subn(pattern, replacement, current_text, count=1)
            reverted += n
        if reverted > 0:
            logger.debug("[Voice] Connector ratio corrected: reverted %d formal connectors", reverted)

    # ── 4. Sentence length drift correction ─────────────────────────────────
    target_length = voice_profile.get("preferred_sentence_length", 18)
    target_variance = voice_profile.get("length_variance", 8)

    sentences = sent_tokenize(current_text)
    if len(sentences) >= 3:
        current_lengths = [len(word_tokenize(s)) for s in sentences]
        current_mean = float(np.mean(current_lengths))

        # If pipeline pushed sentences >30% longer than author's preference, shorten some
        if current_mean > target_length * 1.3 and target_length < 25:
            new_sentences = []
            sp = _get_nlp()
            for s in sentences:
                s_words = word_tokenize(s)
                split_done = False
                if len(s_words) > target_length * 1.5 and len(s_words) > 20 and sp:
                    doc = sp(s)
                    split_token = None
                    # Search for root-level coordinating conjunction
                    for token in doc:
                        if token.pos_ == "CCONJ" and token.head.dep_ == "ROOT" and token.text.strip().lower() in ["and", "but", "so", "yet", "or"]:
                            split_token = token
                            break
                    if split_token:
                        p1_text = "".join([t.text_with_ws for t in doc if t.i < split_token.i]).strip()
                        p2_text = "".join([t.text_with_ws for t in doc if t.i > split_token.i]).strip()
                        if p1_text and p2_text:
                            part1 = p1_text.rstrip(",; ") + "."
                            part2 = p2_text.capitalize()
                            new_sentences.append(part1)
                            new_sentences.append(part2)
                            split_done = True
                
                if not split_done:
                    new_sentences.append(s)
            current_text = " ".join(new_sentences)

    return current_text
