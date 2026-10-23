"""
Style passes — style overlay, nuance injection, determiner scramble,
epistemic hedging, hedging uncertainty, and cross-referencing.
"""

import re
from typing import Tuple, Optional, Dict, Any
from nltk.tokenize import sent_tokenize, word_tokenize
from app.core.passes.utils import (
    get_rng, get_used_phrases, load_spacy,
    DETERMINER_ALTERNATIVES, TRANSITION_ALTERNATIVES,
)


def pass_style_overlay(text: str, profile: Optional[Dict[str, Any]], intensity: float = 1.0) -> Tuple[str, int]:
    """Apply style overlay using detected voice profile patterns."""
    if not profile:
        return text, 0

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    new_paragraphs = []
    changes = 0
    opening_patterns = profile.get("opening_patterns", [])
    hedging = profile.get("hedging_phrases", [])

    for p in paragraphs:
        sentences = sent_tokenize(p)
        new_sentences = []
        for s_idx, sent in enumerate(sentences):
            if s_idx == 0 and opening_patterns and get_rng().random() < 0.3 * intensity:
                pattern = get_rng().choice(opening_patterns)
                pattern_words = word_tokenize(pattern)[:4]
                sent = " ".join(pattern_words) + ", " + sent[0].lower() + sent[1:]
                changes += 1
            if hedging and s_idx % 5 == 0 and s_idx > 0 and get_rng().random() < 0.4 * intensity:
                phrase = get_rng().choice(hedging)
                sent = phrase.capitalize() + ", " + sent[0].lower() + sent[1:]
                changes += 1
            new_sentences.append(sent)
        new_paragraphs.append(" ".join(new_sentences))

    return "\n\n".join(new_paragraphs), changes


def pass_nuance_injection(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Injects human-like nuance and subtle hedging to disrupt AI's over-confident tone."""
    sentences = sent_tokenize(text)
    new_sentences = []
    changes = 0
    used = get_used_phrases()
    nuances = [
        "one might argue,", "it seems that", "arguably,",
        "to some extent,", "in a sense,", "potentially,",
        "it appears that", "broadly speaking,", "in many respects,",
        "one could suggest that", "seemingly,", "ostensibly,",
        "it would appear that", "for the most part,", "it is possible that",
        "in light of this,", "one may conclude that", "conceivably,",
    ]

    for i, sent in enumerate(sentences):
        if i % 4 == 0 and get_rng().random() < 0.4 * intensity and len(sent.split()) > 10:
            available = [n for n in nuances if n.lower() not in used]
            if not available:
                available = nuances
            nuance = get_rng().choice(available)
            used.add(nuance.lower())
            prefix = nuance.capitalize()
            sent = prefix + " " + sent[0].lower() + sent[1:]
            changes += 1
        new_sentences.append(sent)

    return " ".join(new_sentences), changes


def pass_determiner_scramble(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Scrambles 'The X' openers and varies transition words."""
    sentences = sent_tokenize(text)
    new_sentences = []
    changes = 0

    for i, sent in enumerate(sentences):
        words = sent.split()
        if not words:
            new_sentences.append(sent)
            continue
        first_word = words[0].lower().rstrip(",;:")
        modified = False

        if first_word == "the" and len(words) > 3 and get_rng().random() < 0.35 * intensity:
            alts = DETERMINER_ALTERNATIVES.get("the", [])
            replacement = get_rng().choice(alts)
            words[0] = replacement.capitalize() if words[0][0].isupper() else replacement
            modified = True
            changes += 1

        if first_word in TRANSITION_ALTERNATIVES and get_rng().random() < 0.5 * intensity:
            alts = TRANSITION_ALTERNATIVES[first_word]
            replacement = get_rng().choice(alts)
            had_comma = words[0].endswith(",")
            words[0] = replacement.capitalize() + ("," if had_comma and not replacement.endswith(",") else "")
            modified = True
            changes += 1

        if modified:
            new_sentences.append(" ".join(words))
        else:
            new_sentences.append(sent)

    return " ".join(new_sentences), changes


def pass_epistemic_hedging(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Targeted caution markers on reporting verbs.
    Identifies reporting verbs and injects academic hedges.
    """
    sp = load_spacy()
    if not sp:
        return text, 0
    doc = sp(text)
    new_sentences, changes = [], 0
    reporting_verbs = [
        "suggest", "indicate", "show", "demonstrate", "reveal",
        "argue", "claim", "assert", "propose", "imply",
    ]
    hedges = [
        "tentatively", "arguably", "notably", "potentially", "likely",
        "ostensibly", "plausibly", "conceivably",
    ]

    for sent in doc.sents:
        sent_text = sent.text.strip()
        for token in sent:
            if token.lemma_ in reporting_verbs and get_rng().random() < (0.3 * intensity):
                hedge = get_rng().choice(hedges)
                pattern = re.compile(rf"\b{token.text}\b")
                sent_text = pattern.sub(f"{hedge} {token.text}", sent_text, count=1)
                changes += 1
                break
        new_sentences.append(sent_text)
    return " ".join(new_sentences), changes


def pass_hedging_uncertainty(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Add uncertainty markers to sentences."""
    sentences = sent_tokenize(text)
    new_sentences, changes = [], 0
    hedges = [
        "it seems that", "perhaps", "potentially,", "arguably,",
        "one might suggest that",
    ]

    for sent in sentences:
        chance = 1.0 if intensity >= 1.0 else (0.5 * intensity)
        if get_rng().random() < chance:
            h = get_rng().choice(hedges)
            sent = h.capitalize() + " " + sent[0].lower() + sent[1:]
            changes += 1
        new_sentences.append(sent)
    return " ".join(new_sentences), changes


def pass_cross_referencing(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Inject cross-referencing phrases between paragraphs."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    new_paragraphs, changes = [], 0
    markers = [
        "as previously noted,", "consistent with earlier points,",
        "as we saw before,", "bearing in mind the previous discussion,",
    ]

    for i, p in enumerate(paragraphs):
        sentences = sent_tokenize(p)
        chance = 1.0 if intensity >= 1.0 else (0.4 * intensity)
        if i > 0 and sentences and get_rng().random() < chance:
            m = get_rng().choice(markers)
            sentences[0] = m.capitalize() + " " + sentences[0][0].lower() + sentences[0][1:]
            changes += 1
        new_paragraphs.append(" ".join(sentences))
    return "\n\n".join(new_paragraphs), changes
