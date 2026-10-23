"""
Rhythm passes — burstiness enforcement, rhythm sculpting, and paragraph wave.
"""

import re
import numpy as np
from typing import Tuple, Optional, Dict, Any
from nltk.tokenize import sent_tokenize, word_tokenize
from app.core.passes.utils import (
    get_rng, get_used_phrases, load_spacy, safe_split_sentence, is_valid_sentence,
)


def pass_burstiness(text: str, profile: Optional[Dict[str, Any]], intensity: float = 1.0) -> Tuple[str, int]:
    """Enforce structural extremes: short punchy + long labyrinthine sentences."""
    sp = load_spacy()
    if not sp:
        return text, 0

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    processed_paragraphs = []
    total_changes = 0
    used = get_used_phrases()

    for paragraph in paragraphs:
        doc = sp(paragraph)
        sentences = list(doc.sents)
        if not sentences:
            processed_paragraphs.append(paragraph)
            continue

        p_words = word_tokenize(paragraph)
        p_word_count = len(p_words)
        lengths = [len(word_tokenize(s.text)) for s in sentences]
        has_very_short = any(l < 5 for l in lengths)
        has_very_long = any(l > 35 for l in lengths)
        new_sentences = [s.text.strip() for s in sentences]
        changes = 0

        if p_word_count > 60:
            # Rule 1 (The Anchor): Force one punchy 3-6 word sentence
            # INSERT a new short sentence rather than splitting an existing one
            if not has_very_short and len(new_sentences) > 1:
                short_anchors = [
                    "This bears emphasis.",
                    "The implications are clear.",
                    "This warrants further study.",
                    "The trend is unmistakable.",
                    "Such evidence accumulates.",
                    "These findings converge.",
                ]
                insert_idx = min(2, len(new_sentences))
                new_sentences.insert(insert_idx, get_rng().choice(short_anchors))
                changes += 1
                has_very_short = True

            # Rule 2 (The Labyrinth): Force one 35+ word hyper-nested sentence
            if not has_very_long and len(new_sentences) >= 2:
                i = 0
                while i < len(new_sentences) - 1:
                    s1, s2 = new_sentences[i], new_sentences[i + 1]
                    s1_words = len(word_tokenize(s1))
                    s2_words = len(word_tokenize(s2))
                    if s1_words + s2_words > 20:  # Both must be substantial
                        connector = get_rng().choice([
                            ", and in this regard, ",
                            ", which further suggests that ",
                            "; in a related vein, ",
                            ", and when considered alongside this, ",
                        ])
                        merged = s1.rstrip(".!?") + connector + s2[0].lower() + s2[1:]
                        new_sentences[i] = merged
                        new_sentences.pop(i + 1)
                        changes += 1
                        if len(word_tokenize(merged)) > 35:
                            has_very_long = True
                            break
                    else:
                        i += 1

        # Qualifier injection for short sentences
        current_sentences = []
        i = 0
        while i < len(new_sentences):
            s_text = new_sentences[i]
            s_len = len(s_text.split())
            if 3 < s_len < 10 and get_rng().random() < 0.5 * intensity:
                qualifiers = [
                    " — at least in the cases examined here",
                    ", though this varies by context",
                    ", a finding consistent with earlier work",
                    ", which remains a point of debate",
                    ", something researchers keep exploring",
                    " — an observation that merits further study",
                ]
                available = [q for q in qualifiers if q.lower() not in used]
                if not available:
                    available = qualifiers
                q = get_rng().choice(available)
                used.add(q.lower())
                s_text = s_text.rstrip('.!?') + q + "."
                changes += 1
            current_sentences.append(s_text)
            i += 1

        processed_paragraphs.append(" ".join(current_sentences))
        total_changes += changes

    return "\n\n".join(processed_paragraphs), total_changes


def pass_rhythm_sculpting(text: str, profile: Optional[Dict[str, Any]], intensity: float = 1.0) -> Tuple[str, int]:
    """DNA-aware rhythm matching — split sentences to match academic variance."""
    sp = load_spacy()
    if not sp:
        return text, 0

    # Load DNA stats if available
    try:
        from app.core.humanizer import ACADEMIC_DNA
        dna_stats = ACADEMIC_DNA.get("sentence_stats", {})
    except (ImportError, AttributeError):
        dna_stats = {}

    target_std = dna_stats.get("std_dev_length", dna_stats.get("std", 9))
    doc = sp(text)
    sentences = list(doc.sents)
    if len(sentences) < 5:
        return text, 0

    lengths = [len(s) for s in sentences]
    current_std = float(np.std(lengths))
    changes = 0
    new_sentences = []

    for s in sentences:
        s_len = len(s)
        if current_std < (target_std - 3) and s_len > 25 and get_rng().random() < intensity:
            split_token = next(
                (t for t in s if t.pos_ == "CCONJ" and t.head.dep_ == "ROOT"), None
            )
            if split_token:
                is_list = False
                for prev in reversed(s[:split_token.i - s.start]):
                    if prev.text.lower() in ("such", "including", "like"):
                        is_list = True
                        break
                    if prev.pos_ == "PUNCT" or prev.i < split_token.i - 5:
                        break
                if is_list:
                    new_sentences.append(s.text.strip())
                    continue

                p1 = [t for t in s if t.i < split_token.i]
                p2 = [t for t in s if t.i >= split_token.i]
                if is_valid_sentence(p1) and is_valid_sentence(p2):
                    new_sentences.append(
                        "".join([t.text_with_ws for t in p1]).strip().rstrip(",; ") + "."
                    )
                    p2_text = "".join([t.text_with_ws for t in p2]).strip()
                    p2_text = re.sub(r'^(and|but|or|yet|so)\s+', '', p2_text, flags=re.IGNORECASE)
                    if p2_text:
                        new_sentences.append(p2_text[0].upper() + p2_text[1:])
                    changes += 1
                    continue
        new_sentences.append(s.text.strip())
    return " ".join(new_sentences), changes


def pass_paragraph_wave(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Enforce structural extremes within paragraphs by merging clauses."""
    paragraphs = text.split("\n")
    new_paragraphs, total_changes = [], 0
    for para in paragraphs:
        if not para.strip():
            new_paragraphs.append(para)
            continue
        sentences = sent_tokenize(para)
        if len(sentences) < 3:
            new_paragraphs.append(para)
            continue
        idx = get_rng().randint(0, len(sentences) - 2)
        s1, s2 = sentences[idx], sentences[idx + 1]
        if len(s1.split()) + len(s2.split()) > 30:
            connectors = ["and", "while", "whereas", "consequently,"]
            conn = get_rng().choice(connectors)
            merged = s1.rstrip(".") + f", {conn} " + s2[0].lower() + s2[1:]
            sentences[idx] = merged
            sentences.pop(idx + 1)
            total_changes += 1
        new_paragraphs.append(" ".join(sentences))
    return "\n".join(new_paragraphs), total_changes
