"""
Adversarial passes — punctuation manipulation, zero-width character injection,
invisible padding, whitespace jitter.
"""

import re
from typing import Tuple
from nltk.tokenize import sent_tokenize
from app.core.passes.utils import get_rng


def pass_adversarial_punctuation(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Upgrade commas to semicolons in compound lists and
    enclose parenthetical appositives in em-dashes (—).
    """
    new_text = text
    changes = 0

    # Compound list commas → semicolons
    if get_rng().random() < (0.4 * intensity):
        pattern = re.compile(r"([^,;]+),\s*([^,;]+),\s*and\s+([^,;.]+)")
        new_text, count = pattern.subn(r"\1; \2; and \3", text)
        changes += count

    # Parenthetical appositives → em-dashes
    pattern2 = re.compile(r",\s*(which|who|that|namely)\s+([^,.]+),")
    new_text, count2 = pattern2.subn(r" — \1 \2 — ", new_text)
    changes += count2

    return new_text, changes


def pass_zwj_jitter(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Inject ZWNJ (\\u200C) inside words that are AI triggers."""
    triggers = [
        "furthermore", "moreover", "additionally", "consequently",
        "therefore", "essential", "crucial",
    ]
    new_text, changes = text, 0
    for t in triggers:
        if t in new_text.lower() and get_rng().random() < intensity:
            mid = len(t) // 2
            jittered = t[:mid] + "\u200C" + t[mid:]
            new_text = re.sub(rf"\b{t}\b", jittered, new_text, flags=re.IGNORECASE)
            changes += 1
    return new_text, changes


def pass_invisible_padding(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Use Hair Spaces (U+200A) between words in common bigrams."""
    common_bigrams = [
        ("of", "the"), ("in", "the"), ("to", "the"),
        ("on", "the"), ("and", "the"),
    ]
    new_text, changes = text, 0
    for b1, b2 in common_bigrams:
        pattern = re.compile(rf"\b{b1}\s+{b2}\b", re.IGNORECASE)
        if get_rng().random() < intensity:
            new_text = pattern.sub(f"{b1}\u200A{b2}", new_text)
            changes += 1
    return new_text, changes


def pass_whitespace_jitter(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Disrupt bigram/trigram analysis with non-standard spaces."""
    new_chars, changes = [], 0
    for char in text:
        if char == " " and get_rng().random() < (0.05 * intensity):
            jitter_type = get_rng().choice(["hair", "thin"])
            new_chars.append("\u200A" if jitter_type == "hair" else "\u2009")
            changes += 1
        else:
            new_chars.append(char)
    return "".join(new_chars), changes


def pass_punctuation_personality(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Replace some commas with em-dashes or semicolons."""
    new_sentences, changes = [], 0
    for sent in sent_tokenize(text):
        chance = 1.0 if intensity >= 1.0 else (0.3 * intensity)
        if "," in sent and get_rng().random() < chance:
            choices = [" — ", "; "]
            sent = sent.replace(",", get_rng().choice(choices), 1)
            changes += 1
        new_sentences.append(sent)
    return " ".join(new_sentences), changes
