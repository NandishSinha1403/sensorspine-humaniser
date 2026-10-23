"""
Shared utilities for NLP passes — RNG, spaCy loader, sentence validators,
citation protection, and shared constant data.
"""

import random
import hashlib
import re
import os
import logging
import contextvars
import numpy as np
import spacy
from typing import List, Dict, Any, Optional, Tuple
from nltk.corpus import wordnet

logger = logging.getLogger("humaniser.nlp")

# ---------------------------------------------------------------------------
# Context-local RNG & phrase dedup
# ---------------------------------------------------------------------------
rng_var = contextvars.ContextVar('rng', default=random.Random())

def get_rng():
    return rng_var.get()

used_phrases_var = contextvars.ContextVar('used_phrases', default=set())

def get_used_phrases():
    return used_phrases_var.get()

# ---------------------------------------------------------------------------
# spaCy model loader
# ---------------------------------------------------------------------------
_nlp_model = None

def load_spacy():
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_md")
        except OSError:
            try:
                _nlp_model = spacy.load("en_core_web_sm")
            except Exception:
                _nlp_model = False
        except Exception:
            _nlp_model = False
    if _nlp_model is False:
        logger.error("[SpaCy] Model not available.")
        return None
    return _nlp_model

# ---------------------------------------------------------------------------
# Seed generator
# ---------------------------------------------------------------------------
def get_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)

# ---------------------------------------------------------------------------
# Sentence validation
# ---------------------------------------------------------------------------
def is_valid_sentence(tokens: List[Any]) -> bool:
    """
    Ensure a sentence fragment is truly self-standing.
    Must have: a FINITE verb (not just a participle), a subject, and 6+ tokens.
    """
    if not tokens or len(tokens) < 6:
        return False
    has_finite_verb = False
    has_subj = False
    for t in tokens:
        # Must be a finite verb (VB, VBD, VBP, VBZ) — not a participle acting as modifier
        if t.pos_ in ("VERB", "AUX") and t.dep_ in ("ROOT", "conj", "ccomp", "advcl", "relcl"):
            if t.tag_ not in ("VBG", "VBN") or t.dep_ == "ROOT":
                has_finite_verb = True
        if t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl"):
            has_subj = True
    return has_finite_verb and has_subj

def safe_split_sentence(sent_doc):
    """Splits a sentence doc into two parts at a safe point to avoid artifacts like 'Artificial. Intelligence'."""
    if not sent_doc or len(sent_doc) < 6:
        return sent_doc.text if sent_doc else "", ""

    best_idx = -1
    for token in sent_doc:
        idx = token.i - sent_doc[0].i
        if 2 < idx < len(sent_doc) - 3:
            if token.text == ",":
                best_idx = idx + 1
                break
            if token.pos_ == "CCONJ":
                best_idx = idx
                break

    if best_idx == -1:
        mid = len(sent_doc) // 2
        best_idx = mid
        for i in range(mid, min(mid + 3, len(sent_doc) - 2)):
            t1, t2 = sent_doc[i], sent_doc[i + 1]
            if t1.pos_ == "ADJ" and t2.pos_ in ("NOUN", "PROPN"):
                continue
            if t1.pos_ in ("NOUN", "PROPN") and t2.pos_ in ("NOUN", "PROPN"):
                continue
            if t1.dep_ == "det":
                continue
            best_idx = i + 1
            break

    part1 = "".join([t.text_with_ws for t in sent_doc[:best_idx]]).strip().rstrip(",; ") + "."
    part2 = "".join([t.text_with_ws for t in sent_doc[best_idx:]]).strip()
    if part2:
        part2 = re.sub(r'^(and|but|or|yet|so)\s+', '', part2, flags=re.IGNORECASE)
        part2 = part2[0].upper() + part2[1:] if part2 else ""
    return part1, part2

def extract_primary_subject(text: str) -> str:
    sp = load_spacy()
    if not sp:
        return "the subject matter"
    doc = sp(text)
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.dep_ == "ROOT":
            subtree = [t.text for t in token.subtree if t.i <= token.i]
            return " ".join(subtree).lower()
    return "this system"

def fix_caps(m):
    return m.group(1) + m.group(2).upper()

# ---------------------------------------------------------------------------
# Citation protection
# ---------------------------------------------------------------------------
CITATION_PATTERNS = [
    r'\[\d+(?:,\s*\d+)*\]',
    r'\(\w+(?:(?:\s+et\s+al\.?)?,\s*\d{4})?\)',
]

def protect_citations(text: str) -> Tuple[str, Dict[str, str]]:
    citations: Dict[str, str] = {}
    protected_text = text
    for pattern in CITATION_PATTERNS:
        matches = re.findall(pattern, protected_text)
        for match in set(matches):
            token = f"XCIT{len(citations)}REF"
            citations[token] = match
            protected_text = protected_text.replace(match, token)
    return protected_text, citations

def restore_citations(text: str, citations: Dict[str, str]) -> str:
    restored = text
    for token, original in citations.items():
        restored = restored.replace(token, original)
    return restored

# ---------------------------------------------------------------------------
# POS mapping
# ---------------------------------------------------------------------------
SPACY_TO_WORDNET = {
    "NOUN": wordnet.NOUN,
    "VERB": wordnet.VERB,
    "ADJ": wordnet.ADJ,
    "ADV": wordnet.ADV,
}

# ---------------------------------------------------------------------------
# Shared constant data
# ---------------------------------------------------------------------------
DETERMINER_ALTERNATIVES = {
    "the": ["this", "that", "one particular", "a given", "each"],
    "these": ["such", "those particular", "the mentioned"],
    "this": ["that", "the", "one such"],
    "many": ["a handful of", "quite a few", "several", "numerous"],
    "various": ["different", "assorted", "multiple"],
}

TRANSITION_ALTERNATIVES = {
    "however": ["but then", "still", "that said", "yet"],
    "nevertheless": ["still", "even so", "all the same"],
    "furthermore": ["also", "plus", "and"],
    "moreover": ["on top of that", "also", "and"],
    "therefore": ["so", "thus", "for that reason"],
    "consequently": ["so", "as a result"],
    "additionally": ["also", "and", "plus"],
    "nonetheless": ["still", "even so"],
}

DEEPENING_PHRASES = [
    ", which implies that the situation, as observed, is multidimensional",
    ", an observation that, while seemingly minor, complicates the framework",
    ", precisely because the underlying modality, though complex, remains relevant",
    ", suggesting that the inherent complexity, as it stands, warrants further analysis",
    ", primarily because the contextual factors, although varied, play a decisive role",
    ", which further indicates that the conceptual structure, as discussed, is quite intricate",
    ", largely because the multifaceted nature of the issue, though subtle, is significant",
    ", specifically because the foundational elements, while diverse, are interconnected",
]

MODERN_ACADEMIC_JARGON = [
    "granularity", "robustness", "paradigm shift", "leveraging",
    "contextualization", "methodological rigor", "nuanced", "multidimensional",
    "transformative", "synergistic", "empirical evidence", "theoretical framework",
    "optimization", "scalability", "interdisciplinary", "holistic",
    "ontology", "epistemological", "discursive", "intersectionality",
    "agential", "post-structural", "phenomenological", "heuristic", "reification",
    "interoperability", "modality", "synchronicity", "performativity", "normativity",
    "teleological", "hermeneutical", "axiomatic", "juxtaposition", "reductive",
    "hegemonic", "paradoxical", "recapitulate", "dichotomy", "divergence",
]

JARGON_REPLACEMENT_MAP = {
    # Only words that are truly safe to replace in academic context
    "detail": "granularity",
    "precision": "granularity",
    "strength": "robustness",
    "stability": "robustness",
    "rigor": "methodological rigor",
    "complexity": "multidimensionality",
    "methods": "methodological approaches",
    "framework": "theoretical framework",
    "study": "investigation",
    "connection": "interoperability",
    "different": "divergent",
    "similar": "analogous",
    "common": "prevalent",
    "rare": "infrequent",
}
