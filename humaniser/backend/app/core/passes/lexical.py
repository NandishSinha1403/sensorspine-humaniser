"""
Lexical passes — synonym substitution, academic n-gram replacement (Aho-Corasick),
and jargon injection.
"""

import re
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from nltk.corpus import wordnet
from lemminflect import getInflection
from app.core.passes.utils import (
    get_rng, load_spacy, SPACY_TO_WORDNET,
    MODERN_ACADEMIC_JARGON, JARGON_REPLACEMENT_MAP,
)

# ===========================================================================
# Protected words — NEVER replace these via synonym substitution.
# These are domain-specific terms whose synonyms would be semantically wrong.
# ===========================================================================
PROTECTED_WORDS = {
    # Scientific / technical terms
    "gas", "gases", "level", "levels", "pattern", "patterns", "event", "events",
    "temperature", "pressure", "concentration", "molecule", "atom", "cell",
    "gene", "protein", "species", "ecosystem", "habitat", "climate",
    "carbon", "dioxide", "methane", "nitrous", "oxide", "ozone",
    "glacier", "ocean", "sea", "atmosphere", "radiation", "wavelength",
    "photon", "electron", "neutron", "proton", "nucleus", "ion",
    "virus", "bacteria", "vaccine", "antibody", "enzyme", "hormone",
    "neural", "cortex", "neuron", "synapse", "receptor",
    # Measurement / units
    "degree", "degrees", "celsius", "fahrenheit", "kelvin",
    "meter", "kilogram", "second", "ampere", "mole",
    "percent", "percentage", "ratio", "rate", "frequency",
    # Academic structure words
    "hypothesis", "theory", "model", "framework", "paradigm",
    "variable", "parameter", "coefficient", "constant", "function",
    "equation", "algorithm", "dataset", "sample", "population",
    "correlation", "regression", "variance", "deviation", "distribution",
    # Common words whose synonyms are misleading
    "floor", "ceiling", "wall", "door", "table", "figure",
    "effect", "cause", "result", "outcome", "factor", "limit",
    "limiting", "warming", "cooling", "emission", "emissions",
    "fossil", "fuel", "fuels", "renewable", "solar", "wind",
    "policy", "regulation", "law", "standard", "protocol",
    # Proper nouns and abbreviations (these should never be replaced)
    "ipcc", "who", "un", "eu", "us", "uk", "nasa", "nih",
}

# Words that should never have jargon prepended
JARGON_BLACKLIST_CONTEXTS = {
    "due", "to", "by", "in", "on", "at", "for", "with", "from",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
}


def pass_lexical(text: str, profile: Optional[Dict[str, Any]], intensity: float = 1.0) -> Tuple[str, int]:
    """
    Context-aware synonym replacement using WordNet + spaCy vectors.
    Now with protected-word filtering and higher similarity threshold.
    """
    sp = load_spacy()
    if not sp:
        return text, 0

    doc = sp(text)
    new_tokens = []
    changes = 0
    avoid_words = set(profile.get("top_vocab", []) if profile else [])
    replace_chance = 0.35 * intensity  # Reduced from 0.55 — less aggressive

    for token in doc:
        t_lower = token.text.lower()

        # Skip protected words, short words, non-alpha, and stop words
        if (token.pos_ not in SPACY_TO_WORDNET or
            not token.is_alpha or
            t_lower in avoid_words or
            t_lower in PROTECTED_WORDS or
            len(t_lower) < 4 or
            token.is_stop):
            new_tokens.append(token.text_with_ws)
            continue

        # Skip words that are part of named entities
        if token.ent_type_:
            new_tokens.append(token.text_with_ws)
            continue

        # Skip words inside parenthetical expressions (chemical formulas, etc.)
        if any(c in token.text for c in "₂₃₄₅₆₇₈₉₀()[]{}"):
            new_tokens.append(token.text_with_ws)
            continue

        # WordNet synonym substitution with HIGH similarity check
        wn_pos = SPACY_TO_WORDNET[token.pos_]
        if get_rng().random() < replace_chance:
            synsets = wordnet.synsets(t_lower, pos=wn_pos)
            if synsets:
                synonyms = []
                for syn in synsets:
                    for lemma in syn.lemmas():
                        syn_name = lemma.name().replace('_', ' ')
                        if ' ' not in syn_name and syn_name.lower() not in PROTECTED_WORDS:
                            synonyms.append(syn_name)

                synonyms = list(dict.fromkeys(synonyms))
                synonyms = [s for s in synonyms if s.lower() != t_lower]

                if synonyms:
                    # Use BOTH sentence context AND original word similarity
                    token_vec = token.vector
                    token_norm = np.linalg.norm(token_vec)
                    valid_synonyms = []

                    for syn_word in synonyms:
                        syn_vec_doc = sp.vocab[syn_word]
                        if syn_vec_doc.has_vector:
                            syn_vec = syn_vec_doc.vector
                            syn_norm = np.linalg.norm(syn_vec)
                            if token_norm > 0 and syn_norm > 0:
                                # Direct word-to-word similarity (more precise than sentence-context)
                                similarity = np.dot(token_vec, syn_vec) / (token_norm * syn_norm)
                                if similarity >= 0.75:  # Raised from 0.65 → 0.75
                                    valid_synonyms.append((syn_word, similarity))

                    if valid_synonyms:
                        valid_synonyms.sort(key=lambda x: x[1], reverse=True)
                        # Only pick from top 3 most similar (was top 5)
                        top_n = min(3, len(valid_synonyms))
                        replacement_lemma = get_rng().choice(valid_synonyms[:top_n])[0]
                        inflected_replacement = replacement_lemma
                        inflection = getInflection(replacement_lemma, tag=token.tag_)
                        if inflection:
                            inflected_replacement = inflection[0]
                        if token.text[0].isupper():
                            inflected_replacement = inflected_replacement.capitalize()
                        new_tokens.append(inflected_replacement + token.whitespace_)
                        changes += 1
                        continue

        new_tokens.append(token.text_with_ws)

    return "".join(new_tokens), changes


def pass_academic_ngrams(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Use the preloaded Aho-Corasick automaton for rapid O(N) phrase replacement.
    Falls back to a local replacement map if automaton is unavailable.
    """
    try:
        from app.core.humanizer import ACADEMIC_AUTOMATON
        from flashtext import KeywordProcessor
    except ImportError:
        ACADEMIC_AUTOMATON = None

    # Curated AI → Academic phrase map for targeted replacement
    academic_map = {
        "furthermore": "moreover",
        "additionally": "further",
        "in conclusion": "consequently",
        "it is worth noting": "notably",
        "it is important to note": "critically",
        "research has shown": "empirical evidence suggests",
        "in today's world": "in the contemporary landscape",
        "it is clear that": "the evidence indicates that",
        "a wide range of": "a broad spectrum of",
        "plays a crucial role": "serves as a cornerstone",
        "has been shown": "has been empirically demonstrated",
        "it is essential": "it remains imperative",
        "in order to": "so as to",
        "due to the fact that": "owing to the circumstance that",
        "in light of": "in consideration of",
        "with regard to": "with reference to",
        "in terms of": "from the standpoint of",
        "as previously mentioned": "as earlier articulated",
        "this demonstrates": "this substantiates",
        "this indicates": "this intimates",
    }

    changes = 0

    # Try using the preloaded automaton for O(N) performance
    if ACADEMIC_AUTOMATON and hasattr(ACADEMIC_AUTOMATON, 'replace_keywords'):
        new_text = ACADEMIC_AUTOMATON.replace_keywords(text)
        if new_text != text:
            changes = len(ACADEMIC_AUTOMATON.extract_keywords(text))
            return new_text, changes

    # Fallback: build a local automaton with academic map
    try:
        from flashtext import KeywordProcessor
        local_automaton = KeywordProcessor()
        for k, v in academic_map.items():
            local_automaton.add_keyword(k, v)
        new_text = local_automaton.replace_keywords(text)
        if new_text != text:
            changes = len(local_automaton.extract_keywords(text))
        return new_text, changes
    except ImportError:
        return text, 0


def pass_jargon_injection(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Injects modern academic jargon into suitable positions.
    Now context-aware — avoids injecting before prepositions, determiners,
    and inside technical/scientific phrases.
    """
    sp = load_spacy()
    if not sp:
        return text, 0
    doc = sp(text)
    new_tokens, changes = [], 0

    for token in doc:
        t_lower = token.text.lower()

        # JARGON_REPLACEMENT_MAP substitution — only for non-protected, non-entity words
        if (t_lower in JARGON_REPLACEMENT_MAP and
            t_lower not in PROTECTED_WORDS and
            not token.ent_type_ and
            get_rng().random() < 0.25 * intensity):
            replacement = JARGON_REPLACEMENT_MAP[t_lower]
            if token.text[0].isupper():
                replacement = replacement.capitalize()
            new_tokens.append(replacement + token.whitespace_)
            changes += 1

        # Adjective qualifier injection — ONLY for suitable adjective positions
        # Must be: modifying a non-technical noun, not inside NER, not before a preposition
        elif (token.pos_ == "ADJ" and
              token.dep_ == "amod" and  # Must be an attributive adjective
              not token.ent_type_ and
              t_lower not in PROTECTED_WORDS and
              len(t_lower) > 4 and      # Skip short adjectives like "due"
              token.head.text.lower() not in PROTECTED_WORDS and  # Don't modify technical nouns
              get_rng().random() < 0.02 * intensity):  # Very low probability
            injection = "nuanced"  # Removed "thoroughly contextualized" — too long/awkward
            replacement = injection + " " + token.text
            new_tokens.append(replacement + token.whitespace_)
            changes += 1
        else:
            new_tokens.append(token.text_with_ws)

    return "".join(new_tokens), changes
