"""
humanizer.py — Slim orchestrator for the multi-pass NLP pipeline.

All individual passes live in app.core.passes/. This module:
1. Loads Academic DNA (Pickle/Aho-Corasick)
2. Composes the pipeline (light vs. full)
3. Runs the self-audit loop
4. Applies voice correction & final cleanup
"""

import re
import os
import logging
import pickle
import contextvars
import random
from typing import Dict, Any, Optional
from nltk.tokenize import sent_tokenize, word_tokenize

# Configure logging FIRST (bug fix #1)
logger = logging.getLogger("humaniser.nlp")
logger.setLevel(logging.DEBUG)

from flashtext import KeywordProcessor

from app.core.detector import (
    detect_ai_score, score_sentences, _split_into_segments, score_segment,
)
from app.core.voice import extract_voice, apply_voice

# Import all passes from modular package
from app.core.passes.utils import (
    get_rng, rng_var, used_phrases_var, get_seed,
    protect_citations, restore_citations, fix_caps,
)
from app.core.passes.phrase_replacement import (
    pass_signature_phrase_breaker, pass_phrase_replacement,
)
from app.core.passes.structural import (
    pass_restructuring, pass_structural_reset, pass_structural_chaos,
    pass_morphological_shifting, pass_appositive_injection,
)
from app.core.passes.rhythm import (
    pass_burstiness, pass_rhythm_sculpting, pass_paragraph_wave,
)
from app.core.passes.syntactic import (
    pass_syntactic_clefting, pass_syntactic_variance, pass_clause_reorder,
)
from app.core.passes.lexical import (
    pass_lexical, pass_academic_ngrams, pass_jargon_injection,
)
from app.core.passes.style import (
    pass_style_overlay, pass_nuance_injection, pass_determiner_scramble,
    pass_epistemic_hedging, pass_hedging_uncertainty, pass_cross_referencing,
)
from app.core.passes.adversarial import (
    pass_adversarial_punctuation, pass_zwj_jitter, pass_invisible_padding,
    pass_whitespace_jitter, pass_punctuation_personality,
)
from app.core.passes.cleanup import (
    pass_final_cleanup, pass_human_errors,
)

# ===========================================================================
# Load Harvested Academic DNA (High-Performance Pickle Format)
# ===========================================================================
DNA_PATH = os.path.join(os.path.dirname(__file__), "academic_dna.pkl")
try:
    with open(DNA_PATH, "rb") as f:
        DNA_DATA = pickle.load(f)
    ACADEMIC_DNA = DNA_DATA.get("sentence_stats", {})
    ACADEMIC_AUTOMATON = DNA_DATA.get("automaton", KeywordProcessor())
    ACADEMIC_VOCAB = set(DNA_DATA.get("top_vocab", []))
except FileNotFoundError:
    logger.warning(f"Academic DNA Pickle not found at {DNA_PATH}. Using empty defaults.")
    ACADEMIC_DNA = {}
    ACADEMIC_AUTOMATON = KeywordProcessor()
    ACADEMIC_VOCAB = set()
except Exception as e:
    logger.error(f"Error loading Academic DNA Pickle: {e}")
    ACADEMIC_DNA = {}
    ACADEMIC_AUTOMATON = KeywordProcessor()
    ACADEMIC_VOCAB = set()


# ===========================================================================
# Pipeline composers
# ===========================================================================

def apply_light_passes(
    txt: str, intensity: float, changes: Dict[str, Any],
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Minimal pass set for low-confidence text."""
    c_txt, c0 = pass_signature_phrase_breaker(txt)
    changes["signature_phrase_breaks"] += c0
    c_txt, c1 = pass_phrase_replacement(c_txt, intensity)
    changes["phrases_replaced"] += c1
    c_txt, c4 = pass_lexical(c_txt, profile, intensity)
    changes["lexical_substitutions"] += c4
    return c_txt


def apply_full_passes(
    txt: str, intensity: float, changes: Dict[str, Any],
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Full NLP pipeline — all passes in optimal order with tuned intensities."""
    # Phase 1: Core signature removal (always full intensity)
    c_txt, c0 = pass_signature_phrase_breaker(txt)
    changes["signature_phrase_breaks"] += c0
    c_txt, c1 = pass_phrase_replacement(c_txt, intensity)
    changes["phrases_replaced"] += c1
    c_txt, c_an = pass_academic_ngrams(c_txt, intensity)
    changes["academic_ngrams"] += c_an

    # Phase 2: Structural transformation (full intensity)
    c_txt, c_ms = pass_morphological_shifting(c_txt, intensity * 0.6)
    changes["morphological_shifting"] += c_ms
    c_txt, c2 = pass_restructuring(c_txt, intensity)
    changes["sentences_restructured"] += c2
    c_txt, c7 = pass_clause_reorder(c_txt, intensity * 0.7)
    changes["clause_reorders"] += c7
    c_txt, c_sc = pass_syntactic_clefting(c_txt, intensity * 0.5)
    changes["syntactic_clefting"] += c_sc

    # Phase 3: Rhythm and burstiness
    c_txt, c3 = pass_burstiness(c_txt, profile, intensity)
    changes["burstiness_injections"] += c3
    c_txt, c35 = pass_rhythm_sculpting(c_txt, profile, intensity)
    changes["rhythm_adjustments"] += c35
    c_txt, c_pw = pass_paragraph_wave(c_txt, intensity * 0.6)
    changes["paragraph_wave"] += c_pw

    # Phase 4: Lexical enrichment
    c_txt, c4 = pass_lexical(c_txt, profile, intensity * 0.7)
    changes["lexical_substitutions"] += c4
    c_txt, cji = pass_jargon_injection(c_txt, intensity * 0.3)
    changes["jargon_injections"] += cji

    # Phase 5: Style and hedging (reduced intensity to avoid over-stacking)
    c_txt, c5 = pass_style_overlay(c_txt, profile, intensity * 0.5)
    changes["style_overlays"] += c5
    c_txt, c_ai = pass_appositive_injection(c_txt, intensity * 0.15)
    changes["appositive_injections"] += c_ai
    c_txt, cn = pass_nuance_injection(c_txt, intensity * 0.3)
    changes["nuance_injections"] += cn
    c_txt, c_det = pass_determiner_scramble(c_txt, intensity * 0.5)
    changes["determiner_scrambles"] += c_det
    c_txt, c_eh = pass_epistemic_hedging(c_txt, intensity * 0.4)
    changes["epistemic_hedging"] += c_eh
    c_txt, c_cr = pass_cross_referencing(c_txt, intensity * 0.3)
    changes["cross_referencing"] += c_cr

    # Phase 6: Adversarial (subtle — these are invisible to human readers)
    c_txt, c_zwj = pass_zwj_jitter(c_txt, intensity * 0.5)
    changes["zwj_jitters"] += c_zwj
    c_txt, c_inv = pass_invisible_padding(c_txt, intensity * 0.4)
    changes["invisible_padding"] += c_inv
    c_txt, c_ap = pass_adversarial_punctuation(c_txt, intensity * 0.4)
    changes["adversarial_punctuation"] += c_ap
    c_txt, c_pp = pass_punctuation_personality(c_txt, intensity * 0.25)
    changes["punctuation_personality"] += c_pp

    # Phase 7: Enforcement and variance
    c_txt, c9 = pass_syntactic_variance(c_txt, intensity * 0.6)
    changes["syntactic_variance"] += c9
    c_txt, c10 = pass_structural_reset(c_txt, intensity * 0.5)
    changes["structural_resets"] += c10
    c_txt, che = pass_human_errors(c_txt, intensity * 0.3)
    changes["human_errors"] += che
    return c_txt


# ===========================================================================
# Main Pipeline
# ===========================================================================

def humanize_text(text: str, intensity: float = 0.7) -> Dict[str, Any]:
    """
    Main humanization pipeline with forced full passes at intensity >= 0.7.
    """
    voice_profile = extract_voice(text)
    initial_score = detect_ai_score(text)
    word_count = len(word_tokenize(text))

    # Chunked processing for very long texts (>5000 words)
    if word_count > 5000:
        sentences = sent_tokenize(text)
        chunks, current_chunk, current_len = [], [], 0
        for sent in sentences:
            sent_len = len(word_tokenize(sent))
            if current_len + sent_len > 1000 and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_len = [sent], sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        chunk_results = [humanize_text(chunk, intensity) for chunk in chunks]
        final_text = " ".join([c["humanized_text"] for c in chunk_results])
        final_changes = {}
        for c in chunk_results:
            for k, v in c["changes_made"].items():
                if isinstance(v, (int, float)):
                    final_changes[k] = final_changes.get(k, 0) + v
        return {
            "humanized_text": final_text,
            "passes_applied": chunk_results[0]["passes_applied"] if chunk_results else [],
            "changes_made": final_changes,
            "original_score": initial_score,
            "humanized_score": detect_ai_score(final_text),
            "voice_profile": voice_profile,
        }

    # Seed RNG for deterministic results per input
    get_rng().seed(get_seed(text))
    used_phrases_var.set(set())
    profile = None
    current_text, citations_map = protect_citations(text)

    changes = {
        "phrases_replaced": 0, "sentences_restructured": 0,
        "rhythm_adjustments": 0, "burstiness_injections": 0,
        "lexical_substitutions": 0, "style_overlays": 0,
        "clause_reorders": 0, "syntactic_variance": 0,
        "structural_resets": 0, "determiner_scrambles": 0,
        "audit_iterations": 0, "voice_adjustments": 0,
        "final_cleanups": 0, "paragraph_rhythm_fixes": 0,
        "nuance_injections": 0, "jargon_injections": 0,
        "whitespace_jitter": 0, "structural_chaos": 0,
        "human_errors": 0, "signature_phrase_breaks": 0,
        "final_enforcement": 0,
        "morphological_shifting": 0,
        "appositive_injections": 0, "zwj_jitters": 0,
        "invisible_padding": 0,
        "punctuation_personality": 0, "cross_referencing": 0,
        "hedging_uncertainty": 0,
        "academic_ngrams": 0, "syntactic_clefting": 0,
        "paragraph_wave": 0, "epistemic_hedging": 0,
        "adversarial_punctuation": 0,
    }

    is_high_intensity = intensity >= 0.7
    is_short_text = word_count < 50

    if is_short_text:
        if is_high_intensity:
            current_text = apply_full_passes(current_text, intensity, changes, profile)
        else:
            current_text = apply_light_passes(current_text, intensity, changes, profile)
        passes_applied = ["full_pipeline"] if is_high_intensity else ["light_pipeline"]
    else:
        segments = _split_into_segments(current_text, target_words=250)
        processed_segments = []
        for seg in segments:
            if is_high_intensity or score_segment(seg) >= 25:
                processed_segments.append(apply_full_passes(seg, intensity, changes, profile))
            else:
                processed_segments.append(apply_light_passes(seg, intensity, changes, profile))
        current_text = " ".join(processed_segments)
        passes_applied = [
            "citation_guard", "confidence_gradient", "signature_phrase_breaker",
            "phrase_replacement", "academic_ngrams", "morphological_shifting",
            "restructuring", "burstiness", "rhythm_sculpting",
            "paragraph_wave", "lexical", "style_overlay",
            "clause_reorder", "syntactic_clefting",
            "appositive_injection", "nuance_injection",
            "determiner_scramble", "epistemic_hedging",
            "zwj_jitter", "invisible_padding",
            "adversarial_punctuation", "punctuation_personality",
            "cross_referencing", "hedging_uncertainty",
            "syntactic_variance", "structural_reset", "jargon_injection",
        ]

    # Self-audit recursive loop
    if not is_short_text:
        max_iterations = 5
        last_score = detect_ai_score(restore_citations(current_text, citations_map))
        for iteration in range(max_iterations):
            if last_score <= 15:
                break
            changes["audit_iterations"] += 1
            if last_score > 20:
                current_text, csc = pass_structural_chaos(current_text)
                changes["structural_chaos"] += csc
                current_text, csr = pass_structural_reset(current_text, intensity)
                changes["structural_resets"] += csr
                current_text, cwj = pass_whitespace_jitter(current_text, intensity * 0.8)
                changes["whitespace_jitter"] += cwj

            new_sentences = []
            for s in score_sentences(current_text):
                sent_text = s["text"]
                if s["score"] > 30:
                    sent_text, _ = pass_signature_phrase_breaker(sent_text)
                    sent_text, _ = pass_phrase_replacement(sent_text, intensity)
                    sent_text, can = pass_academic_ngrams(sent_text, intensity)
                    changes["academic_ngrams"] += can
                    sent_text, cms = pass_morphological_shifting(sent_text, intensity)
                    changes["morphological_shifting"] += cms
                    sent_text, csc = pass_syntactic_clefting(sent_text, intensity)
                    changes["syntactic_clefting"] += csc
                    sent_text, cai = pass_appositive_injection(sent_text, intensity)
                    changes["appositive_injections"] += cai
                    sent_text, _ = pass_structural_reset(sent_text, intensity)
                    sent_text, _ = pass_lexical(sent_text, profile, intensity)
                    sent_text, _ = pass_nuance_injection(sent_text, intensity)
                    sent_text, ceh = pass_epistemic_hedging(sent_text, intensity)
                    changes["epistemic_hedging"] += ceh
                    sent_text, cap = pass_adversarial_punctuation(sent_text, intensity)
                    changes["adversarial_punctuation"] += cap
                    sent_text, cji = pass_jargon_injection(sent_text, intensity)
                    changes["jargon_injections"] += cji
                    if s["score"] > 50:
                        sent_text, cwj = pass_whitespace_jitter(sent_text, intensity)
                        changes["whitespace_jitter"] += cwj
                new_sentences.append(sent_text)

            current_text = " ".join(new_sentences)
            # Clean up duplicate connectors
            current_text = re.sub(
                r'\b(yet|but|so|and|still)\s*[,]*\s*(yet|but|so|and|still|however|moreover|furthermore|nevertheless)\b',
                r'\2', current_text, flags=re.IGNORECASE
            )
            current_text = re.sub(
                r'\b(however|moreover|furthermore|nevertheless|consequently|therefore)\s*,\s*\b(however|moreover|furthermore|nevertheless|consequently|therefore)\b',
                r'\1', current_text, flags=re.IGNORECASE
            )
            current_restored = restore_citations(current_text, citations_map)
            new_score = detect_ai_score(current_restored)
            if new_score >= last_score and iteration > 1:
                break
            last_score = new_score

        if changes["audit_iterations"] > 0:
            passes_applied.append("self_audit")

    # Voice preservation
    if voice_profile:
        current_text = apply_voice(current_text, voice_profile)
        changes["voice_adjustments"] += 1
        passes_applied.append("voice_preserved")

    # Final cleanup
    current_text, ccl = pass_final_cleanup(current_text)
    if ccl > 0:
        changes["final_cleanups"] += ccl
        passes_applied.append("artifact_cleanup")

    current_text = restore_citations(current_text, citations_map)
    current_text = current_text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    current_text = re.sub(r'([.!?]\s+)([a-z])', fix_caps, current_text)

    return {
        "humanized_text": current_text,
        "passes_applied": passes_applied,
        "changes_made": changes,
        "original_score": initial_score,
        "humanized_score": detect_ai_score(current_text),
        "voice_profile": voice_profile,
    }
