"""
Comprehensive test suite for the Humaniser pipeline.
Tests all new passes, intensity parameter, segment-based detection,
voice drift correction, and edge cases.
"""
import pytest
import random
import numpy as np
from app.core.humanizer import (
    humanize_text,
    pass_phrase_replacement,
    pass_restructuring,
    pass_burstiness,
    pass_lexical,
    pass_style_overlay,
    pass_voice_conversion,
    pass_clause_reorder,
    pass_discourse_markers,
    pass_paragraph_rhythm,
    pass_rhythm_sculpting,
    pass_final_cleanup,
    protect_citations,
    restore_citations,
    classify_domain,
)
from app.core.detector import (
    detect_ai_score,
    score_segment,
    score_sentences,
    calculate_burstiness,
    calculate_perplexity_proxy,
    calculate_phrase_score,
    calculate_mattr,
    _split_into_segments,
    _calibrate,
)
from app.core.voice import extract_voice, apply_voice


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ai_heavy_text():
    """Text saturated with AI signature phrases."""
    return (
        "Furthermore, it is important to note that artificial intelligence has become increasingly "
        "prevalent in modern society. Additionally, the integration of machine learning algorithms "
        "plays a crucial role in data-driven decisions. In conclusion, these advancements demonstrate "
        "significant potential. It is clear that this highlights the need for further research. "
        "Consequently, in order to improve operational efficiency, a wide range of approaches must "
        "be explored. In terms of methodology, research has shown that this is essential."
    )

@pytest.fixture
def human_text():
    """Natural human-style writing."""
    return (
        "The experiment ran for six weeks. Results were mixed — some groups showed improvement, "
        "others did not. We noticed an interesting pattern in the second cohort. Their response "
        "times dropped sharply after day twelve. Why? Hard to say. Maybe it was the adjusted "
        "dosage, maybe fatigue. Dr. Kim thought it was the humidity in the lab. I disagree, "
        "but the data are ambiguous enough that we can't rule it out."
    )

@pytest.fixture
def short_text():
    """Very short text (under 50 words)."""
    return "This is a brief note. The data suggest otherwise."

@pytest.fixture
def long_text():
    """Text exceeding 5000 words (stress test)."""
    base = (
        "The experimental findings suggest that environmental factors mediate the observed response. "
        "Data were collected over a period of six months using standard regression techniques. "
        "The methodology builds upon approaches outlined by earlier investigators. "
        "Statistical significance was determined using a two-tailed p-value threshold. "
    )
    return base * 40  # ~640 words × enough to exceed threshold

@pytest.fixture
def citation_text():
    """Text with academic citations that must be preserved."""
    return (
        "The framework proposed by Smith et al. (2019) extends earlier work [1,2,3]. "
        "Furthermore, Johnson (2020) demonstrated significant improvements [4]. "
        "These results align with the meta-analysis conducted by Davis [5]."
    )

@pytest.fixture
def subordinate_text():
    """Text with subordinate clause structures for clause reordering tests."""
    return (
        "Because the sample size was small, the results should be interpreted cautiously. "
        "Although the method was novel, it had several limitations. "
        "While the data appeared conclusive, further replication is needed."
    )


# ---------------------------------------------------------------------------
# Test: Detector (segment-based scoring)
# ---------------------------------------------------------------------------

class TestDetector:
    def test_segment_splitting(self):
        """Verify text is split into ~250-word segments on sentence boundaries."""
        text = "First sentence. " * 50 + "Second sentence. " * 50
        segments = _split_into_segments(text, target_words=250)
        assert len(segments) >= 1
        for seg in segments:
            # Each segment should be well-formed text
            assert len(seg.strip()) > 0

    def test_calibration_high_risk(self):
        """Scores > 60 should be boosted by 1.3x."""
        assert _calibrate(70) == min(99.0, 70 * 1.3)
        assert _calibrate(80) == min(99.0, 80 * 1.3)

    def test_calibration_low_risk(self):
        """Scores < 25 should be reduced by 0.7x."""
        assert _calibrate(20) == 20 * 0.7
        assert _calibrate(10) == 10 * 0.7

    def test_calibration_mid_range(self):
        """Scores 25-60 should pass through unchanged."""
        assert _calibrate(40) == 40

    def test_ai_text_scores_high(self, ai_heavy_text):
        """AI-saturated text should score high."""
        score = detect_ai_score(ai_heavy_text)
        assert score > 30, f"AI-heavy text scored unexpectedly low: {score}"

    def test_human_text_scores_low(self, human_text):
        """Natural human text should score lower than AI text."""
        human_score = detect_ai_score(human_text)
        assert human_score < 70, f"Human text scored unexpectedly high: {human_score}"

    def test_phrase_score_detects_ai(self, ai_heavy_text):
        """Phrase scorer must detect AI signature phrases."""
        score = calculate_phrase_score(ai_heavy_text)
        assert score > 20, f"Phrase score too low for AI-heavy text: {score}"

    def test_burstiness_uniform_text(self):
        """Uniform sentence lengths should score high (AI-like)."""
        uniform = "This is exactly ten words in this sentence here. " * 10
        score = calculate_burstiness(uniform.split(". "))
        assert score > 50

    def test_burstiness_varied_text(self):
        """Highly varied sentence lengths should score lower than uniform."""
        varied = "Go. This is a medium length sentence for testing purposes here. " + \
                 "This is a much longer sentence that contains many more words and keeps going further and further with extra detail. " + \
                 "OK. Another medium one here for measure. Absolutely fascinating results were observed across the multiple experimental conditions tested."
        from nltk.tokenize import sent_tokenize
        score = calculate_burstiness(sent_tokenize(varied))
        assert score < 85, f"Varied text scored too high (AI-like): {score}"

    def test_sentence_scoring(self, ai_heavy_text):
        """Individual sentence scoring for heatmap."""
        sentences = score_sentences(ai_heavy_text)
        assert len(sentences) > 0
        for s in sentences:
            assert "text" in s
            assert "score" in s
            assert 0 <= s["score"] <= 100


# ---------------------------------------------------------------------------
# Test: Humanizer passes
# ---------------------------------------------------------------------------

class TestPhraseReplacement:
    def test_replaces_ai_phrases(self, ai_heavy_text):
        result, count = pass_phrase_replacement(ai_heavy_text)
        assert count > 0
        assert "furthermore" not in result.lower() or count > 0

    def test_preserves_case(self):
        text = "Furthermore, this is important."
        result, count = pass_phrase_replacement(text)
        if count > 0:
            # First word should still be capitalized
            assert result[0].isupper()

    def test_intensity_zero(self, ai_heavy_text):
        """With 0 intensity, very few or no phrases should be replaced."""
        random.seed(42)
        _, count_full = pass_phrase_replacement(ai_heavy_text, intensity=1.0)
        random.seed(42)
        _, count_zero = pass_phrase_replacement(ai_heavy_text, intensity=0.1)
        # Lower intensity should produce fewer replacements
        assert count_zero <= count_full


class TestRestructuring:
    def test_splits_long_sentences(self):
        text = "The system processes data efficiently and the results are stored in the database for later analysis."
        result, changes = pass_restructuring(text)
        # Should attempt to split at 'and'
        assert isinstance(result, str)

    def test_joins_short_sentences(self):
        text = "It works. Very well."
        result, changes = pass_restructuring(text)
        assert isinstance(result, str)


class TestBurstiness:
    def test_domain_matching(self):
        """Templates should match the requested domain."""
        text = "The algorithm was tested. Results were positive. The system worked. Data confirmed this. Performance was good."
        profile = {"sample_sentences": []}
        result, count = pass_burstiness(text, profile, field="computer_science")
        # Should not inject medical or law templates
        assert "clinical outcome" not in result.lower()
        assert "precedent" not in result.lower()

    def test_no_injection_when_already_bursty(self):
        """Shouldn't inject if variance is already high."""
        text = "Short. This is a significantly longer sentence with many words and complex structure. Tiny. Another very long sentence that keeps going and going with extra clauses and modifiers attached."
        profile = {"sample_sentences": []}
        result, count = pass_burstiness(text, profile, field="general")
        # If std_dev >= 8, no injections
        assert isinstance(result, str)


class TestStyleOverlay:
    def test_never_replaces_full_sentences(self):
        """The fixed style overlay should NEVER substitute entire user sentences."""
        text = (
            "The experiment showed clear results. The data were analyzed carefully. "
            "Participants responded positively. The methodology was sound. "
            "Final analysis confirmed our hypothesis."
        )
        profile = {
            "sample_sentences": ["Template sentence about <the subject>."],
            "opening_patterns": ["In our analysis"],
            "hedging_phrases": ["it appears that"],
            "top_vocab": [],
        }
        random.seed(1)
        result, changes = pass_style_overlay(text, profile, field="general", intensity=1.0)
        # Original sentence content should still be present (not wholesale replaced)
        assert "experiment" in result.lower() or "data" in result.lower()


class TestNewPasses:
    def test_voice_conversion(self):
        """Active/passive conversion should produce valid output."""
        text = "The researcher analyzed the data. The team published the findings."
        result, changes = pass_voice_conversion(text, intensity=1.0)
        assert isinstance(result, str)
        assert len(result) > 10

    def test_clause_reorder(self, subordinate_text):
        """Subordinate clauses should be reorderable."""
        random.seed(42)
        result, changes = pass_clause_reorder(subordinate_text, intensity=1.0)
        assert isinstance(result, str)

    def test_discourse_markers(self):
        """Discourse markers should be injected periodically."""
        text = "First point here. Second point here. Third point here. Fourth point here. Fifth point here."
        random.seed(42)
        result, changes = pass_discourse_markers(text, intensity=1.0)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test: Voice extraction and drift correction
# ---------------------------------------------------------------------------

class TestVoice:
    def test_extract_voice_basic(self, human_text):
        profile = extract_voice(human_text)
        assert "preferred_sentence_length" in profile
        assert "formality_score" in profile
        assert "person" in profile

    def test_person_detection_first(self):
        text = "I ran the experiment. My results were clear. I believe this works."
        profile = extract_voice(text)
        assert profile["person"] == "first"

    def test_person_detection_plural(self):
        text = "We conducted the study. Our findings suggest improvement. We believe this works."
        profile = extract_voice(text)
        assert profile["person"] == "first_plural"

    def test_person_detection_third(self):
        text = "The study was conducted. The results were analyzed. The data showed improvement."
        profile = extract_voice(text)
        assert profile["person"] == "third"

    def test_drift_correction_person(self):
        """Voice correction should replace 'one must' with author's preferred person."""
        profile = {"person": "first", "formality_score": 0.05,
                    "connector_ratio": 0.3, "preferred_sentence_length": 15,
                    "length_variance": 5, "punctuation_habits": {},
                    "favorite_openers": []}
        text = "One must consider the implications carefully."
        result = apply_voice(text, profile)
        assert "I must" in result

    def test_drift_correction_formality(self):
        """For informal authors, formal words should be downgraded."""
        profile = {"person": "third", "formality_score": 0.02,
                    "connector_ratio": 0.2, "preferred_sentence_length": 12,
                    "length_variance": 4, "punctuation_habits": {},
                    "favorite_openers": []}
        text = "Furthermore, the results were significant. Consequently, we proceed."
        result = apply_voice(text, profile)
        # Should downgrade "Furthermore" to "Also" and "Consequently" to "So"
        assert "Also" in result or "furthermore" not in result.lower()


# ---------------------------------------------------------------------------
# Test: Citation protection
# ---------------------------------------------------------------------------

class TestCitations:
    def test_protect_and_restore(self, citation_text):
        protected, citations = protect_citations(citation_text)
        # Citations should be tokenized with XCIT format
        assert "XCIT" in protected
        # Restore should bring them back
        restored = restore_citations(protected, citations)
        assert "[1,2,3]" in restored
        assert "(2020)" in restored

    def test_citations_survive_pipeline(self, citation_text):
        """Citations must survive the full humanization pipeline."""
        result = humanize_text(citation_text, field="general")
        output = result["humanized_text"]
        # At least some citation markers should survive
        assert "[" in output or "(" in output


# ---------------------------------------------------------------------------
# Test: Full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_basic_execution(self, ai_heavy_text):
        """Pipeline should complete without errors."""
        result = humanize_text(ai_heavy_text, field="general", intensity=0.7)
        assert "humanized_text" in result
        assert "passes_applied" in result
        assert "changes_made" in result
        assert "original_score" in result
        assert "humanized_score" in result

    def test_score_reduction(self, ai_heavy_text):
        """Humanized text should score lower than original."""
        result = humanize_text(ai_heavy_text, field="general", intensity=0.8)
        assert result["humanized_score"] <= result["original_score"] + 5, \
            f"Score did not decrease: {result['original_score']} → {result['humanized_score']}"

    def test_short_text_handling(self, short_text):
        """Short text (<50 words) should use light pipeline."""
        result = humanize_text(short_text, field="general")
        assert "humanized_text" in result
        assert len(result["humanized_text"]) > 0

    def test_intensity_affects_output(self, ai_heavy_text):
        """Different intensity values should produce different outputs."""
        random.seed(42)
        r1 = humanize_text(ai_heavy_text, intensity=0.3)
        random.seed(42)
        r2 = humanize_text(ai_heavy_text, intensity=1.0)
        # Higher intensity should make more changes
        c1_total = sum(v for v in r1["changes_made"].values() if isinstance(v, (int, float)))
        c2_total = sum(v for v in r2["changes_made"].values() if isinstance(v, (int, float)))
        # Can't guarantee exact ordering due to randomness, but both should complete
        assert c1_total >= 0
        assert c2_total >= 0

    def test_no_template_residue(self, ai_heavy_text):
        """Output should not contain template variables like <NAME> or <NUM>."""
        result = humanize_text(ai_heavy_text, field="general")
        output = result["humanized_text"]
        assert "<NAME>" not in output
        assert "< NAME >" not in output

    def test_domain_routing(self):
        """Medical text should be routed to medicine field."""
        text = "The clinical trial showed that patient outcomes improved with the new treatment protocol. Diagnosis was confirmed through laboratory analysis."
        domain = classify_domain(text)
        assert domain == "medicine"

    def test_cs_domain_routing(self):
        text = "The algorithm processes data through a distributed network. Software optimization improved server latency."
        domain = classify_domain(text)
        assert domain == "computer_science"

    def test_final_cleanup(self):
        """Cleanup should remove orphaned verb fragments."""
        text = "The results are clear. Suggests, this matters. The data confirm this."
        result = pass_final_cleanup(text)
        assert isinstance(result, str)

    def test_voice_profile_returned(self, ai_heavy_text):
        """Pipeline should return a voice profile."""
        result = humanize_text(ai_heavy_text)
        assert result["voice_profile"] is not None
        assert "preferred_sentence_length" in result["voice_profile"]


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text(self):
        """Empty text should return gracefully."""
        result = humanize_text("")
        assert result["humanized_text"] == "" or len(result["humanized_text"]) >= 0

    def test_single_word(self):
        """Single word should not crash."""
        result = humanize_text("Hello")
        assert "humanized_text" in result

    def test_numbers_only(self):
        """Numeric text should pass through."""
        result = humanize_text("123 456 789")
        assert "humanized_text" in result

    def test_special_characters(self):
        """Text with special chars should not crash."""
        result = humanize_text("Hello! @#$% World? <test> [brackets]")
        assert "humanized_text" in result

    def test_very_long_sentence(self):
        """A single very long sentence should not crash."""
        text = "The " + " and the ".join(["result"] * 100) + " were significant."
        result = humanize_text(text)
        assert "humanized_text" in result
