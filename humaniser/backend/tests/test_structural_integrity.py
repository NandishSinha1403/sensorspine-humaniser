import pytest
from app.core.humanizer import humanize_text, extract_primary_subject
from app.corpus.style_profile import load_profile

def test_sentence_boundary_integrity():
    """Verify that long technical sentences are not broken into ungrammatical fragments."""
    input_text = "Furthermore, the integration of automated routing protocols ensures that high-priority logistical issues are instantaneously directed to the appropriate departments."
    result = humanize_text(input_text, field="engineering")
    humanized = result["humanized_text"]
    
    # Check for common fragment indicators
    assert "ensures. That" not in humanized
    assert "ensures. that" not in humanized
    
    # Ensure it's still a valid set of sentences (at least one verb/subject per sentence)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(humanized)
    for sent in doc.sents:
        has_verb = any(t.pos_ == "VERB" or t.pos_ == "AUX" for t in sent)
        has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in sent)
        assert has_verb, f"Fragment found (no verb): {sent.text}"
        assert has_subj, f"Fragment found (no subject): {sent.text}"

def test_template_resolution():
    """Verify that < the subject > and other placeholders are resolved correctly."""
    # We'll mock a profile with a template
    test_profile = {
        "sample_sentences": ["In this study, < the subject > provides a foundation for future research."],
        "opening_patterns": [],
        "punctuation_profile": {"semicolon_rate": 0, "dash_rate": 0},
        "hedging_phrases": [],
        "top_vocab": []
    }
    
    input_text = "The integrated circuit design minimizes power consumption while maximizing throughput."
    # Force the style overlay to use our template by overriding load_profile
    import app.core.humanizer as humanizer
    original_load = humanizer.load_profile
    humanizer.load_profile = lambda x: test_profile
    
    try:
        # Intensity 1.0 to increase chance of overlay
        result = humanize_text(input_text, intensity=1.0)
        humanized = result["humanized_text"]
        
        # Check that no < > tags remain
        assert "<" not in humanized
        assert ">" not in humanized
        
        # Check that it resolved to something sensible (either extracted subject or fallback)
        assert "integrated circuit design" in humanized or "this system" in humanized or "the subject matter" in humanized
    finally:
        humanizer.load_profile = original_load

def test_subject_extraction():
    """Test the extraction of primary subjects from complex sentences."""
    text1 = "The distributed ledger technology allows for transparent transactions."
    assert "distributed ledger technology" in extract_primary_subject(text1)
    
    text2 = "High-speed rail networks across Europe have transformed travel habits."
    assert "rail networks" in extract_primary_subject(text2)

def test_fallback_subject():
    """Verify fallback when no subject is found."""
    # Extremely short or weird text
    assert extract_primary_subject("Running fast.") == "this system"
