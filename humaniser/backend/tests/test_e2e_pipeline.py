import pytest
import logging
import numpy as np
from app.core.humanizer import humanize_text
from app.core.detector import (
    detect_ai_score, 
    calculate_burstiness, 
    calculate_perplexity_proxy,
    score_sentences
)
from nltk.tokenize import sent_tokenize, word_tokenize

# Configure Logging for Benchmarking
logger = logging.getLogger("humaniser.benchmark")
logger.setLevel(logging.INFO)

@pytest.fixture
def technical_corpus():
    """A rigid, highly predictable technical corpus mimicking administrative AI output."""
    return (
        "The municipal administrative routing infrastructure operates via a centralized "
        "hierarchical distribution network. Furthermore, it is important to note that "
        "high-priority logistical data packets are instantaneously processed and directed "
        "to the appropriate departmental nodes. Additionally, the system ensures that "
        "operational efficiency is maintained through constant monitoring. It is clear that "
        "automated protocols play a crucial role in modern governance. Consequently, "
        "the integration of data-driven decisions helps to improve the overall workflow. "
        "In conclusion, the facts demonstrate that the current architecture is robust."
    )

def test_full_pipeline_execution(technical_corpus):
    """Verify that the 7-pass pipeline completes and maintains content integrity."""
    # 1. Execute Pipeline
    result = humanize_text(technical_corpus, field="general", intensity=0.8)
    humanized_text = result["humanized_text"]
    
    # 2. Assertions: Integrity & Completion
    assert "humanized_text" in result
    assert len(result["passes_applied"]) >= 7, "Not all 7 passes were applied or recorded."
    
    orig_len = len(technical_corpus)
    final_len = len(humanized_text)
    drift = abs(final_len - orig_len) / orig_len
    
    # Assert length is within +/- 15%
    assert drift <= 0.15, f"Content drift too high: {drift:.2%} (Limit 15%)"
    
    # Ensure no template variables remained
    assert "<" not in humanized_text and ">" not in humanized_text

def test_statistical_benchmarking(technical_corpus):
    """Measure the statistical shift in Burstiness and Perplexity."""
    # Baseline Metrics
    base_sentences = sent_tokenize(technical_corpus)
    base_words = word_tokenize(technical_corpus.lower())
    
    # detect_ai_score returns an AI score (0-100). 
    # For Burstiness, calculate_burstiness returns a score where 100 = low variance (AI-like).
    base_burstiness_score = calculate_burstiness(base_sentences)
    base_perplexity_score = calculate_perplexity_proxy(base_words)
    base_ai_score = detect_ai_score(technical_corpus)
    
    # Process
    result = humanize_text(technical_corpus, field="general", intensity=1.0)
    humanized_text = result["humanized_text"]
    
    # Post-processing Metrics
    final_sentences = sent_tokenize(humanized_text)
    final_words = word_tokenize(humanized_text.lower())
    
    final_burstiness_score = calculate_burstiness(final_sentences)
    final_perplexity_score = calculate_perplexity_proxy(final_words)
    final_ai_score = detect_ai_score(humanized_text)
    
    # Logging the Delta
    logger.info(f"\n[Benchmark] Pipeline Delta Report")
    logger.info(f"-------------------------------")
    logger.info(f"AI Overall Score: {base_ai_score:.1f}% -> {final_ai_score:.1f}%")
    logger.info(f"Burstiness (AI Score): {base_burstiness_score:.1f} -> {final_burstiness_score:.1f}")
    logger.info(f"Perplexity (AI Score): {base_perplexity_score:.1f} -> {final_perplexity_score:.1f}")
    logger.info(f"-------------------------------")
    
    # Assertions: Statistical Shift
    # A successful humanization should LOWER the AI score for each component.
    assert final_ai_score < base_ai_score, "Overall AI detection score did not decrease."
    assert final_burstiness_score < base_burstiness_score, "Burstiness variance did not improve (AI score stayed high)."
    
    # Verify significant reduction in detector confidence
    if base_ai_score > 60:
        assert final_ai_score < 55, f"Detector confidence remains too high: {final_ai_score}%"

def test_detector_hotspot_reduction(technical_corpus):
    """Audit the reduction of high-risk 'hotspots' identified by the detector."""
    base_sentences = score_sentences(technical_corpus)
    high_risk_base = [s for s in base_sentences if s["score"] > 60]
    
    result = humanize_text(technical_corpus, intensity=1.0)
    final_sentences = score_sentences(result["humanized_text"])
    high_risk_final = [s for s in final_sentences if s["score"] > 60]
    
    logger.info(f"[Audit] High-risk segments reduced from {len(high_risk_base)} to {len(high_risk_final)}")
    
    # We expect high-risk segments to be reduced or eliminated
    assert len(high_risk_final) < len(high_risk_base), "Hotspots were not successfully mitigated."
