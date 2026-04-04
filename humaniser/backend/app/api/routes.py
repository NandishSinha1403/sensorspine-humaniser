from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import shutil
import os
import uuid
import time
import logging
from app.core.detector import detect_ai_score, score_sentences, calculate_burstiness, calculate_perplexity_proxy
from nltk.tokenize import sent_tokenize, word_tokenize

router = APIRouter()
logger = logging.getLogger("humaniser.api")

class MetricStats(BaseModel):
    burstiness: float
    perplexity: float
    ai_score: float

class HumanizationMetrics(BaseModel):
    baseline: MetricStats
    humanized: MetricStats
    latency: float

class DetectionRequest(BaseModel):
    text: str

class HumanizationRequest(BaseModel):
    text: str
    intensity: float = 0.7

class HumanizationResponse(BaseModel):
    original_text: str
    humanized_text: str
    original_score: float
    humanized_score: float
    passes_applied: List[str]
    changes_made: Dict[str, Any]
    sentences: List[Dict[str, Any]]
    voice_profile: Optional[Dict[str, Any]] = None
    metrics: Optional[HumanizationMetrics] = None

from app.core.humanizer import humanize_text as humanize_core

@router.post("/detect")
async def detect_text(request: DetectionRequest):
    score = detect_ai_score(request.text)
    label = "AI Generated" if score > 60 else "Human Written" if score < 35 else "Mixed"
    sentences = score_sentences(request.text)
    return {
        "score": score, 
        "label": label, 
        "sentences": sentences
    }

@router.post("/humanize", response_model=HumanizationResponse)
async def humanize_text(request: HumanizationRequest):
    start_time = time.time()
    
    # Calculate baseline metrics
    base_sentences = sent_tokenize(request.text)
    base_words = word_tokenize(request.text.lower())
    baseline_metrics = MetricStats(
        burstiness=calculate_burstiness(base_sentences),
        perplexity=calculate_perplexity_proxy(base_words),
        ai_score=detect_ai_score(request.text)
    )
    
    # Execute Humanization
    result = humanize_core(request.text, request.intensity)
    
    # Calculate humanized metrics
    hum_sentences = sent_tokenize(result["humanized_text"])
    hum_words = word_tokenize(result["humanized_text"].lower())
    humanized_metrics = MetricStats(
        burstiness=calculate_burstiness(hum_sentences),
        perplexity=calculate_perplexity_proxy(hum_words),
        ai_score=result["humanized_score"]
    )
    
    latency = time.time() - start_time
    logger.info(f"[Performance] Pipeline execution completed in {latency:.2f}s")
    
    metrics = HumanizationMetrics(
        baseline=baseline_metrics,
        humanized=humanized_metrics,
        latency=latency
    )
    
    # Get sentence level scores for heatmap
    final_sentences = score_sentences(result["humanized_text"])
    
    return {
        "original_text": request.text,
        "humanized_text": result["humanized_text"],
        "original_score": result["original_score"],
        "humanized_score": result["humanized_score"],
        "passes_applied": result["passes_applied"],
        "changes_made": result["changes_made"],
        "sentences": final_sentences,
        "voice_profile": result.get("voice_profile"),
        "metrics": metrics
    }


