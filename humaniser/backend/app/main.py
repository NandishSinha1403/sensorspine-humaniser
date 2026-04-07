import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nltk

from app.api.routes import router as api_router
from app.core.detector import load_brown_corpus_data

logger = logging.getLogger("humaniser.app")

# ---------------------------------------------------------------------------
# NLTK resource bootstrap (only downloads if missing)
# ---------------------------------------------------------------------------
NLTK_RESOURCES = ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                   "averaged_perceptron_tagger_eng", "wordnet", "stopwords", "brown"]

def _ensure_nltk_resources():
    logger.info("[NLTK] Checking required resources...")
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else resource)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            # Use quiet=True but log progress
            nltk.download(resource, quiet=True)
    
    # Reload Brown corpus data in detector after download
    load_brown_corpus_data()


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure NLTK resources are downloaded once at startup
    _ensure_nltk_resources()
    logger.info("Humaniser API ready and listening")
    yield
    logger.info("Humaniser API shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Humaniser API", version="2.1.0", lifespan=lifespan)

# CORS — configurable via CORS_ORIGINS env var (comma-separated)
# Added vercel production URL as default to avoid preflight failures
default_origins = [
    "http://localhost:3000",
    "https://sensorspine-humaniser-ukqw.vercel.app",
    "https://sensorspine-humaniser-ukqw-git-main-nandishsinha1403s-projects.vercel.app"
]
env_origins = os.environ.get("CORS_ORIGINS", "").split(",")
# Clean and ensure protocol for all origins
cors_origins = [o.strip() for o in env_origins if o.strip()] + default_origins
# Ensure each has protocol
cors_origins = [o if o.startswith("http") else f"https://{o}" for o in cors_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix="/api")


@app.get("/api")
async def api_root():
    return {"status": "api_online", "message": "Humaniser API router is active"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.1.0"}
