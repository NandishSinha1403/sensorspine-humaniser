import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nltk

from app.api.routes import router as api_router

logger = logging.getLogger("humaniser.app")

# ---------------------------------------------------------------------------
# NLTK resource bootstrap (only downloads if missing)
# ---------------------------------------------------------------------------
NLTK_RESOURCES = ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                   "averaged_perceptron_tagger_eng", "wordnet", "stopwords", "brown"]

def _ensure_nltk_resources():
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else resource)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Application lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_nltk_resources()
    logger.info("Humaniser API ready")
    yield
    logger.info("Humaniser API shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Humaniser API", version="2.1.0", lifespan=lifespan)

# CORS — configurable via CORS_ORIGINS env var (comma-separated)
cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.1.0"}
