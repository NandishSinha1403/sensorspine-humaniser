from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import tempfile
import random
from typing import List, Dict, Any
from app.corpus.ingester import ingest_pdf, extract_features, clean_text
from app.corpus.style_profile import list_profiles, load_profile, PROFILES_DIR
from app.core.detector import detect_ai_score
import fitz

app = FastAPI(title="Humaniser Trainer Server")

trainer_origins = os.environ.get(
    "TRAINER_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8001,http://127.0.0.1:8001"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in trainer_origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train/upload")
async def upload_pdf(field: str = Form(...), file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        print(f"Processing file: {file.filename} for field: {field}")
        result = ingest_pdf(tmp_path, field)
        features = result["features"]
        
        return {
            "filename": file.filename,
            "field": field,
            "sentences": len(features["sample_sentences"]),
            "vocab_size": len(features["top_vocab"]),
            "phrases": len(features["transition_phrases"]) + len(features["hedging_phrases"]),
            "stats": features["sentence_stats"]
        }
    except Exception as e:
        import traceback
        print(f"Error processing upload: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/train/upload-batch")
async def upload_batch(field: str = Form(...), files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        res = await upload_pdf(field=field, file=file)
        results.append(res)
    return results

@app.get("/train/profiles")
async def get_profiles():
    names = list_profiles()
    profiles = []
    for name in names:
        data = load_profile(name)
        if data:
            profiles.append({
                "field": name,
                "sentence_count": len(data.get("sample_sentences", [])),
                "vocab_size": len(data.get("top_vocab", [])),
                "phrase_count": len(data.get("transition_phrases", [])) + len(data.get("hedging_phrases", [])),
                "punctuation_stats": data.get("punctuation_profile", {})
            })
    return profiles

@app.get("/train/profile/{field}")
async def get_profile(field: str):
    data = load_profile(field)
    if not data:
        raise HTTPException(status_code=404, detail="Profile not found")
    return data

@app.delete("/train/profile/{field}")
async def delete_profile(field: str):
    file_path = os.path.join(PROFILES_DIR, f"{field}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "deleted", "field": field}
    raise HTTPException(status_code=404, detail="Profile not found")

@app.get("/train/quality-report")
async def quality_report():
    names = list_profiles()
    report = []
    for name in names:
        data = load_profile(name)
        if not data: continue
        
        samples = data.get("sample_sentences", [])
        if not samples: continue
        
        random_samples = random.sample(samples, min(5, len(samples)))
        scores = [detect_ai_score(s) for s in random_samples]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        report.append({
            "field": name,
            "sample_scores": scores,
            "average": avg_score,
            "verdict": "good corpus" if avg_score < 35 else "review papers"
        })
    return report

@app.get("/train/health")
async def health():
    return {"status": "healthy", "port": 8001}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
