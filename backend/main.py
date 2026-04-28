"""
DeepShield API — FastAPI backend for sports media deepfake detection.
Integrates forensic analysis + Google Gemini for explainable AI verdicts.
"""

from __future__ import annotations

import os
import uuid
import shutil
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from detect import analyze_media

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepshield")

# ── Config ───────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepShield API",
    description="AI-powered deepfake detection for digital sports media integrity.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response models ───────────────────────────────────────────────────────────
class AnalysisResult(BaseModel):
    file: str
    authenticity_score: float
    verdict: str
    risk_level: str
    details: str
    gemini_explanation: str | None = None
    score_breakdown: dict | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def validate_file(file: UploadFile, size: int) -> None:
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size / 1024 / 1024:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB."
        )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        message="DeepShield API is running. POST a video or image to /analyze."
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness probe — used by Google Cloud Run / App Engine."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        message="DeepShield is healthy."
    )


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    """
    Analyze a video or image for deepfake manipulation.

    - Runs forensic analysis (sharpness, artifacts, optical flow, face consistency)
    - Runs AI model (ResNet-18 based deepfake detector)
    - Calls Google Gemini for a human-readable explanation
    """
    # Read file bytes once so we can check size before writing to disk
    file_bytes = await file.read()
    validate_file(file, len(file_bytes))

    # Save with a unique name to avoid collisions
    ext = Path(file.filename).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / safe_name

    try:
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        logger.info(f"Analyzing file: {file.filename} ({len(file_bytes) / 1024:.1f} KB)")
        result = analyze_media(str(file_path))

        return AnalysisResult(
            file=file.filename,
            authenticity_score=result["authenticity_score"],
            verdict=result["verdict"],
            risk_level=result["risk_level"],
            details=result["details"],
            gemini_explanation=result.get("gemini_explanation"),
            score_breakdown=result.get("score_breakdown"),
        )

    except ValueError as e:
        logger.warning(f"Analysis failed for {file.filename}: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error analyzing {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal analysis error. Please try again.")

    finally:
        # Always clean up uploaded file
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temp file: {safe_name}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."}
    )
