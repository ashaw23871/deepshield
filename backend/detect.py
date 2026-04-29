"""
DeepShield detection pipeline.
Combines forensic signal analysis with a neural deepfake detector,
then calls Google Gemini to produce a human-readable explanation.
"""

from __future__ import annotations

import os
import logging
import cv2
import numpy as np
from typing import List, Dict

from video_utils import extract_frames
from face_utils import detect_faces
from deepfake_model import DeepfakeModel
from gemini_utils import get_gemini_explanation

logger = logging.getLogger("deepshield.detect")
deepfake_model = DeepfakeModel()


# ── Forensic signals ──────────────────────────────────────────────────────────

def laplacian_variance(frame: np.ndarray) -> float:
    """Higher = sharper frame. Low sharpness may indicate compression or generation artifacts."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def jpeg_artifact_score(frame: np.ndarray) -> float:
    """
    Measures DCT high-frequency energy in 8×8 blocks.
    High energy = significant JPEG artifacts, common in re-encoded or AI-generated content.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255.0
    h, w = gray.shape
    block_size = 8
    dct_energy = 0.0
    count = 0

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i + block_size, j:j + block_size]
            dct_block = cv2.dct(block)
            dct_energy += np.sum(np.abs(dct_block[1:, 1:]))
            count += 1

    return float(dct_energy / count) if count > 0 else 0.0


def temporal_consistency(frames: List[np.ndarray]) -> float:
    """
    Uses Farneback optical flow to measure motion variance between frames.
    High variance = inconsistent motion, a common deepfake tell.
    Returns a score in [0, 1] where 1 = perfectly consistent.
    """
    if len(frames) < 2:
        return 0.0

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    motion_scores = []

    for frame in frames[1:]:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_scores.append(np.mean(magnitude))
        prev_gray = curr_gray

    if not motion_scores:
        return 0.0

    variance = np.var(motion_scores)
    return float(1.0 / (1.0 + variance))


def face_presence_consistency(frames: List[np.ndarray]) -> float:
    """
    Counts detected faces per frame. Flickering face count is a deepfake signal.
    Returns a score in [0, 1] where 1 = perfectly consistent face presence.
    """
    if not frames:
        return 0.0

    face_counts = [len(detect_faces(frame)) for frame in frames]
    mean_faces = np.mean(face_counts)

    if mean_faces == 0:
        return 0.0

    variance = np.var(face_counts)
    return float(1.0 / (1.0 + variance))


def normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val - min_val == 0:
        return 0.0
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_video(video_path: str) -> Dict[str, object]:
    frames = extract_frames(video_path, max_frames=10)

    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    # --- Forensic signals ---
    sharpness_raw = float(np.mean([laplacian_variance(f) for f in frames]))
    artifacts_raw = float(np.mean([jpeg_artifact_score(f) for f in frames]))
    temporal_raw = temporal_consistency(frames)
    face_raw = face_presence_consistency(frames)

    # --- AI model ---
    deepfake_score = deepfake_model.predict_video(frames)
    ai_authenticity = 1.0 - deepfake_score

    # --- Normalize to [0, 1] ---
    sharpness_norm = normalize(sharpness_raw, 50, 500)
    artifacts_norm = normalize(artifacts_raw, 0.01, 0.2)
    temporal_norm = normalize(temporal_raw, 0.0, 1.0)
    face_norm = normalize(face_raw, 0.0, 1.0)

    # --- Weighted final score (0–100) ---
    # AI model carries most weight; forensic signals provide corroborating evidence
    final_score = (
        sharpness_norm  * 0.10 +
        artifacts_norm  * 0.10 +
        temporal_norm   * 0.20 +
        face_norm       * 0.20 +
        ai_authenticity * 0.40
    ) * 100.0

    final_score = float(np.clip(final_score, 0.0, 100.0))

    # --- Verdict ---
    if final_score >= 80:
        verdict, risk = "Authentic", "Low"
    elif final_score >= 60:
        verdict, risk = "Likely Authentic (Minor Edits Possible)", "Low-Medium"
    elif final_score >= 40:
        verdict, risk = "Suspicious — Possible Manipulation", "Medium-High"
    else:
        verdict, risk = "High Probability of AI Manipulation", "High"

    score_breakdown = {
        "sharpness":           round(sharpness_norm * 100, 1),
        "compression_artifacts": round(artifacts_norm * 100, 1),
        "temporal_consistency":  round(temporal_norm * 100, 1),
        "face_consistency":      round(face_norm * 100, 1),
        "ai_model_authenticity": round(ai_authenticity * 100, 1),
    }

    # --- Gemini explanation ---
    gemini_explanation = get_gemini_explanation(
        verdict=verdict,
        score=final_score,
        breakdown=score_breakdown,
    )

    return {
        "authenticity_score": round(final_score, 2),
        "verdict": verdict,
        "risk_level": risk,
        "details": "DeepShield hybrid AI + forensic analysis (ResNet-18 + optical flow + DCT artifacts).",
        "score_breakdown": score_breakdown,
        "gemini_explanation": gemini_explanation,
    }


def analyze_media(file_path: str) -> Dict[str, object]:
    """Entry point for both video and image files."""
    return analyze_video(file_path)
