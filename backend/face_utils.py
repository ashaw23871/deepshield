"""
face_utils.py — Haar cascade face detection for DeepShield.

Used to isolate face regions before running the deepfake model,
and to track face-count consistency across video frames.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger("deepshield.face_utils")

# Load once at import time — cv2.CascadeClassifier is not thread-safe,
# but in a single-worker FastAPI deployment this is fine.
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

if face_cascade.empty():
    logger.error(f"Failed to load Haar cascade from: {_CASCADE_PATH}")


def detect_faces(frame: np.ndarray):
    """
    Detect frontal faces in a BGR frame using Haar cascades.

    Returns:
        List of (x, y, w, h) bounding boxes, or an empty list on failure.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
        )
        # detectMultiScale returns () when nothing is found, not []
        return faces if len(faces) > 0 else []
    except Exception as e:
        logger.warning(f"Face detection failed on frame: {e}")
        return []
