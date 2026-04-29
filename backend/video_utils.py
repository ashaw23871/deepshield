"""
video_utils.py — Frame extraction utilities for DeepShield.

Handles both video files and static images uniformly,
so the detection pipeline receives a list of frames in both cases.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger("deepshield.video_utils")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_frames(file_path: str, max_frames: int = 10) -> List[np.ndarray]:
    """
    Extract up to `max_frames` evenly-spaced frames from a video,
    or return a single frame list for image files.

    Args:
        file_path: Path to a video or image file.
        max_frames: Maximum number of frames to extract from video.

    Returns:
        List of BGR numpy arrays (H × W × 3).
    """
    ext = Path(file_path).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return _extract_image(file_path)

    return _extract_video_frames(file_path, max_frames)


def _extract_image(file_path: str) -> List[np.ndarray]:
    """Read a single image and return it as a one-element list."""
    img = cv2.imread(file_path)
    if img is None:
        logger.error(f"Could not read image: {file_path}")
        return []
    logger.info(f"Loaded image {file_path} — shape: {img.shape}")
    return [img]


def _extract_video_frames(video_path: str, max_frames: int) -> List[np.ndarray]:
    """Sample evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []

    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return frames

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, total // max_frames)

    logger.info(
        f"Extracting frames from {video_path} — "
        f"total={total}, fps={fps:.1f}, step={step}, target={max_frames}"
    )

    count = 0
    i = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            frames.append(frame)
            count += 1
        i += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames
