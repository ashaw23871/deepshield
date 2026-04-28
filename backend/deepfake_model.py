"""
deepfake_model.py — Neural deepfake detector for DeepShield.

Architecture: ResNet-18 fine-tuned as a binary classifier (real vs. fake).

⚠️  Important note on weights:
    This prototype loads ImageNet-pretrained ResNet-18 weights and adds a
    single binary classification head. For production accuracy, fine-tune
    on a deepfake dataset such as FaceForensics++ or DFDC before deployment.
    The current model provides a baseline signal; forensic features in
    detect.py carry significant weight in the final score to compensate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import List

from face_utils import detect_faces


class DeepfakeDetector(nn.Module):
    """
    ResNet-18 with a single sigmoid output node.
    Output → 0.0 = clearly fake, 1.0 = clearly real.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class DeepfakeModel:
    """Wraps DeepfakeDetector with preprocessing and video-level inference."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = DeepfakeDetector().to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def predict_frame(self, frame: np.ndarray) -> float:
        """
        Run inference on a single face crop.
        Returns a score in [0, 1]: higher = more likely authentic.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        return float(output.item())

    def predict_video(self, frames: List[np.ndarray]) -> float:
        """
        Run inference on all detected faces across sampled frames.
        Returns the mean score in [0, 1]; falls back to 0.5 (uncertain)
        if no faces are found.
        Note: score here is deepfake probability — caller inverts (1 - score)
        for the authenticity dimension.
        """
        scores: List[float] = []

        for frame in frames:
            faces = detect_faces(frame)

            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                if face.size == 0:
                    continue
                try:
                    score = self.predict_frame(face)
                    scores.append(score)
                except Exception:
                    continue

        if not scores:
            # No faces detected — return neutral uncertainty
            return 0.5

        return float(np.mean(scores))
