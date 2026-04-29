# 🛡️ DeepShield

## 🌐 Live Demo
**https://deepshield-sooty.vercel.app**

> AI-powered deepfake detection for digital sports media integrity.

DeepShield helps journalists, broadcasters, and sports organizations instantly verify whether a sports video or image has been AI-manipulated — protecting fans, athletes, and the integrity of sport itself.

> **Google Solution Challenge 2026 — UN SDG #16: Peace, Justice, and Strong Institutions**
> *Combating misinformation in sports media through transparent, explainable AI.*

---

## 🎯 Problem Statement

AI-generated deepfakes of sporting moments — fake goals, fabricated fouls, staged controversies — are increasingly being shared as real footage on social media. This erodes fan trust, damages athletes' reputations, and undermines the credibility of sports journalism.

---

## 🏗️ Architecture

User Upload (video/image)
│
▼
┌───────────────────┐
│   FastAPI Backend  │  ← Python, hosted on Railway
└───────────────────┘
│
├──► 🔬 Forensic Pipeline (OpenCV)
│       ├── Sharpness analysis (Laplacian variance)
│       ├── Compression artifact detection (DCT)
│       ├── Temporal consistency (optical flow)
│       └── Face presence consistency (Haar cascades)
│
├──► 🤖 AI Deepfake Detector (PyTorch ResNet-18)
│       └── Per-face binary classification
│
└──► 💬 Google Gemini 1.5 Flash
└── Plain-English explanation for journalists
---

## ✅ Google Technology Used

| Technology | Role |
|---|---|
| **Google Gemini 1.5 Flash** | Generates journalist-friendly explanations of each analysis result |
| **Google Cloud Run** (planned) | Serverless deployment of the FastAPI backend |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ashaw23871/deepshield.git
cd deepshield
```

### 2. Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key-here"
uvicorn main:app --reload
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 📡 API Reference

### `GET /health`
```json
{ "status": "ok", "version": "1.0.0", "message": "DeepShield is healthy." }
```

### `POST /analyze`
```bash
curl -X POST https://deepshield-production-3f11.up.railway.app/analyze \
  -F "file=@match_clip.mp4"
```

**Response:**
```json
{
  "file": "match_clip.mp4",
  "authenticity_score": 73.4,
  "verdict": "Likely Authentic (Minor Edits Possible)",
  "risk_level": "Low-Medium",
  "score_breakdown": {
    "sharpness": 82.0,
    "compression_artifacts": 61.5,
    "temporal_consistency": 88.3,
    "face_consistency": 75.0,
    "ai_model_authenticity": 64.2
  },
  "gemini_explanation": "This video shows strong temporal consistency and stable face detection throughout..."
}
```

---

## 📊 Scoring Methodology

| Signal | Weight | What it measures |
|---|---|---|
| Sharpness | 10% | Natural blur vs. generation artifacts |
| Compression artifacts | 10% | DCT block energy |
| Temporal consistency | 20% | Optical flow variance between frames |
| Face consistency | 20% | Stable face detection throughout |
| AI model (ResNet-18) | 40% | Neural deepfake classifier |

| Score | Verdict | Risk |
|---|---|---|
| 80–100 | Authentic | Low |
| 60–79 | Likely Authentic | Low-Medium |
| 40–59 | Suspicious | Medium-High |
| 0–39 | High Probability of AI Manipulation | High |

---

## 🌍 UN SDG Alignment

**SDG #16 — Peace, Justice and Strong Institutions:**
DeepShield promotes media integrity and access to accurate information — cornerstones of a just society.

---

## 🔗 Links

| | |
|---|---|
| 🌐 Live Demo | https://deepshield-sooty.vercel.app |
| 💻 GitHub | https://github.com/ashaw23871/deepshield |
| 🔧 Backend API | https://deepshield-production-3f11.up.railway.app |

---

## 📄 License
MIT License
