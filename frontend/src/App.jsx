import { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  Upload, Shield, AlertTriangle, CheckCircle,
  XCircle, Activity, Eye, Zap, FileVideo, X
} from "lucide-react";
import "./App.css";

// ── Score ring ────────────────────────────────────────────────────────────────
function ScoreRing({ score }) {
  const r = 70;
  const circ = 2 * Math.PI * r;
  const [displayed, setDisplayed] = useState(0);

  useEffect(() => {
    let start = null;
    const duration = 1200;
    const animate = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setDisplayed(Math.round(ease * score));
      if (progress < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [score]);

  const color =
    score >= 80 ? "#22d3a5" :
    score >= 60 ? "#38bdf8" :
    score >= 40 ? "#f59e0b" : "#ef4444";

  const offset = circ - (displayed / 100) * circ;

  return (
    <div className="score-ring-wrap">
      <svg width="180" height="180" viewBox="0 0 180 180">
        <circle cx="90" cy="90" r={r} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="12" />
        <circle
          cx="90" cy="90" r={r} fill="none"
          stroke={color} strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          transform="rotate(-90 90 90)"
          style={{ transition: "stroke 0.4s" }}
        />
      </svg>
      <div className="score-ring-center">
        <span className="score-number" style={{ color }}>{displayed}</span>
        <span className="score-label">/ 100</span>
      </div>
    </div>
  );
}

// ── Breakdown bar ─────────────────────────────────────────────────────────────
function BreakdownBar({ label, value, delay }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setWidth(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);

  const color =
    value >= 70 ? "#22d3a5" :
    value >= 45 ? "#f59e0b" : "#ef4444";

  return (
    <div className="breakdown-row">
      <span className="breakdown-label">{label}</span>
      <div className="breakdown-track">
        <div
          className="breakdown-fill"
          style={{ width: `${width}%`, background: color }}
        />
      </div>
      <span className="breakdown-value" style={{ color }}>{value.toFixed(0)}</span>
    </div>
  );
}

// ── Verdict badge ─────────────────────────────────────────────────────────────
function VerdictBadge({ verdict, risk }) {
  const isGood = risk === "Low";
  const isMed = risk === "Low-Medium" || risk === "Medium-High";
  const icon = isGood ? <CheckCircle size={20} /> :
    isMed ? <AlertTriangle size={20} /> : <XCircle size={20} />;
  const cls = isGood ? "badge-good" : isMed ? "badge-warn" : "badge-danger";

  return (
    <div className={`verdict-badge ${cls}`}>
      {icon}
      <div>
        <div className="verdict-text">{verdict}</div>
        <div className="verdict-risk">Risk: {risk}</div>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    if (f.type.startsWith("image/")) {
      setPreview(URL.createObjectURL(f));
    } else {
      setPreview(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    try {
      const res = await axios.post("https://deepshield-production-3f11.up.railway.app/analyze", formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Analysis failed. Make sure the backend is running on port 8000.");
    }
    setLoading(false);
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  const breakdownLabels = {
    sharpness: "Sharpness",
    compression_artifacts: "Compression Artifacts",
    temporal_consistency: "Temporal Consistency",
    face_consistency: "Face Consistency",
    ai_model_authenticity: "AI Model Score",
  };

  return (
    <div className="app">
      {/* Background grid */}
      <div className="bg-grid" />

      <div className="container">
        {/* Header */}
        <div className="header">
          <div className="logo-wrap">
            <Shield size={36} className="logo-icon" />
            <span className="logo-text">Deep<span>Shield</span></span>
          </div>
          <p className="tagline">Protecting the integrity of digital sports media</p>
        </div>

        {/* Upload zone */}
        {!result && (
          <div
            className={`upload-zone ${dragOver ? "drag-over" : ""} ${file ? "has-file" : ""}`}
            onClick={() => inputRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/*,image/*"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])}
            />

            {preview ? (
              <img src={preview} alt="preview" className="preview-img" />
            ) : (
              <div className="upload-idle">
                <div className="upload-icon-wrap">
                  <FileVideo size={40} className="upload-icon" />
                </div>
                <p className="upload-title">
                  {file ? file.name : "Drop a video or image here"}
                </p>
                <p className="upload-sub">
                  {file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "MP4, MOV, JPG, PNG — up to 200 MB"}
                </p>
              </div>
            )}

            {file && (
              <button
                className="clear-btn"
                onClick={(e) => { e.stopPropagation(); reset(); }}
              >
                <X size={14} />
              </button>
            )}
          </div>
        )}

        {/* Analyze button */}
        {!result && (
          <button
            className="analyze-btn"
            onClick={handleUpload}
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <span className="spinner" />
                Analyzing...
              </>
            ) : (
              <>
                <Zap size={18} />
                Analyze Media
              </>
            )}
          </button>
        )}

        {/* Results */}
        {result && (
          <div className="results">
            <div className="results-header">
              <h2 className="results-title">
                <Activity size={20} /> Analysis Complete
              </h2>
              <button className="new-btn" onClick={reset}>
                <Upload size={14} /> New File
              </button>
            </div>

            <p className="results-filename">
              <FileVideo size={14} /> {result.file}
            </p>

            {/* Score + verdict */}
            <div className="score-verdict-row">
              <ScoreRing score={result.authenticity_score} />
              <div className="verdict-col">
                <div className="authenticity-label">Authenticity Score</div>
                <VerdictBadge verdict={result.verdict} risk={result.risk_level} />
                <p className="details-text">{result.details}</p>
              </div>
            </div>

            {/* Score breakdown */}
            {result.score_breakdown && (
              <div className="breakdown-card">
                <h3 className="card-title"><Eye size={16} /> Signal Breakdown</h3>
                {Object.entries(result.score_breakdown).map(([key, val], i) => (
                  <BreakdownBar
                    key={key}
                    label={breakdownLabels[key] || key}
                    value={val}
                    delay={i * 120}
                  />
                ))}
              </div>
            )}

            {/* Gemini explanation */}
            {result.gemini_explanation && (
              <div className="gemini-card">
                <h3 className="card-title">
                  <span className="gemini-dot" />
                  Gemini AI Explanation
                </h3>
                <p className="gemini-text">{result.gemini_explanation}</p>
              </div>
            )}
          </div>
        )}

        <p className="footer-note">
          Powered by ResNet-18 · OpenCV · Google Gemini 1.5 Flash · v1.0
        </p>
      </div>
    </div>
  );
}
