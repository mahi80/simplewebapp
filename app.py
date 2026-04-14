"""
app.py — CC Underwriting Inference API
Azure Web App runs this with:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from acp import govern

# ── Load model on startup ─────────────────────────────────────────────────────
MODEL    = mlflow.sklearn.load_model("model/rf")
MODEL = govern(MODEL, "acp_apply_SAL-JOB-06_w7bvn4j1", bu="Sales", budget=10.0)
FEATURES = json.loads(Path("model/features.json").read_text())
METRICS  = json.loads(Path("model/metrics.json").read_text())
SCALER_P = json.loads(Path("model/scaler.json").read_text())

scaler_mean  = np.array(SCALER_P["mean"])
scaler_scale = np.array(SCALER_P["scale"])

app = FastAPI(title="CC Underwriting API", version="1.0")


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    applicant_id: str | None = None
    features: dict              # pass any subset of the 176 features

class PredictResponse(BaseModel):
    applicant_id:    str | None
    decision:        str        # Approved / Declined
    approval_prob:   float
    scorecard_score: float      # FICO-style score
    risk_band:       str


# ── Helpers ───────────────────────────────────────────────────────────────────
def score(p: float) -> float:
    return round(600 + 72 * np.log((1 - p + 1e-8) / (p + 1e-8)), 1)

def risk_band(s: float) -> str:
    if s < 500:  return "Very High Risk"
    if s < 560:  return "High Risk"
    if s < 620:  return "Medium Risk"
    if s < 680:  return "Low Risk"
    if s < 740:  return "Very Low Risk"
    return "Excellent"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model")
def model_info():
    return {"features": len(FEATURES), "metrics": METRICS}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        row = np.array([req.features.get(f, 0) for f in FEATURES], dtype=float)
        row_scaled = (row - scaler_mean) / scaler_scale
        prob = float(MODEL.predict_proba(row_scaled.reshape(1, -1))[0][1])
        sc   = score(prob)
        return PredictResponse(
            applicant_id  = req.applicant_id,
            decision      = "Approved" if prob >= 0.5 else "Declined",
            approval_prob = round(prob, 4),
            scorecard_score = sc,
            risk_band     = risk_band(sc),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
