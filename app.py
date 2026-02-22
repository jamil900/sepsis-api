from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Sepsis Ensemble Prediction API",
    description="API untuk prediksi risiko sepsis pasien menggunakan ensemble model",
    version="1.0"
)

# ==== LOAD MODEL SAAT STARTUP ====
try:
    scaler = joblib.load("scaler.pkl")
    rf = joblib.load("rf_model.pkl")
    svc = joblib.load("svc_model.pkl")
    nb = joblib.load("nb_model.pkl")
    w_rf, w_svc, w_nb = joblib.load("ensemble_weights.pkl")
except Exception as e:
    raise RuntimeError(f"Gagal load model artefak: {e}")

# ==== SKEMA INPUT ====
class PatientData(BaseModel):
    heart_rate: float = Field(..., example=95)
    respiratory_rate: float = Field(..., example=22)
    temperature: float = Field(..., example=38.1)
    wbc_count: float = Field(..., example=12)
    lactate_level: float = Field(..., example=2.1)
    age: float = Field(..., example=67)
    num_comorbidities: int = Field(..., example=2)

FEATURES = [
    "heart_rate",
    "respiratory_rate",
    "temperature",
    "wbc_count",
    "lactate_level",
    "age",
    "num_comorbidities"
]

@app.get("/")
def home():
    return {"message": "Sepsis Ensemble API running"}

# ==== ENDPOINT PREDIKSI ====
@app.post("/predict")
def predict(data: PatientData):
    try:
        # convert ke dataframe sesuai urutan training
        df = pd.DataFrame([[getattr(data, f) for f in FEATURES]], columns=FEATURES)

        # preprocessing
        X = scaler.transform(df)

        # probabilitas tiap model
        p_rf = rf.predict_proba(X)[0][1]
        p_svc = svc.predict_proba(X)[0][1]
        p_nb = nb.predict_proba(X)[0][1]

        # ensemble weighted average
        prob = (w_rf*p_rf + w_svc*p_svc + w_nb*p_nb) / (w_rf+w_svc+w_nb)
        pred = int(prob >= 0.5)

        return {
            "sepsis_risk_prediction": pred,
            "risk_probability": float(prob)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
