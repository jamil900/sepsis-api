from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Sepsis Ensemble API")

# load artefak
scaler = joblib.load("scaler.pkl")
rf = joblib.load("rf_model.pkl")
svc = joblib.load("svc_model.pkl")
nb = joblib.load("nb_model.pkl")

w_rf, w_svc, w_nb = joblib.load("ensemble_weights.pkl")

@app.get("/")
def home():
    return {"message": "Sepsis Ensemble API running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])
    X = scaler.transform(df)

    # ambil probabilitas masing-masing model
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