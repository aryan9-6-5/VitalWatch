"""
Run this from backend/ folder:
python diagnose.py
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("models")

print("Loading models...")
regressor  = joblib.load(MODELS_DIR / "xgb_regressor.pkl")
classifier = joblib.load(MODELS_DIR / "lgbm_classifier.pkl")
scaler     = joblib.load(MODELS_DIR / "scaler.pkl")
feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")

print(f"Feature names ({len(feature_names)}): {feature_names}")
print()

def test(label, vals):
    df = pd.DataFrame([vals], columns=feature_names)
    X  = scaler.transform(df)
    score = float(regressor.predict(X)[0])
    cls   = classifier.predict(X)[0]
    try:
        cls_label = label_encoder.inverse_transform([cls])[0]
    except:
        cls_label = str(cls)
    proba = classifier.predict_proba(X)[0]
    print(f"{label}: score={score:.4f}  class={cls_label}  confidence={max(proba)*100:.1f}%")

# Build feature vectors manually
# feature order from pkl
fn = feature_names

def make(sbp,dbp,hr,spo2,temp,rr,ecg,co,steps=0,cals=0):
    pp   = sbp - dbp
    si   = hr / sbp
    od   = 100 - spo2
    htr  = hr / temp
    return {
        "systolic_bp": sbp, "diastolic_bp": dbp, "heart_rate": hr,
        "spo2": spo2, "temperature": temp, "respiratory_rate": rr,
        "ecg": ecg, "cardiac_output": co, "steps": steps, "calories": cals,
        "pulse_pressure": pp, "shock_index": si, "oxygen_deficit": od,
        "hr_temp_ratio": htr,
        "hr_trend": 0, "spo2_trend": 0, "temp_trend": 0, "rr_trend": 0,
        "risk_trend": 0,
        "hr_mean_3d": hr, "spo2_mean_3d": spo2, "temp_mean_3d": temp,
    }

print("=== TEST CASES ===")
test("NORMAL  (healthy)",    make(112, 72, 68, 99, 36.6, 14, 1.0, 5.8))
test("WARNING (borderline)", make(118, 76, 92, 97, 37.1, 16, 1.02, 5.3))
test("CRITICAL (dangerous)", make(85,  50, 128, 83, 39.2, 28, 0.6, 2.1))
test("EXTREME (near death)", make(70,  40, 150, 75, 40.0, 35, 0.4, 1.5))

print()
print("=== FEATURE ORDER CHECK ===")
print("Expected features:", fn)
sample = make(112, 72, 68, 99, 36.6, 14, 1.0, 5.8)
missing = [f for f in fn if f not in sample]
extra   = [f for f in sample if f not in fn]
print(f"Missing from sample: {missing}")
print(f"Extra in sample:     {extra}")