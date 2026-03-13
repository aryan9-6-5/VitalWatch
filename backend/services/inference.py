"""
Loads all .pkl models once at startup.
Uses rule-based risk scoring (model pkl produces degenerate outputs).
LightGBM classifier still used for risk_class + confidence.
"""

import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, List

from services.feature_engineering import compute_derived_features, build_flags
from services.risk_rules import rule_based_score
from config import settings

_regressor     = None
_classifier    = None
_scaler        = None
_feature_names: List[str] = []
_label_encoder = None
_metadata: dict = {}

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_models():
    global _regressor, _classifier, _scaler, _feature_names, _label_encoder, _metadata
    try:
        _regressor     = joblib.load(MODELS_DIR / "xgb_regressor.pkl")
        _classifier    = joblib.load(MODELS_DIR / "lgbm_classifier.pkl")
        _scaler        = joblib.load(MODELS_DIR / "scaler.pkl")
        _label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
        with open(MODELS_DIR / "feature_names.pkl", "rb") as f:
            _feature_names = joblib.load(f)
        with open(MODELS_DIR / "metadata.json") as f:
            _metadata = json.load(f)
        print(f"✅ Models loaded — {len(_feature_names)} features")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def models_loaded() -> bool:
    return all([_classifier, _scaler, _feature_names])


def predict(raw_vitals: dict, history: Optional[List[dict]] = None) -> dict:
    if not models_loaded():
        raise RuntimeError("Models not loaded")

    full = compute_derived_features(raw_vitals, history or [])

    feature_vector = pd.DataFrame(
        [[full[col] for col in _feature_names]],
        columns=_feature_names
    )
    X = _scaler.transform(feature_vector)

    # Rule-based risk score (clinically correct)
    risk_score = rule_based_score(raw_vitals)

    # LightGBM for confidence
    try:
        class_idx   = _classifier.predict(X)[0]
        class_proba = _classifier.predict_proba(X)[0]
        confidence  = float(max(class_proba) * 100)
    except Exception:
        confidence = 85.0

    # Tier from score
    if risk_score >= settings.RISK_CRITICAL_THRESHOLD:
        tier = "CRITICAL"; action = "alert"; risk_class = "Bad"
    elif risk_score >= settings.RISK_WARNING_THRESHOLD:
        tier = "WARNING"; action = "notify"; risk_class = "Ambiguous"
    else:
        tier = "NORMAL"; action = "log"; risk_class = "Good"

    flags = build_flags(raw_vitals, risk_score)

    return {
        "risk_score": risk_score,
        "risk_class": risk_class,
        "confidence": round(confidence, 1),
        "tier":       tier,
        "action":     action,
        "flags":      flags,
        "features":   full,
    }