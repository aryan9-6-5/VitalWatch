"""
Loads all .pkl models once at startup.
Exposes a single predict() function used by all routes.
"""

import joblib
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, List

from services.feature_engineering import compute_derived_features, build_flags
from config import settings

# ── Model registry (loaded once at startup) ───────────────────────────────────
_regressor  = None
_classifier = None
_scaler     = None
_feature_names: List[str] = []
_label_encoder = None
_metadata: dict = {}

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_models():
    """Called once at FastAPI startup. Loads all pkl files into module globals."""
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
        print(f"   Feature order: {_feature_names[:5]}...")
        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def models_loaded() -> bool:
    return all([_regressor, _classifier, _scaler, _feature_names])


def predict(raw_vitals: dict, history: Optional[List[dict]] = None) -> dict:
    """
    Full inference pipeline:
    1. Compute 12 derived features
    2. Assemble 20-feature array in training column order
    3. Scale with StandardScaler
    4. Run XGBoost (risk_score) + LightGBM (risk_class + confidence)
    5. Return structured result
    """
    if not models_loaded():
        raise RuntimeError("Models not loaded — call load_models() at startup")

    # Step 1 — derive features
    full = compute_derived_features(raw_vitals, history or [])

    # Step 2 — reindex to training column order (CRITICAL — one wrong column = wrong prediction)
    try:
        feature_vector = np.array([full[col] for col in _feature_names], dtype=np.float64)
    except KeyError as e:
        raise ValueError(f"Missing feature: {e}. Available: {list(full.keys())}")

    # Step 3 — scale
    X = _scaler.transform(feature_vector.reshape(1, -1))

    # Step 4 — inference
    risk_score = float(_regressor.predict(X)[0])
    risk_score = max(0.0, min(1.0, risk_score))   # clamp to [0, 1]

    class_idx   = _classifier.predict(X)[0]
    class_proba = _classifier.predict_proba(X)[0]
    confidence  = float(max(class_proba) * 100)

    # Decode class label
    try:
        risk_class = _label_encoder.inverse_transform([class_idx])[0]
    except Exception:
        # Fallback: derive class from score
        if risk_score >= settings.RISK_CRITICAL_THRESHOLD:
            risk_class = "Bad"
        elif risk_score >= settings.RISK_WARNING_THRESHOLD:
            risk_class = "Ambiguous"
        else:
            risk_class = "Good"

    # Step 5 — tier + action
    if risk_score >= settings.RISK_CRITICAL_THRESHOLD:
        tier   = "CRITICAL"
        action = "alert"
    elif risk_score >= settings.RISK_WARNING_THRESHOLD:
        tier   = "WARNING"
        action = "notify"
    else:
        tier   = "NORMAL"
        action = "log"

    flags = build_flags(raw_vitals, risk_score)

    return {
        "risk_score":  risk_score,
        "risk_class":  risk_class,
        "confidence":  round(confidence, 1),
        "tier":        tier,
        "action":      action,
        "flags":       flags,
        "features":    full,       # full 22-value dict for explanation agent
    }