"""
Computes the 12 derived features from 10 raw vital inputs.
Pure Python math — no ML, no API calls.
"""

from typing import List, Optional
import numpy as np


REQUIRED_PARAMS = [
    "systolic_bp", "diastolic_bp", "heart_rate", "spo2",
    "temperature", "respiratory_rate", "ecg", "cardiac_output"
]

OPTIONAL_PARAMS = ["steps", "calories"]


def compute_derived_features(raw: dict, history: Optional[List[dict]] = None) -> dict:
    """
    Takes 10 raw vitals, returns full 22-value dict (10 raw + 12 derived).
    history: list of past reading dicts (newest first), used for trend features.
    """
    h = history or []

    sbp   = raw["systolic_bp"]
    dbp   = raw["diastolic_bp"]
    hr    = raw["heart_rate"]
    spo2  = raw["spo2"]
    temp  = raw["temperature"]
    rr    = raw["respiratory_rate"]

    # ── Clinical indices ─────────────────────────────────────────────────────
    pulse_pressure  = sbp - dbp
    shock_index     = hr / sbp if sbp > 0 else 0.0
    oxygen_deficit  = 100.0 - spo2
    hr_temp_ratio   = hr / temp if temp > 0 else 0.0

    # ── Temporal trends (day-over-day delta) ─────────────────────────────────
    if h:
        prev = h[0]
        hr_trend   = hr   - prev.get("heart_rate",   hr)
        spo2_trend = spo2 - prev.get("spo2",         spo2)
        temp_trend = temp - prev.get("temperature",  temp)
        rr_trend   = rr   - prev.get("respiratory_rate", rr)
        risk_trend = 0.0  # computed after prediction; default 0 for new readings
    else:
        hr_trend = spo2_trend = temp_trend = rr_trend = risk_trend = 0.0

    # ── Rolling 3-day averages ────────────────────────────────────────────────
    if len(h) >= 2:
        window = h[:2]
        hr_mean_3d   = (hr   + sum(r.get("heart_rate",   hr)   for r in window)) / (len(window) + 1)
        spo2_mean_3d = (spo2 + sum(r.get("spo2",         spo2) for r in window)) / (len(window) + 1)
        temp_mean_3d = (temp + sum(r.get("temperature",  temp) for r in window)) / (len(window) + 1)
    else:
        hr_mean_3d   = hr
        spo2_mean_3d = spo2
        temp_mean_3d = temp

    derived = {
        "pulse_pressure":  pulse_pressure,
        "shock_index":     shock_index,
        "oxygen_deficit":  oxygen_deficit,
        "hr_temp_ratio":   hr_temp_ratio,
        "hr_trend":        hr_trend,
        "spo2_trend":      spo2_trend,
        "temp_trend":      temp_trend,
        "rr_trend":        rr_trend,
        "risk_trend":      risk_trend,
        "hr_mean_3d":      hr_mean_3d,
        "spo2_mean_3d":    spo2_mean_3d,
        "temp_mean_3d":    temp_mean_3d,
    }

    return {**raw, **derived}


def build_flags(raw: dict, risk_score: float) -> list:
    """Return a list of human-readable clinical flags for display."""
    flags = []

    if raw.get("spo2", 100) < 92:
        flags.append(f"SpO2 critical ({raw['spo2']}%)")
    elif raw.get("spo2", 100) < 95:
        flags.append(f"SpO2 borderline ({raw['spo2']}%)")

    if raw.get("heart_rate", 70) > 120:
        flags.append(f"HR elevated ({raw['heart_rate']} bpm)")
    elif raw.get("heart_rate", 70) < 50:
        flags.append(f"HR low ({raw['heart_rate']} bpm)")

    sbp = raw.get("systolic_bp", 120)
    if sbp > 160:
        flags.append(f"BP high ({sbp}/{raw.get('diastolic_bp', 80)} mmHg)")
    elif sbp < 90:
        flags.append(f"BP low ({sbp}/{raw.get('diastolic_bp', 80)} mmHg)")

    if raw.get("temperature", 37) > 38.5:
        flags.append(f"Fever ({raw['temperature']}°C)")
    elif raw.get("temperature", 37) < 36.0:
        flags.append(f"Hypothermia ({raw['temperature']}°C)")

    if raw.get("respiratory_rate", 16) > 24:
        flags.append(f"Tachypnoea ({raw['respiratory_rate']}/min)")

    shock = raw.get("heart_rate", 70) / max(raw.get("systolic_bp", 120), 1)
    if shock > 1.0:
        flags.append(f"Shock index high ({shock:.2f})")

    if not flags and risk_score > 0.4:
        flags.append("Multiple borderline values combined")

    return flags