"""
Exact risk score formula from the training notebook.
Thresholds: Good < 0.06 | Ambiguous 0.06-0.30 | Bad >= 0.30
"""

def rule_based_score(raw: dict) -> float:
    score = 0
    max_possible_score = 4.5

    sbp  = raw.get("systolic_bp", 120)
    dbp  = raw.get("diastolic_bp", 80)
    hr   = raw.get("heart_rate", 70)
    spo2 = raw.get("spo2", 98)
    temp = raw.get("temperature", 37.0)
    rr   = raw.get("respiratory_rate", 16)
    ecg  = raw.get("ecg", 1.0)
    co   = raw.get("cardiac_output", 5.0)

    if sbp < 90:     score += (90  - sbp) / 30
    elif sbp > 140:  score += (sbp - 140) / 40
    if dbp < 60:     score += (60  - dbp) / 20
    elif dbp > 90:   score += (dbp - 90)  / 30

    if hr < 50:      score += (50 - hr)  / 25
    elif hr > 110:   score += (hr - 110) / 40
    elif hr > 100:   score += (hr - 100) / 50

    if spo2 < 90:    score += (90  - spo2) / 10
    elif spo2 < 95:  score += (95  - spo2) / 15
    elif spo2 < 100: score += (100 - spo2) / 100

    if temp < 35:     score += (35   - temp) / 5
    elif temp > 38:   score += (temp - 38)   / 3
    elif temp > 37.2: score += (temp - 37.2) / 10

    if rr < 10:      score += (10 - rr)  / 5
    elif rr > 24:    score += (rr - 24)  / 11
    elif rr > 20:    score += (rr - 20)  / 20

    if ecg < 0.5:    score += (0.5 - ecg) / 2.5
    elif ecg > 1.5:  score += (ecg - 1.5) / 2.5

    if co < 3:       score += (3  - co) / 3
    elif co > 9:     score += (co - 9)  / 3

    return round(min(score / max_possible_score, 1.0), 3)