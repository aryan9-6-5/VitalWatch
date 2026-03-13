"""
Explanation Agent — uses Groq to turn ML output into plain English.
Different prompt per tier (CRITICAL / WARNING / NORMAL).
"""

from groq import Groq
from config import settings

client = Groq(api_key=settings.GROQ_API_KEY)

SYSTEM_CRITICAL = """You are a medical AI assistant explaining a CRITICAL vital signs alert.
Be direct, clear, and urgent — but not alarmist. 
Focus on which specific values are dangerous and what they indicate.
Write 2-3 sentences max. Mention that emergency services have been notified.
Do not use medical jargon the patient won't understand."""

SYSTEM_WARNING = """You are a medical AI assistant explaining a WARNING vital signs alert.
Be clear and informative. Explain which values are outside normal range and why that matters.
Suggest the patient follow their doctor's guidance. Write 2-3 sentences max.
Do not cause unnecessary panic."""

SYSTEM_NORMAL = """You are a medical AI assistant providing health feedback for normal vitals.
Be warm, encouraging, and give one practical health tip.
Write 2 sentences. Keep it positive and motivating."""


async def explain_prediction(
    vitals: dict,
    risk_score: float,
    risk_class: str,
    tier: str,
    flags: list,
    features: dict,
) -> str:
    """
    Takes prediction result and vital values, returns plain-English explanation.
    """
    system_prompt = {
        "CRITICAL": SYSTEM_CRITICAL,
        "WARNING":  SYSTEM_WARNING,
        "NORMAL":   SYSTEM_NORMAL,
    }.get(tier, SYSTEM_NORMAL)

    # Build a compact summary to send to Groq
    shock_index = features.get("shock_index", 0)
    pulse_press = features.get("pulse_pressure", 0)

    user_content = f"""
Patient vitals:
- Blood pressure: {vitals.get('systolic_bp')}/{vitals.get('diastolic_bp')} mmHg
- Heart rate: {vitals.get('heart_rate')} bpm
- SpO2: {vitals.get('spo2')}%
- Temperature: {vitals.get('temperature')}°C
- Respiratory rate: {vitals.get('respiratory_rate')}/min
- ECG: {vitals.get('ecg')} mV
- Cardiac output: {vitals.get('cardiac_output')} L/min

Computed clinical indices:
- Shock index: {shock_index:.2f}
- Pulse pressure: {pulse_press:.0f} mmHg

Risk score: {risk_score:.2f}/1.0
Risk class: {risk_class}
Alert tier: {tier}
Flags: {', '.join(flags) if flags else 'None'}

Please explain this result to the patient in plain English.
"""

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Fallback explanation if Groq fails
        if tier == "CRITICAL":
            return (
                f"Your vitals show critical values (risk score: {risk_score:.2f}). "
                f"Flags: {', '.join(flags)}. Emergency services have been notified."
            )
        elif tier == "WARNING":
            return (
                f"Some of your vitals need attention (risk score: {risk_score:.2f}). "
                f"Flags: {', '.join(flags)}. Your doctor has been notified."
            )
        else:
            return (
                f"Your vitals look good (risk score: {risk_score:.2f}). "
                "Keep up your healthy habits and stay hydrated."
            )