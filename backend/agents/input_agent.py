"""
Input Agent — uses Groq to extract vital signs from any text/paste.
Returns structured JSON + list of missing params.
"""

import json
import uuid
from groq import Groq
from config import settings
from db.redis_client import save_session, get_session

client = Groq(api_key=settings.GROQ_API_KEY)

REQUIRED = ["systolic_bp", "diastolic_bp", "heart_rate", "spo2",
            "temperature", "respiratory_rate", "ecg", "cardiac_output"]
OPTIONAL = ["steps", "calories"]
ALL_PARAMS = REQUIRED + OPTIONAL

PARAM_LABELS = {
    "systolic_bp":       "systolic blood pressure (mmHg)",
    "diastolic_bp":      "diastolic blood pressure (mmHg)",
    "heart_rate":        "heart rate (bpm)",
    "spo2":              "oxygen saturation SpO2 (%)",
    "temperature":       "body temperature (°C)",
    "respiratory_rate":  "respiratory rate (breaths/min)",
    "ecg":               "ECG amplitude (mV)",
    "cardiac_output":    "cardiac output (L/min)",
    "steps":             "daily steps",
    "calories":          "daily calories",
}

SYSTEM_PROMPT = """You are a medical vital signs extractor.
Extract vital signs from the user's message and return ONLY valid JSON.

Schema (all numeric or null):
{
  "systolic_bp": null,
  "diastolic_bp": null,
  "heart_rate": null,
  "spo2": null,
  "temperature": null,
  "respiratory_rate": null,
  "ecg": null,
  "cardiac_output": null,
  "steps": null,
  "calories": null
}

Rules:
- BP like "118/76" → systolic_bp: 118, diastolic_bp: 76
- "HR 92" or "pulse 92" → heart_rate: 92
- "SpO2 97%" or "oxygen 97" → spo2: 97
- "temp 37.1" or "37.1°C" → temperature: 37.1
- "RR 16" or "breathing 16" → respiratory_rate: 16
- Set any param you cannot find to null
- Return ONLY the JSON object, no explanation, no markdown
"""


async def extract_vitals(text: str, session_id: str = None) -> dict:
    """
    Extract vitals from text. Merges with existing session if session_id given.
    Returns: { session_id, extracted, missing, complete, message }
    """
    # Load existing session state (partial extraction from previous turn)
    session_id = session_id or str(uuid.uuid4())
    existing = await get_session(session_id) or {}

    # Call Groq
    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        raw_json = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]

        new_params = json.loads(raw_json)
    except Exception as e:
        # Groq failed — return existing state with error
        new_params = {p: None for p in ALL_PARAMS}

    # Merge: existing values take precedence only where new extraction is null
    merged = {}
    for param in ALL_PARAMS:
        new_val = new_params.get(param)
        old_val = existing.get(param)
        merged[param] = new_val if new_val is not None else old_val

    # Save merged state to Redis
    await save_session(session_id, merged)

    # Compute missing (required only)
    missing = [p for p in REQUIRED if merged.get(p) is None]
    complete = len(missing) == 0

    # Fill optional defaults
    if merged.get("steps") is None:
        merged["steps"] = 0.0
    if merged.get("calories") is None:
        merged["calories"] = 0.0

    # Build chatbot message
    message = _build_message(missing, complete, merged)

    return {
        "session_id": session_id,
        "extracted":  merged,
        "missing":    missing,
        "complete":   complete,
        "message":    message,
    }


def _build_message(missing: list, complete: bool, extracted: dict) -> str:
    if complete:
        return (
            "✅ Got all your vitals! Running analysis now...\n"
            f"BP: {extracted['systolic_bp']}/{extracted['diastolic_bp']} mmHg | "
            f"HR: {extracted['heart_rate']} bpm | "
            f"SpO2: {extracted['spo2']}% | "
            f"Temp: {extracted['temperature']}°C"
        )

    filled = [p for p in REQUIRED if extracted.get(p) is not None]

    if not filled:
        return (
            "Hi! I'm your VitalWatch assistant. Please share your current vital signs.\n\n"
            "You can type them, paste from a report, or upload a PDF/image.\n"
            "Example: *BP 118/76, HR 92, SpO2 97%, temp 37.1, RR 16*"
        )

    missing_labels = [PARAM_LABELS[p] for p in missing]
    got_count = len(filled)
    return (
        f"Got {got_count}/{len(REQUIRED)} readings so far. "
        f"Still need: **{', '.join(missing_labels)}**.\n"
        "Please provide the missing values to complete your assessment."
    )