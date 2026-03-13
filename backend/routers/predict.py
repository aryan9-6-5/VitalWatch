from fastapi import APIRouter, HTTPException
from schemas import (
    PredictRequest, PredictResponse,
    FullPredictRequest, FullPredictResponse,
)
from services.inference import predict, models_loaded
from agents.input_agent import extract_vitals
from agents.explanation_agent import explain_prediction
from services.alert_service import route_alert
from db.supabase_client import save_reading
from db.redis_client import clear_session

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
async def predict_vitals(req: PredictRequest):
    """
    Direct prediction — pass 10 raw vitals, get risk score + class.
    No Groq, no DB write. Fast internal route.
    """
    if not models_loaded():
        raise HTTPException(503, "ML models not loaded")

    raw = req.model_dump(exclude={"patient_id", "history"})
    result = predict(raw, req.history or [])

    return PredictResponse(
        risk_score  = result["risk_score"],
        risk_class  = result["risk_class"],
        confidence  = result["confidence"],
        tier        = result["tier"],
        flags       = result["flags"],
        action      = result["action"],
    )


@router.post("/full", response_model=FullPredictResponse)
async def full_pipeline(req: FullPredictRequest):
    """
    One-shot chatbot endpoint:
    text → extract vitals → (if complete) infer → explain → alert → save → respond
    """
    # Step 1 — extract vitals from text
    extracted = await extract_vitals(req.text, req.session_id)

    if not extracted["complete"]:
        # Not all required params found — ask the user for more
        return FullPredictResponse(
            complete   = False,
            session_id = extracted["session_id"],
            missing    = extracted["missing"],
            message    = extracted["message"],
        )

    # Step 2 — run ML inference
    if not models_loaded():
        raise HTTPException(503, "ML models not loaded")

    vitals = extracted["extracted"]
    result = predict(vitals, req.history or [])

    # Step 3 — explain with Groq
    explanation = await explain_prediction(
        vitals     = vitals,
        risk_score = result["risk_score"],
        risk_class = result["risk_class"],
        tier       = result["tier"],
        flags      = result["flags"],
        features   = result["features"],
    )

    # Step 4 — save reading to DB
    reading_id = await save_reading({
        "patient_id":       req.patient_id,
        "raw_inputs":       vitals,
        "derived_features": result["features"],
        "risk_score":       result["risk_score"],
        "risk_class":       result["risk_class"],
        "confidence":       result["confidence"],
        "tier":             result["tier"],
        "explanation":      explanation,
        "alert_fired":      result["tier"] != "NORMAL",
        "flags":            result["flags"],
    })

    # Step 5 — route alert (email / push / ticket)
    alert_result = await route_alert(
        patient_id  = req.patient_id,
        reading_id  = reading_id,
        risk_score  = result["risk_score"],
        tier        = result["tier"],
        flags       = result["flags"],
        vitals      = vitals,
        explanation = explanation,
    )

    # Step 6 — clear session
    await clear_session(extracted["session_id"])

    # Build chatbot reply
    tier_emoji = {"CRITICAL": "🚨", "WARNING": "⚠️", "NORMAL": "✅"}
    message = (
        f"{tier_emoji.get(result['tier'], '📊')} **{result['tier']}** "
        f"(risk score: {result['risk_score']:.2f})\n\n"
        f"{explanation}"
    )
    if alert_result.get("ticket_id"):
        message += "\n\n🚑 Emergency services have been dispatched."
    elif alert_result.get("alert_id"):
        message += "\n\n👨‍⚕️ Your doctor has been notified."

    return FullPredictResponse(
        complete   = True,
        session_id = extracted["session_id"],
        missing    = [],
        message    = message,
        prediction = PredictResponse(
            risk_score  = result["risk_score"],
            risk_class  = result["risk_class"],
            confidence  = result["confidence"],
            tier        = result["tier"],
            flags       = result["flags"],
            action      = result["action"],
            explanation = explanation,
        ),
        reading_id = reading_id,
    )