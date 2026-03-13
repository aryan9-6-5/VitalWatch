from fastapi import APIRouter, HTTPException
from schemas import PatientCreate, PatientResponse
from db.supabase_client import (
    create_patient, get_patient, get_all_patients,
    get_readings, get_alerts,
)

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.post("")
async def new_patient(body: PatientCreate):
    patient_id = await create_patient(body.model_dump())
    return {"patient_id": patient_id}


@router.get("")
async def list_patients():
    return await get_all_patients()


@router.get("/{patient_id}")
async def patient_detail(patient_id: str):
    p = await get_patient(patient_id)
    if not p:
        raise HTTPException(404, "Patient not found")
    return p


@router.get("/{patient_id}/readings")
async def patient_readings(patient_id: str, limit: int = 30):
    return await get_readings(patient_id, limit=limit)


@router.get("/{patient_id}/alerts")
async def patient_alerts_history(patient_id: str):
    return await get_alerts(patient_id=patient_id)


@router.get("/{patient_id}/trend")
async def patient_trend(patient_id: str):
    """7-day risk score + HR + SpO2 trend for charts."""
    readings = await get_readings(patient_id, limit=7)
    trend = []
    for r in reversed(readings):
        raw = r.get("raw_inputs", {})
        trend.append({
            "timestamp":   r["timestamp"],
            "risk_score":  r.get("risk_score", 0),
            "heart_rate":  raw.get("heart_rate"),
            "spo2":        raw.get("spo2"),
            "systolic_bp": raw.get("systolic_bp"),
        })
    return trend