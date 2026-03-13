from fastapi import APIRouter
from db.supabase_client import get_alerts, acknowledge_alert, save_doctor_action

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/active")
async def active_alerts():
    """All unacknowledged alerts — for doctor dashboard."""
    return await get_alerts(active_only=True)


@router.get("/{patient_id}")
async def patient_alerts(patient_id: str):
    """All alerts for a specific patient."""
    return await get_alerts(patient_id=patient_id)


@router.put("/{alert_id}/acknowledge")
async def ack_alert(alert_id: str):
    """Doctor acknowledges an alert."""
    await acknowledge_alert(alert_id)
    return {"acknowledged": True}


@router.post("/{alert_id}/action")
async def doctor_action(alert_id: str, action_type: str, patient_id: str, note: str = ""):
    """Doctor takes action: test / visit / increase_monitoring."""
    action_id = await save_doctor_action({
        "alert_id":    alert_id,
        "patient_id":  patient_id,
        "action_type": action_type,
        "note":        note,
    })
    return {"action_id": action_id}