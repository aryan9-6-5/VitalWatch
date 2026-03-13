"""
Supabase client — all database reads and writes.
Uses the supabase-py client for PostgreSQL operations.
"""

import uuid
from datetime import datetime
from typing import Optional
from supabase import create_client, Client
from config import settings

_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    return _client


# ── Patients ─────────────────────────────────────────────────────────────────

async def create_patient(data: dict) -> str:
    db = get_client()
    result = db.table("patients").insert(data).execute()
    return result.data[0]["id"]


async def get_patient(patient_id: str) -> Optional[dict]:
    db = get_client()
    result = db.table("patients").select("*").eq("id", patient_id).execute()
    return result.data[0] if result.data else None


async def get_all_patients() -> list:
    db = get_client()
    result = db.table("patients").select("*").order("created_at", desc=True).execute()
    return result.data or []


# ── Readings ─────────────────────────────────────────────────────────────────

async def save_reading(data: dict) -> str:
    db = get_client()
    reading_id = str(uuid.uuid4())
    payload = {
        "id":               reading_id,
        "patient_id":       data["patient_id"],
        "timestamp":        datetime.utcnow().isoformat(),
        "raw_inputs":       data.get("raw_inputs", {}),
        "derived_features": data.get("derived_features", {}),
        "risk_score":       data.get("risk_score", 0.0),
        "risk_class":       data.get("risk_class", "Good"),
        "confidence":       data.get("confidence", 0.0),
        "tier":             data.get("tier", "NORMAL"),
        "explanation":      data.get("explanation", ""),
        "alert_fired":      data.get("alert_fired", False),
        "flags":            data.get("flags", []),
    }
    db.table("readings").insert(payload).execute()
    return reading_id


async def get_readings(patient_id: str, limit: int = 30) -> list:
    db = get_client()
    result = (
        db.table("readings")
        .select("*")
        .eq("patient_id", patient_id)
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


async def get_all_readings(limit: int = 100) -> list:
    db = get_client()
    result = (
        db.table("readings")
        .select("*, patients(name)")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


# ── Alerts ────────────────────────────────────────────────────────────────────

async def save_alert(data: dict) -> str:
    db = get_client()
    alert_id = str(uuid.uuid4())
    payload = {
        "id":           alert_id,
        "patient_id":   data["patient_id"],
        "reading_id":   data.get("reading_id"),
        "tier":         data["tier"],
        "status":       data.get("status", "OPEN"),
        "risk_score":   data.get("risk_score", 0.0),
        "flags":        data.get("flags", []),
        "fired_at":     datetime.utcnow().isoformat(),
        "acknowledged": False,
    }
    db.table("alerts").insert(payload).execute()
    return alert_id


async def get_alerts(patient_id: str = None, active_only: bool = False) -> list:
    db = get_client()
    query = db.table("alerts").select("*, patients(name)").order("fired_at", desc=True)
    if patient_id:
        query = query.eq("patient_id", patient_id)
    if active_only:
        query = query.eq("acknowledged", False)
    return query.execute().data or []


async def acknowledge_alert(alert_id: str) -> bool:
    db = get_client()
    db.table("alerts").update({
        "acknowledged":     True,
        "acknowledged_at":  datetime.utcnow().isoformat(),
    }).eq("id", alert_id).execute()
    return True


# ── Ambulance tickets ─────────────────────────────────────────────────────────

async def create_ambulance_ticket(data: dict) -> str:
    db = get_client()
    ticket_id = str(uuid.uuid4())
    payload = {
        "id":               ticket_id,
        "patient_id":       data["patient_id"],
        "alert_id":         data.get("alert_id"),
        "status":           "OPEN",
        "patient_name":     data.get("patient_name", ""),
        "patient_address":  data.get("patient_address", ""),
        "vitals_snapshot":  data.get("vitals_snapshot", {}),
        "risk_score":       data.get("risk_score", 0.0),
        "flags":            data.get("flags", []),
        "fired_at":         datetime.utcnow().isoformat(),
    }
    db.table("ambulance_tickets").insert(payload).execute()
    return ticket_id


async def get_tickets(open_only: bool = True) -> list:
    db = get_client()
    query = db.table("ambulance_tickets").select("*").order("fired_at", desc=True)
    if open_only:
        query = query.neq("status", "RESOLVED")
    return query.execute().data or []


async def get_ticket(ticket_id: str) -> Optional[dict]:
    db = get_client()
    result = db.table("ambulance_tickets").select("*").eq("id", ticket_id).execute()
    return result.data[0] if result.data else None


async def update_ticket_status(ticket_id: str, status: str) -> bool:
    db = get_client()
    db.table("ambulance_tickets").update({"status": status}).eq("id", ticket_id).execute()
    return True


# ── Doctor actions ────────────────────────────────────────────────────────────

async def save_doctor_action(data: dict) -> str:
    db = get_client()
    action_id = str(uuid.uuid4())
    payload = {
        "id":          action_id,
        "alert_id":    data.get("alert_id"),
        "patient_id":  data.get("patient_id"),
        "action_type": data["action_type"],
        "note":        data.get("note", ""),
        "created_at":  datetime.utcnow().isoformat(),
    }
    db.table("doctor_actions").insert(payload).execute()
    return action_id