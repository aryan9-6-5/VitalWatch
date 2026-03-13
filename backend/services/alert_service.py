"""
Alert service — routes alerts based on risk tier.
CRITICAL: email doctor + push family + create ambulance ticket
WARNING:  email doctor only
NORMAL:   no alert, just log
"""

import httpx
import json
from config import settings
from db.supabase_client import (
    save_alert,
    create_ambulance_ticket,
    get_patient,
)


async def route_alert(
    patient_id: str,
    reading_id: str,
    risk_score: float,
    tier: str,
    flags: list,
    vitals: dict,
    explanation: str,
) -> dict:
    """
    Main entry point. Returns dict with alert_id and ticket_id (if CRITICAL).
    """
    result = {"alert_fired": False, "alert_id": None, "ticket_id": None}

    if tier == "NORMAL":
        return result

    # Fetch patient info for the email / ticket
    patient = await get_patient(patient_id)
    if not patient:
        patient = {"name": "Unknown", "doctor_email": "", "address": "Unknown", "age": "N/A"}

    # Save alert to DB
    alert_id = await save_alert({
        "patient_id":  patient_id,
        "reading_id":  reading_id,
        "tier":        tier,
        "status":      "OPEN",
        "risk_score":  risk_score,
        "flags":       flags,
        "acknowledged": False,
    })
    result["alert_id"] = alert_id
    result["alert_fired"] = True

    if tier == "WARNING":
        await _send_doctor_email(patient, risk_score, flags, vitals, explanation, tier)

    elif tier == "CRITICAL":
        await _send_doctor_email(patient, risk_score, flags, vitals, explanation, tier)
        await _send_push_notification(patient, risk_score, flags)
        ticket_id = await _create_ticket(patient_id, alert_id, patient, risk_score, flags, vitals)
        result["ticket_id"] = ticket_id

    return result


async def _send_doctor_email(patient: dict, risk_score: float, flags: list,
                              vitals: dict, explanation: str, tier: str):
    doctor_email = patient.get("doctor_email", "")
    if not doctor_email or not settings.RESEND_API_KEY:
        print(f"⚠️  Email skipped — no doctor_email or RESEND_API_KEY")
        return

    color = "#dc2626" if tier == "CRITICAL" else "#d97706"
    subject = (
        f"🚨 CRITICAL: {patient['name']} needs immediate attention"
        if tier == "CRITICAL"
        else f"⚠️ WARNING: {patient['name']} vitals need review"
    )

    flags_html = "".join(f"<li>{f}</li>" for f in flags)
    vitals_html = "".join(
        f"<tr><td style='padding:4px 12px'>{k.replace('_',' ').title()}</td>"
        f"<td style='padding:4px 12px'><b>{v}</b></td></tr>"
        for k, v in vitals.items() if k not in ("steps", "calories")
    )

    html_body = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:0 auto">
      <div style="background:{color};color:white;padding:16px 24px;border-radius:8px 8px 0 0">
        <h2 style="margin:0">{subject}</h2>
      </div>
      <div style="padding:24px;border:1px solid #e5e7eb;border-top:none;border-radius:0 0 8px 8px">
        <p><b>Patient:</b> {patient['name']}, Age {patient.get('age','N/A')}</p>
        <p><b>Risk score:</b> {risk_score:.2f} / 1.0</p>
        <p><b>AI assessment:</b> {explanation}</p>
        <h3>Clinical flags</h3>
        <ul>{flags_html}</ul>
        <h3>Latest vitals</h3>
        <table style="border-collapse:collapse;width:100%">
          <thead><tr style="background:#f3f4f6">
            <th style="padding:8px 12px;text-align:left">Parameter</th>
            <th style="padding:8px 12px;text-align:left">Value</th>
          </tr></thead>
          <tbody>{vitals_html}</tbody>
        </table>
        <p style="margin-top:24px;color:#6b7280;font-size:12px">
          Sent by VitalWatch · Post-discharge monitoring system
        </p>
      </div>
    </div>
    """

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "from":    settings.ALERT_EMAIL_FROM,
                    "to":      [doctor_email],
                    "subject": subject,
                    "html":    html_body,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                print(f"✅ Alert email sent to {doctor_email}")
            else:
                print(f"⚠️  Email failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"⚠️  Email exception: {e}")


async def _send_push_notification(patient: dict, risk_score: float, flags: list):
    if not settings.NTFY_TOPIC:
        return

    message = (
        f"CRITICAL ALERT: {patient['name']}\n"
        f"Risk score: {risk_score:.2f}\n"
        f"Flags: {', '.join(flags[:3])}"
    ).encode("utf-8", errors="replace").decode("utf-8")

    try:
        async with httpx.AsyncClient() as http:
            await http.post(
                f"https://ntfy.sh/{settings.NTFY_TOPIC}",
                content=message.encode("ascii", errors="replace"),
                headers={
                    "Title":    f"VitalWatch CRITICAL — {patient['name']}",
                    "Priority": "urgent",
                    "Tags":     "rotating_light",
                },
                timeout=8,
            )
            print(f"✅ Push notification sent")
    except Exception as e:
        print(f"⚠️  Push failed: {e}")


async def _create_ticket(patient_id, alert_id, patient, risk_score, flags, vitals) -> str:
    ticket_data = {
        "patient_id":       patient_id,
        "alert_id":         alert_id,
        "status":           "OPEN",
        "patient_name":     patient.get("name", "Unknown"),
        "patient_address":  patient.get("address", ""),
        "vitals_snapshot":  vitals,
        "risk_score":       risk_score,
        "flags":            flags,
    }
    ticket_id = await create_ambulance_ticket(ticket_data)
    print(f"✅ Ambulance ticket created: {ticket_id}")
    return ticket_id