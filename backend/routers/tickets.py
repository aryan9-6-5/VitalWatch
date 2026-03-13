from fastapi import APIRouter, HTTPException
from schemas import TicketStatusUpdate, TicketResponse
from db.supabase_client import get_tickets, get_ticket, update_ticket_status

router = APIRouter(prefix="/api/tickets", tags=["tickets"])


@router.get("/open")
async def open_tickets():
    """All open/en-route tickets — for ambulance dashboard."""
    return await get_tickets(open_only=True)


@router.get("/all")
async def all_tickets():
    """All tickets including resolved."""
    return await get_tickets(open_only=False)


@router.get("/{ticket_id}")
async def ticket_detail(ticket_id: str):
    """Full ticket detail — patient info, vitals, location."""
    ticket = await get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    return ticket


@router.put("/{ticket_id}/status")
async def update_status(ticket_id: str, body: TicketStatusUpdate):
    """Paramedic updates ticket status: EN_ROUTE → ON_SCENE → RESOLVED."""
    valid = {"EN_ROUTE", "ON_SCENE", "RESOLVED"}
    if body.status not in valid:
        raise HTTPException(400, f"Status must be one of {valid}")
    await update_ticket_status(ticket_id, body.status)
    return {"ticket_id": ticket_id, "status": body.status}