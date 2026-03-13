from fastapi import APIRouter, HTTPException
from schemas import ParseTextRequest, ParseTextResponse
from agents.input_agent import extract_vitals
from db.redis_client import get_session, clear_session

router = APIRouter(prefix="/api/input", tags=["input"])


@router.post("/parse-text", response_model=ParseTextResponse)
async def parse_text(req: ParseTextRequest):
    """Extract vital signs from raw text using Groq."""
    result = await extract_vitals(req.text, req.session_id)
    return ParseTextResponse(**result)


@router.get("/session/{session_id}")
async def get_session_state(session_id: str):
    """Return current session — which params are filled vs null."""
    data = await get_session(session_id)
    if not data:
        raise HTTPException(404, "Session not found or expired")
    return {"session_id": session_id, "params": data}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clear a session after successful prediction."""
    await clear_session(session_id)
    return {"cleared": True}