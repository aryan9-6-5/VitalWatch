"""
Upstash Redis (HTTP) — stores partial vital extraction sessions.
Each session is a JSON object keyed by session_id.
TTL: 30 minutes (1800 seconds).
"""

import json
import httpx
from config import settings

SESSION_TTL = 1800  # 30 min


async def save_session(session_id: str, data: dict) -> bool:
    if not settings.UPSTASH_REDIS_REST_URL:
        return False
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.UPSTASH_REDIS_REST_URL}/set/{session_id}",
                headers={"Authorization": f"Bearer {settings.UPSTASH_REDIS_REST_TOKEN}"},
                json={"value": json.dumps(data), "ex": SESSION_TTL},
                timeout=5,
            )
        return resp.status_code == 200
    except Exception as e:
        print(f"Redis save error: {e}")
        return False


async def get_session(session_id: str) -> dict | None:
    if not settings.UPSTASH_REDIS_REST_URL:
        return None
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.UPSTASH_REDIS_REST_URL}/get/{session_id}",
                headers={"Authorization": f"Bearer {settings.UPSTASH_REDIS_REST_TOKEN}"},
                timeout=5,
            )
        body = resp.json()
        if body.get("result"):
            return json.loads(body["result"])
    except Exception as e:
        print(f"Redis get error: {e}")
    return None


async def clear_session(session_id: str) -> bool:
    if not settings.UPSTASH_REDIS_REST_URL:
        return False
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.UPSTASH_REDIS_REST_URL}/del/{session_id}",
                headers={"Authorization": f"Bearer {settings.UPSTASH_REDIS_REST_TOKEN}"},
                timeout=5,
            )
        return resp.status_code == 200
    except Exception as e:
        print(f"Redis del error: {e}")
        return False