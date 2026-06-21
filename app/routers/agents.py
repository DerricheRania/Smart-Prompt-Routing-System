import uuid

from fastapi import APIRouter, HTTPException

from app.routers.router import reset_chat_session, route_prompt

router = APIRouter(
    prefix="/agents",
    tags=["agents"],
    responses={404: {"description": "Not found"}},
)


@router.get("/ask")
async def chat_endpoint(query: str, session_id: str | None = None):
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # session_id groups multi-turn conversations together (mainly used by the
    # chat intent, which keeps a running history per session). If the caller
    # doesn't supply one, generate a fresh one so each browser tab/session
    # doesn't accidentally share chat history with another.
    sid = session_id or str(uuid.uuid4())

    intent, answer = route_prompt(query, session_id=sid)

    return {
        "intent": intent,
        "agent_name": f"{intent.upper()} AGENT",
        "response": answer,
        "session_id": sid,
    }


@router.post("/reset")
async def reset_endpoint(session_id: str):
    """Clear stored chat history for a session — call this when starting a new conversation."""
    reset_chat_session(session_id)
    return {"status": "ok"}
