"""
Ocean Air demo API — Vercel serverless function.

Does NOT modify agent.py — imports its step functions (call_router,
call_planner, call_executor, call_validator) and runs them in the same
order agent.py's own handle_user_message() does.

Stateless by design: conversation history is NOT stored server-side.
Vercel functions can spin up a fresh instance per request with no shared
memory, so the frontend sends the full running history with every call,
and gets the updated history back to send on the next turn.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import (
    OTHER_RESPONSE,
    call_executor,
    call_planner,
    call_router,
    call_validator,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
    trace: dict
    history: list[dict]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    history = [*req.history, {"role": "user", "content": req.message}]
    trace: dict = {}

    routing = call_router(history)
    trace["router"] = routing

    if routing["intent"] != "flight":
        reply = OTHER_RESPONSE
        history.append({"role": "assistant", "content": reply})
        return ChatResponse(reply=reply, trace=trace, history=history)

    decision = call_planner(history)
    trace["planner"] = decision

    result = call_executor(decision, history)
    draft_reply, tool_result = result["reply"], result["tool_result"]
    trace["executor_draft"] = draft_reply

    validation = call_validator(draft_reply, tool_result)
    trace["validator"] = validation

    reply = validation["final_response"] if validation["approved"] else validation["safe_fallback"]
    history.append({"role": "assistant", "content": reply})

    return ChatResponse(reply=reply, trace=trace, history=history)
