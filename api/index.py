"""
Ocean Air demo API — Vercel serverless function.

Self-contained: agent logic (router/planner/executor/validator) is inlined
here rather than imported from a separate agent.py. This avoids a Vercel
Python bundling issue where sibling modules next to the entrypoint aren't
reliably included in the deployed function, even with includeFiles config.

If you edit the agent's behavior, this file and your canonical agent.py
(used for local dev / the CLI loop) need to be updated together — they're
no longer the same file.

Stateless by design: conversation history is NOT stored server-side.
Vercel functions can spin up a fresh instance per request with no shared
memory, so the frontend sends the full running history with every call,
and gets the updated history back to send on the next turn.
"""

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-5"
ROUTER_MODEL = "claude-haiku-4-5-20251001"


def _parse_json_response(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# TOOLS (placeholder — this is a demo agent, no real API calls behind it)
# ---------------------------------------------------------------------------


def _example_tool() -> dict:
    return {"status": "placeholder — real tool logic added once the use case is known"}


TOOLS = {
    "example_tool": {
        "description": "Placeholder tool demonstrating the planner/executor/validator flow. Not a real capability — takes no arguments.",
        "run": lambda args: _example_tool(),
    },
}

TOOL_CATALOG = "\n".join(f"- {name}: {tool['description']}" for name, tool in TOOLS.items())

# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """You are the ROUTER for a support agent for
Ocean Air, a fictional travel and hospitality booking platform.

You do not answer the user directly and you do not call tools yourself.
Your only job is to classify the user's latest message into exactly one
of two categories, and output that classification as JSON — nothing else.
No prose, no markdown fences.

Categories:
- "flight": any flight-related request — booking a new flight, changing
  or cancelling an existing flight, flight status, baggage on a flight,
  etc.
- "other": anything not related to flights — e.g. loyalty program
  questions, billing/transaction disputes, hotel or car bookings, or
  unrelated complaints.

Respond with JSON matching exactly this shape:
{"intent": "<flight|other>"}
"""

VALID_INTENTS = {"flight", "other"}

OTHER_RESPONSE = (
    "I'm a flight-focused assistant, so I'm not able to help with that directly — "
    "I'll get you connected with someone who can."
)


def call_router(conversation_history: list[dict]) -> dict:
    response = client.messages.create(
        model=ROUTER_MODEL,
        max_tokens=32,
        system=ROUTER_SYSTEM_PROMPT,
        messages=conversation_history,
    )
    raw_text = "\n".join(block.text for block in response.content if block.type == "text").strip()

    try:
        decision = _parse_json_response(raw_text)
    except json.JSONDecodeError:
        decision = {"intent": "other"}

    if decision.get("intent") not in VALID_INTENTS:
        decision["intent"] = "other"

    return decision


# ---------------------------------------------------------------------------
# PLANNER
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = f"""You are the PLANNER for a support agent for
Ocean Air, a fictional travel and hospitality booking platform.

You do not answer the user directly and you do not call tools yourself.
Your only job is to decide what should happen next, and output that
decision as JSON — nothing else. No prose, no markdown fences.

Available tools:
{TOOL_CATALOG}

Respond with JSON matching exactly one of these shapes:

1. To answer directly (no tool needed):
{{"action": "respond_directly", "direct_response": "<your reply to the user>"}}

2. To call a tool:
{{"action": "call_tool", "tool": "<tool name>", "args": {{...}}}}
"""


def call_planner(conversation_history: list[dict]) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=PLANNER_SYSTEM_PROMPT,
        messages=conversation_history,
    )
    raw_text = "\n".join(block.text for block in response.content if block.type == "text").strip()

    try:
        return _parse_json_response(raw_text)
    except json.JSONDecodeError:
        return {"action": "respond_directly", "direct_response": raw_text}


# ---------------------------------------------------------------------------
# EXECUTOR
# ---------------------------------------------------------------------------

EXECUTOR_SYSTEM_PROMPT = """You are the EXECUTOR for an Ocean Air support
agent. You have just received a raw tool result. Turn it into a short,
natural, helpful reply to the user. Do not mention that a tool was called."""


def call_executor(decision: dict, conversation_history: list[dict]) -> dict:
    if decision["action"] == "respond_directly":
        return {"reply": decision["direct_response"], "tool_result": None}

    if decision["action"] == "call_tool":
        tool = TOOLS.get(decision["tool"])
        if tool is None:
            raise RuntimeError(f"Planner requested unknown tool: {decision['tool']}")

        tool_result = tool["run"](decision.get("args", {}))

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=EXECUTOR_SYSTEM_PROMPT,
            messages=[
                *conversation_history,
                {
                    "role": "user",
                    "content": f'Tool "{decision["tool"]}" returned: {json.dumps(tool_result)}',
                },
            ],
        )
        reply = "\n".join(block.text for block in response.content if block.type == "text")
        return {"reply": reply, "tool_result": tool_result}

    raise RuntimeError(f"Unknown planner action: {decision['action']}")


# ---------------------------------------------------------------------------
# VALIDATOR
# ---------------------------------------------------------------------------

VALIDATOR_SYSTEM_PROMPT = """You are the VALIDATOR for an Ocean Air support
agent. You will be shown a draft reply and, if a tool was used, the raw
tool result it should be based on.

Reject the draft if it does any of the following:
- States a specific date, price, or confirmation detail not present in the
  tool result
- Claims an action was completed (e.g. "I've cancelled your booking") when
  no tool call actually performed it
- Makes a firm commitment about cost, dates, or availability beyond what
  the tool data supports

If there was no tool result (the planner answered directly), only reject
if the draft makes a false claim of already having taken an action.

Respond with JSON matching exactly one of these shapes:

1. If the draft is fine as-is:
{"approved": true, "final_response": "<the draft, unchanged>"}

2. If the draft should be blocked:
{"approved": false, "reason": "<short internal note, not shown to user>", "safe_fallback": "<a safe, honest reply to send instead>"}
"""


def call_validator(draft_reply: str, tool_result: dict | None) -> dict:
    if tool_result is not None:
        context = f"Tool result the draft should be grounded in: {json.dumps(tool_result)}\n\nDraft reply: {draft_reply}"
    else:
        context = f"No tool was called for this turn.\n\nDraft reply: {draft_reply}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=VALIDATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )
    raw_text = "\n".join(block.text for block in response.content if block.type == "text").strip()

    try:
        return _parse_json_response(raw_text)
    except json.JSONDecodeError:
        return {
            "approved": False,
            "reason": f"Validator returned non-JSON output: {raw_text}",
            "safe_fallback": "I want to make sure I give you accurate information — let me get a human to confirm the details on this one.",
        }


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------

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
