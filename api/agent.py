"""
Travel & hospitality agent skeleton (Python port).

Architecture: planner, executor, and validator, each
a SEPARATE call to Claude. No specific use case is wired in yet — that
arrives at the start of the real session. The tools below are placeholders
that exist only so the full flow is demonstratable end-to-end; they get
replaced by whatever real API/action the session calls for.

The comments marked [STEP 0] / [STEP 1] / [STEP 2] / [STEP 3] appear in
that order top-to-bottom in this file, one per call to Claude, in the
order they run:

  [STEP 0] ROUTER — classifies the message as booking / change /
           cancellation / other before anything else runs. Only "booking"
           continues into the flow below; the rest get a templated
           deferral reply and never reach the planner.
  [STEP 1] PLANNER — Decomposes message and decides WHAT should happen next.
  [STEP 2] EXECUTOR — carries out that decision: calls a tool if needed,
           then phrases the reply.
  [STEP 3] VALIDATOR — checks the executor's draft reply against the guardrails we have set.

"""

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-5"
# The router only classifies intent — it never reasons about task
# details — so it runs on a smaller, faster model than the rest of the flow.
ROUTER_MODEL = "claude-haiku-4-5-20251001"


def _parse_json_response(raw_text: str) -> dict:
    # Every Claude call in this file is told "no markdown fences" but
    # sometimes wraps its JSON in a ```json ... ``` block anyway. Strip a
    # fence if present before parsing, so that doesn't get treated the same
    # as genuinely malformed output.
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()
    return json.loads(text)

# ---------------------------------------------------------------------------
# TOOLS
# Fake/placeholder. Real tools (flight search, booking lookup, cancellation,
# etc.) get registered here once the actual use case is known. Kept
# deliberately generic so it doesn't presume flights vs. hotels vs. cars.
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
# [STEP 0] ROUTER
# Runs before the planner on every message and decides whether it's
# "flight" (booking, change, cancellation, or any other flight-related
# request) or "other" (anything not flight-related — loyalty program
# questions, billing disputes, unrelated complaints). Only "flight"
# continues into the planner/executor/validator flow; "other" gets a
# short templated response, no tool or execution logic behind it.
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

# Static reply for anything outside flights. Deliberately not a Claude
# call — there's no tool or execution logic behind this branch yet.
OTHER_RESPONSE = (
    "I'm a flight-focused assistant, so I'm not able to help with that directly — "
    "I'll get you connected with someone who can."
)


def call_router(conversation_history: list[dict]) -> dict:
    # Same call pattern as call_planner, just a lighter model since this is
    # classification, not reasoning about task details.
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
        # Can't trust an unparsable classification — fail to "other" so the
        # user gets a clarifying question instead of silently mis-routing.
        decision = {"intent": "other"}

    if decision.get("intent") not in VALID_INTENTS:
        decision["intent"] = "other"

    return decision


# --- end of STEP 0 ---

# ---------------------------------------------------------------------------
# [STEP 1] PLANNER
# Decides WHAT should happen next. Never touches a tool, never talks to the
# user directly. Outputs plain JSON so the decision is inspectable — you can
# print it, log it, point at it — rather than a hidden model behavior.
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
    # One stateless request to Claude, fed the full running history.
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
        # The planner sometimes replies in plain text instead of using the
        # respond_directly JSON shape. Treat that as a direct response rather
        # than erroring, so it still passes through the validator below.
        return {"action": "respond_directly", "direct_response": raw_text}


# ---------------------------------------------------------------------------
# [STEP 2] EXECUTOR
# Carries out whatever the planner decided. This is the ONLY place that
# touches tools. Returns the tool_result alongside the draft reply (or None,
# if no tool was called) so the validator below has something concrete to
# check the reply against.
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

        # This is the one line in the entire file where the agent touches the
        # outside world. A real integration replaces tool["run"](...) itself —
        # everything around it (planner, validator, loop) stays the same.
        tool_result = tool["run"](decision.get("args", {}))

        # A second, separate call to Claude — the executor's own call — to
        # phrase the raw tool result as a natural-language reply.
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
# [STEP 3] VALIDATOR
# Sits between the executor's draft reply and the user. A third, separate
# Claude call — never calls a tool, never talks to the user directly. Its
# only job is to check the draft against what the tool actually returned:
# does this reply say anything the data doesn't support (a wrong date, a
# claimed cancellation/refund that didn't happen, a commitment the tool
# result doesn't back up)?
#
# Without this step, the executor's phrasing call could hallucinate
# something ("You're all set, I've rebooked your flight!") and it would go
# straight out the door with nothing checking it first.
#
# Chosen failure mode here (deliberately the simple option): if the
# validator rejects a reply, swap in a safe fallback and stop — no retry
# loop back to the executor yet. That's a reasonable next step, but it
# adds branching complexity this skeleton doesn't need.
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
        # If the validator itself is malformed, fail SAFE — block the draft
        # rather than let an unverifiable reply through.
        return {
            "approved": False,
            "reason": f"Validator returned non-JSON output: {raw_text}",
            "safe_fallback": "I want to make sure I give you accurate information — let me get a human to confirm the details on this one.",
        }


# --- end of STEP 3 ---

# ---------------------------------------------------------------------------
# AGENT LOOP
# Three calls deep: planner -> executor -> validator. Only the validator's
# final, approved text ever gets pushed into memory or shown to the user.
# This list is the agent's entire memory — nothing else persists between
# turns, since the API itself is stateless.
# ---------------------------------------------------------------------------

conversation_history: list[dict] = []


def handle_user_message(user_message: str) -> str:
    conversation_history.append({"role": "user", "content": user_message})

    routing = call_router(conversation_history)
    intent = routing["intent"]
    print(f"  [router decision]    {json.dumps(routing)}")

    if intent != "flight":
        final_reply = OTHER_RESPONSE
        conversation_history.append({"role": "assistant", "content": final_reply})
        return final_reply

    decision = call_planner(conversation_history)
    print(f"  [planner decision]   {json.dumps(decision)}")

    result = call_executor(decision, conversation_history)
    draft_reply, tool_result = result["reply"], result["tool_result"]
    print(f"  [executor draft]     {draft_reply}")

    validation = call_validator(draft_reply, tool_result)
    print(f"  [validator decision] {json.dumps(validation)}")

    final_reply = validation["final_response"] if validation["approved"] else validation["safe_fallback"]

    conversation_history.append({"role": "assistant", "content": final_reply})
    return final_reply


def main():
    print('Ocean Air travel agent skeleton (Python). Type a message to try it out. Type "exit" to quit.\n')

    while True:
        user_message = input("You: ")
        if user_message.strip().lower() == "exit":
            break

        try:
            reply = handle_user_message(user_message)
            print(f"Agent: {reply}\n")
        except Exception as err:
            # Even at this skeleton stage, an unhandled error here would crash
            # the loop. Surface it and keep going instead.
            print(f"Error: {err}\n")


if __name__ == "__main__":
    main()
