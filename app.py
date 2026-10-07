"""Pantry Pal: a chat agent that plans a week of dinners and prices them.

Gemini on Vertex AI through LiteLLM. Run locally:
    uv run --env-file .env uvicorn app:app --reload --port 8080
"""
import json
import uuid
from pathlib import Path

import litellm
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import tools

MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are Pantry Pal, a meal-planning assistant for a young adult living in a
city who wants to eat well on a budget.

How you work:
0. If the user mentions groceries they already have, call update_pantry first (before
   searching or pricing). Those items are free in every cost breakdown, and you should say
   which ingredients were counted as owned. When they ask what they can make with what they
   have, or want the cheapest option, use suggest_pantry_meals.
   Each owned item only covers 2 more meals by default; once a committed meal uses its last
   one, tell the user it is used up and will cost money from then on.
1. When the user describes a craving, dish, or ingredient, call search_recipes and pick the
   best fit. Offer one or two options rather than a long list.
2. Call estimate_recipe_cost for the recipe you recommend, with servings set to how many
   people the user is cooking for. Report the tool's headline figures exactly: the cash to
   spend and the cost per serving. Never recompute them or swap in value_used_per_serving;
   mention value_used only if the user asks, as the value of just the amount used. Also mention any
   if ingredients had low confidence or no price match. If there are more than two ingredients,
   no need to list them all out.
3. When the user decides to cook a meal, call estimate_recipe_cost again with
   add_to_cart=true, then call suggest_leftover_meals and show what each follow-up meal
   would cost per serving. Point out how the cost per meal drops as leftovers get reused.
4. Offer a wine pairing for the meal they chose. Only call suggest_wine_pairing when the
   user asks about wine or says yes to your offer. Choose the wine style or appellation
   yourself (for example Chianti for a tomato pasta) and pass it as wine_query; the wine
   database cannot search by food.
5. If they ask how their week is going, call weekly_summary.

Rules:
- Prices are estimates from a Trader Joe's price list. Say so briefly.
- Never invent prices, recipes, or wines. Use only what the tools return.
- If a tool returns an error, read its hint, fix the call if you can, and otherwise tell the
  user plainly what is unavailable and carry on with the rest.
- Be concise and friendly. Use short paragraphs and simple lists."""


MAX_EMPTY_RETRIES = 2
NUDGE = {"role": "user",
         "content": "Please reply to my last message now, using any tool results above."}


def ask_model(messages: list[dict]):
    """One model call. Gemini sometimes returns a completely empty reply (no text and no tool
    call), so retry a couple of times. The last retry adds a short nudge that is NOT saved to
    the conversation. Returns None if every attempt came back empty."""
    for attempt in range(MAX_EMPTY_RETRIES + 1):
        sent = messages + [NUDGE] if attempt == MAX_EMPTY_RETRIES else messages
        response = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=sent,
            tools=tools.TOOLS,
        )
        reply = response.choices[0].message
        if reply.tool_calls or (reply.content or "").strip():
            return reply
        print(f"[empty model reply, attempt {attempt + 1}] "
              f"finish_reason={response.choices[0].finish_reason} reply={reply.model_dump()}")
    return None


def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """Complete until the model answers without asking for a tool.

    Returns the final text and a record of every tool call made along the way.
    """
    tool_calls = []

    for _ in range(MAX_TOOL_ROUNDS):
        reply = ask_model(messages)
        if reply is None:
            return "The model sent back an empty reply. Please send your message again.", tool_calls

        # Append assistant's reply (text, tool calls, or both) to the context.
        # model_dump() keeps it a plain dict: the raw object carries provider-specific
        # fields that trip Pydantic when LiteLLM re-serializes it next round.
        messages += [reply.model_dump()]

        if not reply.tool_calls:
            return reply.content, tool_calls

        # The harness, not the model, runs each tool and appends the result
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = tools.run_tool(call.function.name, args)
            tool_calls += [{"name": call.function.name, "args": args, "result": result}]

            messages += [{"role": "tool", "tool_call_id": call.id, "content": result}]

    return "Sorry, I hit my tool-call limit before finishing.", tool_calls


# --- Session store ---

# session_id -> {"messages": [...], "state": this session's kitchen and cart}.
# In memory, single process.
sessions: dict[str, dict] = {}

# --- FastAPI app ---

app = FastAPI(title="Pantry Pal")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create the session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "state": tools.new_state(),
        }
    session = sessions[session_id]
    messages = session["messages"]

    checkpoint = len(messages)
    messages.append({"role": "user", "content": request.message})

    token = tools.begin_request(session["state"])
    try:
        response, tool_calls = run_agent(messages)
    except Exception as e:
        # Auth, billing, a model that is not running: show it in the chat, not as a 500.
        print(f"[chat error] {type(e).__name__}: {e}")
        del messages[checkpoint:]  # drop the half-finished turn so the next message starts clean
        response, tool_calls = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", []
    finally:
        tools.end_request(token)

    return ChatResponse(response=response or "I didn't get a response. Please try again.",
                        session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port= 8080)
