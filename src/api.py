"""
Phase 6 — Web API for the agent, plus the web page.

    GET  /               -> the chat page (web/index.html)
    GET  /api/health     -> {"status": "ok"}
    POST /api/ask        -> streams the agent's progress as NDJSON (one JSON object per line):
        {"type": "step",       "tool": "...", "args": {...}}       the agent calls a tool
        {"type": "tool_error", "tool": "...", "message": "..."}    a tool call failed (agent will retry)
        {"type": "answer",     "text": "...", "sql": "...", "columns": [...], "rows": [...], ...}
        {"type": "error",      "message": "..."}                   something went wrong

Run from the project root:
    uvicorn api:app --app-dir src --reload
Then open http://localhost:8000
"""
import json
import os
import re
import time
from collections import deque
from contextlib import closing
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from langchain.messages import HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

import graph_agent
from schema import get_connection
from tools import execute_with_timeout

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
RESULT_ROWS = 200  # rows shown in the result table

# Cost guards for a public demo: every question costs real API money.
MAX_QUESTION_CHARS = 500
MAX_PER_IP_PER_HOUR = int(os.getenv("MAX_QUESTIONS_PER_HOUR", "20"))
MAX_TOTAL_PER_DAY = int(os.getenv("MAX_QUESTIONS_PER_DAY", "300"))

app = FastAPI(title="SQL Analyst Agent")


# ---------- rate limiting (in memory; resets when the server restarts) ----------

_lock = Lock()
_per_ip: dict[str, deque] = {}
_all: deque = deque()


def _client_ip(request: Request) -> str:
    # Behind a host's proxy, the real client is the first address in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    with _lock:
        while _all and now - _all[0] > 86_400:
            _all.popleft()
        if len(_all) >= MAX_TOTAL_PER_DAY:
            raise HTTPException(429, "The demo has reached its daily question limit. Try again tomorrow.")
        history = _per_ip.setdefault(ip, deque())
        while history and now - history[0] > 3_600:
            history.popleft()
        if len(history) >= MAX_PER_IP_PER_HOUR:
            raise HTTPException(429, f"Limit of {MAX_PER_IP_PER_HOUR} questions per hour reached. Try again later.")
        history.append(now)
        _all.append(now)


# ---------- turning the agent's run into events ----------

def _event(**fields) -> str:
    return json.dumps(fields, default=str) + "\n"


def _clean_answer(text: str) -> tuple[str, str | None, bool]:
    """Split the agent's final text into (prose, sql, cannot_answer)."""
    blocks = SQL_BLOCK.findall(text)
    sql = blocks[-1].strip() if blocks else None
    prose = SQL_BLOCK.sub("", text).strip()
    cannot_answer = prose.upper().startswith("CANNOT_ANSWER")
    if cannot_answer:
        prose = re.sub(r"^CANNOT_ANSWER[\s:.\-]*", "", prose, flags=re.IGNORECASE).strip()
        prose = prose[:1].upper() + prose[1:]
    return prose, sql, cannot_answer


def _run_final_sql(sql: str) -> dict:
    """Re-run the agent's final SQL so the page can show the full result table."""
    try:
        with closing(get_connection()) as conn:
            columns, rows = execute_with_timeout(conn, sql, RESULT_ROWS + 1)
        return {"columns": columns, "rows": rows[:RESULT_ROWS],
                "truncated": len(rows) > RESULT_ROWS, "query_error": None}
    except Exception as error:
        return {"columns": [], "rows": [], "truncated": False, "query_error": str(error)}


def _stream_agent(question: str):
    start = time.perf_counter()
    config = {"recursion_limit": 2 * graph_agent.MAX_STEPS + 1}
    handler = graph_agent._langfuse_handler()
    if handler:
        config["callbacks"] = [handler]

    llm_calls = 0
    final_text = None
    tool_names = {}  # tool_call_id -> tool name, so errors can say which tool failed
    try:
        # stream_mode="updates" yields {node_name: what_that_node_returned} after each node runs.
        for update in graph_agent.graph.stream(
            {"messages": [HumanMessage(question)], "llm_calls": 0},
            config=config,
            stream_mode="updates",
        ):
            for node, output in update.items():
                for message in output.get("messages", []):
                    if node == "call_model":
                        llm_calls += 1
                        if message.tool_calls:
                            for call in message.tool_calls:
                                tool_names[call["id"]] = call["name"]
                                yield _event(type="step", tool=call["name"], args=call["args"])
                        else:
                            final_text = message.text
                    elif isinstance(message, ToolMessage) and message.status == "error":
                        yield _event(type="tool_error", tool=tool_names.get(message.tool_call_id, "tool"),
                                     message=str(message.content))
    except GraphRecursionError:
        yield _event(type="error", message=f"The agent stopped after {graph_agent.MAX_STEPS} steps without an answer.")
        return
    except Exception as error:
        yield _event(type="error", message=f"Agent failed: {error}")
        return
    finally:
        if handler:
            from langfuse import get_client

            get_client().flush()

    prose, sql, cannot_answer = _clean_answer(final_text or "")
    result = _run_final_sql(sql) if sql else {"columns": [], "rows": [], "truncated": False, "query_error": None}
    yield _event(
        type="answer",
        text=prose,
        sql=sql,
        cannot_answer=cannot_answer,
        llm_calls=llm_calls,
        elapsed_s=round(time.perf_counter() - start, 1),
        **result,
    )


# ---------- routes ----------

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask")
def ask(body: AskRequest, request: Request) -> StreamingResponse:
    _check_rate_limit(_client_ip(request))
    return StreamingResponse(_stream_agent(body.question.strip()), media_type="application/x-ndjson")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
