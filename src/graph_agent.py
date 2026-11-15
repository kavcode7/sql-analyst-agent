"""
Phase 3 — The same agent as agent.py, rebuilt as a LangGraph graph and traced in Langfuse.

    START -> call_model -> (tool calls?) -> call_tools -> call_model -> ... -> END

Run from the project root:
    python src/graph_agent.py "How many orders were cancelled?"
    python src/graph_agent.py --graph        # print the graph as a Mermaid diagram
"""
import json
import operator
import os
import sys
from contextlib import closing
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

import tools as db_tools
from agent import SYSTEM_PROMPT  # same prompt as Phase 2, so the comparison is fair
from schema import get_connection

load_dotenv()

MODEL = "claude-haiku-5-5"
MAX_STEPS = 10


# ---------- tools ----------
# Same functions as Phase 2, wrapped with @tool. The docstring becomes the
# description the LLM reads, and the type hints become the input schema.
# Each call opens its own read-only connection: SQLite connections can't be
# shared across threads, and LangGraph may run nodes in worker threads.

def _with_db(fn, *args):
    with closing(get_connection()) as conn:
        return fn(conn, *args)


@tool
def list_tables() -> str:
    """List every table in the database with its row count. Call this first to see what data exists."""
    return _with_db(db_tools.list_tables)


@tool
def describe_table(table: str) -> str:
    """Show a table's columns, their types, and 3 example rows. Use it to learn column
    names and how values are formatted before writing SQL."""
    return _with_db(db_tools.describe_table, table)


@tool
def get_distinct_values(table: str, column: str, limit: int = 30) -> str:
    """Show the most common values in a column with their counts, plus how many distinct
    values exist. Use it before filtering on a text value (a status, category, city or
    state) to get the exact spelling stored in the data."""
    return _with_db(db_tools.get_distinct_values, table, column, limit)


@tool
def run_query(sql: str) -> str:
    """Run one read-only SQLite SELECT and return up to 50 rows. If the query fails, the
    error message is returned so you can fix the query and try again."""
    return _with_db(db_tools.run_query, sql)


TOOLS = [list_tables, describe_table, get_distinct_values, run_query]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# timeout: give up on a single API call after 60s (the default is much longer);
# max_retries: retry a failed or throttled call up to 3 times.
model = init_chat_model(MODEL, model_provider="anthropic", timeout=60, max_retries=3)
model_with_tools = model.bind_tools(TOOLS)


# ---------- state ----------
# The state is everything the graph carries between nodes.
# `operator.add` means: when a node returns new messages, APPEND them (don't replace).

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# ---------- nodes ----------

def call_model(state: AgentState) -> dict:
    """Ask the LLM what to do next: call tools, or answer."""
    response = model_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}


def call_tools(state: AgentState) -> dict:
    """Run every tool the LLM asked for and return the results as ToolMessages."""
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        try:
            output = TOOLS_BY_NAME[tool_call["name"]].invoke(tool_call["args"])
            results.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
        except Exception as error:
            # Same idea as Phase 2: hand the error back so the model can fix itself.
            results.append(
                ToolMessage(content=f"Error: {error}", tool_call_id=tool_call["id"], status="error")
            )
    return {"messages": results}


# ---------- edges ----------

def should_continue(state: AgentState) -> Literal["call_tools", "__end__"]:
    """If the LLM asked for tools, run them; otherwise we're done."""
    if state["messages"][-1].tool_calls:
        return "call_tools"
    return END


builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("call_tools", call_tools)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, ["call_tools", END])
builder.add_edge("call_tools", "call_model")
graph = builder.compile()


# ---------- running it ----------

def _langfuse_handler():
    """Return a Langfuse callback if keys are configured, else None (tracing is optional)."""
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def run_graph_agent(question: str) -> dict:
    # Each loop is 2 graph steps (call_model + call_tools), so this caps us at MAX_STEPS loops.
    config = {"recursion_limit": 2 * MAX_STEPS + 1}
    handler = _langfuse_handler()
    if handler:
        config["callbacks"] = [handler]

    try:
        final = graph.invoke(
            {"messages": [HumanMessage(question)], "llm_calls": 0}, config=config
        )
    except GraphRecursionError:
        return {"answer": f"Stopped: no answer after {MAX_STEPS} steps.", "llm_calls": MAX_STEPS, "messages": []}

    return {
        "answer": final["messages"][-1].text,
        "llm_calls": final["llm_calls"],
        "messages": final["messages"],
    }


def main() -> None:
    if sys.argv[1:] == ["--graph"]:
        print(graph.get_graph().draw_mermaid())
        return

    question = " ".join(sys.argv[1:]) or input("Question: ")
    result = run_graph_agent(question)

    # Print a trace like Phase 2's, rebuilt from the message history.
    for message in result["messages"]:
        for tool_call in getattr(message, "tool_calls", None) or []:
            print(f"[tool] {tool_call['name']}({json.dumps(tool_call['args'])})")
        if isinstance(message, ToolMessage) and message.status == "error":
            print(f"       -> {message.content}")

    print(f"\n{result['answer']}\n\n(LLM calls: {result['llm_calls']})")

    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse import get_client

        get_client().flush()  # short scripts can exit before traces are sent


if __name__ == "__main__":
    main()
