"""
Phase 2 — A hand-written agent loop.

    question -> LLM -> wants a tool? -> run it -> send result back -> LLM -> ... -> answer

Run from the project root:
    python src/agent.py "How many orders were cancelled?"
"""
import json
import sqlite3
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

from schema import get_connection
from tools import TOOLS, run_tool

load_dotenv()

MODEL = "claude-haiku-5-5"
MAX_STEPS = 10  # hard stop so a confused agent can't loop (and spend) forever

SYSTEM_PROMPT = """You are a data analyst answering questions about an e-commerce SQLite database.
You cannot see the database directly. Use the tools to explore it, then answer.

How to work:
1. Find the relevant tables with list_tables and describe_table.
2. Before filtering on a text value (a status, category, city or state), check its exact \
spelling with get_distinct_values. Never guess a value.
3. Run your query with run_query. If it errors or the result looks wrong, fix it and try again.
4. 4. Finish with a short answer to the question, then the final SQL you used in a ```sql block. \
The final SQL must return only what the question asks for: for "which X has the most/least", \
return one row (LIMIT 1). You may mention runners-up in your text, but not in the final SQL.

If the data cannot answer the question, start your answer with CANNOT_ANSWER, explain what \
data is missing, and do not include a ```sql block. You may suggest a related question the \
data can answer."""


def run_agent(question: str, client: Anthropic, conn: sqlite3.Connection, verbose: bool = True) -> str:
    # The conversation history IS the agent's memory: every step sees everything before it.
    messages = [{"role": "user", "content": question}]

    for step in range(1, MAX_STEPS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # No tool requested -> the model is done; return its text answer.
        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        # Keep the model's turn (including its tool_use blocks) in the history.
        messages.append({"role": "assistant", "content": response.content})

        # The model may ask for several tools at once: run all, reply with all results together.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"[step {step}] {block.name}({json.dumps(block.input)})")
            try:
                output = run_tool(conn, block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
            except (sqlite3.Error, ValueError, KeyError) as error:
                # Don't crash: hand the error back so the model can correct itself.
                if verbose:
                    print(f"         -> error: {error}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {error}",
                        "is_error": True,
                    }
                )

        messages.append({"role": "user", "content": tool_results})

    return f"Stopped: no answer after {MAX_STEPS} steps."


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Question: ")
    client = Anthropic()
    conn = get_connection()
    answer = run_agent(question, client, conn)
    print("\n" + answer)
    conn.close()


if __name__ == "__main__":
    main()
