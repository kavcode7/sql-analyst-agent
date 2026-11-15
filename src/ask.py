"""
Phase 1, Step 1.2 — One-shot text-to-SQL.

    question -> [system prompt + schema] -> LLM -> SQL -> run on DB -> result

Run from the project root:
    python src/ask.py "How many orders were delivered?"
"""
import sqlite3
import sys

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from schema import get_connection, get_schema

load_dotenv()  # loads ANTHROPIC_API_KEY from the .env file in the project root

MODEL = "claude-haiku-5-5"  # cheapest current model; fine for Phase 1
MAX_ROWS = 20


class SQLAnswer(BaseModel):
    """The exact shape we want back from the LLM. The API guarantees it matches."""

    # Order matters: the model writes `reasoning` first, so it plans before it writes SQL.
    reasoning: str = Field(
        description="Which tables and joins are needed and why, in 1-3 sentences."
    )
    sql: str = Field(description="A single SQLite SELECT query that answers the question.")
    answerable: bool = Field(
        description="False if the database has no data that can answer the question."
    )


SYSTEM_PROMPT = """You are a data analyst. You answer questions by writing SQLite queries \
against the database described below.

Rules:
- Write exactly one SELECT statement. Never modify data.
- Use only tables and columns that appear in the schema.
- Return at most {max_rows} rows unless the question asks for more.
- Return only what the question asks for: for "which X has the most/least", return one row (LIMIT 1).

Database schema:
{schema}
"""


def generate_sql(client: Anthropic, schema: str, question: str) -> SQLAnswer:
    """One LLM call: question in, structured {reasoning, sql} out."""
    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT.format(schema=schema, max_rows=MAX_ROWS),
        messages=[{"role": "user", "content": question}],
        output_format=SQLAnswer,
    )
    return response.parsed_output


def run_sql(conn: sqlite3.Connection, sql: str) -> tuple[list[str], list[tuple]]:
    """Execute the query and return (column names, up to MAX_ROWS rows)."""
    cursor = conn.execute(sql)
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchmany(MAX_ROWS)
    return columns, rows


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Question: ")

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    conn = get_connection()
    schema = get_schema(conn)

    answer = generate_sql(client, schema, question)
    print(f"\nReasoning: {answer.reasoning}")
    print(f"\nSQL:\n{answer.sql}\n")

    try:
        columns, rows = run_sql(conn, answer.sql)
    except sqlite3.Error as error:
        # Phase 1 just reports the error. Phase 3 will feed it back to the LLM to fix.
        print(f"SQL error: {error}")
        return

    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(str(value) for value in row))

    conn.close()


if __name__ == "__main__":
    main()
