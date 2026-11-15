"""
Phase 2 — The tools the agent can call.

Each tool is a plain Python function (what actually runs) plus a JSON definition
(what the LLM reads to decide when and how to call it). Tool descriptions are
prompts: vague descriptions produce bad tool choices.
"""
import json
import sqlite3
import time

MAX_ROWS = 50
QUERY_TIMEOUT_S = 15  # any query slower than this is stopped


def execute_with_timeout(
    conn: sqlite3.Connection, sql: str, max_rows: int, timeout_s: float = QUERY_TIMEOUT_S
) -> tuple[list[str], list[tuple]]:
    """Run a query, but stop it if it runs longer than timeout_s seconds.

    SQLite calls the progress handler every 10,000 internal steps. If the handler
    returns 1, SQLite aborts the query with "interrupted". Without this, one badly
    written query (e.g. joining the 1M-row geolocation table) can run for many minutes.
    """
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cursor = conn.execute(sql)
        columns = [col[0] for col in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows)
    except sqlite3.OperationalError as error:
        if "interrupted" in str(error):
            raise sqlite3.OperationalError(
                f"Query was stopped after {timeout_s:.0f}s. Simplify it: filter earlier, "
                "avoid joining large tables you don't need (e.g. geolocation)."
            ) from error
        raise
    finally:
        conn.set_progress_handler(None, 0)  # remove the handler for the next query
    return columns, rows


# ---------- helpers ----------

def _table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
    ]


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [col[1] for col in conn.execute(f'PRAGMA table_info("{table}")')]


def _check_table(conn: sqlite3.Connection, table: str) -> None:
    # The LLM's input is untrusted: never paste a name into SQL without checking it exists.
    if table not in _table_names(conn):
        raise ValueError(f"Unknown table '{table}'. Call list_tables to see valid names.")


# ---------- tools ----------

def list_tables(conn: sqlite3.Connection) -> str:
    """Every table and how many rows it has."""
    tables = []
    for table in _table_names(conn):
        count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        tables.append({"table": table, "rows": count})
    return json.dumps(tables)


def describe_table(conn: sqlite3.Connection, table: str) -> str:
    """Columns, types and 3 example rows for one table."""
    _check_table(conn, table)
    columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    samples = conn.execute(f'SELECT * FROM "{table}" LIMIT 3').fetchall()
    return json.dumps(
        {
            "table": table,
            "columns": [{"name": col[1], "type": col[2]} for col in columns],
            "example_rows": samples,
        },
        default=str,
    )


def get_distinct_values(
    conn: sqlite3.Connection, table: str, column: str, limit: int = 30
) -> str:
    """The most common values in a column, with counts. Shows real spellings."""
    _check_table(conn, table)
    if column not in _column_names(conn, table):
        raise ValueError(
            f"Unknown column '{column}' in '{table}'. Call describe_table to see columns."
        )
    rows = conn.execute(
        f'SELECT "{column}", COUNT(*) AS n FROM "{table}" '
        f'GROUP BY "{column}" ORDER BY n DESC LIMIT ?',
        (limit,),
    ).fetchall()
    distinct_count = conn.execute(
        f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"'
    ).fetchone()[0]
    return json.dumps(
        {
            "table": table,
            "column": column,
            "distinct_count": distinct_count,
            "top_values": [{"value": value, "count": n} for value, n in rows],
        },
        default=str,
    )


def run_query(conn: sqlite3.Connection, sql: str) -> str:
    """Run one read-only SELECT and return up to MAX_ROWS rows."""
    # read-only connection: writes fail; slow queries are stopped after QUERY_TIMEOUT_S
    columns, rows = execute_with_timeout(conn, sql, MAX_ROWS + 1)
    return json.dumps(
        {
            "columns": columns,
            "rows": rows[:MAX_ROWS],
            "truncated": len(rows) > MAX_ROWS,  # tells the LLM there were more rows
        },
        default=str,
    )


def run_tool(conn: sqlite3.Connection, name: str, tool_input: dict) -> str:
    """Dispatch a tool call from the LLM to the matching Python function."""
    if name == "list_tables":
        return list_tables(conn)
    if name == "describe_table":
        return describe_table(conn, tool_input["table"])
    if name == "get_distinct_values":
        return get_distinct_values(
            conn, tool_input["table"], tool_input["column"], tool_input.get("limit", 30)
        )
    if name == "run_query":
        return run_query(conn, tool_input["sql"])
    raise ValueError(f"Unknown tool: {name}")


# ---------- definitions the LLM sees ----------

TOOLS = [
    {
        "name": "list_tables",
        "description": "List every table in the database with its row count. "
        "Call this first to see what data exists.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "describe_table",
        "description": "Show a table's columns, their types, and 3 example rows. "
        "Use it to learn column names and how values are formatted before writing SQL.",
        "input_schema": {
            "type": "object",
            "properties": {"table": {"type": "string", "description": "Exact table name."}},
            "required": ["table"],
        },
    },
    {
        "name": "get_distinct_values",
        "description": "Show the most common values in a column with their counts, plus how "
        "many distinct values exist. Use it before filtering on a text value (a status, "
        "category, city, state) to get the exact spelling stored in the data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "column": {"type": "string"},
                "limit": {"type": "integer", "description": "Max values to return (default 30)."},
            },
            "required": ["table", "column"],
        },
    },
    {
        "name": "run_query",
        "description": f"Run one SQLite SELECT query and return up to {MAX_ROWS} rows. "
        "The database is read-only. If the query fails, the error message is returned so "
        "you can fix the query and try again.",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["sql"],
        },
    },
]
