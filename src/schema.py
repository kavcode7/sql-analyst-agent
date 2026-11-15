"""
Phase 1, Step 1.1 — Turn the database schema into text an LLM can read.

Run from the project root:
    python src/schema.py
"""
import sqlite3
from pathlib import Path

DB_PATH = Path("data/olist.db")


def get_connection() -> sqlite3.Connection:
    """Open the database READ-ONLY. Any INSERT/UPDATE/DELETE/DROP will fail."""
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def get_schema(conn: sqlite3.Connection, sample_rows: int = 2) -> str:
    """Describe every table: column names, types, and a few example rows."""
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
    ]

    parts = []
    for table in tables:
        # PRAGMA table_info returns one row per column: (cid, name, type, notnull, default, pk)
        columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        column_lines = [f"  {col[1]} {col[2]}" for col in columns]

        samples = conn.execute(f'SELECT * FROM "{table}" LIMIT {sample_rows}').fetchall()

        parts.append(
            f"TABLE {table}\n"
            + "\n".join(column_lines)
            + f"\n  -- example rows: {samples}"
        )

    return "\n\n".join(parts)


if __name__ == "__main__":
    conn = get_connection()
    schema = get_schema(conn)
    print(schema)
    # Rough rule of thumb: ~4 characters per token
    print(f"\nSchema length: {len(schema):,} characters (~{len(schema) // 4:,} tokens)")
    conn.close()
