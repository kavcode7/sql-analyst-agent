"""
Phase 0 — Load the Olist CSVs into a single SQLite database.

Run from the project root:
    python scripts/load_data.py
"""
from pathlib import Path
import sqlite3

import pandas as pd

RAW_DIR = Path("data/raw")          # where you unzipped the Kaggle CSVs
DB_PATH = Path("data/olist.db")     # the database the agent will query


def table_name(csv_path: Path) -> str:
    """'olist_order_items_dataset.csv' -> 'order_items'"""
    name = csv_path.stem
    name = name.removeprefix("olist_").removesuffix("_dataset")
    return name


def main() -> None:
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSVs found in {RAW_DIR.resolve()} — did you unzip the dataset there?")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    for csv_path in csv_files:
        name = table_name(csv_path)
        df = pd.read_csv(csv_path)
        df.to_sql(name, conn, if_exists="replace", index=False)
        print(f"{name:<40} {len(df):>10,} rows  {len(df.columns):>3} columns")

    conn.close()
    print(f"\nDone. Database written to {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()
