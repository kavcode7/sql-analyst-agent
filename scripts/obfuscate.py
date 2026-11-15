"""
Contamination test — build a disguised copy of the database.

The LLM has seen the public Olist dataset during training, so it can answer from memory.
This script makes a copy with the SAME DATA but different names:
  - tables and columns renamed      (orders -> purchases, order_status -> p_state)
  - some text codes renamed         ('canceled' -> 'CXL', 'credit_card' -> 'CARD')
It also rewrites every gold query to use the new names, then CHECKS that each rewritten
gold query returns the same answer on the copy as the original does on the real database.

Run from the project root:
    python scripts/obfuscate.py
Creates:
    data/olist_obf.db
    benchmark/questions_obf.json
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

SRC_DB = Path("data/olist.db")
OUT_DB = Path("data/olist_obf.db")
SRC_BENCHMARK = Path("benchmark/questions.json")
OUT_BENCHMARK = Path("benchmark/questions_obf.json")

# old table name -> new table name
TABLES = {
    "orders": "purchases",
    "customers": "clients",
    "order_items": "purchase_lines",
    "order_payments": "settlements",
    "order_reviews": "feedback",
    "products": "catalog",
    "product_category_name_translation": "cat_labels",
    "sellers": "vendors",
    "geolocation": "geo",
}

# old column name -> new column name (the same old name always gets the same new name,
# so join keys like order_id stay joinable: order_id -> purchase_id everywhere)
COLUMNS = {
    # orders
    "order_id": "purchase_id",
    "customer_id": "client_ref",
    "order_status": "p_state",
    "order_purchase_timestamp": "created_ts",
    "order_approved_at": "approved_ts",
    "order_delivered_carrier_date": "carrier_ts",
    "order_delivered_customer_date": "received_ts",
    "order_estimated_delivery_date": "eta_ts",
    # customers
    "customer_unique_id": "person_key",
    "customer_zip_code_prefix": "zip5",
    "customer_city": "city",
    "customer_state": "region",
    # order_items
    "order_item_id": "line_no",
    "product_id": "sku",
    "seller_id": "vendor_ref",
    "shipping_limit_date": "ship_by_ts",
    "price": "unit_amt",
    "freight_value": "ship_amt",
    # order_payments
    "payment_sequential": "seq",
    "payment_type": "method",
    "payment_installments": "n_parts",
    "payment_value": "amt",
    # order_reviews
    "review_id": "feedback_id",
    "review_score": "stars",
    "review_comment_title": "title",
    "review_comment_message": "body",
    "review_creation_date": "fb_created_ts",
    "review_answer_timestamp": "fb_answered_ts",
    # products
    "product_category_name": "cat_code",
    "product_name_lenght": "name_len",
    "product_description_lenght": "desc_len",
    "product_photos_qty": "photo_n",
    "product_weight_g": "wt_g",
    "product_length_cm": "len_cm",
    "product_height_cm": "ht_cm",
    "product_width_cm": "wd_cm",
    # translation
    "product_category_name_english": "label_en",
    # sellers
    "seller_zip_code_prefix": "zip5",
    "seller_city": "city",
    "seller_state": "region",
    # geolocation
    "geolocation_zip_code_prefix": "zip5",
    "geolocation_lat": "lat",
    "geolocation_lng": "lng",
    "geolocation_city": "city",
    "geolocation_state": "region",
}

# (old table, old column) -> {old value: new value}
# State codes (SP, RJ...) and English category names are NOT changed, because the
# benchmark questions mention them by name.
VALUES = {
    ("orders", "order_status"): {
        "delivered": "DLV", "shipped": "SHP", "canceled": "CXL", "unavailable": "UNV",
        "invoiced": "INV", "processing": "PRC", "created": "CRT", "approved": "APR",
    },
    ("order_payments", "payment_type"): {
        "credit_card": "CARD", "boleto": "BLT", "voucher": "VCH",
        "debit_card": "DEBIT", "not_defined": "NA",
    },
}

ALL_VALUES = {old: new for mapping in VALUES.values() for old, new in mapping.items()}


def build_database() -> None:
    if OUT_DB.exists():
        OUT_DB.unlink()
    conn = sqlite3.connect(f"file:{OUT_DB}", uri=True)  # uri=True lets ATTACH use a read-only file: URI
    conn.execute("ATTACH DATABASE ? AS src", (f"file:{SRC_DB}?mode=ro",))  # never modify the original

    tables = [r[0] for r in conn.execute("SELECT name FROM src.sqlite_master WHERE type = 'table'")]
    for table in tables:
        new_table = TABLES.get(table)
        if new_table is None:
            sys.exit(f"Unknown table '{table}' — add it to TABLES before continuing.")

        select_parts = []
        for col in conn.execute(f'PRAGMA src.table_info("{table}")'):
            name, col_type = col[1], col[2]
            new_name = COLUMNS.get(name)
            if new_name is None:
                sys.exit(f"Unknown column '{table}.{name}' — add it to COLUMNS before continuing.")
            value_map = VALUES.get((table, name))
            if value_map:
                cases = " ".join(f"WHEN '{old}' THEN '{new}'" for old, new in value_map.items())
                expr = f'CAST(CASE "{name}" {cases} ELSE "{name}" END AS TEXT)'
            else:
                expr = f'"{name}"'
            select_parts.append(f'{expr} AS "{new_name}"')

        conn.execute(f'CREATE TABLE "{new_table}" AS SELECT {", ".join(select_parts)} FROM src."{table}"')
        count = conn.execute(f'SELECT COUNT(*) FROM "{new_table}"').fetchone()[0]
        print(f"  {table:<36} -> {new_table:<16} {count:>10,} rows")

    conn.commit()
    conn.close()


def rewrite_sql(sql: str) -> str:
    """Translate a gold query from the real names to the disguised names."""
    for old, new in ALL_VALUES.items():
        sql = sql.replace(f"'{old}'", f"'{new}'")
    # Whole-word replacement, longest names first (so 'order_items' isn't hit by 'order').
    names = {**TABLES, **COLUMNS}
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b")
    return pattern.sub(lambda m: names[m.group(1)], sql)


def _normalise(rows: list[tuple], map_values: bool) -> list[tuple]:
    out = []
    for row in rows:
        cells = []
        for cell in row:
            if map_values and isinstance(cell, str):
                cell = ALL_VALUES.get(cell, cell)
            cells.append(round(cell, 6) if isinstance(cell, float) else cell)
        out.append(tuple(cells))
    return sorted(out, key=repr)


def build_benchmark() -> None:
    questions = json.loads(SRC_BENCHMARK.read_text(encoding="utf-8"))
    src = sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)
    obf = sqlite3.connect(f"file:{OUT_DB}?mode=ro", uri=True)

    problems = 0
    for item in questions:
        if item["gold_sql"] is None:
            continue
        new_sql = rewrite_sql(item["gold_sql"])
        # Self-check: same answer on both databases (after mapping renamed codes).
        expected = _normalise(src.execute(item["gold_sql"]).fetchall(), map_values=True)
        try:
            actual = _normalise(obf.execute(new_sql).fetchall(), map_values=False)
        except sqlite3.Error as error:
            actual = f"SQL error: {error}"
        if expected != actual:
            problems += 1
            print(f"  MISMATCH [{item['id']}]\n    {new_sql}\n    expected {expected[:3]}\n    got      {actual if isinstance(actual, str) else actual[:3]}")
        item["gold_sql"] = new_sql

    if problems:
        sys.exit(f"\n{problems} gold queries don't match after renaming — benchmark NOT written.")
    OUT_BENCHMARK.write_text(json.dumps(questions, indent=2), encoding="utf-8")
    print(f"  all gold queries verified: same answers on both databases")


def main() -> None:
    print(f"Building {OUT_DB} ...")
    build_database()
    print(f"\nRewriting gold queries -> {OUT_BENCHMARK} ...")
    build_benchmark()
    print("\nDone.")


if __name__ == "__main__":
    main()
