"""
Make a small, compressed copy of the database for deployment.

data/ is gitignored, so a host that builds from GitHub never sees data/olist.db.
This writes deploy/olist_demo.db.gz (committed to git), which the Dockerfile unpacks.
The 1M-row geolocation table is left out: no benchmark question needs it, and
dropping it shrinks the file a lot (and removes the slowest table to join by mistake).

Run from the project root:
    python scripts/make_demo_db.py
"""
import gzip
import shutil
import sqlite3
from pathlib import Path

SRC_DB = Path("data/olist.db")
TMP_DB = Path("data/olist_demo.db")
OUT = Path("deploy/olist_demo.db.gz")
SKIP_TABLES = {"geolocation"}
GITHUB_FILE_LIMIT_MB = 100


def main() -> None:
    if TMP_DB.exists():
        TMP_DB.unlink()
    shutil.copyfile(SRC_DB, TMP_DB)

    conn = sqlite3.connect(TMP_DB)
    for table in SKIP_TABLES:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.commit()
    conn.execute("VACUUM")  # actually reclaim the space from the dropped table
    conn.close()

    OUT.parent.mkdir(exist_ok=True)
    with open(TMP_DB, "rb") as src, gzip.open(OUT, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    TMP_DB.unlink()

    size_mb = OUT.stat().st_size / 1_000_000
    print(f"Wrote {OUT} ({size_mb:.1f} MB)")
    if size_mb > GITHUB_FILE_LIMIT_MB * 0.95:
        print(f"WARNING: GitHub rejects files over {GITHUB_FILE_LIMIT_MB} MB. Drop more tables or use Git LFS.")


if __name__ == "__main__":
    main()
