"""
Task 1.1: Load the Metro Interstate Traffic Volume dataset into SQLite.

Usage:
    python load_data.py

Reads ../data/Metro_Interstate_Traffic_Volume.csv and writes traffic.db
in the same folder as this script, with a single table: traffic.
"""

import sqlite3
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "data" / "Metro_Interstate_Traffic_Volume.csv"
DB_PATH = Path(__file__).parent / "traffic.db"
TABLE_NAME = "traffic"


def load_csv_to_sqlite(csv_path: Path, db_path: Path, table_name: str) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse date_time now so it's stored as a proper TEXT ISO timestamp
    # and can be sorted/filtered correctly in SQL.
    df["date_time"] = pd.to_datetime(df["date_time"])

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns into "
          f"'{table_name}' table in {db_path}")


def verify_load(db_path: Path, table_name: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cur.fetchone()[0]

        cur.execute(f"PRAGMA table_info({table_name})")
        columns = cur.fetchall()

        cur.execute(f"SELECT * FROM {table_name} LIMIT 3")
        sample_rows = cur.fetchall()

        print("\n--- Verification ---")
        print(f"Row count: {row_count:,}")
        print(f"Column count: {len(columns)}")
        print("Columns:", [c[1] for c in columns])
        print("\nSample rows:")
        for row in sample_rows:
            print(row)
    finally:
        conn.close()


if __name__ == "__main__":
    load_csv_to_sqlite(CSV_PATH, DB_PATH, TABLE_NAME)
    verify_load(DB_PATH, TABLE_NAME)