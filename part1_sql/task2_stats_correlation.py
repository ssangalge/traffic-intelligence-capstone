"""
Task 2: Descriptive Statistics and Correlation

2.1 Traffic volume statistics: mean, median, standard deviation, variance, range
2.2 Correlation analysis: temperature vs. traffic volume

Uses the de-duplicated dataset (one row per unique timestamp) for consistency
with Task 1.2's approach, since this dataset contains 5,445 duplicate
timestamp groups that would otherwise distort variability measures.
"""

import os
import sqlite3
import sys
import pandas as pd

DB_PATH = "traffic.db"


def load_deduped_data(db_path: str) -> pd.DataFrame:
    """Load and de-duplicate traffic data from the SQLite database.

    Raises a clear, specific error and exits gracefully rather than letting
    a raw traceback surface, if the database is missing, unreadable, or
    doesn't contain the expected table.
    """
    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found at '{db_path}'. "
              f"Run load_data.py first to create it.")
        sys.exit(1)

    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        print(f"ERROR: Could not connect to database '{db_path}': {e}")
        sys.exit(1)

    try:
        # Collapse duplicate timestamps by averaging, same approach as Task 1.2
        query = """
            SELECT date_time, AVG(temp) AS temp, AVG(traffic_volume) AS traffic_volume
            FROM traffic
            GROUP BY date_time
        """
        df = pd.read_sql_query(query, conn)
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        print(f"ERROR: Query failed. Does the 'traffic' table exist? Details: {e}")
        sys.exit(1)
    finally:
        conn.close()

    if df.empty:
        print("ERROR: Query returned no rows. Check that the database was loaded correctly.")
        sys.exit(1)

    return df


def traffic_volume_statistics(df: pd.DataFrame) -> dict:
    volume = df["traffic_volume"]
    return {
        "mean": round(volume.mean(), 2),
        "median": round(volume.median(), 2),
        "std_dev": round(volume.std(), 2),
        "variance": round(volume.var(), 2),
        "range": round(volume.max() - volume.min(), 2),
        "min": round(volume.min(), 2),
        "max": round(volume.max(), 2),
    }


def correlation_analysis(df: pd.DataFrame) -> float:
    return round(df["temp"].corr(df["traffic_volume"]), 4)


if __name__ == "__main__":
    df = load_deduped_data(DB_PATH)
    print(f"Rows used (de-duplicated): {len(df):,}\n")

    try:
        print("=== 2.1 Traffic Volume Statistics ===")
        stats = traffic_volume_statistics(df)
        for k, v in stats.items():
            print(f"{k}: {v}")

        print("\n=== 2.2 Correlation: Temperature vs. Traffic Volume ===")
        corr = correlation_analysis(df)
        print(f"Pearson correlation coefficient: {corr}")
    except KeyError as e:
        print(f"ERROR: Expected column missing from data: {e}")
        sys.exit(1)
