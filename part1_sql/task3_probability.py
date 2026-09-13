"""
Task 3: Probability and Congestion Analysis

Congestion is defined as: traffic_volume > 5,500 vehicles.
High temperature is defined as: temp > 292K.

NOTE ON DATA: this analysis uses the FULL raw dataset (not de-duplicated
like Tasks 1.2/2), because the "duplicate" timestamps in this dataset are
not data errors -- they represent multiple simultaneous weather conditions
reported for the same hour (e.g. "Rain" and "Mist" both logged for the same
timestamp, with identical temperature and traffic volume). Collapsing these
would arbitrarily discard legitimate weather information needed for this
task's probability calculations.
"""

import os
import sqlite3
import sys
import pandas as pd

DB_PATH = "traffic.db"
CONGESTION_THRESHOLD = 5500
HIGH_TEMP_THRESHOLD = 292


def load_raw_data(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found at '{db_path}'. Run load_data.py first.")
        sys.exit(1)
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM traffic", conn)
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        print(f"ERROR: Failed to load data: {e}")
        sys.exit(1)
    finally:
        conn.close()
    if df.empty:
        print("ERROR: No rows found in traffic table.")
        sys.exit(1)
    return df


def run_probability_analysis(df: pd.DataFrame) -> None:
    n = len(df)

    congestion = df["traffic_volume"] > CONGESTION_THRESHOLD
    clear_weather = df["weather_main"] == "Clear"
    cloudy_weather = df["weather_main"] == "Clouds"
    high_temp = df["temp"] > HIGH_TEMP_THRESHOLD

    # 3.1 Basic probability
    p_congestion = congestion.mean()
    p_clear = clear_weather.mean()
    p_congestion_and_clear = (congestion & clear_weather).mean()

    print("=== 3.1 Basic Probability ===")
    print(f"P(Congestion)              = {p_congestion:.4f}")
    print(f"P(Clear Weather)           = {p_clear:.4f}")
    print(f"P(Congestion AND Clear)    = {p_congestion_and_clear:.4f}")

    # 3.2 Conditional probability
    p_clear_given_congestion = (congestion & clear_weather).sum() / congestion.sum()
    p_hightemp_given_congestion = (congestion & high_temp).sum() / congestion.sum()

    print("\n=== 3.2 Conditional Probability ===")
    print(f"P(Clear Weather | Congestion)     = {p_clear_given_congestion:.4f}")
    print(f"P(High Temperature | Congestion)  = {p_hightemp_given_congestion:.4f}")

    # Independence check: P(A and B) vs P(A) * P(B)
    independent_product = p_congestion * p_clear
    print(f"\nIndependence check (Congestion vs. Clear Weather):")
    print(f"  P(Congestion AND Clear) actual   = {p_congestion_and_clear:.4f}")
    print(f"  P(Congestion) * P(Clear) expected = {independent_product:.4f}")
    diff = abs(p_congestion_and_clear - independent_product)
    print(f"  Difference                        = {diff:.4f}")
    verdict = "approximately independent" if diff < 0.01 else "NOT independent"
    print(f"  Verdict: variables appear {verdict}")

    # Odds ratio: congestion in clear vs. cloudy weather
    p_congestion_given_clear = (congestion & clear_weather).sum() / clear_weather.sum()
    p_congestion_given_cloudy = (congestion & cloudy_weather).sum() / cloudy_weather.sum()

    odds_clear = p_congestion_given_clear / (1 - p_congestion_given_clear)
    odds_cloudy = p_congestion_given_cloudy / (1 - p_congestion_given_cloudy)
    odds_ratio = odds_clear / odds_cloudy

    print(f"\nOdds ratio (Congestion: Clear vs. Cloudy weather):")
    print(f"  P(Congestion | Clear)  = {p_congestion_given_clear:.4f}")
    print(f"  P(Congestion | Cloudy) = {p_congestion_given_cloudy:.4f}")
    print(f"  Odds(Congestion | Clear)  = {odds_clear:.4f}")
    print(f"  Odds(Congestion | Cloudy) = {odds_cloudy:.4f}")
    print(f"  Odds Ratio = {odds_ratio:.4f}")


if __name__ == "__main__":
    df = load_raw_data(DB_PATH)
    print(f"Total rows analysed: {len(df):,}\n")
    run_probability_analysis(df)
