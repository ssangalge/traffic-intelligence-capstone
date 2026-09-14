"""
proxy_risk_label.py

Constructs a PROXY accident-risk label from traffic congestion and adverse
weather conditions, since no real accident dataset exists for this project.

IMPORTANT: This is explicitly a proxy/synthetic label for educational purposes,
NOT a validated real-world accident predictor. It should never be presented
or used as if it reflects actual accident data or outcomes.

Construction logic:
  - Congestion quartile: traffic_volume binned into 4 quartiles (Q1-Q4)
  - Severe/low-visibility weather: weather conditions historically associated
    with reduced visibility or hazardous driving (Fog, Snow, Thunderstorm,
    Squall) OR heavy precipitation (rain_1h or snow_1h above a threshold)
  - proxy_risk_score: a simple weighted combination of the above
  - proxy_risk_label: binarized High/Low risk from that score

Usage:
    python proxy_risk_label.py

Input:  featured_traffic_data.csv (from Part 2)
Output: risk_labeled_data.csv
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

INPUT_CSV_PATH = Path(__file__).parent.parent / "capstone_part2" / "featured_traffic_data.csv"
OUTPUT_CSV_PATH = Path(__file__).parent / "risk_labeled_data.csv"

# Weather conditions treated as "severe/low-visibility" for this proxy label
SEVERE_WEATHER_CONDITIONS = {"Fog", "Snow", "Thunderstorm", "Squall"}
HEAVY_PRECIPITATION_THRESHOLD_MM = 2.0  # rain_1h or snow_1h above this counts as "heavy"


def load_featured_data(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading featured data from {csv_path}")
    if not csv_path.exists():
        logger.error(f"Featured data file not found at {csv_path}. Run Part 2's feature_engineering.py first.")
        raise FileNotFoundError(f"Expected featured CSV at {csv_path}, but it does not exist.")

    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    df["date_time"] = pd.to_datetime(df["date_time"])
    logger.info(f"Loaded {len(df):,} rows.")
    return df


def add_congestion_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Bin traffic_volume into 4 quartiles: Q1 (lowest) to Q4 (highest)."""
    logger.debug("Computing congestion quartiles.")

    df["congestion_quartile"] = pd.qcut(
        df["traffic_volume"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )

    logger.info("Added congestion_quartile (Q1-Q4, based on traffic_volume quartiles).")
    return df


def add_severe_weather_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with severe/low-visibility weather or heavy precipitation."""
    logger.debug("Flagging severe/low-visibility weather conditions.")

    is_severe_condition = df["weather_main"].isin(SEVERE_WEATHER_CONDITIONS)
    is_heavy_precip = (
        (df["rain_1h"] > HEAVY_PRECIPITATION_THRESHOLD_MM)
        | (df["snow_1h"] > HEAVY_PRECIPITATION_THRESHOLD_MM)
    )
    df["is_severe_weather"] = is_severe_condition | is_heavy_precip

    severe_rate = df["is_severe_weather"].mean()
    logger.info(f"Added is_severe_weather flag ({severe_rate:.2%} of rows flagged).")
    return df


def add_proxy_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """Combine congestion quartile and severe weather into a proxy risk score/label.

    proxy_risk_score: 0-3 scale.
      +2 if congestion_quartile is Q4 (highest traffic)
      +1 if congestion_quartile is Q3
      +1 if is_severe_weather is True
    proxy_risk_label: "High" if score >= 2, else "Low".

    This is a simple, transparent, documented heuristic -- not a fitted or
    validated model. It exists purely to give Part 3's supervised learning
    tasks a target variable to work with, standing in for real accident data.
    """
    logger.debug("Computing proxy risk score and label.")

    score = pd.Series(0, index=df.index)
    score += (df["congestion_quartile"] == "Q4").astype(int) * 2
    score += (df["congestion_quartile"] == "Q3").astype(int) * 1
    score += df["is_severe_weather"].astype(int) * 1

    df["proxy_risk_score"] = score
    df["proxy_risk_label"] = (score >= 2).map({True: "High", False: "Low"})

    label_distribution = df["proxy_risk_label"].value_counts(normalize=True)
    logger.info(
        f"Added proxy_risk_score (0-3) and proxy_risk_label. "
        f"Distribution: {label_distribution.to_dict()}"
    )
    logger.warning(
        "proxy_risk_label is a SYNTHETIC/PROXY target constructed from congestion "
        "and weather heuristics. It does NOT represent real accident data and "
        "must not be presented or interpreted as a validated accident predictor."
    )
    return df


def run_proxy_labeling(
    input_csv_path: Path = INPUT_CSV_PATH,
    output_csv_path: Path = OUTPUT_CSV_PATH,
) -> pd.DataFrame:
    logger.info("=== Starting proxy risk label construction ===")

    df = load_featured_data(input_csv_path)
    df = add_congestion_quartile(df)
    df = add_severe_weather_flag(df)
    df = add_proxy_risk_label(df)

    try:
        df.to_csv(output_csv_path, index=False)
    except OSError as e:
        logger.error(f"Failed to write risk-labeled CSV to {output_csv_path}: {e}")
        raise

    logger.info(f"Risk-labeled data written to {output_csv_path} ({len(df):,} rows).")
    logger.info("=== Proxy risk label construction completed successfully ===")
    return df


if __name__ == "__main__":
    configure_logging()
    try:
        run_proxy_labeling()
    except Exception as e:
        logger.error(f"Proxy risk labeling failed: {e}")
        sys.exit(1)