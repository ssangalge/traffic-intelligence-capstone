"""
feature_engineering.py

Builds model-ready features from the cleaned traffic dataset produced by
pipeline.py. Adds time-based features, weather-based features, scaled
numeric columns, and the congestion target variable used later in Part 3.

Usage:
    python feature_engineering.py

Input:  cleaned_traffic_data.csv (produced by pipeline.py)
Output: featured_traffic_data.csv
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

INPUT_CSV_PATH = Path(__file__).parent / "cleaned_traffic_data.csv"
OUTPUT_CSV_PATH = Path(__file__).parent / "featured_traffic_data.csv"

# Same thresholds used in Part 1 (Task 3 probability analysis and the Power BI
# Traffic_Category column), kept consistent here for continuity across parts.
CONGESTION_THRESHOLD = 5500
LOW_TRAFFIC_THRESHOLD = 4500


def load_cleaned_data(csv_path: Path) -> pd.DataFrame:
    """Load the cleaned CSV produced by pipeline.py."""
    logger.info(f"Loading cleaned data from {csv_path}")

    if not csv_path.exists():
        logger.error(f"Cleaned data file not found at {csv_path}. Run pipeline.py first.")
        raise FileNotFoundError(f"Expected cleaned CSV at {csv_path}, but it does not exist.")

    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    df["date_time"] = pd.to_datetime(df["date_time"])

    logger.info(f"Loaded {len(df):,} rows for feature engineering.")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day of week, month, weekend flag, and a cyclical encoding
    of hour from date_time.

    Hour is cyclical (23 and 0 are actually adjacent, not 23 apart), so a
    raw integer 0-23 misrepresents that adjacency to a model. Encoding it
    as sine/cosine pairs preserves the correct circular distance.
    """
    logger.debug("Adding time-based features.")

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.day_name()
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = df["date_time"].dt.dayofweek >= 5  # Saturday=5, Sunday=6

    # Cyclical encoding of hour (24-hour cycle)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    logger.info(
        "Added time features: hour, day_of_week, month, is_weekend, "
        "hour_sin, hour_cos (cyclical encoding)."
    )
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature in Celsius and simplified weather condition flags."""
    logger.debug("Adding weather-based features.")

    df["temp_celsius"] = df["temp"] - 273.15
    df["is_clear"] = df["weather_main"] == "Clear"
    df["is_precipitating"] = (df["rain_1h"] > 0) | (df["snow_1h"] > 0)

    logger.info("Added weather features: temp_celsius, is_clear, is_precipitating.")
    return df


def add_scaled_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Add standardized (z-score) versions of key numeric columns for later ML use."""
    logger.debug("Scaling numeric columns: temp_celsius, traffic_volume.")

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[["temp_celsius", "traffic_volume"]])
    df["temp_celsius_scaled"] = scaled_values[:, 0]
    df["traffic_volume_scaled"] = scaled_values[:, 1]

    logger.info("Added scaled numeric features: temp_celsius_scaled, traffic_volume_scaled.")
    return df


def add_congestion_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the congestion target variable(s).

    is_congested: binary target (traffic_volume > 5,500), using the fixed
    threshold established in Part 1's Power BI Traffic_Category column,
    kept for consistency across parts.

    traffic_category: 3-class label (Low/Medium/High) using those same
    fixed Part 1 thresholds.

    congestion_category_quartile: a DATA-DRIVEN 4-class label (Low/Medium/
    High/Severe) based on the quartiles of traffic_volume itself, rather
    than fixed thresholds -- this is the "data-driven congestion category"
    requested by the spec. The quartile boundary VALUES are intermediate
    calculations not needed in the final output, so they are logged at
    DEBUG level only (not persisted as their own column).
    """
    logger.debug("Building congestion target variable(s).")

    df["is_congested"] = df["traffic_volume"] > CONGESTION_THRESHOLD

    def categorize(volume: float) -> str:
        if volume < LOW_TRAFFIC_THRESHOLD:
            return "Low"
        elif volume <= CONGESTION_THRESHOLD:
            return "Medium"
        else:
            return "High"

    df["traffic_category"] = df["traffic_volume"].apply(categorize)

    # Data-driven quartile thresholds -- intermediate values, logged at
    # DEBUG only since they aren't part of the persisted output themselves.
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values
    logger.debug(f"Congestion quartile thresholds calculated: Q1={q1:.1f}, Q2(median)={q2:.1f}, Q3={q3:.1f}")

    def quartile_bucket(volume: float) -> str:
        if volume <= q1:
            return "Low"
        elif volume <= q2:
            return "Medium"
        elif volume <= q3:
            return "High"
        return "Severe"

    df["congestion_category_quartile"] = df["traffic_volume"].apply(quartile_bucket)

    congestion_rate = df["is_congested"].mean()
    logger.info(
        f"Added target variables: is_congested ({congestion_rate:.2%} of rows), "
        f"traffic_category (Low/Medium/High, fixed thresholds), "
        f"congestion_category_quartile (Low/Medium/High/Severe, data-driven quartiles)."
    )
    return df


def run_feature_engineering(
    input_csv_path: Path = INPUT_CSV_PATH,
    output_csv_path: Path = OUTPUT_CSV_PATH,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline end-to-end."""
    logger.info("=== Starting feature engineering ===")

    df = load_cleaned_data(input_csv_path)
    logger.info(f"Dataset shape BEFORE feature engineering: {df.shape[0]:,} rows, {df.shape[1]} columns.")

    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_scaled_numerics(df)
    df = add_congestion_target(df)

    logger.info(f"Dataset shape AFTER feature engineering: {df.shape[0]:,} rows, {df.shape[1]} columns.")

    try:
        df.to_csv(output_csv_path, index=False)
    except OSError as e:
        logger.error(f"Failed to write featured CSV to {output_csv_path}: {e}", exc_info=True)
        raise

    logger.info(f"Featured data written to {output_csv_path} ({len(df):,} rows, {len(df.columns)} columns).")
    logger.info("=== Feature engineering completed successfully ===")
    return df


if __name__ == "__main__":
    configure_logging()
    try:
        run_feature_engineering()
    except Exception:
        logger.error("Feature engineering failed and could not complete.", exc_info=True)
        sys.exit(1)