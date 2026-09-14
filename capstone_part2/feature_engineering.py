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
    """Add hour, day of week, month, and weekend flag from date_time."""
    logger.debug("Adding time-based features.")

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.day_name()
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = df["date_time"].dt.dayofweek >= 5  # Saturday=5, Sunday=6

    logger.info("Added time features: hour, day_of_week, month, is_weekend.")
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
    """Add the congestion target variable(s) using thresholds established in Part 1.

    is_congested: binary target (traffic_volume > 5,500)
    traffic_category: 3-class target (Low / Medium / High), matching the
    Power BI Traffic_Category column from Part 1 for consistency.
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

    congestion_rate = df["is_congested"].mean()
    logger.info(
        f"Added target variables: is_congested ({congestion_rate:.2%} of rows), "
        f"traffic_category (Low/Medium/High)."
    )
    return df


def run_feature_engineering(
    input_csv_path: Path = INPUT_CSV_PATH,
    output_csv_path: Path = OUTPUT_CSV_PATH,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline end-to-end."""
    logger.info("=== Starting feature engineering ===")

    df = load_cleaned_data(input_csv_path)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_scaled_numerics(df)
    df = add_congestion_target(df)

    try:
        df.to_csv(output_csv_path, index=False)
    except OSError as e:
        logger.error(f"Failed to write featured CSV to {output_csv_path}: {e}")
        raise

    logger.info(f"Featured data written to {output_csv_path} ({len(df):,} rows, {len(df.columns)} columns).")
    logger.info("=== Feature engineering completed successfully ===")
    return df


if __name__ == "__main__":
    configure_logging()
    try:
        run_feature_engineering()
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        sys.exit(1)