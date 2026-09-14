"""
pipeline.py

Core data pipeline for the Smart City Traffic Intelligence capstone.
Loads the raw traffic CSV, validates it, cleans known data quality issues,
and writes a cleaned CSV ready for feature engineering (Part 2) and
downstream ML modelling (Part 3).

Usage:
    python pipeline.py

Known data quality issues this pipeline handles (identified in Part 1):
  1. rain_1h / snow_1h sometimes get inferred as integer-like; enforced as float.
  2. July 23, 2016 contains a run of physically implausible near-zero traffic
     readings (likely sensor malfunction) bookending a plausible morning window.
  3. Duplicate timestamps exist because multiple simultaneous weather conditions
     are logged as separate rows for the same hour (not true duplicate errors).
  4. holiday column uses the literal string "None" rather than a true null.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

RAW_CSV_PATH = Path(__file__).parent.parent / "data" / "Metro_Interstate_Traffic_Volume.csv"
CLEANED_CSV_PATH = Path(__file__).parent / "cleaned_traffic_data.csv"

# Threshold below which hourly traffic volume is considered physically implausible
# for this interstate corridor, based on the July 23, 2016 anomaly found in Part 1.
IMPLAUSIBLE_VOLUME_THRESHOLD = 20


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load the raw traffic CSV, with error handling for common failure points."""
    logger.info(f"Loading raw data from {csv_path}")

    if not csv_path.exists():
        logger.error(f"Raw data file not found at {csv_path}")
        raise FileNotFoundError(f"Expected CSV at {csv_path}, but it does not exist.")

    try:
        # keep_default_na=False prevents pandas from silently converting the
        # literal string "None" in the holiday column into an actual NaN --
        # this is a real behaviour we discovered while building this pipeline:
        # pandas treats "None" as a missing-value marker by default, which
        # would have wrongly nulled out 48,143 rows in the holiday column.
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty.")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise

    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Check that expected columns are present before proceeding."""
    expected_columns = {
        "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
        "weather_main", "weather_description", "date_time", "traffic_volume",
    }
    missing = expected_columns - set(df.columns)
    if missing:
        logger.error(f"Missing expected columns: {missing}")
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")
    logger.info("Schema validation passed: all expected columns present.")


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce correct data types for known-problematic columns.

    rain_1h and snow_1h contain decimal millimetre values but can be
    misinterpreted as integers if a column happens to be all whole numbers
    in a given sample -- we saw this exact issue in Power BI during Part 1.
    """
    logger.debug("Fixing data types for rain_1h, snow_1h, and date_time.")

    df["rain_1h"] = df["rain_1h"].astype(float)
    df["snow_1h"] = df["snow_1h"].astype(float)
    df["date_time"] = pd.to_datetime(df["date_time"])

    logger.info("Data types enforced: rain_1h and snow_1h as float, date_time as datetime.")
    return df


def check_missing_values(df: pd.DataFrame) -> None:
    """Log a report of missing values per column. Does not modify the data."""
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls == 0:
        logger.info("No true null values found in any column.")
    else:
        logger.warning(f"Found {total_nulls} total null values across columns:")
        for col, count in null_counts[null_counts > 0].items():
            logger.warning(f"  {col}: {count} nulls")

    # Known quirk from Part 1: 'holiday' uses the string "None", not a true null
    none_string_count = (df["holiday"] == "None").sum()
    logger.info(
        f"'holiday' column: {none_string_count:,} rows use the string 'None' "
        f"(not a true null) to represent non-holiday hours."
    )


def detect_and_handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and handle physically implausible traffic_volume readings.

    Rather than silently dropping these rows (which would lose the surrounding
    context), we flag them with a boolean column so downstream users/models
    can decide how to treat them, and we log exactly what was found.
    """
    implausible_mask = df["traffic_volume"] < IMPLAUSIBLE_VOLUME_THRESHOLD
    implausible_count = implausible_mask.sum()

    df["is_implausible_volume"] = implausible_mask

    if implausible_count > 0:
        logger.warning(
            f"Found {implausible_count} rows with traffic_volume < "
            f"{IMPLAUSIBLE_VOLUME_THRESHOLD} (physically implausible for an "
            f"interstate highway). Flagged via 'is_implausible_volume' column "
            f"rather than dropped, to preserve context."
        )
        # Log the specific dates affected, for transparency (e.g. the known
        # July 23, 2016 sensor malfunction found during Part 1 dashboard work)
        affected_dates = df.loc[implausible_mask, "date_time"].dt.date.unique()
        logger.info(f"Dates with implausible readings: {list(affected_dates)[:10]}"
                    f"{'...' if len(affected_dates) > 10 else ''}")
    else:
        logger.info("No implausible traffic_volume readings found.")

    return df


def check_duplicate_timestamps(df: pd.DataFrame) -> None:
    """Log duplicate timestamp groups. These are NOT dropped, since Part 1
    established they represent legitimate simultaneous weather conditions
    reported for the same hour, not erroneous duplicate rows.
    """
    dup_counts = df.groupby("date_time").size()
    dup_groups = (dup_counts > 1).sum()

    if dup_groups > 0:
        logger.info(
            f"Found {dup_groups:,} timestamps with multiple rows (multiple "
            f"simultaneous weather conditions reported for the same hour). "
            f"These are preserved, not dropped, per the Part 1 analysis."
        )
    else:
        logger.info("No duplicate timestamps found.")


def run_pipeline(raw_csv_path: Path = RAW_CSV_PATH, output_csv_path: Path = CLEANED_CSV_PATH) -> pd.DataFrame:
    """Run the full cleaning and validation pipeline end-to-end."""
    logger.info("=== Starting data pipeline ===")

    df = load_raw_data(raw_csv_path)
    validate_schema(df)
    df = fix_data_types(df)
    check_missing_values(df)
    df = detect_and_handle_outliers(df)
    check_duplicate_timestamps(df)

    try:
        df.to_csv(output_csv_path, index=False)
    except OSError as e:
        logger.error(f"Failed to write cleaned CSV to {output_csv_path}: {e}")
        raise

    logger.info(f"Cleaned data written to {output_csv_path} ({len(df):,} rows).")
    logger.info("=== Pipeline completed successfully ===")
    return df


if __name__ == "__main__":
    configure_logging()
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)