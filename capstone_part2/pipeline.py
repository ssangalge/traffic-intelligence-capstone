"""
pipeline.py

Core data pipeline for the Smart City Traffic Intelligence capstone.
Loads the raw traffic CSV, validates it, cleans known data quality issues,
and writes a cleaned CSV ready for feature engineering (Part 2) and
downstream ML modelling (Part 3).

Usage:
    python pipeline.py

Known data quality issues this pipeline handles:
  1. rain_1h / snow_1h sometimes get inferred as integer-like; enforced as float.
  2. 17 exact full-row duplicates exist in the raw file and are removed.
  3. Duplicate TIMESTAMPS (different from exact duplicate rows) exist because
     multiple simultaneous weather conditions are logged as separate rows for
     the same hour -- these are preserved, not dropped (see Part 1 analysis).
  4. holiday column uses the literal string "None" rather than a true null.
  5. 10 rows record temp == 0 Kelvin (absolute zero -- a physically impossible
     sensor reading). Imputed using the median temperature for that month.
  6. 1 row records rain_1h = 9,831.3mm (a known data bug in this dataset --
     physically impossible for a single hour). Imputed using the median
     rain_1h for that month.
  7. Categorical text columns are standardised (stripped whitespace, and
     weather_description is lower-cased for consistency).
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

RAW_CSV_PATH = Path(__file__).parent.parent / "data" / "Metro_Interstate_Traffic_Volume.csv"
CLEANED_CSV_PATH = Path(__file__).parent / "cleaned_traffic_data.csv"

IMPLAUSIBLE_VOLUME_THRESHOLD = 20      # vehicles/hour; see Part 1 July 2016 anomaly
IMPOSSIBLE_TEMP_KELVIN = 0             # absolute zero is not a real weather reading
IMPLAUSIBLE_RAIN_MM = 300              # highest ever recorded global 1-hour rainfall is ~305mm


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load the raw traffic CSV, with error handling for common failure points."""
    logger.info(f"Loading raw data from {csv_path}")

    if not csv_path.exists():
        logger.error(f"Raw data file not found at {csv_path}")
        raise FileNotFoundError(f"Expected CSV at {csv_path}, but it does not exist.")

    try:
        # keep_default_na=False prevents pandas from silently converting the
        # literal string "None" in the holiday column into an actual NaN.
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
    """Enforce correct data types for known-problematic columns."""
    logger.debug("Fixing data types for rain_1h, snow_1h, and date_time.")

    df["rain_1h"] = df["rain_1h"].astype(float)
    df["snow_1h"] = df["snow_1h"].astype(float)
    df["date_time"] = pd.to_datetime(df["date_time"])

    logger.info("Data types enforced: rain_1h and snow_1h as float, date_time as datetime.")
    return df


def standardise_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise inconsistent categorical text values: strip stray
    whitespace and normalise casing so identical categories aren't
    accidentally treated as distinct (e.g. 'Clear' vs 'clear ').
    """
    logger.debug("Standardising categorical text columns.")

    before_unique = {
        col: df[col].nunique() for col in ["holiday", "weather_main", "weather_description"]
    }

    for col in ["holiday", "weather_main", "weather_description"]:
        df[col] = df[col].str.strip()
    # weather_main is a clean category label (e.g. "Clear", "Snow") -- keep
    # its casing as-is. weather_description is free-text and benefits from
    # consistent lower-casing.
    df["weather_description"] = df["weather_description"].str.lower()

    after_unique = {
        col: df[col].nunique() for col in ["holiday", "weather_main", "weather_description"]
    }

    for col in before_unique:
        if before_unique[col] != after_unique[col]:
            logger.warning(
                f"Standardising '{col}' reduced distinct categories from "
                f"{before_unique[col]} to {after_unique[col]} (removed casing/whitespace inconsistencies)."
            )
    logger.info("Categorical text columns standardised (whitespace stripped, weather_description lower-cased).")
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

    none_string_count = (df["holiday"] == "None").sum()
    logger.info(
        f"'holiday' column: {none_string_count:,} rows use the string 'None' "
        f"(not a true null) to represent non-holiday hours."
    )


def remove_exact_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and remove rows that are IDENTICAL across every column.

    This is distinct from duplicate TIMESTAMPS (handled separately below) --
    a duplicate timestamp with a different weather_main value is a legitimate
    multi-condition reading, not an erroneous duplicate. An exact duplicate
    row (every single column matching another row) is genuinely redundant
    and is safe to remove.
    """
    logger.debug("Checking for exact full-row duplicates.")

    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = duplicate_mask.sum()

    if duplicate_count > 0:
        df = df.loc[~duplicate_mask].reset_index(drop=True)
        logger.warning(
            f"Removed {duplicate_count} exact duplicate row(s) (identical across "
            f"every column) -- reason: redundant repeated records, safe to drop "
            f"without losing information."
        )
    else:
        logger.info("No exact duplicate rows found.")

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


def impute_impossible_values_by_month(
    df: pd.DataFrame, column: str, is_impossible: pd.Series, reason: str
) -> pd.DataFrame:
    """Impute impossible values in `column` using the MEDIAN for that same
    calendar month, rather than a single global median -- this respects
    seasonal variation (e.g. a winter month's median temperature is very
    different from a summer month's).

    Uses an explicit loop over months, as required, rather than a single
    vectorised global calculation.
    """
    affected_count = is_impossible.sum()
    if affected_count == 0:
        logger.info(f"No impossible values found in '{column}'.")
        return df

    months_affected = sorted(df.loc[is_impossible, "date_time"].dt.month.unique())

    for month in months_affected:
        month_mask = df["date_time"].dt.month == month
        valid_month_values = df.loc[month_mask & ~is_impossible, column]

        if valid_month_values.empty:
            logger.warning(f"No valid '{column}' values found for month {month} to compute a median from.")
            continue

        month_median = valid_month_values.median()
        rows_to_fix = month_mask & is_impossible
        n_fixed = rows_to_fix.sum()

        if n_fixed > 0:
            df.loc[rows_to_fix, column] = month_median
            logger.warning(
                f"Imputed {n_fixed} impossible '{column}' value(s) in month {month} "
                f"with that month's median ({month_median:.2f}) -- reason: {reason}"
            )

    return df


def detect_and_handle_impossible_temp(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and impute temp == 0 Kelvin (absolute zero -- not a real reading)."""
    logger.debug("Checking for impossible temperature readings (0 Kelvin).")
    is_impossible = df["temp"] == IMPOSSIBLE_TEMP_KELVIN
    return impute_impossible_values_by_month(
        df, "temp", is_impossible,
        reason=f"temp == {IMPOSSIBLE_TEMP_KELVIN}K (absolute zero) is not a physically real weather reading",
    )


def detect_and_handle_impossible_rain(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and impute rain_1h values exceeding the physically plausible
    maximum ever recorded for a single hour (~305mm globally)."""
    logger.debug("Checking for impossible rainfall readings.")
    is_impossible = df["rain_1h"] > IMPLAUSIBLE_RAIN_MM
    return impute_impossible_values_by_month(
        df, "rain_1h", is_impossible,
        reason=f"rain_1h exceeding {IMPLAUSIBLE_RAIN_MM}mm/hour exceeds the highest rainfall ever recorded globally in one hour",
    )


def detect_and_flag_implausible_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """Flag (not impute) physically implausible near-zero traffic_volume
    readings. Unlike temp/rain, these are flagged rather than imputed,
    because they occur in long connected runs (see the July 23, 2016
    sensor malfunction found in Part 1) where a monthly median would
    erase a real, documentable data quality event rather than correct
    a one-off sensor glitch.
    """
    logger.debug("Checking for implausible near-zero traffic volume readings.")
    implausible_mask = df["traffic_volume"] < IMPLAUSIBLE_VOLUME_THRESHOLD
    implausible_count = implausible_mask.sum()

    df["is_implausible_volume"] = implausible_mask

    if implausible_count > 0:
        logger.warning(
            f"Found {implausible_count} rows with traffic_volume < "
            f"{IMPLAUSIBLE_VOLUME_THRESHOLD} (physically implausible for an "
            f"interstate highway) -- reason: likely sensor malfunction (see "
            f"July 23, 2016 event documented in Part 1). Flagged via "
            f"'is_implausible_volume' column rather than imputed, since these "
            f"occur in connected runs where a single median would obscure a "
            f"genuine, documentable data quality event."
        )
    else:
        logger.info("No implausible traffic_volume readings found.")

    return df


def run_pipeline(raw_csv_path: Path = RAW_CSV_PATH, output_csv_path: Path = CLEANED_CSV_PATH) -> pd.DataFrame:
    """Run the full cleaning and validation pipeline end-to-end."""
    logger.info("=== Starting data pipeline ===")

    df = load_raw_data(raw_csv_path)
    validate_schema(df)
    df = fix_data_types(df)
    df = standardise_categorical_values(df)
    check_missing_values(df)
    df = remove_exact_duplicate_rows(df)
    check_duplicate_timestamps(df)
    df = detect_and_handle_impossible_temp(df)
    df = detect_and_handle_impossible_rain(df)
    df = detect_and_flag_implausible_traffic(df)

    try:
        df.to_csv(output_csv_path, index=False)
    except OSError as e:
        logger.error(f"Failed to write cleaned CSV to {output_csv_path}: {e}", exc_info=True)
        raise

    logger.info(f"Cleaned data written to {output_csv_path} ({len(df):,} rows).")
    logger.info("=== Pipeline completed successfully ===")
    return df


if __name__ == "__main__":
    configure_logging()
    try:
        run_pipeline()
    except Exception:
        logger.error("Pipeline failed and could not complete.", exc_info=True)
        sys.exit(1)