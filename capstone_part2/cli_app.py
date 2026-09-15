"""
cli_app.py

A small menu-driven command-line app for exploring the featured traffic
dataset: view overall stats, filter by date range or weather condition,
and check congestion rates.

Usage:
    python cli_app.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "featured_traffic_data.csv"

MENU_TEXT = """
=== Traffic Intelligence CLI ===
1. View overall traffic statistics
2. Filter by date range
3. Filter by weather condition
4. View congestion rate
5. Exit
"""


def load_data(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {csv_path}")
    if not csv_path.exists():
        logger.error(f"Data file not found at {csv_path}. Run feature_engineering.py first.")
        raise FileNotFoundError(f"Expected data at {csv_path}, but it does not exist.")

    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    df["date_time"] = pd.to_datetime(df["date_time"])
    logger.info(f"Loaded {len(df):,} rows.")
    return df


def show_overall_stats(df: pd.DataFrame) -> None:
    logger.debug("Displaying overall traffic statistics.")
    print("\n--- Overall Traffic Volume Statistics ---")
    print(f"Rows:    {len(df):,}")
    print(f"Mean:    {df['traffic_volume'].mean():,.1f}")
    print(f"Median:  {df['traffic_volume'].median():,.1f}")
    print(f"Std Dev: {df['traffic_volume'].std():,.1f}")
    print(f"Min:     {df['traffic_volume'].min():,.0f}")
    print(f"Max:     {df['traffic_volume'].max():,.0f}")


def filter_by_date_range(df: pd.DataFrame) -> None:
    logger.debug("Prompting user for date range filter.")
    start_str = input("Enter start date (YYYY-MM-DD): ").strip()
    end_str = input("Enter end date (YYYY-MM-DD): ").strip()

    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
    except ValueError:
        logger.error(f"Invalid date format entered: '{start_str}' / '{end_str}'. Expected YYYY-MM-DD.")
        print("Invalid date format. Please use YYYY-MM-DD (e.g. 2017-01-01).")
        return

    if start_date > end_date:
        logger.error(f"Invalid date range: start date {start_date} is after end date {end_date}.")
        print("Start date must be before or equal to end date.")
        return

    mask = (df["date_time"] >= start_date) & (df["date_time"] <= end_date)
    filtered = df.loc[mask]

    if filtered.empty:
        logger.info(f"No rows found between {start_date.date()} and {end_date.date()}.")
        print("No data found in that date range.")
        return

    print(f"\n--- Traffic Summary: {start_date.date()} to {end_date.date()} ---")
    print(f"Rows:               {len(filtered):,}")
    print(f"Average volume:     {filtered['traffic_volume'].mean():,.1f}")
    print(f"Congestion rate:    {filtered['is_congested'].mean():.2%}")
    logger.info(f"Displayed date range summary for {start_date.date()} to {end_date.date()} "
                f"({len(filtered):,} rows).")


def filter_by_weather(df: pd.DataFrame) -> None:
    logger.debug("Prompting user for weather condition filter.")
    available = sorted(df["weather_main"].unique())
    print(f"\nAvailable weather conditions: {', '.join(available)}")
    choice = input("Enter a weather condition: ").strip()

    matches = df[df["weather_main"].str.lower() == choice.lower()]
    if matches.empty:
        logger.error(f"Invalid input: unrecognised weather condition '{choice}'.")
        print(f"'{choice}' not found. Please choose from the list shown above.")
        return

    print(f"\n--- Traffic Summary: Weather = {choice} ---")
    print(f"Rows:               {len(matches):,}")
    print(f"Average volume:     {matches['traffic_volume'].mean():,.1f}")
    print(f"Congestion rate:    {matches['is_congested'].mean():.2%}")
    logger.info(f"Displayed weather summary for '{choice}' ({len(matches):,} rows).")


def show_congestion_rate(df: pd.DataFrame) -> None:
    logger.debug("Displaying overall congestion rate.")
    rate = df["is_congested"].mean()
    print(f"\nOverall congestion rate: {rate:.2%} of all recorded hours.")
    print(f"({df['is_congested'].sum():,} congested hours out of {len(df):,} total)")


def run_cli(data_path: Path = DATA_PATH) -> None:
    logger.info("=== Starting Traffic Intelligence CLI ===")
    df = load_data(data_path)

    while True:
        print(MENU_TEXT)
        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            logger.info("Command invoked: 1 (view overall traffic statistics), args=none")
            show_overall_stats(df)
        elif choice == "2":
            logger.info("Command invoked: 2 (filter by date range), args=prompted interactively")
            filter_by_date_range(df)
        elif choice == "3":
            logger.info("Command invoked: 3 (filter by weather condition), args=prompted interactively")
            filter_by_weather(df)
        elif choice == "4":
            logger.info("Command invoked: 4 (view congestion rate), args=none")
            show_congestion_rate(df)
        elif choice == "5":
            logger.info("Command invoked: 5 (exit)")
            print("Goodbye!")
            break
        else:
            logger.error(f"Invalid menu choice entered: '{choice}'. Expected 1-5.")
            print("Invalid choice. Please enter a number from 1 to 5.")


if __name__ == "__main__":
    configure_logging()
    try:
        run_cli()
    except FileNotFoundError:
        logger.error("CLI could not start.", exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("CLI interrupted by user (Ctrl+C).")
        print("\nExiting.")
        sys.exit(0)