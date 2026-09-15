"""
travel_recommendation.py

A travel-timing recommendation system: given a day of week (and optionally
current weather conditions), recommends the hours with the lowest predicted
congestion/risk, using the patterns learned across Parts 1-3.

This is a rule-based/statistical recommender built directly on the
risk-labeled historical data (not a new trained model) -- it answers
"historically, when has this day+weather combination been safest/least
congested?"

Usage:
    python travel_recommendation.py
    (interactive CLI prompts for day of week and weather condition)

Input: risk_labeled_data.csv (from proxy_risk_label.py)
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

INPUT_CSV_PATH = Path(__file__).parent / "risk_labeled_data.csv"

VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_data(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading risk-labeled data from {csv_path}")
    if not csv_path.exists():
        logger.error(f"Data file not found at {csv_path}. Run proxy_risk_label.py first.")
        raise FileNotFoundError(f"Expected data at {csv_path}, but it does not exist.")

    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    logger.info(f"Loaded {len(df):,} rows.")
    return df


def generate_plain_language_recommendation(
    day_of_week: str, best_hours: pd.DataFrame, weather_filter: str = None
) -> str:
    """Generate a human-readable recommendation sentence, e.g.:
    'For a weekday journey, consider travelling between 10:00 and 11:00 AM,
    when historical traffic volumes are typically lower.'
    """
    if best_hours.empty:
        return "No historical data available to generate a recommendation."

    top_hour = int(best_hours.iloc[0]["hour"])
    end_hour = (top_hour + 1) % 24

    def format_hour(h: int) -> str:
        period = "AM" if h < 12 else "PM"
        display_hour = h % 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:00 {period}"

    day_type = "weekend" if day_of_week in ("Saturday", "Sunday") else "weekday"
    weather_clause = f" under {weather_filter.lower()} conditions" if weather_filter else ""

    return (
        f"For a {day_type} journey on {day_of_week}{weather_clause}, consider travelling "
        f"between {format_hour(top_hour)} and {format_hour(end_hour)}, when historical "
        f"traffic volumes and risk levels are typically lower."
    )


def recommend_travel_times(
    df: pd.DataFrame, day_of_week: str, weather_filter: str = None, top_n: int = 5
) -> pd.DataFrame:
    """Return the top_n hours with the lowest historical risk/congestion for
    the given day (optionally filtered to a specific weather condition).
    """
    logger.debug(f"Computing recommendations for day={day_of_week}, weather={weather_filter}")

    subset = df[df["day_of_week"] == day_of_week]
    if weather_filter:
        subset = subset[subset["weather_main"] == weather_filter]

    if subset.empty:
        logger.warning(f"No historical data found for day={day_of_week}, weather={weather_filter}.")
        return pd.DataFrame()

    hourly_summary = (
        subset.groupby("hour")
        .agg(
            avg_traffic_volume=("traffic_volume", "mean"),
            high_risk_rate=("proxy_risk_label", lambda x: (x == "High").mean()),
            sample_size=("traffic_volume", "count"),
        )
        .reset_index()
    )

    # Sort by risk first, then by traffic volume as a tiebreaker
    ranked = hourly_summary.sort_values(["high_risk_rate", "avg_traffic_volume"])

    logger.info(
        f"Top {top_n} recommended hours for {day_of_week}"
        f"{f' ({weather_filter} weather)' if weather_filter else ''}:"
    )
    for _, row in ranked.head(top_n).iterrows():
        logger.info(
            f"  {int(row['hour']):02d}:00 -- avg volume: {row['avg_traffic_volume']:.0f}, "
            f"high-risk rate: {row['high_risk_rate']:.1%} (n={int(row['sample_size'])})"
        )

    return ranked.head(top_n)


def recommend_worst_travel_times(
    df: pd.DataFrame, day_of_week: str, weather_filter: str = None, top_n: int = 3
) -> pd.DataFrame:
    """Return the hours to AVOID -- highest historical risk -- as a complement
    to the main recommendation."""
    subset = df[df["day_of_week"] == day_of_week]
    if weather_filter:
        subset = subset[subset["weather_main"] == weather_filter]

    if subset.empty:
        return pd.DataFrame()

    hourly_summary = (
        subset.groupby("hour")
        .agg(
            avg_traffic_volume=("traffic_volume", "mean"),
            high_risk_rate=("proxy_risk_label", lambda x: (x == "High").mean()),
            sample_size=("traffic_volume", "count"),
        )
        .reset_index()
    )

    ranked = hourly_summary.sort_values(["high_risk_rate", "avg_traffic_volume"], ascending=False)

    logger.info(f"Hours to AVOID for {day_of_week}:")
    for _, row in ranked.head(top_n).iterrows():
        logger.info(
            f"  {int(row['hour']):02d}:00 -- avg volume: {row['avg_traffic_volume']:.0f}, "
            f"high-risk rate: {row['high_risk_rate']:.1%}"
        )

    return ranked.head(top_n)


def run_interactive_cli(input_csv_path: Path = INPUT_CSV_PATH) -> None:
    logger.info("=== Starting travel-timing recommendation system ===")
    df = load_data(input_csv_path)

    print("\n=== Travel Timing Recommendation System ===")
    print("(Based on historical traffic risk patterns, not real-time data)")
    available_weather = sorted(df["weather_main"].unique())
    print(f"Available weather conditions: {', '.join(available_weather)}\n")

    while True:
        day_input = input(f"Enter day of week ({', '.join(VALID_DAYS)}) or 'quit': ").strip().title()
        if day_input.lower() == "quit":
            logger.info("User exited the recommendation system.")
            break
        if day_input not in VALID_DAYS:
            logger.warning(f"Invalid day entered: '{day_input}'")
            print(f"'{day_input}' is not a valid day. Please try again.")
            continue

        weather_filter = None
        while True:
            weather_input = input(
                f"Filter by weather condition ({', '.join(available_weather)}), "
                f"or press Enter to skip: "
            ).strip()

            if not weather_input:
                break

            matched = next((w for w in available_weather if w.lower() == weather_input.lower()), None)
            if matched is None:
                logger.warning(f"Invalid weather condition entered: '{weather_input}'")
                print(f"'{weather_input}' is not a valid option. Choose from: {', '.join(available_weather)}")
                continue

            weather_filter = matched
            break

        best = recommend_travel_times(df, day_input, weather_filter)
        worst = recommend_worst_travel_times(df, day_input, weather_filter)

        if best.empty:
            print(f"No historical data available for {day_input}"
                  f"{f' with {weather_filter} weather' if weather_filter else ''}.")
            continue

        print(f"\nBest times to travel on {day_input}"
              f"{f' ({weather_filter} weather)' if weather_filter else ''}:")
        for _, row in best.iterrows():
            print(f"  {int(row['hour']):02d}:00 -- avg volume: {row['avg_traffic_volume']:.0f}, "
                  f"high-risk rate: {row['high_risk_rate']:.1%}")

        recommendation = generate_plain_language_recommendation(day_input, best, weather_filter)
        print(f"\nRecommendation: {recommendation}")

        print(f"\nTimes to AVOID on {day_input}:")
        for _, row in worst.iterrows():
            print(f"  {int(row['hour']):02d}:00 -- avg volume: {row['avg_traffic_volume']:.0f}, "
                  f"high-risk rate: {row['high_risk_rate']:.1%}")
        print()


if __name__ == "__main__":
    configure_logging()
    try:
        run_interactive_cli()
    except FileNotFoundError as e:
        logger.error(f"Recommendation system could not start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)