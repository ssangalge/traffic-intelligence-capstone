"""
visualizations.py

Generates the required Matplotlib visualizations from the featured traffic
dataset. Saves each figure as a PNG into the figures/ subfolder.

Usage:
    python visualizations.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts/CI
import matplotlib.pyplot as plt
import pandas as pd

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

INPUT_CSV_PATH = Path(__file__).parent / "featured_traffic_data.csv"
FIGURES_DIR = Path(__file__).parent / "figures"


def load_featured_data(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading featured data from {csv_path}")
    if not csv_path.exists():
        logger.error(f"Featured data file not found at {csv_path}. Run feature_engineering.py first.")
        raise FileNotFoundError(f"Expected featured CSV at {csv_path}, but it does not exist.")

    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    df["date_time"] = pd.to_datetime(df["date_time"])
    logger.info(f"Loaded {len(df):,} rows for visualization.")
    return df


def plot_hourly_traffic_pattern(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart: average traffic volume by hour of day."""
    logger.debug("Building hourly traffic pattern chart.")

    hourly_avg = df.groupby("hour")["traffic_volume"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hourly_avg.index, hourly_avg.values, color="#4C72B0")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Traffic Volume")
    ax.set_title("Average Traffic Volume by Hour of Day")
    ax.set_xticks(range(0, 24))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved hourly traffic pattern chart to {output_path}")


def plot_traffic_volume_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Histogram of traffic_volume, with mean/median lines to show skew."""
    logger.debug("Building traffic volume distribution chart.")

    mean_val = df["traffic_volume"].mean()
    median_val = df["traffic_volume"].median()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["traffic_volume"], bins=50, color="#55A868", edgecolor="white")
    ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean ({mean_val:,.0f})")
    ax.axvline(median_val, color="black", linestyle="--", label=f"Median ({median_val:,.0f})")
    ax.set_xlabel("Traffic Volume (vehicles/hour)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Traffic Volume")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved traffic volume distribution chart to {output_path}")


def plot_congestion_by_weather(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart: congestion rate (% of hours) by weather condition."""
    logger.debug("Building congestion rate by weather condition chart.")

    congestion_by_weather = (
        df.groupby("weather_main")["is_congested"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(congestion_by_weather.index, congestion_by_weather.values * 100, color="#C44E52")
    ax.set_xlabel("Weather Condition")
    ax.set_ylabel("Congestion Rate (%)")
    ax.set_title("Congestion Rate by Weather Condition")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved congestion-by-weather chart to {output_path}")


def run_visualizations(input_csv_path: Path = INPUT_CSV_PATH, figures_dir: Path = FIGURES_DIR) -> None:
    logger.info("=== Starting visualization generation ===")

    figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_featured_data(input_csv_path)

    plot_hourly_traffic_pattern(df, figures_dir / "hourly_traffic_pattern.png")
    plot_traffic_volume_distribution(df, figures_dir / "traffic_volume_distribution.png")
    plot_congestion_by_weather(df, figures_dir / "congestion_by_weather.png")

    logger.info(f"All figures saved to {figures_dir}")
    logger.info("=== Visualization generation completed successfully ===")


if __name__ == "__main__":
    configure_logging()
    try:
        run_visualizations()
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        sys.exit(1)
