"""
train_deployment_model.py

Trains the Random Forest classifier (best-performing model from Part 3's
supervised learning) and saves it as a model artifact, along with the exact
feature column order, for use by the FastAPI deployment service.

Usage:
    python train_deployment_model.py

Input:  risk_labeled_data.csv
Output: model_artifact.joblib
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

INPUT_CSV_PATH = Path(__file__).parent / "risk_labeled_data.csv"
MODEL_ARTIFACT_PATH = Path(__file__).parent / "model_artifact.joblib"
RANDOM_STATE = 42

TIME_FEATURES = ["hour", "month"]
NUMERIC_WEATHER_FEATURES = ["temp_celsius", "clouds_all", "rain_1h", "snow_1h"]
CATEGORICAL_FEATURES = ["day_of_week", "weather_main"]
BOOLEAN_FEATURES = ["is_weekend", "is_clear", "is_precipitating"]


def load_data(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading risk-labeled data from {csv_path}")
    if not csv_path.exists():
        logger.error(f"Data file not found at {csv_path}. Run proxy_risk_label.py first.")
        raise FileNotFoundError(f"Expected data at {csv_path}, but it does not exist.")
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    logger.info(f"Loaded {len(df):,} rows.")
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = TIME_FEATURES + NUMERIC_WEATHER_FEATURES + BOOLEAN_FEATURES
    X = df[feature_cols].copy()
    for col in BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)
    X = pd.get_dummies(X.join(df[CATEGORICAL_FEATURES]), columns=CATEGORICAL_FEATURES, drop_first=True)
    return X


def train_and_save_model(input_csv_path: Path = INPUT_CSV_PATH, output_path: Path = MODEL_ARTIFACT_PATH) -> None:
    logger.info("=== Training deployment model ===")

    df = load_data(input_csv_path)
    X = build_feature_matrix(df)
    y = (df["proxy_risk_label"] == "High").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    test_accuracy = model.score(X_test, y_test)
    logger.info(f"Model trained. Test accuracy: {test_accuracy:.4f}")

    artifact = {
        "model": model,
        "feature_columns": list(X.columns),
        "day_of_week_options": sorted(df["day_of_week"].unique().tolist()),
        "weather_main_options": sorted(df["weather_main"].unique().tolist()),
    }

    try:
        joblib.dump(artifact, output_path)
    except OSError as e:
        logger.error(f"Failed to save model artifact to {output_path}: {e}")
        raise

    logger.info(f"Model artifact saved to {output_path}")
    logger.info("=== Deployment model training completed successfully ===")


if __name__ == "__main__":
    configure_logging()
    try:
        train_and_save_model()
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)