"""
supervised_models.py

Trains and evaluates supervised learning models:
  - Classification: predicts proxy_risk_label (High/Low) using Logistic
    Regression and Random Forest.
  - Regression: predicts traffic_volume using Linear Regression and
    Random Forest Regressor.

IMPORTANT (data leakage): proxy_risk_label is directly derived from
congestion_quartile, proxy_risk_score, is_severe_weather, and traffic_volume.
None of those columns are used as model features -- doing so would let the
model trivially reconstruct the label instead of learning a genuine pattern.
Only time-based and raw weather features (independent of the label's own
construction) are used as predictors.

Usage:
    python supervised_models.py

Input:  risk_labeled_data.csv (from proxy_risk_label.py)
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

INPUT_CSV_PATH = Path(__file__).parent / "risk_labeled_data.csv"
RANDOM_STATE = 42

# Features used for BOTH tasks -- deliberately excludes any column derived
# from or used to construct proxy_risk_label / traffic_volume itself.
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
    """Build the shared, leakage-free feature matrix used by all models."""
    logger.debug("Building feature matrix (time + raw weather features only).")

    feature_cols = TIME_FEATURES + NUMERIC_WEATHER_FEATURES + BOOLEAN_FEATURES
    X = df[feature_cols].copy()

    for col in BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)

    X = pd.get_dummies(X.join(df[CATEGORICAL_FEATURES]), columns=CATEGORICAL_FEATURES, drop_first=True)

    logger.info(f"Feature matrix built: {X.shape[1]} features, {len(X):,} rows.")
    logger.debug(f"Feature columns: {list(X.columns)}")
    return X


def run_classification(df: pd.DataFrame, X: pd.DataFrame) -> None:
    """Train and evaluate 2 classifiers predicting proxy_risk_label."""
    logger.info("--- Classification: predicting proxy_risk_label (High/Low) ---")

    y = (df["proxy_risk_label"] == "High").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train/test split: {len(X_train):,} train, {len(X_test):,} test rows.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    for name, model in models.items():
        logger.info(f"Training {name}...")
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(
            f"{name} results -- Accuracy: {acc:.4f}, Precision: {prec:.4f}, "
            f"Recall: {rec:.4f}, F1: {f1:.4f}"
        )
        logger.info(f"{name} confusion matrix (rows=actual, cols=predicted):\n{cm}")


def run_regression(df: pd.DataFrame, X: pd.DataFrame) -> None:
    """Train and evaluate 2 regressors predicting traffic_volume."""
    logger.info("--- Regression: predicting traffic_volume ---")

    y = df["traffic_volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    logger.info(f"Train/test split: {len(X_train):,} train, {len(X_test):,} test rows.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    for name, model in models.items():
        logger.info(f"Training {name}...")
        if name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"{name} results -- RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")


def run_supervised_models(input_csv_path: Path = INPUT_CSV_PATH) -> None:
    logger.info("=== Starting supervised model training ===")

    df = load_data(input_csv_path)
    X = build_feature_matrix(df)

    run_classification(df, X)
    run_regression(df, X)

    logger.info("=== Supervised model training completed successfully ===")


if __name__ == "__main__":
    configure_logging()
    try:
        run_supervised_models()
    except Exception as e:
        logger.error(f"Supervised model training failed: {e}")
        sys.exit(1)