"""
advanced_mlflow.py

Advanced technique: MLflow experiment tracking. Re-trains the classification
models from supervised_models.py, but this time logs parameters, metrics,
and the trained model itself to MLflow, so runs are comparable and
reproducible over time -- a standard MLOps practice for tracking model
experiments rather than just printing results to a log.

Usage:
    python advanced_mlflow.py

View results with:
    mlflow ui
Then open http://localhost:5000 in a browser.

Input: risk_labeled_data.csv (from proxy_risk_label.py)
"""

import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

INPUT_CSV_PATH = Path(__file__).parent / "risk_labeled_data.csv"
RANDOM_STATE = 42
EXPERIMENT_NAME = "traffic_risk_classification"

TIME_FEATURES = ["hour", "month", "hour_sin", "hour_cos"]
NUMERIC_WEATHER_FEATURES = ["temp_celsius", "clouds_all", "rain_1h", "snow_1h"]
CATEGORICAL_FEATURES = ["day_of_week", "weather_main"]
BOOLEAN_FEATURES = ["is_weekend", "is_clear", "is_precipitating"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


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

    X["is_holiday"] = (df["holiday"] != "None").astype(int)
    dow_numeric = df["day_of_week"].map({day: i for i, day in enumerate(DAY_ORDER)})
    X["dow_sin"] = np.sin(2 * np.pi * dow_numeric / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow_numeric / 7)

    X = pd.get_dummies(X.join(df[CATEGORICAL_FEATURES]), columns=CATEGORICAL_FEATURES, drop_first=True)
    return X


def run_tracked_experiment(model_name: str, model, X_train, X_test, y_train, y_test, params: dict, use_scaling: bool) -> None:
    """Train one model inside an MLflow run, logging params/metrics/model."""
    logger.info(f"--- MLflow run: {model_name} ---")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("random_state", RANDOM_STATE)
        for k, v in params.items():
            mlflow.log_param(k, v)

        if use_scaling:
            scaler = StandardScaler()
            X_train_input = scaler.fit_transform(X_train)
            X_test_input = scaler.transform(X_test)
            mlflow.log_param("scaled_features", True)
        else:
            X_train_input = X_train
            X_test_input = X_test
            mlflow.log_param("scaled_features", False)

        model.fit(X_train_input, y_train)
        y_pred = model.predict(X_test_input)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, name="model")

        logger.info(
            f"{model_name} -- Accuracy: {acc:.4f}, Precision: {prec:.4f}, "
            f"Recall: {rec:.4f}, F1: {f1:.4f} (logged to MLflow)"
        )


def run_mlflow_experiments(input_csv_path: Path = INPUT_CSV_PATH) -> None:
    logger.info("=== Starting MLflow-tracked experiments ===")

    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow experiment set to '{EXPERIMENT_NAME}'.")

    df = load_data(input_csv_path)
    X = build_feature_matrix(df)
    y = (df["proxy_risk_label"] == "High").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train/test split: {len(X_train):,} train, {len(X_test):,} test rows.")

    run_tracked_experiment(
        "Logistic_Regression",
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        X_train, X_test, y_train, y_test,
        params={"max_iter": 1000},
        use_scaling=True,
    )

    for n_estimators in [50, 100, 200]:
        run_tracked_experiment(
            f"Random_Forest_{n_estimators}trees",
            RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE),
            X_train, X_test, y_train, y_test,
            params={"n_estimators": n_estimators},
            use_scaling=False,
        )

    logger.info("=== MLflow-tracked experiments completed successfully ===")
    logger.info("Run 'mlflow ui' and open http://localhost:5000 to view and compare all runs.")


if __name__ == "__main__":
    configure_logging()
    try:
        run_mlflow_experiments()
    except Exception as e:
        logger.error(f"MLflow experiment run failed: {e}")
        sys.exit(1)