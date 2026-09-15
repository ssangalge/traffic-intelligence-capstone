"""
deep_learning_model.py

Trains a small feedforward neural network (Keras/TensorFlow) to predict
proxy_risk_label (High/Low), then explains its predictions using SHAP.

Uses the same leakage-free feature set as supervised_models.py (time and
raw weather features only -- no congestion_quartile, proxy_risk_score,
is_severe_weather, or traffic_volume as inputs).

Usage:
    python deep_learning_model.py

Input:  risk_labeled_data.csv (from proxy_risk_label.py)
Output: figures/shap_summary.png
"""

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress noisy TF startup logs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

INPUT_CSV_PATH = Path(__file__).parent / "risk_labeled_data.csv"
FIGURES_DIR = Path(__file__).parent / "figures"
RANDOM_STATE = 42

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
    logger.debug("Building feature matrix (time + weather + holiday features).")

    feature_cols = TIME_FEATURES + NUMERIC_WEATHER_FEATURES + BOOLEAN_FEATURES
    X = df[feature_cols].copy()

    for col in BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)

    X["is_holiday"] = (df["holiday"] != "None").astype(int)

    dow_numeric = df["day_of_week"].map({day: i for i, day in enumerate(DAY_ORDER)})
    X["dow_sin"] = np.sin(2 * np.pi * dow_numeric / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow_numeric / 7)

    X = pd.get_dummies(X.join(df[CATEGORICAL_FEATURES]), columns=CATEGORICAL_FEATURES, drop_first=True)

    logger.info(f"Feature matrix built: {X.shape[1]} features, {len(X):,} rows.")
    return X


def build_model(input_dim: int) -> tf.keras.Model:
    """A small feedforward network: 2 hidden layers with dropout for regularization."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> tuple:
    logger.info("--- Training deep learning model (feedforward NN) ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train/test split: {len(X_train):,} train, {len(X_test):,} test rows.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tf.random.set_seed(RANDOM_STATE)
    model = build_model(input_dim=X_train_scaled.shape[1])

    logger.info("Training for 20 epochs...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=256,
        verbose=0,
    )
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    logger.info(f"Training complete. Final train accuracy: {final_train_acc:.4f}, val accuracy: {final_val_acc:.4f}")

    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(
        f"Test set results -- Accuracy: {acc:.4f}, Precision: {prec:.4f}, "
        f"Recall: {rec:.4f}, F1: {f1:.4f}"
    )

    return model, scaler, X_train, X_test_scaled


def explain_with_shap(model: tf.keras.Model, X_train: pd.DataFrame, X_test_scaled: np.ndarray, feature_names: list, output_path: Path) -> None:
    """Generate SHAP explanations for the trained neural network."""
    logger.info("--- Generating SHAP explanations ---")

    background = X_test_scaled[:100]
    explain_sample = X_test_scaled[:200]

    def predict_fn(x):
        return model.predict(x, verbose=0)

    explainer = shap.KernelExplainer(predict_fn, background)
    logger.info("Computing SHAP values (this may take a minute)...")
    shap_values = explainer.shap_values(explain_sample, nsamples=100)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values).reshape(explain_sample.shape[0], -1)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_order = np.argsort(mean_abs_shap)[::-1]

    logger.info("Top 5 most important features by mean |SHAP value|:")
    for idx in importance_order[:5]:
        logger.info(f"  {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, explain_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"SHAP summary plot saved to {output_path}")


def run_deep_learning(input_csv_path: Path = INPUT_CSV_PATH) -> None:
    logger.info("=== Starting deep learning model training and explanation ===")

    df = load_data(input_csv_path)
    X = build_feature_matrix(df)
    y = (df["proxy_risk_label"] == "High").astype(int)

    model, scaler, X_train, X_test_scaled = train_and_evaluate(X, y)
    explain_with_shap(model, X_train, X_test_scaled, list(X.columns), FIGURES_DIR / "shap_summary.png")

    logger.info("=== Deep learning model training and explanation completed successfully ===")


if __name__ == "__main__":
    configure_logging()
    try:
        run_deep_learning()
    except Exception as e:
        logger.error(f"Deep learning model training failed: {e}")
        sys.exit(1)