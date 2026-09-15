"""
app.py

FastAPI deployment simulation for the traffic risk classifier.

Endpoints:
    POST /predict      -- predict proxy_risk_label for given conditions
    GET  /health        -- service health check
    GET  /monitoring    -- view accumulated monitoring stats since startup

MONITORING/ALERTING SIMULATION:
This is a simplified, in-memory simulation of production monitoring, suitable
for a capstone demo -- NOT a production-grade monitoring system (which would
use a real time-series store, persistent storage, and proper alerting
channels like email/Slack/PagerDuty).

It tracks:
  - Total request count and error count
  - A rolling window of the last N predictions' risk labels
  - An ALERT is logged (at WARNING level) if the rolling high-risk rate
    exceeds a threshold, simulating a data/model drift alert a real
    deployment would want to catch (e.g. if conditions become unusually
    risky more often than the model's training data would suggest).

Usage:
    uvicorn app:app --reload
Then visit http://localhost:8000/docs for interactive API documentation.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

MODEL_ARTIFACT_PATH = Path(__file__).parent / "model_artifact.joblib"

# Monitoring configuration
ROLLING_WINDOW_SIZE = 50
HIGH_RISK_ALERT_THRESHOLD = 0.60  # alert if >60% of the last N predictions are "High"

app = FastAPI(title="Traffic Risk Prediction API", version="1.0")

# Loaded once at startup
try:
    logger.info(f"Loading model artifact from {MODEL_ARTIFACT_PATH}")
    artifact = joblib.load(MODEL_ARTIFACT_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    valid_days = artifact["day_of_week_options"]
    valid_weather = artifact["weather_main_options"]
    logger.info("Model artifact loaded successfully.")
except FileNotFoundError:
    logger.error(f"Model artifact not found at {MODEL_ARTIFACT_PATH}. Run train_deployment_model.py first.")
    raise

# In-memory monitoring state (resets on service restart -- see module docstring)
monitoring_state = {
    "total_requests": 0,
    "total_errors": 0,
    "recent_predictions": deque(maxlen=ROLLING_WINDOW_SIZE),
}


class TrafficConditions(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of day, 0-23")
    month: int = Field(..., ge=1, le=12, description="Month, 1-12")
    day_of_week: str = Field(..., description="e.g. Monday, Tuesday, ...")
    weather_main: str = Field(..., description="e.g. Clear, Clouds, Rain, Snow, ...")
    temp_celsius: float = Field(..., description="Temperature in Celsius")
    clouds_all: int = Field(..., ge=0, le=100, description="Cloud coverage percentage")
    rain_1h: float = Field(0.0, ge=0, description="Rainfall in the last hour (mm)")
    snow_1h: float = Field(0.0, ge=0, description="Snowfall in the last hour (mm)")
    is_weekend: bool = Field(...)
    is_clear: bool = Field(...)
    is_precipitating: bool = Field(...)


class PredictionResponse(BaseModel):
    predicted_label: str
    high_risk_probability: float
    warning: Optional[str] = None


def build_feature_row(conditions: TrafficConditions) -> pd.DataFrame:
    """Convert a validated request into the exact one-hot-encoded feature
    row the model expects, matching build_feature_matrix() used at training
    time in train_deployment_model.py.
    """
    raw = {
        "hour": conditions.hour,
        "month": conditions.month,
        "temp_celsius": conditions.temp_celsius,
        "clouds_all": conditions.clouds_all,
        "rain_1h": conditions.rain_1h,
        "snow_1h": conditions.snow_1h,
        "is_weekend": int(conditions.is_weekend),
        "is_clear": int(conditions.is_clear),
        "is_precipitating": int(conditions.is_precipitating),
    }
    row = pd.DataFrame([raw])

    for day in valid_days:
        col = f"day_of_week_{day}"
        if col in feature_columns:
            row[col] = 1 if conditions.day_of_week == day else 0

    for weather in valid_weather:
        col = f"weather_main_{weather}"
        if col in feature_columns:
            row[col] = 1 if conditions.weather_main == weather else 0

    # Ensure exact column order/set the model was trained on
    row = row.reindex(columns=feature_columns, fill_value=0)
    return row


def check_for_drift_alert() -> Optional[str]:
    """Simulated monitoring check: alert if recent predictions are skewing
    unusually high-risk compared to what the training data would suggest."""
    recent = monitoring_state["recent_predictions"]
    if len(recent) < ROLLING_WINDOW_SIZE:
        return None

    high_risk_rate = sum(recent) / len(recent)
    if high_risk_rate > HIGH_RISK_ALERT_THRESHOLD:
        alert_msg = (
            f"ALERT: rolling high-risk prediction rate is {high_risk_rate:.1%} "
            f"over the last {ROLLING_WINDOW_SIZE} requests, exceeding the "
            f"{HIGH_RISK_ALERT_THRESHOLD:.0%} threshold. This could indicate "
            f"a genuine spike in risky conditions, or model/data drift worth investigating."
        )
        logger.warning(alert_msg)
        return alert_msg
    return None


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(conditions: TrafficConditions):
    monitoring_state["total_requests"] += 1

    if conditions.day_of_week not in valid_days:
        monitoring_state["total_errors"] += 1
        logger.warning(f"Invalid day_of_week in request: '{conditions.day_of_week}'")
        raise HTTPException(status_code=422, detail=f"day_of_week must be one of {valid_days}")

    if conditions.weather_main not in valid_weather:
        monitoring_state["total_errors"] += 1
        logger.warning(f"Invalid weather_main in request: '{conditions.weather_main}'")
        raise HTTPException(status_code=422, detail=f"weather_main must be one of {valid_weather}")

    try:
        X = build_feature_row(conditions)
        proba = model.predict_proba(X)[0][1]
        label = "High" if proba > 0.5 else "Low"
    except Exception as e:
        monitoring_state["total_errors"] += 1
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    monitoring_state["recent_predictions"].append(1 if label == "High" else 0)
    alert = check_for_drift_alert()

    logger.info(
        f"Prediction served: label={label}, probability={proba:.3f}, "
        f"hour={conditions.hour}, day={conditions.day_of_week}, weather={conditions.weather_main}"
    )

    return PredictionResponse(
        predicted_label=label,
        high_risk_probability=round(float(proba), 4),
        warning=alert,
    )


@app.get("/monitoring")
def get_monitoring_stats():
    recent = monitoring_state["recent_predictions"]
    rolling_high_risk_rate = (sum(recent) / len(recent)) if recent else None
    return {
        "total_requests": monitoring_state["total_requests"],
        "total_errors": monitoring_state["total_errors"],
        "rolling_window_size": len(recent),
        "rolling_high_risk_rate": round(rolling_high_risk_rate, 4) if rolling_high_risk_rate is not None else None,
        "alert_threshold": HIGH_RISK_ALERT_THRESHOLD,
    }
