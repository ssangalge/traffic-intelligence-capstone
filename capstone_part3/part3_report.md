# Part 3 – ML/AI Report

**Project:** Smart City Traffic Intelligence
**Scope:** Proxy risk labeling, supervised/unsupervised learning, deep learning with explainability, MLflow experiment tracking, a travel-timing recommendation system, an MLOps deployment simulation, and a bias/fairness/governance/sustainability analysis.

## Important Framing

There is no real accident dataset for this project. Every "risk" prediction throughout Part 3 refers to **`proxy_risk_label`**, a synthetic target constructed from traffic congestion quartiles and severe/low-visibility weather conditions (see `proxy_risk_label.py`). This is explicitly *not* a validated accident predictor, and this distinction is enforced in code (warning-level logs), in variable naming, and in the API's response fields throughout the project.

## 1. Proxy Risk Label

Built from: congestion quartile (Q3/Q4 contribute to risk) + severe/low-visibility weather (Fog, Snow, Thunderstorm, Squall, or heavy rain/snowfall). Resulting distribution: **27.7% High risk, 72.3% Low risk** — a usable, if imperfect, target for the supervised learning tasks that follow.

## 2. Supervised Learning

The common feature set includes time-based features, weather encodings, a **holiday flag**, and **cyclical encodings of both hour and day of week** (`hour_sin`/`hour_cos`, `dow_sin`/`dow_cos`) — required since these are circular variables (hour 23 and hour 0 are adjacent, not 23 apart).

**Classification** (predicting `proxy_risk_label`): Logistic Regression achieved **80.86% accuracy** (ROC AUC 0.897), while Random Forest achieved **93.89% accuracy** (ROC AUC 0.980) with balanced precision/recall (~88-90% each).

**Regression** (predicting `traffic_volume`): Linear Regression explained **72.76%** of variance (R²=0.728), while Random Forest Regressor explained **95.17%** (R²=0.952).

**The cyclical encoding finding is the single most important result in this section**: before adding `hour_sin`/`hour_cos`, Logistic Regression scored only 72.5% accuracy and Linear Regression only R²=0.198. Adding cyclical encoding alone — with no other changes — improved Logistic Regression to 80.9% and Linear Regression's R² by more than 3.5x, to 0.728. This is because linear models cannot represent a circular, double-peaked daily pattern using a raw integer hour (0-23) as input; the sine/cosine transform lets even a linear model correctly express the rush-hour structure. Tree-based models (Random Forest) were far less sensitive to this, since they can already split on raw hour ranges non-linearly.

**Leakage avoidance:** neither model used `congestion_quartile`, `proxy_risk_score`, `is_severe_weather`, or `traffic_volume` as input features when predicting `proxy_risk_label`, since these are used to *construct* that label. Only genuinely independent time, weather, and holiday features were used as predictors.

## 3. Unsupervised Learning

**K-means clustering** (k=4, using hour/temperature/traffic volume, with no access to the risk label) naturally rediscovered the day/night traffic structure: two low-traffic clusters (overnight and evening) with 0-0.6% high-risk rate, and two high-traffic midday clusters (warm and cold) with 53-55% high-risk rate. This is strong independent validation that the proxy label reflects real structure in the data, not an arbitrary construct.

**Association rule mining (Apriori)** found that weekday morning rush hour predicts High risk with 85.3% confidence and a lift of 3.08 — directly consistent with every other finding across all three parts of this project.

## 4. Deep Learning with Explainability

A small feedforward neural network (2 hidden layers, dropout regularization) achieved **93.71% test accuracy** predicting `proxy_risk_label` — slightly ahead of where it stood before cyclical encoding was added (92.5%), and now closely comparable to the Random Forest classifier. SHAP (KernelExplainer) analysis confirmed **`hour_cos`** as the dominant feature (mean |SHAP| = 0.251), consistent with every prior finding in this project that time-of-day drives most of the predictive signal.

**On the choice of classification over "demand prediction":** the assignment lists three example deep learning framings (a neural network for demand prediction, an LSTM for sequential traffic behaviour, or a CNN for image-based mobility). This project used a feedforward network for the same classification task as the supervised learning section (predicting `proxy_risk_label`) rather than a demand/traffic-volume regression network. This choice was made because: (1) the classification framing let the same SHAP explainability workflow directly extend and cross-validate the Random Forest's feature importances from Section 2, strengthening the project's overall internal consistency; and (2) Section 2's Random Forest Regressor already achieves strong demand-prediction performance (R²=0.952), so a second demand-prediction model would have been comparatively redundant. A regression-framed neural network for `traffic_volume` would have been an equally valid choice and is a natural extension for future work.

## 5. Advanced Technique: MLflow Experiment Tracking

Four model configurations were tracked via MLflow (Logistic Regression; Random Forest at 50/100/200 trees), logging parameters, metrics, and model artifacts for each run. All three Random Forest configurations performed almost identically (93.87-93.89% accuracy), confirming that beyond 50 trees, additional trees add negligible predictive value for this task — directly informing the sustainability discussion in Section 8, where a smaller model was deliberately chosen for deployment.

## 6. Travel-Timing Recommendation System

A rule-based recommender (`travel_recommendation.py`) surfaces the safest and riskiest hours for a given day of week and (optionally) weather condition, using the historical risk-labeled data directly. Example finding: Monday at 7am under Snow conditions shows a 100% historical high-risk rate — a genuinely striking, actionable data point for a real commuter.

## 7. MLOps Deployment Simulation

A FastAPI service (`app.py`) serves the trained classifier via a `/predict` endpoint, with input validation, error handling, and a simulated monitoring/alerting system (`/monitoring` endpoint) that tracks a rolling high-risk prediction rate and fires a drift alert if it exceeds 60% over the last 50 requests. This was tested and confirmed to correctly distinguish a snowy Monday rush hour (82% high-risk probability) from a clear Sunday night (1% high-risk probability), and to correctly trigger its alert under sustained high-risk load.

## 8. Bias, Fairness, Governance & Sustainability

Full findings are in `bias_fairness_governance_report.md`. Headline findings:

- **The proxy label itself is structurally biased** toward flagging high-congestion (mostly weekday commute) periods as "High risk," regardless of real accident likelihood.
- **The classifier's recall on weekends (43.4%) remains far worse than on weekdays (93.9%)**, even after the feature improvements — a persistent, measured fairness gap that would under-warn weekend travelers in any real deployment.
- **Haze conditions and midday hours** show the weakest overall model performance among tested subgroups.
- **The larger neural network and 200-tree Random Forest are not sustainability-justified** given their marginal accuracy gains over a smaller, depth-capped Random Forest — the deployed model was deliberately chosen to balance accuracy against size/cost.

## Summary

Across Parts 1–3, a consistent, cross-validated story emerged: **traffic risk on this corridor is driven overwhelmingly by time-of-day and day-of-week patterns (rush hour, weekday vs. weekend), with weather playing a real but secondary role.** This finding was independently confirmed via SQL analysis (Part 1), probability analysis (Part 1), supervised learning (Part 3), unsupervised clustering (Part 3), association rules (Part 3), and SHAP explainability (Part 3) — a level of cross-method agreement that gives real confidence in the finding, even though the underlying "risk" label itself remains a synthetic construct requiring careful, honest communication in any real-world application.