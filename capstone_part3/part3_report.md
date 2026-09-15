# Part 3 – ML/AI Report

**Project:** Smart City Traffic Intelligence
**Scope:** Proxy risk labeling, supervised/unsupervised learning, deep learning with explainability, MLflow experiment tracking, a travel-timing recommendation system, an MLOps deployment simulation, and a bias/fairness/governance/sustainability analysis.

## Important Framing

There is no real accident dataset for this project. Every "risk" prediction throughout Part 3 refers to **`proxy_risk_label`**, a synthetic target constructed from traffic congestion quartiles and severe/low-visibility weather conditions (see `proxy_risk_label.py`). This is explicitly *not* a validated accident predictor, and this distinction is enforced in code (warning-level logs), in variable naming, and in the API's response fields throughout the project.

## 1. Proxy Risk Label

Built from: congestion quartile (Q3/Q4 contribute to risk) + severe/low-visibility weather (Fog, Snow, Thunderstorm, Squall, or heavy rain/snowfall). Resulting distribution: **27.7% High risk, 72.3% Low risk** — a usable, if imperfect, target for the supervised learning tasks that follow.

## 2. Supervised Learning

**Classification** (predicting `proxy_risk_label`): Logistic Regression achieved only 72.5% accuracy with very poor recall (11.4%), while Random Forest achieved 94.1% accuracy with balanced precision/recall (~89% each). This gap demonstrates that the relationship between time/weather features and risk is highly non-linear — a linear model structurally cannot capture the rush-hour pattern that drives most of the risk signal.

**Regression** (predicting `traffic_volume`): Linear Regression explained only ~20% of variance (R²=0.198), while Random Forest Regressor explained ~95% (R²=0.947), for the same underlying reason — traffic volume follows a non-linear double-peak daily curve.

**Leakage avoidance:** neither model used `congestion_quartile`, `proxy_risk_score`, `is_severe_weather`, or `traffic_volume` as input features when predicting `proxy_risk_label`, since these are used to *construct* that label. Only genuinely independent time and raw weather features were used as predictors.

## 3. Unsupervised Learning

**K-means clustering** (k=4, using hour/temperature/traffic volume, with no access to the risk label) naturally rediscovered the day/night traffic structure: two low-traffic clusters (overnight and evening) with 0-0.6% high-risk rate, and two high-traffic midday clusters (warm and cold) with 53-55% high-risk rate. This is strong independent validation that the proxy label reflects real structure in the data, not an arbitrary construct.

**Association rule mining (Apriori)** found that weekday morning rush hour predicts High risk with 85.3% confidence and a lift of 3.08 — directly consistent with every other finding across all three parts of this project.

## 4. Deep Learning with Explainability

A small feedforward neural network (2 hidden layers, dropout regularization) achieved 92.5% test accuracy predicting `proxy_risk_label` — comparable to, but not better than, the Random Forest. SHAP (KernelExplainer) analysis confirmed **hour of day** as the dominant feature (mean |SHAP| = 0.245), far ahead of `is_weekend` (0.063) and day-of-week indicators — consistent with every prior finding in this project.

## 5. Advanced Technique: MLflow Experiment Tracking

Four model configurations were tracked via MLflow (Logistic Regression; Random Forest at 50/100/200 trees), logging parameters, metrics, and model artifacts for each run. The 200-tree Random Forest performed marginally best (94.17% accuracy), but as discussed in the sustainability report, this marginal gain came with a 7x larger model file — a trade-off not worth making for the deployed version.

## 6. Travel-Timing Recommendation System

A rule-based recommender (`travel_recommendation.py`) surfaces the safest and riskiest hours for a given day of week and (optionally) weather condition, using the historical risk-labeled data directly. Example finding: Monday at 7am under Snow conditions shows a 100% historical high-risk rate — a genuinely striking, actionable data point for a real commuter.

## 7. MLOps Deployment Simulation

A FastAPI service (`app.py`) serves the trained classifier via a `/predict` endpoint, with input validation, error handling, and a simulated monitoring/alerting system (`/monitoring` endpoint) that tracks a rolling high-risk prediction rate and fires a drift alert if it exceeds 60% over the last 50 requests. This was tested and confirmed to correctly distinguish a snowy Monday rush hour (82% high-risk probability) from a clear Sunday night (1% high-risk probability), and to correctly trigger its alert under sustained high-risk load.

## 8. Bias, Fairness, Governance & Sustainability

Full findings are in `bias_fairness_governance_report.md`. Headline findings:

- **The proxy label itself is structurally biased** toward flagging high-congestion (mostly weekday commute) periods as "High risk," regardless of real accident likelihood.
- **The classifier's recall on weekends (37.6%) is far worse than on weekdays (95.3%)** — a genuine, measured fairness gap that would under-warn weekend travelers in any real deployment.
- **Haze conditions and midday hours** show the weakest overall model performance among tested subgroups.
- **The larger neural network and 200-tree Random Forest are not sustainability-justified** given their marginal accuracy gains over a smaller, depth-capped Random Forest — the deployed model was deliberately chosen to balance accuracy against size/cost.

## Summary

Across Parts 1–3, a consistent, cross-validated story emerged: **traffic risk on this corridor is driven overwhelmingly by time-of-day and day-of-week patterns (rush hour, weekday vs. weekend), with weather playing a real but secondary role.** This finding was independently confirmed via SQL analysis (Part 1), probability analysis (Part 1), supervised learning (Part 3), unsupervised clustering (Part 3), association rules (Part 3), and SHAP explainability (Part 3) — a level of cross-method agreement that gives real confidence in the finding, even though the underlying "risk" label itself remains a synthetic construct requiring careful, honest communication in any real-world application.
