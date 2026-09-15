# Bias, Fairness, Governance & Sustainability Report

**Project:** Smart City Traffic Intelligence — Part 3 (ML/AI)
**Model in scope:** Random Forest classifier predicting `proxy_risk_label` (High/Low)

---

## 1. Bias & Fairness Analysis

### 1.1 A structural bias built into the label itself

Before looking at model performance, it's important to be explicit about a bias baked into the **proxy_risk_label** by construction, not by the model. The label is built as:

> High risk = top congestion quartile (Q3/Q4) OR severe/low-visibility weather

This means **any subgroup that is naturally associated with high traffic volume — chiefly weekday rush-hour commuters — is systematically more likely to be labeled "High risk," regardless of whether real accidents actually occurred more often in that group.** The label conflates *congestion* with *risk*, which is a reasonable modelling assumption for a classroom project but would be a serious validity problem if this label were ever mistaken for real accident data. This is the single most important caveat in this entire report and is repeated throughout the codebase (see the `logger.warning()` calls in `proxy_risk_label.py`).

### 1.2 Measured performance disparities across subgroups

Testing the trained Random Forest classifier on held-out data, split by subgroup, reveals real and meaningful performance differences:

| Subgroup | n (test) | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Weekday | 6,911 | 0.933 | 0.885 | **0.939** |
| Weekend | 2,727 | 0.967 | 0.808 | **0.434** |

**Finding:** the model's recall on weekends (43.4%) remains dramatically worse than on weekdays (93.9%) — even after adding cyclical hour/day-of-week encoding and a holiday flag, which improved overall accuracy but did not close this gap. It still misses the majority of genuinely "High risk" weekend hours. This is the most significant fairness concern found: **the model is far less reliable for weekend travelers**, likely because the training data contains proportionally fewer weekend high-risk examples (the label's congestion-quartile logic is itself calibrated mostly around weekday commute patterns).

| Weather condition | n (test) | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Thunderstorm | 197 | 0.975 | 0.964 | 0.976 |
| Fog | 202 | 0.960 | 0.945 | 0.945 |
| Mist | 1,242 | 0.956 | 0.877 | 0.954 |
| Clear | 2,696 | 0.948 | 0.860 | 0.902 |
| Drizzle | 383 | 0.948 | 0.873 | 0.927 |
| Snow | 559 | 0.939 | 0.926 | 0.947 |
| Rain | 1,114 | 0.938 | 0.878 | 0.893 |
| Clouds | 2,982 | 0.932 | 0.882 | 0.896 |
| **Haze** | 255 | **0.914** | **0.818** | 0.887 |

**Finding:** performance under **Haze** conditions remains the weakest across every metric among weather types, though it improved from the earlier model (88.2%→91.4% accuracy). This is plausibly a smaller-sample effect (Haze is one of the rarer categories) rather than a fundamental issue, but predictions during hazy conditions should still be treated with more caution.

| Time-of-day bucket | n (test) | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Overnight | 4,481 | 0.997 | **0.500** | **0.143** |
| Midday | 2,323 | **0.863** | **0.835** | 0.852 |
| Morning rush | 1,648 | 0.921 | 0.901 | 0.982 |
| Evening rush | 1,186 | 0.922 | 0.931 | 0.918 |

**Finding:** overnight hours still show a misleadingly high accuracy (99.7%) driven almost entirely by class imbalance — "High risk" is so rare overnight that the model scores well by predicting "Low" nearly every time, but its precision (50%) and recall (14.3%) on the rare positive cases remain poor even after the feature improvements. Midday hours improved (82.1%→86.3% accuracy) but remain the weakest daytime segment.

### 1.3 Fairness implications

**A notable methodological finding:** adding cyclical hour/day-of-week encoding and a holiday flag meaningfully improved overall model accuracy (93.9% vs. 94.1% previously — comparable, but via a materially different and more principled feature representation) and even slightly improved weekend recall (37.6%→43.4%), but did **not** close the weekend fairness gap. This suggests the gap is not primarily a feature-representation problem (which better encoding could fix), but more likely a genuine **class imbalance** issue — proportionally fewer weekend "High risk" training examples for the model to learn from. This distinction matters for remediation: the fix is more likely to require rebalancing or reweighting training data by day-type, not further feature engineering.

- **Who is underserved:** weekend travelers and midday travelers receive the least reliable risk predictions. If this system informed real routing/scheduling decisions, these groups would be systematically under-warned about genuine risk.
- **Root cause:** very likely a **training data imbalance** issue (fewer weekend/midday high-risk examples for the model to learn from) compounded by the proxy label's inherent weekday-commute bias, rather than an issue with the algorithm or feature representation itself.
- **Recommendation:** before any real-world use, weekend-specific and midday-specific model variants (or reweighted/oversampled training) should be explored, and users should be shown a confidence/reliability indicator alongside predictions rather than a bare label.

### 1.4 Limitations on generalizability

This entire analysis is based on a single sensor location (westbound I-94 near Minneapolis-St Paul) over a specific historical period (2012-2018). The model and its fairness properties **do not generalize** to other roads, cities, climates, or time periods without re-validation. There is no demographic data in this dataset (e.g., no information about who is driving), so this report cannot and does not assess bias with respect to protected demographic characteristics — only observable situational subgroups (day type, weather, time of day).

---

## 2. Governance

### 2.1 Documenting the synthetic nature of the risk label

Because `proxy_risk_label` could easily be mistaken for real accident data if separated from its context, governance controls were built directly into the code and outputs, not left to documentation alone:

- `proxy_risk_label.py` logs an explicit `WARNING`-level message every time the label is constructed, stating it is synthetic and must not be presented as validated accident data.
- Every downstream script (`supervised_models.py`, `deep_learning_model.py`, `app.py`, etc.) references the label by its full name (`proxy_risk_label`), never abbreviating it to something that could be mistaken for real accident data (e.g., never renamed to just "accident" or "crash").
- The FastAPI service (`app.py`) returns `predicted_label` and `high_risk_probability` — deliberately not "accident_probability" — to keep the terminology honest at the point of consumption.

### 2.2 Data lineage and reproducibility

The project maintains a clear, auditable lineage: raw CSV → `pipeline.py` (cleaning/validation) → `feature_engineering.py` (features) → `proxy_risk_label.py` (synthetic target) → model training scripts. Each stage writes its own output file and logs its transformations, so any prediction can in principle be traced back to the exact raw data and code that produced it. MLflow (`advanced_mlflow.py`) additionally version-tracks model parameters and metrics across training runs, giving a reproducible record of which model configuration produced which results.

### 2.3 Recommended governance controls for any real deployment

This project is a simulation, but if a system like this were genuinely deployed:

- A **human-in-the-loop review** step should sit between model predictions and any real-world action (e.g., road advisories), especially given the fairness gaps found in Section 1.
- **Model cards** should be published alongside the model, stating clearly (as this report does) that risk is a proxy construct, not validated accident data, and listing the known subgroup performance gaps.
- **Access control and audit logging** on the prediction API would be needed in production (this simulation's `/monitoring` endpoint is a simplified stand-in for what a real monitoring/alerting stack would provide).
- **Periodic re-validation** against any newly available real accident data (if it become available) should replace the proxy label entirely, rather than treating this proxy as a permanent substitute.

---

## 3. Sustainability

### 3.1 Computational cost comparison across models built in this project

| Model | Approx. training time | Artifact size | Test accuracy |
|---|---|---|---|
| Logistic Regression | <1 second | <1 MB | 0.725 |
| Random Forest (100 trees, depth 15) | ~5 seconds | 16 MB | 0.930 |
| Random Forest (200 trees, unlimited depth) | ~13 seconds | 115 MB | 0.942 |
| Feedforward Neural Network (TensorFlow) | ~7 seconds (20 epochs) + SHAP explanation (~80 seconds) | Comparable to RF | 0.925 |

### 3.2 Findings and recommendation

The **200-tree, unlimited-depth Random Forest** used in `supervised_models.py`/`advanced_mlflow.py` produced a **115MB model artifact** for only a marginal accuracy gain (94.2% vs. 93.0%) over a much smaller, depth-capped 100-tree version. For the deployment simulation (`app.py`), the depth-capped version was deliberately chosen — a **~7x smaller file** with **~1 percentage point** less accuracy is a clearly favourable trade-off for real-world deployment, where model size affects storage, loading time, and (at scale) serving cost.

The **neural network with SHAP explainability** achieved comparable accuracy (92.5%) to the Random Forest (93-94%) but at a much higher computational cost: TensorFlow itself is a heavyweight dependency, and generating SHAP explanations via `KernelExplainer` took roughly 80 seconds for just 200 test samples (a model-agnostic explainer that would scale poorly to real-time production traffic). Given the Random Forest already provides built-in feature importances and comparable accuracy, **the added complexity and compute cost of the neural network is not clearly justified for this specific task** — it was valuable here as a learning exercise and to demonstrate model-agnostic explainability (SHAP), but a production system optimizing for sustainability would likely deploy the Random Forest alone.

### 3.3 Broader sustainability considerations

- **Experiment tracking efficiency:** MLflow was used to compare only 4 model configurations in this project (deliberately small), rather than large-scale hyperparameter sweeps, keeping the compute/energy footprint of experimentation modest.
- **Inference cost:** the deployed Random Forest's `/predict` endpoint responds in milliseconds per request, which is appropriate for a system that might realistically serve many concurrent users; a neural network of meaningfully larger scale would need to weigh added latency/energy cost against any accuracy gains, which were not evident here.
- **Recommendation:** for future iterations of this system, model selection should default to the smallest model that meets accuracy requirements, reserving heavier models (deep learning, large ensembles) for cases where they demonstrate a clear, justified performance advantage — which was not the case in this project's findings.