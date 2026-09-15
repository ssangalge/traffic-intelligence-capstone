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
| Weekday | 6,889 | 0.916 | 0.839 | **0.953** |
| Weekend | 2,752 | 0.964 | 0.828 | **0.376** |

**Finding:** the model's recall on weekends (37.6%) is dramatically worse than on weekdays (95.3%) — it misses the majority of genuinely "High risk" weekend hours. This is the most significant fairness concern found: **the model is far less reliable for weekend travelers**, likely because the training data contains proportionally fewer weekend high-risk examples (the label's congestion-quartile logic is itself calibrated mostly around weekday commute patterns).

| Weather condition | n (test) | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Fog | 185 | 0.968 | 0.965 | 0.932 |
| Thunderstorm | 187 | 0.963 | 0.941 | 0.955 |
| Snow | 575 | 0.930 | 0.909 | 0.949 |
| Clear | 2,702 | 0.943 | 0.851 | 0.903 |
| Drizzle | 327 | 0.945 | 0.830 | 0.988 |
| Mist | 1,256 | 0.936 | 0.813 | 0.946 |
| Clouds | 2,978 | 0.915 | 0.827 | 0.915 |
| Rain | 1,172 | 0.922 | 0.796 | 0.941 |
| **Haze** | 254 | **0.882** | **0.761** | 0.785 |

**Finding:** performance under **Haze** conditions is notably weaker across every metric than under other weather types. This is plausibly a smaller-sample effect (Haze is one of the rarer categories) rather than a fundamental issue, but it means predictions during hazy conditions should be treated with more caution.

| Time-of-day bucket | n (test) | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Overnight | 4,459 | 0.997 | **0.500** | **0.167** |
| Midday | 2,343 | **0.821** | **0.744** | 0.897 |
| Morning rush | 1,665 | 0.911 | 0.887 | 0.977 |
| Evening rush | 1,174 | 0.917 | 0.949 | 0.891 |

**Finding:** overnight hours show a misleadingly high accuracy (99.7%) driven almost entirely by class imbalance — "High risk" is so rare overnight that the model can score well just by predicting "Low" nearly every time, but its precision (50%) and recall (16.7%) on the rare positive cases are poor. Midday hours show the weakest overall balance of precision/recall, suggesting this is the hardest segment for the model to characterize.

### 1.3 Fairness implications

- **Who is underserved:** weekend travelers and midday travelers receive the least reliable risk predictions. If this system informed real routing/scheduling decisions, these groups would be systematically under-warned about genuine risk.
- **Root cause:** this is very likely a **training data imbalance** issue (fewer weekend/midday high-risk examples for the model to learn from) compounded by the proxy label's inherent weekday-commute bias, rather than an issue with the algorithm itself.
- **Recommendation:** before any real-world use, weekend-specific and midday-specific model variants (or reweighted training) should be explored, and users should be shown a confidence/reliability indicator alongside predictions rather than a bare label.

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