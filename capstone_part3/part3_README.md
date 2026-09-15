# Part 3 – ML/AI

Proxy risk labeling, supervised/unsupervised learning, deep learning with explainability, MLflow tracking, a travel-timing recommender, an MLOps deployment simulation, and a bias/fairness/governance/sustainability report — all built on the Metro Interstate Traffic Volume dataset, with no real accident data available.

## ⚠️ Important

`proxy_risk_label` throughout this folder is a **synthetic construct**, not real accident data. See `bias_fairness_governance_report.md` Section 1.1 and the warning logs in `proxy_risk_label.py` for full context. Never present model outputs from this folder as validated accident predictions.

## Setup

```bash
pip install pandas scikit-learn matplotlib mlxtend tensorflow shap mlflow joblib fastapi uvicorn
```

## Files

| File | Purpose |
|---|---|
| `logging_setup.py` | Centralized logging config (same pattern as Part 2) |
| `proxy_risk_label.py` | Builds the synthetic risk label from congestion + weather. Writes `risk_labeled_data.csv` |
| `supervised_models.py` | Classification (Logistic Regression, Random Forest) + regression (Linear Regression, Random Forest Regressor) |
| `unsupervised_learning.py` | K-means clustering + Apriori association rule mining |
| `deep_learning_model.py` | Feedforward neural network with SHAP explainability. Writes `figures/shap_summary.png` |
| `advanced_mlflow.py` | MLflow-tracked comparison of 4 classifier configurations |
| `travel_recommendation.py` | Interactive CLI recommending best/worst travel hours by day and weather |
| `train_deployment_model.py` | Trains and saves the production model artifact (`model_artifact.joblib`) |
| `app.py` | FastAPI service: `/predict`, `/health`, `/monitoring` endpoints |
| `bias_fairness_governance_report.md` | Full bias/fairness/governance/sustainability analysis, with real measured subgroup performance |
| `report.md` | Summary report of all Part 3 findings |

## Running everything end-to-end

```bash
python proxy_risk_label.py          # requires ../capstone_part2/featured_traffic_data.csv
python supervised_models.py
python unsupervised_learning.py
python deep_learning_model.py       # slow: SHAP explanation takes ~1-2 minutes
python advanced_mlflow.py
python travel_recommendation.py     # interactive
python train_deployment_model.py
uvicorn app:app --reload            # then visit http://localhost:8000/docs
```

View MLflow results with:
```bash
mlflow ui
```
Then open http://localhost:5000.

## Key Findings

- Time-of-day (especially rush hour) is overwhelmingly the strongest predictor of risk — confirmed independently via supervised learning, unsupervised clustering, association rules, and SHAP explainability.
- The classifier has a real, measured fairness gap: 95.3% recall on weekdays vs. only 37.6% on weekends.
- A smaller, depth-capped Random Forest (16MB) was deliberately chosen for deployment over larger models, since bigger models gave only marginal accuracy gains — see `bias_fairness_governance_report.md` Section 3 for the full sustainability analysis.

See `report.md` for the complete write-up.
