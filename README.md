# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

Capstone project analysing ~48,000 hourly records of westbound I-94 traffic volume near Minneapolis–St Paul, progressing from SQL/statistics through a Python data pipeline to machine learning and AI.

## Repository Structure

```
traffic-intelligence-capstone/
├── data/                              # Shared raw dataset (Metro_Interstate_Traffic_Volume.csv)
├── part1_sql/                         # Part 1: Data Analytics (SQL, stats, probability, Power BI)
│   └── insights_report.md             # Start here for Part 1
├── capstone_part2/                    # Part 2: Python pipeline, features, visualizations, CLI
│   └── report.md                      # Start here for Part 2
└── capstone_part3/                    # Part 3: ML/AI, deep learning, MLOps, bias/fairness
    ├── report.md                      # Start here for Part 3
    └── bias_fairness_governance_report.md
```

Each part builds on the last: Part 2's pipeline reuses Part 1's data-quality findings; Part 3's models reuse Part 2's engineered features.

## Headline Finding

Across all three parts — SQL analysis, probability theory, supervised learning, unsupervised clustering, association rules, and SHAP explainability — the same conclusion emerged independently every time: **traffic risk on this corridor is driven overwhelmingly by time-of-day and day-of-week patterns (rush hour, weekday vs. weekend), with weather playing a real but secondary role.** This level of cross-method agreement, using entirely different techniques in each part, gives real confidence in the finding.

## Important Note on Accident Risk

There is no real accident dataset for this project. All "risk" predictions in Part 3 refer to a **synthetic proxy label** built from traffic congestion and severe weather — not validated accident data. See `capstone_part3/bias_fairness_governance_report.md` for full details on this and other limitations.

## Quick Start

```bash
python3 -m venv capstone-env
source capstone-env/bin/activate
pip install pandas numpy matplotlib scikit-learn seaborn mlxtend tensorflow shap mlflow joblib fastapi uvicorn
```

Then follow each part's own README for run instructions:
- `part1_sql/` — run SQL scripts against `traffic.db` (built via `load_data.py`)
- `capstone_part2/README.md` — pipeline → features → visualizations → CLI
- `capstone_part3/README.md` — proxy label → ML models → deployment simulation

## Reports

| Part | Report |
|---|---|
| 1 | `part1_sql/insights_report.md` |
| 2 | `capstone_part2/report.md` |
| 3 | `capstone_part3/report.md` and `capstone_part3/bias_fairness_governance_report.md` |