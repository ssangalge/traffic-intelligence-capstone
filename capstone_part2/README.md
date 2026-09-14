# Part 2 – Python Pipeline

Data pipeline, feature engineering, visualizations, and an interactive CLI app for the Metro Interstate Traffic Volume dataset.

## Setup

From the repo root, with your virtual environment activated:

```bash
pip install pandas scikit-learn matplotlib
```

## Files

| File | Purpose |
|---|---|
| `logging_setup.py` | Centralized logging configuration (console + file, INFO/WARNING/ERROR/DEBUG). Run standalone to verify: `python logging_setup.py` |
| `pipeline.py` | Loads raw CSV, validates schema, fixes data types, detects outliers, writes `cleaned_traffic_data.csv` |
| `feature_engineering.py` | Adds time/weather features, scaled numerics, and the congestion target. Writes `featured_traffic_data.csv` |
| `visualizations.py` | Generates 3 Matplotlib charts into `figures/` |
| `cli_app.py` | Interactive menu-driven CLI for exploring the featured dataset |
| `sample_pipeline_log.txt` | A frozen snapshot of a full pipeline run's log output (the live `pipeline.log` is gitignored since it grows on every run) |
| `report.md` | Written summary of findings and design decisions |

## Running the pipeline end-to-end

```bash
python pipeline.py
python feature_engineering.py
python visualizations.py
python cli_app.py
```

Run in this order — each step depends on the output of the previous one:
`Metro_Interstate_Traffic_Volume.csv` (in `../data/`) → `cleaned_traffic_data.csv` → `featured_traffic_data.csv` → figures / CLI.

## Notes

- All scripts read the raw CSV with `keep_default_na=False` to prevent pandas from silently treating the literal string `"None"` in the `holiday` column as a missing value.
- Duplicate timestamps in the source data (5,445 groups) are intentionally preserved, not dropped — they represent multiple simultaneous weather conditions logged for the same hour, not data errors (see `report.md` and the Part 1 insights report for details).
- See `report.md` for the full write-up of findings, including data quality issues discovered while building this pipeline.
