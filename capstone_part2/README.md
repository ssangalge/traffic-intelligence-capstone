# Part 2 – Python Pipeline

Data pipeline, feature engineering, visualizations, and an interactive CLI app for the Metro Interstate Traffic Volume dataset.

## Logging Configuration

All scripts share a single logging setup, configured once in `logging_setup.py` and imported by every other module (`logging.getLogger(__name__)` is used in each file, never the bare root logger).

**Where logs are written:**
- **Console (stdout):** shows INFO level and above (INFO, WARNING, ERROR) — this is what you see while a script runs.
- **`pipeline.log` (file):** captures everything from DEBUG level upward — the complete diagnostic trail, including detail not shown on-screen. This file is gitignored since it grows on every run; a frozen example is committed separately as `sample_pipeline_log.txt`.

**Log format:** `timestamp | level | module name | message`

**What each level means in this project:**
| Level | Meaning | Example |
|---|---|---|
| `DEBUG` | Fine-grained internal values, useful only for troubleshooting. Never shown on console, only in `pipeline.log`. | Calculated quartile threshold values before they're used to build a category column |
| `INFO` | Normal, expected milestones. | Data loaded (with row/column counts), a cleaning step completed, a figure saved, a CLI command was invoked |
| `WARNING` | Something unexpected but recoverable — data was changed. | Rows dropped as duplicates, an impossible value imputed, a monitoring alert threshold crossed |
| `ERROR` | Something prevented normal completion, or the user supplied invalid input. | A malformed date entered in the CLI, the pipeline failing to write its output file |

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