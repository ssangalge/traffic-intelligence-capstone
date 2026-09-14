# Part 2 – Python Pipeline Report

**Project:** Smart City Traffic Intelligence
**Scope:** Data pipeline, feature engineering, visualizations, and CLI app built on the Metro Interstate Traffic Volume dataset.

## Pipeline Overview

Four components were built, each with structured logging (console + `pipeline.log`, timestamp/level/module/message format):

1. **`pipeline.py`** — loads the raw CSV, validates schema, fixes data types, checks for missing values, detects outliers, and writes `cleaned_traffic_data.csv`.
2. **`feature_engineering.py`** — reads the cleaned data and adds time features (hour, day of week, month, weekend flag), weather features (Celsius temperature, clear/precipitating flags), scaled numerics (standardized temperature and traffic volume), and the congestion target variables (`is_congested`, `traffic_category`), writing `featured_traffic_data.csv`.
3. **`visualizations.py`** — generates three Matplotlib figures from the featured data.
4. **`cli_app.py`** — a menu-driven interface for exploring the featured dataset interactively.

## Key Data Quality Findings

Building this pipeline surfaced two genuine bugs/issues not visible in Part 1's SQL/Power BI work, since Python's default CSV parsing behaves differently:

**1. Pandas silently nulled the `holiday` column.** By default, `pandas.read_csv()` treats the literal string `"None"` as a missing value (`NaN`), which would have wrongly converted 48,143 rows of legitimate "not a holiday" data into true nulls. Fixed using `keep_default_na=False` on load. This is a good example of why the same dataset can look different depending on the tool: Power BI in Part 1 did not have this issue.

**2. A richer outlier picture than Part 1's dashboard showed.** Part 1's Power BI chart only surfaced one anomalous day (July 23, 2016). Running a systematic threshold check (`traffic_volume < 20`) in the pipeline found **52 rows** across **9 distinct dates** (July 2015; July, August, and September 2016) with physically implausible near-zero traffic volume. Rather than dropping these rows, they are flagged via an `is_implausible_volume` boolean column so downstream analysis or modelling can decide how to treat them, while the surrounding context (e.g. the plausible readings before/after) is preserved.

## Feature Engineering Summary

The `is_congested` target variable (traffic_volume > 5,500) shows a **14.73% congestion rate**, consistent with Task 3's SQL-based probability analysis from Part 1 — a useful cross-check confirming the pipeline behaves correctly end-to-end. The three-class `traffic_category` (Low/Medium/High) reuses the exact thresholds from Part 1's Power BI dashboard for continuity across the whole project.

## Visualization Findings

- **Hourly traffic pattern**: confirms the expected two-peak commuter structure (morning ~7-9am, evening ~4-6pm), consistent with Part 1's Power BI hourly chart.
- **Traffic volume distribution**: revealed the distribution is genuinely **bimodal** (two distinct clusters: low-volume overnight hours and higher-volume daytime hours), a more precise finding than the "left-skewed" description used in Part 1's Task 2, which only compared mean vs. median without visualizing the full shape.
- **Congestion rate by weather**: confirms Part 1's finding that Clouds has the highest congestion rate (~17%) and reinforces that Squall's apparent low rate (0%) is unreliable, given it's based on only 4 total records.

## CLI App

The interactive CLI (`cli_app.py`) provides four functions: overall statistics, date-range filtering, weather-condition filtering, and congestion rate lookup. All user inputs (dates, weather names, menu choices) are validated, with invalid input logged at WARNING level and handled gracefully without crashing.

## Design Decisions Carried Over From Part 1

- Duplicate timestamps (5,445 groups) are preserved, not dropped, since Part 1 established they represent legitimate simultaneous weather conditions for the same hour rather than data errors.
- All thresholds (congestion at 5,500; Low/Medium/High traffic categories) match Part 1's Power BI `Traffic_Category` column exactly, keeping the three-part project internally consistent.
