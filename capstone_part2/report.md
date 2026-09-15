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

Building this pipeline surfaced several genuine data quality issues, some not visible in Part 1's SQL/Power BI work since Python's default CSV parsing behaves differently, others found through systematic outlier checks:

**1. Pandas silently nulled the `holiday` column.** By default, `pandas.read_csv()` treats the literal string `"None"` as a missing value (`NaN`), which would have wrongly converted 48,143 rows of legitimate "not a holiday" data into true nulls. Fixed using `keep_default_na=False` on load.

**2. 17 exact full-row duplicates** were identified and removed — every column identical across each pair. This is distinct from the 5,430 duplicate *timestamps* (see below), which are legitimate multi-weather-condition entries, not errors.

**3. 10 rows recorded `temp == 0 Kelvin`** (absolute zero — a physically impossible weather reading). Imputed using that month's median temperature (4 rows in January, 6 in February), via an explicit month-by-month loop rather than a single global average.

**4. 1 row recorded `rain_1h = 9,831.3mm`** — a known bug in this specific public dataset (the highest rainfall ever recorded globally in one hour is ~305mm). Imputed using July's median rainfall (0.00mm).

**5. Categorical text standardisation**: stripped stray whitespace and lower-cased `weather_description`, which reduced 38 distinct descriptions to 37 (removing a casing/whitespace duplicate).

**6. A richer near-zero-traffic outlier picture than Part 1's dashboard showed.** Part 1's Power BI chart only surfaced one anomalous day (July 23, 2016). A systematic threshold check (`traffic_volume < 20`) found **52 rows** across **9 distinct dates** with physically implausible near-zero traffic volume. These are flagged (not imputed) via an `is_implausible_volume` column, since they occur in connected runs where a median would obscure a genuine, documentable sensor malfunction event rather than correct a one-off glitch.

## Feature Engineering Summary

The `is_congested` target variable (traffic_volume > 5,500) shows a **14.73% congestion rate**, consistent with Task 3's SQL-based probability analysis from Part 1. Two congestion category schemes are provided: `traffic_category` (Low/Medium/High, using Part 1's fixed thresholds, for cross-part consistency) and `congestion_category_quartile` (Low/Medium/High/Severe, data-driven quartile thresholds of Q1=1,192.5, Q2=3,379.0, Q3=4,933.0 — logged at DEBUG level since these are intermediate calculations). A **cyclical encoding of hour** (`hour_sin`/`hour_cos`) was also added, since hour is circular (23:00 and 00:00 are adjacent, not 23 apart) — this pays off significantly in Part 3, where it dramatically improves linear models' ability to represent the rush-hour pattern.

## Visualization Findings

- **Hourly traffic pattern**: confirms the expected two-peak commuter structure (morning ~7-9am, evening ~4-6pm), consistent with Part 1's Power BI hourly chart.
- **Traffic volume distribution**: revealed the distribution is genuinely **bimodal** (two distinct clusters: low-volume overnight hours and higher-volume daytime hours), a more precise finding than the "left-skewed" description used in Part 1's Task 2, which only compared mean vs. median without visualizing the full shape.
- **Congestion rate by weather**: confirms Part 1's finding that Clouds has the highest congestion rate (~17%) and reinforces that Squall's apparent low rate (0%) is unreliable, given it's based on only 4 total records.

## CLI App

The interactive CLI (`cli_app.py`) provides four functions: overall statistics, date-range filtering, weather-condition filtering, and congestion rate lookup. All user inputs (dates, weather names, menu choices) are validated, with invalid input logged at WARNING level and handled gracefully without crashing.

## Design Decisions Carried Over From Part 1

- Duplicate timestamps (5,430 groups, after removing the 17 exact duplicates) are preserved, not dropped, since Part 1 established they represent legitimate simultaneous weather conditions for the same hour rather than data errors.
- All fixed thresholds (congestion at 5,500; Low/Medium/High traffic categories) match Part 1's Power BI `Traffic_Category` column exactly, keeping the three-part project internally consistent, while the new quartile-based category provides a data-driven alternative view.
- Final cleaned dataset: **48,187 rows** (48,204 raw rows minus 17 exact duplicates removed).