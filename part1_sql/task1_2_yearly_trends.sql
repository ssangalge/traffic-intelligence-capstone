-- Task 1.2: Analyse annual traffic trends
-- Compares total yearly traffic volume for 2012-2017.
--
-- IMPORTANT DATA QUALITY NOTE:
-- This dataset contains 5,445 groups of duplicate timestamps (the same
-- hour appears more than once, sometimes up to 6 times). This means a
-- raw SUM(traffic_volume) per year is misleading: a year with more
-- duplicate/duplicated rows will show an inflated total even if actual
-- traffic conditions were similar. AVG(traffic_volume) per year is a
-- fairer comparison because it isn't distorted by row-count differences,
-- but for a fully rigorous analysis, duplicates should be de-duplicated
-- first (e.g. keep one row per date_time). Both views are included below.

-- 1. Raw comparison: total volume, hours recorded, and average volume per year
SELECT
    strftime('%Y', date_time) AS year,
    COUNT(*)                   AS hours_recorded,
    SUM(traffic_volume)        AS total_volume,
    ROUND(AVG(traffic_volume), 1) AS avg_volume
FROM traffic
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- 2. De-duplicated comparison: one row per unique timestamp, using AVG
--    per timestamp group to collapse duplicates before aggregating by year.
--    This gives a cleaner, more defensible year-over-year comparison.
WITH deduped AS (
    SELECT
        date_time,
        AVG(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
)
SELECT
    strftime('%Y', date_time) AS year,
    COUNT(*)                   AS unique_hours,
    ROUND(SUM(traffic_volume), 0) AS total_volume_dedup,
    ROUND(AVG(traffic_volume), 1) AS avg_volume_dedup
FROM deduped
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- 3. Year-over-year change in average volume (using de-duplicated data)
WITH deduped AS (
    SELECT date_time, AVG(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
),
yearly AS (
    SELECT
        strftime('%Y', date_time) AS year,
        AVG(traffic_volume) AS avg_volume
    FROM deduped
    WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
    GROUP BY year
)
SELECT
    year,
    ROUND(avg_volume, 1) AS avg_volume,
    ROUND(avg_volume - LAG(avg_volume) OVER (ORDER BY year), 1) AS change_from_prior_year,
    ROUND(
        100.0 * (avg_volume - LAG(avg_volume) OVER (ORDER BY year)) / LAG(avg_volume) OVER (ORDER BY year),
        2
    ) AS pct_change
FROM yearly
ORDER BY year;
