-- Task 1.3: Analyse temperature around holidays
-- Compares temperature patterns for 2015, 2016, and 2017 during
-- New Year's Day and Labor Day.
--
-- NOTE ON APPROACH: the 'holiday' column in this dataset only flags a
-- single row per holiday (typically the midnight hour), and is
-- inconsistently applied -- e.g. 2015's New Year's Day isn't flagged at
-- all, and 2017's flag lands on Jan 2 because Jan 1 fell on a Sunday
-- and the holiday was observed the next day. To get complete and
-- correct temperature data, we match on the actual calendar date
-- instead of the holiday flag. Labor Day is the first Monday in
-- September, so its calendar date changes each year:
--   2015-09-07, 2016-09-05, 2017-09-04

-- 1. New Year's Day temperature summary (Jan 1, 2015/2016/2017)
SELECT
    strftime('%Y', date_time) AS year,
    COUNT(*) AS hours_recorded,
    ROUND(AVG(temp - 273.15), 2) AS avg_temp_c,
    ROUND(MIN(temp - 273.15), 2) AS min_temp_c,
    ROUND(MAX(temp - 273.15), 2) AS max_temp_c
FROM traffic
WHERE strftime('%m-%d', date_time) = '01-01'
  AND strftime('%Y', date_time) IN ('2015', '2016', '2017')
GROUP BY year
ORDER BY year;

-- 2. Labor Day temperature summary (first Monday of September, per year)
SELECT
    strftime('%Y', date_time) AS year,
    COUNT(*) AS hours_recorded,
    ROUND(AVG(temp - 273.15), 2) AS avg_temp_c,
    ROUND(MIN(temp - 273.15), 2) AS min_temp_c,
    ROUND(MAX(temp - 273.15), 2) AS max_temp_c
FROM traffic
WHERE (strftime('%Y-%m-%d', date_time) = '2015-09-07')
   OR (strftime('%Y-%m-%d', date_time) = '2016-09-05')
   OR (strftime('%Y-%m-%d', date_time) = '2017-09-04')
GROUP BY year
ORDER BY year;

-- 3. Combined view: average traffic volume on these same holiday dates,
--    to see whether temperature differences line up with any traffic difference
SELECT
    CASE
        WHEN strftime('%m-%d', date_time) = '01-01' THEN 'New Years Day'
        ELSE 'Labor Day'
    END AS holiday_name,
    strftime('%Y', date_time) AS year,
    ROUND(AVG(temp - 273.15), 2) AS avg_temp_c,
    ROUND(AVG(traffic_volume), 1) AS avg_traffic_volume
FROM traffic
WHERE strftime('%Y-%m-%d', date_time) IN (
    '2015-01-01', '2016-01-01', '2017-01-01',
    '2015-09-07', '2016-09-05', '2017-09-04'
)
GROUP BY holiday_name, year
ORDER BY holiday_name, year;
