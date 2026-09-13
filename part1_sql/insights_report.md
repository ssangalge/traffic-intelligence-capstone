# Part 1 – Data Analytics Insights Report

**Project:** Smart City Traffic Intelligence
**Dataset:** Metro Interstate Traffic Volume (westbound I-94, Minneapolis–St Paul), Oct 2012 – Sep 2018, 48,204 hourly records

This report summarises the key findings from the SQL, statistical, probability, and Power BI analysis in Part 1, and their implications for the smart-city mobility team.

---

## Task 1.2 – Annual Traffic Trends

**Data quality note:** the dataset contains 5,445 duplicate timestamp groups (the same hour recorded more than once, up to 6 times). A raw yearly SUM is misleading, since years with more recorded rows show inflated totals regardless of actual traffic levels. All figures below use de-duplicated average traffic volume per year for a fair comparison.

**Observation 1:** Average hourly traffic volume grew overall from 3,226.7 vehicles/hour in 2012 to a peak of 3,376.6 in 2017. The sharpest single-year change was **+5.73% from 2016 to 2017**, the largest swing in the six-year period.

**Observation 2:** Traffic dipped across 2014–2016 (-1.19%, -0.37%, -1.98% year-over-year) before rebounding sharply in 2017. This non-linear pattern suggests demand wasn't driven by simple steady growth — it's worth investigating whether external factors (construction, economic conditions, or gaps in data coverage during those years) explain the dip.

**Implication for the mobility team:** capacity planning should account for the possibility of sudden demand rebounds (like the 2017 jump) rather than assuming smooth year-over-year growth.

---

## Task 1.3 – Temperature Around Holidays

**Data note** : New Year's Day has no recorded data for 2015 in this dataset — a genuine gap, not a processing error. 

**Observation 1:** Comparing 2016 to 2017, average temperature rose from -6.12°C to -3.06°C. 

**Observation 2:** Labor Day temperatures declined steadily each year: 22.17°C (2015) → 21.6°C (2016) → 17.48°C (2017). Notably, temperature alone doesn't cleanly predict Labor Day traffic volume — the coldest year (2017) had the highest average traffic, suggesting other factors dominate holiday travel behavior.

---

## Task 2 – Descriptive Statistics and Correlation

**Observation 1:** Mean traffic volume is 3,290.65 vehicles/hour, while the median is higher at 3,427.0 — this gap indicates a left-skewed distribution, likely driven by a long tail of low-traffic overnight hours pulling the mean below the median.

**Observation 2:** Standard deviation (1,984.77) is large relative to the mean, confirming substantial hour-to-hour variability — unsurprising given the mix of rush-hour peaks and overnight lows in a single "hourly volume" metric.

**Correlation:** The Pearson correlation between temperature and traffic volume is r = 0.1369 — positive in direction but weak in strength. This means temperature alone is a poor predictor of traffic volume. Even if this correlation were stronger, it would not establish causation: both temperature and traffic patterns could independently be driven by a third factor such as season (which affects both typical weather and typical commuting/holiday behavior).

---

## Task 3 – Probability and Congestion Analysis

**Data note:** this analysis uses the full raw dataset (not de-duplicated), since duplicate timestamps here represent legitimate simultaneous weather conditions (e.g. "Rain" and "Mist" reported for the same hour with identical traffic volume), not data errors.

**Observation 1:** Congestion occurs in 14.73% of hours. Congestion and clear weather appear approximately independent (P(Congestion AND Clear) = 0.0366 vs. an expected 0.0409 under independence, a difference of only 0.0043) — congestion is more strongly tied to time-of-day/rush hour patterns than to weather conditions.

**Observation 2:** The odds ratio of congestion in clear vs. cloudy weather is 0.7354 — meaning the odds of congestion are actually about 26% lower in clear weather than cloudy weather, a mildly counterintuitive result. Combined with the near-independence finding, weather condition alone is a weak lever for predicting congestion in this corridor; timing-based factors likely dominate.

---

## Task 4 – Power BI Dashboard Findings

**A. Daily traffic trends (2015-2017):** Line chart split by year shows [describe the pattern you observed — e.g. visible weekly cyclicality with dips on certain days, and note whether one year's line sits consistently above/below the others].

- **B. Hourly traffic patterns (2017):** Column chart confirms the expected two-peak commuter pattern — higher average traffic during morning and evening rush hours, with clear overnight lows.

- **C. Weather impact:** Clouds shows the highest average traffic volume (3,618.4 vehicles/hour, n=15,164 — a large, reliable sample). Squall shows the lowest (2,061.8 vehicles/hour), but this is based on only 4 recorded instances and should be treated cautiously rather than as a robust finding. The difference between highest and lowest is 1,556.6 vehicles/hour. Excluding the low-sample Squall category, Fog (2,703.7 vehicles/hour, n=912) is a more statistically meaningful "lowest" comparison point.

- **D. Temperature vs. traffic scatter plot:** Points form a wide, scattered cloud across a temperature range of roughly -28°C to +36°C, with no clean visible trend line — consistent with the weak correlation (r=0.14) found in Task 2. Higher-traffic points are broadly spread across the moderate-to-warm temperature range rather than concentrated in a narrow band, reinforcing that temperature alone is a poor predictor of traffic volume in this corridor.

**4.3 KPI Cards and Filters**

Three KPI cards were added: Total Hours Analysed (48,204), Average Traffic Volume (3,260 vehicles/hour), and Average Temperature (8.06°C) — the temperature figure aligns well with Minneapolis-St Paul's known climate averages, giving confidence in the data pipeline. Three interactive slicers were added: an Hour range slider (0-23), a Weather condition checklist, and a Traffic Category checklist (Low/Medium/High), allowing stakeholders to filter the whole dashboard down to specific conditions of interest (e.g. isolating rush-hour periods, or comparing clear vs. stormy weather).


---

## Summary of Key Implications

This analysis suggests traffic volume on this I-94 corridor is driven far more by **time-based patterns** (hour of day, day of week, and year-over-year demand shifts) than by weather or temperature, which showed only weak statistical relationships (r=0.14 correlation; near-independence between congestion and clear weather). The clearest actionable pattern is the two-peak commuter rush-hour structure, which should anchor any capacity planning or congestion-mitigation strategy. The 2016-2017 demand rebound (+5.73%) and the corridor's non-linear year-over-year trend suggest the mobility team should build monitoring for sudden demand shifts rather than assuming smooth, predictable growth. Data quality findings — duplicate timestamps representing simultaneous weather conditions, and inconsistent holiday flagging — are important caveats for any downstream Part 2/Part 3 modelling work, and should be explicitly handled (not silently dropped) to avoid biasing future predictive models.

