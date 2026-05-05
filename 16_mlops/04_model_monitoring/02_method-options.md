# Methods, Options, and Properties: Model Monitoring

This document outlines common techniques and libraries used for detecting drift.

## 1. Statistical Tests for Drift

When we don't have immediate access to ground truth labels, we monitor the input features (Data Drift) to guess if the model is failing.

### Kolmogorov-Smirnov (K-S) Test
- **Purpose:** Compares the cumulative distributions of two numerical datasets (e.g., Training Data Feature X vs. Live Data Feature X).
- **How it works:** It measures the maximum distance between the cumulative distribution functions of the two samples.
- **Python Implementation:** `scipy.stats.ks_2samp(data1, data2)`
- **Interpretation:** If the p-value is small (typically < 0.05), you reject the null hypothesis that the two samples come from the same distribution -> Drift detected.

### Population Stability Index (PSI)
- **Purpose:** Widely used in finance to measure how much a population's distribution has shifted over time.
- **How it works:** Bins the data into categories (e.g., deciles) and compares the percentage of data falling into each bin between the old and new datasets.
- **Interpretation:** 
  - PSI < 0.1: No significant change.
  - 0.1 <= PSI < 0.2: Minor shift, monitor closely.
  - PSI >= 0.2: Significant shift, action required.

### Chi-Squared Test
- **Purpose:** Used for categorical variables.
- **How it works:** Compares the observed frequencies of categories in live data to the expected frequencies based on training data.
- **Python Implementation:** `scipy.stats.chisquare()`

---

## 2. Dedicated Monitoring Libraries

Instead of writing statistical tests from scratch, several Python libraries are built specifically for ML monitoring.

### Evidently
Evidently is an open-source library that generates interactive reports (in HTML or Jupyter Notebooks) comparing datasets.

- **`DataDriftPreset`**: Automatically selects the right statistical test (K-S for numerical, Chi-Squared for categorical) for every feature and generates a visual dashboard.
- **`TargetDriftPreset`**: Analyzes how the model's predictions have shifted over time.
- **`DataQualityPreset`**: Checks for missing values, constant features, and extreme outliers.

### Alibi Detect
An open-source library focused on outlier, adversarial, and drift detection. Good for more complex data types like images or text using advanced techniques (e.g., Maximum Mean Discrepancy).

### Deepchecks
Another robust open-source tool for testing ML models and data, offering suites for data integrity, train-test validation, and model evaluation.

## Typical Workflow
1. At training time, save the training dataset (or a representative sample) as a baseline.
2. During inference, log incoming feature requests to a database or file.
3. Every 24 hours, run a script that pulls the last 24 hours of live data.
4. Use `scipy` or `evidently` to compare the live data against the baseline.
5. If drift exceeds a threshold (e.g., p-value < 0.05 on key features), send an email/Slack alert to the Data Science team.
