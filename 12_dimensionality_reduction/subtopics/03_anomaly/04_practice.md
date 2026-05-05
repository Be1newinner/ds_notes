# Anomaly Detection Practice Tasks

## Task 1: Contamination Tuning
Load the Boston Housing dataset or generate a synthetic dataset. Fit an Isolation Forest with `contamination=0.01` and another with `contamination=0.10`. How many rows are flagged as anomalies in each case?

## Task 2: LOF vs Isolation Forest
Scikit-learn also has `LocalOutlierFactor`. Apply both `IsolationForest` and `LocalOutlierFactor` to a synthetic dataset containing clusters of different densities. Compare which points each algorithm flags.

## Task 3: Salary Outliers
Create a simple dataframe of employee salaries where 98% earn between $40k and $80k, and 2% earn over $500k. Use Isolation Forest to flag the CEOs/Executives.
