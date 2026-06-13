# Model Monitoring and Drift

## Learning Objective
Students should understand that machine learning models degrade over time after deployment and learn how to identify data drift and concept drift.

## What Is This Topic?
Model monitoring is the continuous observation of a deployed machine learning model's performance, data inputs, and predictions. Unlike software engineering, where code works until it is explicitly broken, ML models break silently because the world changes. 

## Why This Topic Matters
A fraud detection model trained in 2019 would perform terribly in 2021 because COVID-19 drastically changed people's purchasing behavior. If a data science team deploys a model and never looks at it again, it will eventually cost the business money due to inaccurate predictions.

## Core Intuition
Imagine a model trained to predict house prices based on historical data. If inflation suddenly spikes or a new tech company opens an office in a quiet neighborhood, the old rules the model learned no longer apply. The model doesn't know the world changed; it just keeps guessing based on the old rules. We need an alarm system (monitoring) to tell us when the rules have changed.

## Key Concepts
- **Data Drift (Covariate Shift):** The distribution of the *input features* changes over time. The model is receiving data that looks different from what it was trained on. 
  - *Example:* A spam filter trained mostly on emails from English speakers suddenly starts receiving thousands of emails in Spanish.
- **Concept Drift:** The relationship between the input features and the *target variable* changes. The definition of what we are trying to predict has changed.
  - *Example:* Before 2020, buying 50 rolls of toilet paper at once might look like "Fraud/Reseller" behavior. In March 2020, it became "Normal" behavior. The inputs are the same, but the true label changed.
- **Ground Truth Delay:** The time it takes to find out if the model's prediction was actually correct.
  - *Example:* If predicting loan default, you won't have ground truth until the customer actually defaults 6 months later.

## Step-by-Step Explanation (Monitoring Workflow)
1. **Baseline Creation:** When deploying the model, save the statistical distribution of the training data (mean, variance, min, max, etc.).
2. **Logging:** In the API, log every incoming request (features) and every outgoing response (prediction) to a database.
3. **Statistical Comparison:** Set up a scheduled job (e.g., daily) that compares the distribution of the incoming live data to the baseline training data.
4. **Alerting:** If the distributions diverge significantly (using tests like Kolmogorov-Smirnov or Population Stability Index), trigger an alert.
5. **Retraining:** If drift is detected and performance drops, gather the newest data, label it, and retrain the model.

## Output / Result Interpretation
Monitoring tools usually output dashboards showing metrics over time (e.g., a line graph showing the daily average age of customers hitting the API). If the line suddenly jumps, it warrants investigation.

## Real-World Uses
- Monitoring an image recognition system in a factory. If the lighting in the factory changes (Data Drift), the model's accuracy will drop.
- E-commerce product recommendation. If a new viral trend starts on TikTok, user behavior changes rapidly (Concept Drift), requiring fast retraining.

## Advantages
- Prevents silent failures that cost businesses money.
- Builds trust in ML systems among business stakeholders.
- Helps determine *when* to retrain a model (retraining too often is expensive; retraining too rarely is inaccurate).

## Limitations
- Ground truth is often delayed, making it hard to calculate live accuracy. We often have to rely solely on detecting Data Drift instead.
- Requires building extra infrastructure (databases, dashboards) alongside the API.

## Common Mistakes
- **Assuming High Accuracy Lasts Forever:** The most dangerous assumption in ML.
- **Retraining on Bad Data:** If a sensor breaks and starts sending `-999` instead of real data (Data Quality issue), you should fix the sensor, not retrain the model to understand `-999`.

## Code References
- `code/example-01-data-drift.py` — Simulating and detecting Data Drift visually.
- `code/example-02-concept-drift.py` — Simulating Concept Drift and observing model performance decay.


---

## Methods, Options, and Properties: Model Monitoring

This document outlines common techniques and libraries used for detecting drift.

### 1. Statistical Tests for Drift

When we don't have immediate access to ground truth labels, we monitor the input features (Data Drift) to guess if the model is failing.

#### Kolmogorov-Smirnov (K-S) Test
- **Purpose:** Compares the cumulative distributions of two numerical datasets (e.g., Training Data Feature X vs. Live Data Feature X).
- **How it works:** It measures the maximum distance between the cumulative distribution functions of the two samples.
- **Python Implementation:** `scipy.stats.ks_2samp(data1, data2)`
- **Interpretation:** If the p-value is small (typically < 0.05), you reject the null hypothesis that the two samples come from the same distribution -> Drift detected.

#### Population Stability Index (PSI)
- **Purpose:** Widely used in finance to measure how much a population's distribution has shifted over time.
- **How it works:** Bins the data into categories (e.g., deciles) and compares the percentage of data falling into each bin between the old and new datasets.
- **Interpretation:** 
  - PSI < 0.1: No significant change.
  - 0.1 <= PSI < 0.2: Minor shift, monitor closely.
  - PSI >= 0.2: Significant shift, action required.

#### Chi-Squared Test
- **Purpose:** Used for categorical variables.
- **How it works:** Compares the observed frequencies of categories in live data to the expected frequencies based on training data.
- **Python Implementation:** `scipy.stats.chisquare()`

---

### 2. Dedicated Monitoring Libraries

Instead of writing statistical tests from scratch, several Python libraries are built specifically for ML monitoring.

#### Evidently
Evidently is an open-source library that generates interactive reports (in HTML or Jupyter Notebooks) comparing datasets.

- **`DataDriftPreset`**: Automatically selects the right statistical test (K-S for numerical, Chi-Squared for categorical) for every feature and generates a visual dashboard.
- **`TargetDriftPreset`**: Analyzes how the model's predictions have shifted over time.
- **`DataQualityPreset`**: Checks for missing values, constant features, and extreme outliers.

#### Alibi Detect
An open-source library focused on outlier, adversarial, and drift detection. Good for more complex data types like images or text using advanced techniques (e.g., Maximum Mean Discrepancy).

#### Deepchecks
Another robust open-source tool for testing ML models and data, offering suites for data integrity, train-test validation, and model evaluation.

### Typical Workflow
1. At training time, save the training dataset (or a representative sample) as a baseline.
2. During inference, log incoming feature requests to a database or file.
3. Every 24 hours, run a script that pulls the last 24 hours of live data.
4. Use `scipy` or `evidently` to compare the live data against the baseline.
5. If drift exceeds a threshold (e.g., p-value < 0.05 on key features), send an email/Slack alert to the Data Science team.

---

## Examples: Model Monitoring

This document outlines the practical examples provided for learning model monitoring.

### Code References

- `code/example-01-data-drift.py` — **Detecting Data Drift**: Simulates a baseline training dataset and a "live" dataset where the inputs have slowly changed. Uses `scipy.stats` to perform a Kolmogorov-Smirnov test to detect if the drift is statistically significant.
- `code/example-02-concept-drift.py` — **Simulating Concept Drift**: Trains a simple model. Simulates a scenario where the relationship between inputs and outputs changes (the "rules" change), but the inputs look identical. Shows how the model's accuracy slowly decays over time.

### How to use these examples

Run the Python scripts from your terminal. 
- Example 01 demonstrates the mathematical tests running behind the scenes of popular monitoring dashboards.
- Example 02 is purely illustrative to help visualize *why* models fail silently.

---

## Practice: Model Monitoring

These exercises will test your ability to understand and detect drift.

### Exercise 1: Manual Data Drift Detection
1. Create a "Baseline" dataset: Generate an array of 1000 random numbers drawn from a normal distribution with mean=50 and std=10.
2. Create a "Live Week 1" dataset: Generate an array of 1000 numbers drawn from a normal distribution with mean=50 and std=10.
3. Create a "Live Week 2" dataset: Generate an array of 1000 numbers drawn from a normal distribution with mean=55 and std=12 (Simulating a shift).
4. Write a script using `scipy.stats.ks_2samp`.
5. Compare Baseline vs. Week 1. Print the p-value.
6. Compare Baseline vs. Week 2. Print the p-value.
7. Write an `if` statement that prints "ALERT: Drift Detected!" if the p-value is less than 0.05.

### Exercise 2: Understanding Concept Drift
1. Imagine a dataset predicting if a customer will buy a winter coat. Features: `Temperature_Celsius` and `Month`.
2. Model trained on data from 2018-2022. It learns: If `Temperature < 5`, `Buy = True`.
3. In 2024, a massive marketing campaign makes the coats a fashion trend, and people buy them even when it is 15 degrees Celsius.
4. Answer the following:
   - Does the API receive different inputs (Data Drift)?
   - Has the accuracy of the old model dropped?
   - Is this Data Drift or Concept Drift?

### Exercise 3: Planning a Monitoring Strategy
Imagine you are deploying a model that predicts "Probability of Loan Default" based on age, income, and credit score. Ground truth (whether they default) is not known for 12 months.
1. Write a short paragraph explaining how you will monitor this model in the first 11 months before you have any ground truth.
2. What specific features will you run statistical tests on?

---

## Interview Questions: Model Monitoring

### Beginner Questions
1. **What is model decay (or model degradation)?**
   - *Answer concept:* It is the gradual decrease in a machine learning model's predictive performance over time after it has been deployed, usually caused by changes in the real world.
2. **What is Data Drift?**
   - *Answer concept:* When the statistical distribution of the input features sent to the model changes over time (e.g., users get older, incomes rise, sensor calibration changes).
3. **What is Concept Drift?**
   - *Answer concept:* When the relationship between the input features and the target variable changes. The definition of what we are trying to predict has shifted (e.g., what was considered "expensive" 10 years ago is now considered "cheap").

### Conceptual Questions
4. **Why is model monitoring fundamentally different from traditional software monitoring?**
   - *Answer concept:* Traditional software monitoring checks if the server is running, CPU usage is normal, and HTTP 500 errors are low. If the code doesn't crash, it's working. ML models fail *silently*; the server returns HTTP 200, the code runs perfectly, but the prediction itself is wrong because the world changed.
5. **If a deployed classification model's accuracy drops from 95% to 70% over 6 months, how do you decide whether to retrain it?**
   - *Answer concept:* First, investigate the cause. Is it bad data (a broken pipeline sending null values)? If so, fix the pipeline. If it is genuine data/concept drift, gather the newest labeled data and retrain. Do not just retrain blindly.

### Practical Questions
6. **You have deployed a model that predicts whether a customer will default on a 5-year mortgage. How do you monitor this model, given that you won't know if they default for up to 5 years?**
   - *Answer concept:* Since ground truth is delayed by 5 years, we cannot calculate live accuracy. We must rely entirely on monitoring *Data Drift*. We establish a baseline of the customer profiles at training time, and if the incoming live customers' demographics or financial stats diverge significantly (using K-S test or PSI), we trigger a review.
7. **Explain the Kolmogorov-Smirnov (K-S) test in simple terms and how it's used in MLOps.**
   - *Answer concept:* The K-S test compares two datasets to see if they come from the same distribution. In MLOps, we use it to compare the feature values of the training data against the live data hitting the API. If the p-value is low, it indicates the live data looks fundamentally different, alerting us to Data Drift.

---

## Python Code Examples

### `example-01-data-drift.py`

```python
import numpy as np
from scipy import stats

def check_drift(baseline_data, live_data, feature_name):
    """
    Uses the Kolmogorov-Smirnov test to detect data drift.
    """
    # Perform the K-S test
    statistic, p_value = stats.ks_2samp(baseline_data, live_data)
    
    print(f"\n--- Testing Feature: {feature_name} ---")
    print(f"K-S Statistic: {statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    # Typical threshold is 0.05
    if p_value < 0.05:
        print("🚨 ALERT: Data Drift Detected! The distributions are significantly different.")
    else:
        print("✅ No significant drift detected.")

# 1. Simulate Baseline (Training) Data
# E.g., The age of our users when we trained the model
print("Gathering Baseline Data...")
np.random.seed(42)
baseline_age = np.random.normal(loc=35, scale=5, size=1000)

# 2. Simulate Live Data (Week 1)
# The user base looks basically the same
live_age_week1 = np.random.normal(loc=35.2, scale=5.1, size=1000)
check_drift(baseline_age, live_age_week1, "User Age (Week 1)")

# 3. Simulate Live Data (Month 6)
# Our marketing campaign brought in a much younger demographic
live_age_month6 = np.random.normal(loc=28, scale=6, size=1000)
check_drift(baseline_age, live_age_month6, "User Age (Month 6)")

# What does this mean?
# Our model was trained to predict behavior for 35-year-olds.
# It is now mostly seeing 28-year-olds. It might not be accurate anymore.
```

### `example-02-concept-drift.py`

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# SIMULATING CONCEPT DRIFT
# We predict if a user clicks an ad based on 'Time_Spent_on_Site'

# 1. Training Phase (Year 2020)
# Rule: If Time > 5 mins, Click = True
X_train = np.random.uniform(1, 10, 1000).reshape(-1, 1)
y_train = (X_train > 5).astype(int).ravel()

model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Initial Training Accuracy (2020): {accuracy_score(y_train, model.predict(X_train)):.2f}")

# 2. Live Deployment (Year 2021) - No Drift
# The rule remains the same.
X_live_2021 = np.random.uniform(1, 10, 500).reshape(-1, 1)
y_true_2021 = (X_live_2021 > 5).astype(int).ravel()
print(f"Live Accuracy (2021): {accuracy_score(y_true_2021, model.predict(X_live_2021)):.2f}")

# 3. Live Deployment (Year 2023) - CONCEPT DRIFT
# A competitor launched, and users are less patient.
# NEW Rule: If Time > 8 mins, Click = True. (The old rule was > 5)
# Note: The input data (X) still looks exactly the same (Uniform 1-10). Data drift tests would pass!
X_live_2023 = np.random.uniform(1, 10, 500).reshape(-1, 1)
y_true_2023 = (X_live_2023 > 8).astype(int).ravel()

# The model still predicts based on the 2020 rule (> 5)
predictions_2023 = model.predict(X_live_2023)
print(f"Live Accuracy (2023) - After Concept Drift: {accuracy_score(y_true_2023, predictions_2023):.2f}")

print("\nConclusion: The inputs didn't change, but the world changed. The model's accuracy plummeted silently.")
```
