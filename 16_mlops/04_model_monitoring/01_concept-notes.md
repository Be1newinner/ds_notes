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
