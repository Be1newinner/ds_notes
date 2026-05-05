# Interview Questions: Model Monitoring

## Beginner Questions
1. **What is model decay (or model degradation)?**
   - *Answer concept:* It is the gradual decrease in a machine learning model's predictive performance over time after it has been deployed, usually caused by changes in the real world.
2. **What is Data Drift?**
   - *Answer concept:* When the statistical distribution of the input features sent to the model changes over time (e.g., users get older, incomes rise, sensor calibration changes).
3. **What is Concept Drift?**
   - *Answer concept:* When the relationship between the input features and the target variable changes. The definition of what we are trying to predict has shifted (e.g., what was considered "expensive" 10 years ago is now considered "cheap").

## Conceptual Questions
4. **Why is model monitoring fundamentally different from traditional software monitoring?**
   - *Answer concept:* Traditional software monitoring checks if the server is running, CPU usage is normal, and HTTP 500 errors are low. If the code doesn't crash, it's working. ML models fail *silently*; the server returns HTTP 200, the code runs perfectly, but the prediction itself is wrong because the world changed.
5. **If a deployed classification model's accuracy drops from 95% to 70% over 6 months, how do you decide whether to retrain it?**
   - *Answer concept:* First, investigate the cause. Is it bad data (a broken pipeline sending null values)? If so, fix the pipeline. If it is genuine data/concept drift, gather the newest labeled data and retrain. Do not just retrain blindly.

## Practical Questions
6. **You have deployed a model that predicts whether a customer will default on a 5-year mortgage. How do you monitor this model, given that you won't know if they default for up to 5 years?**
   - *Answer concept:* Since ground truth is delayed by 5 years, we cannot calculate live accuracy. We must rely entirely on monitoring *Data Drift*. We establish a baseline of the customer profiles at training time, and if the incoming live customers' demographics or financial stats diverge significantly (using K-S test or PSI), we trigger a review.
7. **Explain the Kolmogorov-Smirnov (K-S) test in simple terms and how it's used in MLOps.**
   - *Answer concept:* The K-S test compares two datasets to see if they come from the same distribution. In MLOps, we use it to compare the feature values of the training data against the live data hitting the API. If the p-value is low, it indicates the live data looks fundamentally different, alerting us to Data Drift.
