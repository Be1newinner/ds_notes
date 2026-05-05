# Practice: Model Monitoring

These exercises will test your ability to understand and detect drift.

## Exercise 1: Manual Data Drift Detection
1. Create a "Baseline" dataset: Generate an array of 1000 random numbers drawn from a normal distribution with mean=50 and std=10.
2. Create a "Live Week 1" dataset: Generate an array of 1000 numbers drawn from a normal distribution with mean=50 and std=10.
3. Create a "Live Week 2" dataset: Generate an array of 1000 numbers drawn from a normal distribution with mean=55 and std=12 (Simulating a shift).
4. Write a script using `scipy.stats.ks_2samp`.
5. Compare Baseline vs. Week 1. Print the p-value.
6. Compare Baseline vs. Week 2. Print the p-value.
7. Write an `if` statement that prints "ALERT: Drift Detected!" if the p-value is less than 0.05.

## Exercise 2: Understanding Concept Drift
1. Imagine a dataset predicting if a customer will buy a winter coat. Features: `Temperature_Celsius` and `Month`.
2. Model trained on data from 2018-2022. It learns: If `Temperature < 5`, `Buy = True`.
3. In 2024, a massive marketing campaign makes the coats a fashion trend, and people buy them even when it is 15 degrees Celsius.
4. Answer the following:
   - Does the API receive different inputs (Data Drift)?
   - Has the accuracy of the old model dropped?
   - Is this Data Drift or Concept Drift?

## Exercise 3: Planning a Monitoring Strategy
Imagine you are deploying a model that predicts "Probability of Loan Default" based on age, income, and credit score. Ground truth (whether they default) is not known for 12 months.
1. Write a short paragraph explaining how you will monitor this model in the first 11 months before you have any ground truth.
2. What specific features will you run statistical tests on?
