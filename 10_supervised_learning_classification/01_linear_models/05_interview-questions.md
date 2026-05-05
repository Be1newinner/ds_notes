# Interview Questions: Logistic Regression

These questions test your practical and theoretical knowledge of Logistic Regression, commonly asked in data science interviews.

## Beginner Questions
1. **What is the difference between Linear Regression and Logistic Regression?**
   *Hint:* Discuss the type of output (continuous vs. discrete probability), the line vs. the S-curve, and the use of the sigmoid function.
2. **What does the output of a Logistic Regression model represent?**
   *Hint:* It represents a probability (from 0 to 1) that the observation belongs to the positive class.
3. **What is the default threshold used in Logistic Regression to make a class prediction?**
   *Hint:* 0.5.

## Conceptual Questions
4. **Why is it called "regression" if it is used for classification?**
   *Hint:* Because under the hood, it fits a linear regression model to the *log-odds* of the probability.
5. **How does the sigmoid function work?**
   *Hint:* It takes any real-valued number and maps it to a value between 0 and 1 using the formula $1 / (1 + e^{-x})$.
6. **Can Logistic Regression handle non-linear decision boundaries?**
   *Hint:* Not natively. It is a linear classifier. However, you can engineer non-linear features (like $x^2$ or $x_1 * x_2$) to allow it to fit non-linear data.

## Practical Questions
7. **Why is feature scaling (like Standardization) important before training a Logistic Regression model in scikit-learn?**
   *Hint:* Scikit-learn applies L2 regularization by default. Regularization penalizes large coefficients. If features are on different scales, the penalty will disproportionately affect features with smaller absolute values.
8. **If you have a highly imbalanced dataset (e.g., 99% class 0, 1% class 1), what parameter in `LogisticRegression` should you change?**
   *Hint:* Set `class_weight='balanced'`.
9. **You train a Logistic Regression model and the solver throws a `ConvergenceWarning`. What does this mean and how do you fix it?**
   *Hint:* It means the optimization algorithm didn't find the minimum of the cost function in the allotted steps. Fix it by increasing `max_iter` or scaling the data.

## Output Interpretation
10. **If the coefficient for the feature "Age" is -0.5, what does that mean?**
    *Hint:* It means that as "Age" increases, the log-odds of the positive class decrease. In plain English, older people are less likely to belong to the positive class.
