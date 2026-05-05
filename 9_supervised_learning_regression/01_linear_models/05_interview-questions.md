# Interview Questions: Linear Models

## Beginner Level
1. **What is Linear Regression?**
   - *Expected Answer*: An algorithm that models the linear relationship between independent variables and a continuous dependent variable by fitting a straight line (or hyperplane) that minimizes the sum of squared errors.

2. **What is the purpose of the intercept/bias term?**
   - *Expected Answer*: It represents the predicted value when all independent variables are exactly zero. Without it, the line would be forced to pass through the origin (0,0), which often leads to a poor fit.

3. **What does a coefficient in Multiple Linear Regression mean?**
   - *Expected Answer*: Assuming no multicollinearity, a coefficient represents the change in the target variable for a one-unit change in that specific feature, while holding all other features constant.

## Intermediate Level
4. **What are the key assumptions of Linear Regression?**
   - *Expected Answer*: 
     1. Linearity: The relationship is linear.
     2. Independence: Residuals are independent.
     3. Homoscedasticity: Constant variance of residuals.
     4. Normality: Residuals are normally distributed.
     5. No multicollinearity: Features are not highly correlated with each other.

5. **Explain L1 and L2 Regularization.**
   - *Expected Answer*: Both add a penalty to the cost function to prevent overfitting by keeping weights small. L1 (Lasso) uses the absolute value of the weights and can push weights to exactly zero (feature selection). L2 (Ridge) uses the squared value of the weights, penalizing large weights heavily but rarely shrinking them to zero.

6. **Why do we need to scale data before using Ridge or Lasso?**
   - *Expected Answer*: The regularization penalty is based on the magnitude of the coefficients. If features are on different scales (e.g., 1-10 vs 1000-100000), features with smaller numerical ranges will need larger coefficients to have an effect, and thus will be unfairly penalized by the regularization term.

## Advanced / Practical Level
7. **If two features in your dataset are perfectly correlated, what happens to your Linear Regression model?**
   - *Expected Answer*: This is perfect multicollinearity. The math breaks down (the matrix becomes non-invertible for OLS). The coefficients become highly unstable and uninterpretable. Ridge regression can help mitigate this by distributing the weights among the correlated features.

8. **You ran a Lasso regression and it dropped a feature that the business domain experts swear is highly predictive. What might have happened?**
   - *Expected Answer*: Lasso arbitrarily selects one feature from a group of highly correlated features and shrinks the others to zero. If the dropped feature was highly correlated with another feature that Lasso kept, it will appear "unimportant" to Lasso, even though it possesses predictive power.

9. **Output Interpretation**: You predict house prices. The coefficient for "number of bathrooms" is 15,000, and for "distance to city center (miles)" is -5,000. Interpret this.
   - *Expected Answer*: For every additional bathroom, the house price is predicted to increase by $15,000, assuming all other factors remain constant. For every extra mile away from the city center, the house price decreases by $5,000.
