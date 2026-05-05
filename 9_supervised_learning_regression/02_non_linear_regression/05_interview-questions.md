# Interview Questions: Non-Linear Regression

## Beginner Level
1. **What is the main difference between Linear Regression and Polynomial Regression?**
   - *Expected Answer*: Linear regression fits a straight line. Polynomial regression introduces higher-degree terms (like $x^2$, $x^3$) allowing the model to fit a curve to the data, while still being solved using linear regression techniques.

2. **What does SVR stand for?**
   - *Expected Answer*: Support Vector Regression. It is the regression counterpart to Support Vector Machines (SVM) used for classification.

## Intermediate Level
3. **What is the risk of using a very high degree in Polynomial Regression?**
   - *Expected Answer*: Overfitting. The curve will pass through almost every single training data point, capturing all the noise. As a result, it will perform terribly on new, unseen data (high variance).

4. **Explain the $\epsilon$-tube (epsilon-tube) in SVR.**
   - *Expected Answer*: In SVR, the model tries to fit a tube with a radius of $\epsilon$ around the data points. Any errors that fall *inside* this tube are ignored (penalty is zero). The model only penalizes points that fall outside the tube.

5. **Why is scaling absolutely crucial for SVR but not necessarily for ordinary Linear Regression (without regularization)?**
   - *Expected Answer*: SVR uses a distance metric (especially with the RBF kernel) to calculate the similarity between data points. If features are on different scales, features with larger ranges will dominate the distance calculations, ruining the model's performance. OLS linear regression just calculates slopes and intercepts analytically, so scale doesn't affect the final line's predictive power.

## Advanced / Practical Level
6. **If your SVR model is underfitting, which parameter should you change?**
   - *Expected Answer*: You could increase `C` (which increases the penalty for points outside the margin, forcing the model to fit the training data tighter). You could also decrease `epsilon` to narrow the tube, making the model more sensitive to small errors. Finally, if using RBF, you might adjust `gamma`.

7. **How does the 'kernel trick' work in simple terms?**
   - *Expected Answer*: It mathematically transforms data points into a higher-dimensional space where a complex, non-linear relationship becomes a simple, linear relationship. It does this by calculating the "similarity" between points without actually performing the expensive calculations to map them to the higher dimension.
