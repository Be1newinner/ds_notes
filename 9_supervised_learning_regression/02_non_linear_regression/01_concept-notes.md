# Non-Linear Regression

## Learning Objective
Students will learn how to model relationships that are not straight lines using Polynomial Regression and Support Vector Regression (SVR).

## What Is This Topic?
While simple linear regression fits a straight line, non-linear regression models can bend and curve to capture more complex patterns in the data (like exponential growth, cyclical patterns, or diminishing returns).

## Why This Topic Matters
In reality, very few relationships are perfectly linear. A company's revenue might grow exponentially at first and then plateau. A car's value drops rapidly in the first year and then depreciates slower. Non-linear models capture these real-world phenomena accurately.

## Core Intuition
- **Polynomial Regression**: Instead of just using $x$ to predict $y$, we create new features like $x^2$ and $x^3$. The model is still "linear" in the mathematical sense, but it draws a curve when plotted against the original $x$.
- **Support Vector Regression (SVR)**: Imagine a straight tube (a margin) around your data points. SVR tries to fit as many data points as possible *inside* this tube, ignoring errors smaller than a certain threshold. By using a "kernel trick," it can warp the space to draw complex non-linear tubes.

## Key Concepts
- **Degree (Polynomial)**: The highest power added. Degree 2 makes a U-shape (parabola). Degree 3 makes an S-shape.
- **Kernel Trick (SVR)**: A mathematical shortcut that transforms data into a higher dimension where a non-linear relationship looks linear.
- **Epsilon ($\epsilon$) Margin (SVR)**: The width of the tube where no penalty is given for errors.

## Important Parameters / Options / Settings
- `degree` (PolynomialFeatures): Controls flexibility. High degree = high chance of overfitting.
- `kernel` (SVR): Usually `'rbf'` (Radial Basis Function) for non-linear relationships.
- `C` (SVR): Regularization parameter. High `C` means strict fitting (risk of overfitting), low `C` means a smoother, more generalized curve.

## Real-World Uses
- Predicting the spread of a viral trend or disease (exponential phase, plateau phase).
- Modeling the relationship between vehicle speed and fuel efficiency (optimal at a certain speed, worse if slower or faster).

## Advantages
- **Polynomial**: Easy to implement using standard linear regression underneath.
- **SVR**: Highly robust to outliers (due to the epsilon margin).

## Limitations
- **Polynomial**: Extremely prone to overfitting if the degree is too high. Extrapolates very poorly outside the training data.
- **SVR**: Computationally expensive on large datasets. Requires careful feature scaling. Hard to interpret coefficients.

## Common Mistakes
- Using a high-degree polynomial (e.g., degree=10) that perfectly hits every training point but fails completely on new data.
- Forgetting to scale data before using SVR (it relies heavily on distance calculations).

## Code References
- `code/example-01-basic-polynomial.py`
- `code/example-02-intermediate-svr.py`
