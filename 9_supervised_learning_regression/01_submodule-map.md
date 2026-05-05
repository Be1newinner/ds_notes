# Submodule Map: Supervised Learning Regression

This document outlines the teaching map for Module 9: Supervised Learning Regression.

## 1. Linear Models (`01_linear_models/`)
- **Topics**: Simple Linear Regression, Multiple Linear Regression, Ridge, Lasso, ElasticNet.
- **Why it is taught**: It is the foundation of predictive modeling. It introduces the concept of weights, bias, and the loss function (MSE). Regularization helps handle multicollinearity and overfitting.
- **Nature**: Theory-heavy (cost function, gradient descent intuition) and Code-heavy.
- **Needs**: Visual explanation of the line of best fit and regularization penalties. Business examples like predicting sales based on marketing spend.

## 2. Non-linear Regression (`02_non_linear_regression/`)
- **Topics**: Polynomial Regression, Support Vector Regression (SVR).
- **Why it is taught**: Real-world relationships are rarely strictly linear. Polynomial regression introduces feature transformations, and SVR introduces margins to regression.
- **Nature**: Moderate theory, moderate code.
- **Needs**: Visual explanations showing curves fitting data better than straight lines. Examples like predicting non-linear growth curves or sensor degradation.

## 3. Tree-based Regression (`03_tree_based_regression/`)
- **Topics**: Decision Tree Regressor, Random Forest Regressor, Gradient Boosting (XGBoost, LightGBM).
- **Why it is taught**: Tree-based models often provide the best performance on tabular data without extensive feature engineering. They capture complex non-linear relationships naturally.
- **Nature**: Moderate theory (splitting criteria for regression like variance reduction), Code-heavy.
- **Needs**: Visual explanation of how trees split numerical space. Business examples like house price prediction where many categorical and numerical features interact.

## 4. Evaluation and Tuning (`04_evaluation_and_tuning/`)
- **Topics**: R-squared, Adjusted R-squared, RMSE, MAE, Grid Search, Random Search for Regression.
- **Why it is taught**: To properly assess model performance in a business context and squeeze out extra accuracy.
- **Nature**: Highly practical and Code-heavy.
- **Needs**: Business examples explaining what an RMSE of $500 actually means to stakeholders compared to an R2 of 0.85.

## Recommended Order of Teaching
1. Start with the intuition of a single variable (Simple Linear).
2. Expand to Multiple Linear Regression.
3. Show how models overfit and introduce Regularization (Lasso/Ridge).
4. Introduce Non-linear methods (Polynomial, SVR) when linear fails.
5. Move to Tree-based Ensembles (Random Forest, XGBoost) as the ultimate toolkit.
6. Conclude by comparing all models and tuning the best one.
