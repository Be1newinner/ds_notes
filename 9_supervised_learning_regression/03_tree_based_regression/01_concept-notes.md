# Tree-Based Regression

## Learning Objective
Students will learn how Decision Trees, Random Forests, and Gradient Boosting algorithms (like XGBoost) partition data to make highly accurate non-linear predictions without requiring extensive feature scaling.

## What Is This Topic?
Tree-based models make predictions by asking a series of "Yes/No" questions about the data. A Decision Tree is a single flowchart. A Random Forest is a crowd of trees voting on the answer. Gradient Boosting is a team of trees where each new tree tries to fix the mistakes of the previous one.

## Why This Topic Matters
In industry, ensemble tree-based models (Random Forest, XGBoost, LightGBM) are the absolute "go-to" algorithms for tabular data (data in spreadsheets or SQL tables). They often win Kaggle competitions and provide the best out-of-the-box accuracy with minimal data cleaning.

## Core Intuition
- **Decision Tree**: "Is the house size > 2000 sq ft? Yes. Is it located in NY? No." At the end of the branches, it predicts the average price of all training houses that fit those exact criteria.
- **Random Forest**: Builds 100 different Decision Trees on random subsets of the data and random subsets of features. The final prediction is the average of all 100 trees' predictions. This stops the model from overfitting.
- **Gradient Boosting (XGBoost)**: Builds Tree 1. Tree 1 gets some predictions wrong. Builds Tree 2 specifically to predict the *errors* of Tree 1. Tree 3 predicts the errors of Tree 1+2. They work together sequentially.

## Key Concepts
- **Splitting Criterion (Variance Reduction/MSE)**: How a tree decides where to split a node. It tries to group data so that points in the resulting groups have similar target values (low variance).
- **Max Depth**: How many layers of questions the tree can ask. Deep trees overfit easily.
- **Bagging (Bootstrap Aggregating)**: The technique used by Random Forest (building multiple independent models in parallel).
- **Boosting**: Building models sequentially to correct past errors.
- **Feature Importance**: Trees can naturally calculate which features were used the most to make splits, making the model highly interpretable at a macro level.

## Advantages
- Does **not** require feature scaling (standardization/normalization).
- Handles non-linear relationships naturally.
- Handles outliers and missing values well (especially XGBoost).
- Provides Feature Importance scores.

## Limitations
- **Decision Trees**: Highly prone to overfitting.
- **Random Forests**: Can be slow to predict if the forest is huge.
- **Gradient Boosting**: Prone to overfitting if not tuned carefully; harder to tune than Random Forests.
- Tree models **cannot extrapolate**. If you train a tree on houses priced between $100k-$500k, it can *never* predict a price of $600k for new data, unlike linear regression.

## Code References
- `code/example-01-decision-tree.py` — basic tree intuition
- `code/example-02-random-forest.py` — the workhorse model
- `code/example-03-xgboost-real-world.py` — the industry standard
