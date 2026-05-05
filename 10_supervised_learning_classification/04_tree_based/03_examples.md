# Examples: Tree-Based Methods

Here is a breakdown of the Python examples provided in the `code/` directory.

## 1. Single Decision Tree (`example-01-decision-tree.py`)
- **Goal:** Understand how a single tree easily overfits and how to visualize its decisions.
- **Dataset:** Synthetic classification dataset.
- **Key Concepts Shown:** 
  - Training an unconstrained `DecisionTreeClassifier`.
  - Seeing the 100% training accuracy vs. poor testing accuracy (classic overfitting).
  - Using `max_depth` to constrain the tree and fix the overfitting.
- **Takeaway:** Never use a single, unconstrained Decision Tree in production.

## 2. Random Forest (`example-02-random-forest.py`)
- **Goal:** Show how ensembling fixes the problems of a single tree.
- **Dataset:** The Wine quality dataset.
- **Key Concepts Shown:** 
  - Training a `RandomForestClassifier`.
  - Understanding the `n_estimators` and `n_jobs` parameters.
- **Takeaway:** Random Forest is robust, powerful, and requires almost no tuning to get a "good enough" baseline result.

## 3. Gradient Boosting (`example-03-gradient-boosting.py`)
- **Goal:** Introduce sequential boosting and the concept of learning rates.
- **Dataset:** The Wine quality dataset (same as above, for comparison).
- **Key Concepts Shown:** 
  - Training a `GradientBoostingClassifier`.
  - Showing how `learning_rate` and `n_estimators` interact.
- **Takeaway:** Boosting can often squeeze out better performance than Random Forest, but is more prone to overfitting if not tuned properly.

## 4. Feature Importance (`example-04-feature-importance.py`)
- **Goal:** Learn how to extract business insights from tree-based models.
- **Dataset:** Simulated Employee Attrition (Churn) dataset.
- **Key Concepts Shown:** 
  - Accessing the `.feature_importances_` attribute.
  - Sorting and plotting the results using `matplotlib`.
- **Takeaway:** Tree models are excellent at telling you *which* features matter most, making them invaluable for exploratory data analysis and feature selection.
