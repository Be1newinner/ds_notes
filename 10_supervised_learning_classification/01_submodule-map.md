# Submodule Map: Supervised Learning Classification

This document provides a structural roadmap for teaching the Supervised Learning Classification module.

## 01_linear_models
- **Why it is taught:** Logistic regression provides a fundamental, highly interpretable baseline model. It helps students transition from linear regression (predicting continuous values) to classification (predicting probabilities).
- **Format:** High on theory, heavy on interpretation of coefficients (log odds). Requires business examples (e.g., predicting churn probability).

## 02_distance_probabilistic
- **Why it is taught:** Introduces K-Nearest Neighbors (KNN) as an intuitive instance-based learner and Naive Bayes for text/probabilistic classification.
- **Format:** Focuses on the impact of distance metrics and scaling (for KNN) and independence assumptions (for Naive Bayes). Strong visual and conceptual components.

## 03_margin_based
- **Why it is taught:** Support Vector Machines (SVM) are powerful for finding optimal decision boundaries and introducing the "kernel trick" for non-linear data.
- **Format:** Theory-heavy. Needs visual explanations of hyperplanes, margins, and support vectors.

## 04_tree_based
- **Why it is taught:** Decision Trees and their ensembles (Random Forests, Gradient Boosting) are the industry standard for structured tabular data.
- **Format:** Mix of theory and code. High emphasis on visualizing trees, understanding feature importance, and explaining how ensembles reduce variance/bias.

## 05_imbalanced_classification
- **Why it is taught:** Most real-world classification problems (e.g., fraud, disease diagnosis) do not have a 50/50 class split. Standard models fail here.
- **Format:** Very practical. Students learn to implement SMOTE, adjust class weights, and understand why accuracy is a bad metric for imbalanced data.

## 06_hyperparameter_tuning
- **Why it is taught:** Out-of-the-box models are rarely optimal. Students must learn systematic ways to search for the best model configurations.
- **Format:** Code-heavy. Introduces GridSearchCV, RandomizedSearchCV, and cross-validation pipelines. Emphasizes preventing data leakage during tuning.

---

### Recommended Teaching Order
1. `01_linear_models` -> Establish baseline.
2. `02_distance_probabilistic` -> Introduce alternative paradigms.
3. `03_margin_based` -> Complex, high-dimensional boundaries.
4. `04_tree_based` -> Industry workhorses.
5. `05_imbalanced_classification` -> Practical data realities.
6. `06_hyperparameter_tuning` -> Model optimization.
