# Tree-Based Classification: Decision Trees & Ensembles

## Learning Objective
Understand how Decision Trees mimic human decision-making, why a single tree is prone to overfitting, and how ensemble methods (Random Forest and Gradient Boosting) combine many weak trees to create state-of-the-art models for tabular data.

## Part 1: Decision Trees

### What Is This Topic?
A Decision Tree classifies data by asking a series of True/False questions about the features. It splits the data into smaller and smaller subsets until it reaches a final decision (a "leaf").

### Core Intuition
Think of a game of "20 Questions." To guess an animal, you ask: "Does it have four legs?" (Yes/No). If Yes, "Does it bark?" (Yes/No). A Decision Tree does exactly this mathematically. It chooses the question that best separates the classes at each step.

### Key Concepts
- **Root Node**: The very first question asked (the top of the tree).
- **Leaf Node**: The final prediction (the bottom of the tree).
- **Gini Impurity / Entropy**: The mathematical metrics used to decide which question to ask. The algorithm wants to split the data so that the resulting groups are as "pure" as possible (e.g., all 1s in one group, all 0s in the other).

### Advantages & Limitations
- **Advantages**: Incredibly interpretable; requires *zero* feature scaling; handles both categorical and numerical data naturally.
- **Limitations**: A single tree is notorious for **overfitting**. It will keep growing until it memorizes every single row in the training data, leading to a massive, complex tree that fails on test data.

---

## Part 2: Ensembles (Random Forest)

### What Is This Topic?
An ensemble model combines the predictions of multiple machine learning models to produce a single, stronger prediction. Random Forest is an ensemble of Decision Trees.

### Core Intuition
If you ask one person to guess the weight of a cow, they might be wildly wrong. If you ask 1,000 people and take the average, the guess will be incredibly accurate. Random Forest creates 100+ different Decision Trees and lets them "vote" on the final classification.

### Key Concepts
- **Bagging (Bootstrap Aggregating)**: Each tree in the forest is trained on a random *sample* of the training data (with replacement).
- **Random Subspace**: At each split in a tree, only a random *subset* of features is considered. 
- **Result**: Because of this randomness, every tree is slightly different. When they vote together, the variance (overfitting) is drastically reduced.

---

## Part 3: Ensembles (Gradient Boosting)

### What Is This Topic?
Gradient Boosting is another way to ensemble trees, but instead of building them all at once (like Random Forest), it builds them sequentially.

### Core Intuition
1. Build a short, weak tree. It makes some mistakes.
2. Build a second tree *specifically designed to fix the mistakes of the first tree*.
3. Build a third tree to fix the mistakes of the second tree.
4. Repeat 100+ times.

### Key Concepts
- **Boosting**: Combining weak learners sequentially to minimize errors.
- **Learning Rate**: Controls how much each tree is allowed to contribute to the final answer. A small learning rate requires more trees but usually results in a better, more robust model.

## Real-World Uses
- **Random Forest**: The ultimate "baseline" model for tabular data. Often used for feature selection because it can rank the importance of features.
- **Gradient Boosting (XGBoost, LightGBM, CatBoost)**: The algorithm that wins almost all Kaggle competitions for structured/tabular data (credit scoring, pricing, etc.).

## Code References
- `code/example-01-decision-tree.py`
- `code/example-02-random-forest.py`
- `code/example-03-gradient-boosting.py`
- `code/example-04-feature-importance.py`
