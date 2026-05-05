# Margin-Based Methods: Support Vector Machines (SVM)

## Learning Objective
Understand how Support Vector Machines classify data by finding the optimal hyperplane that maximizes the margin between classes, and learn how the "Kernel Trick" allows SVMs to solve complex non-linear problems.

## What Is This Topic?
A Support Vector Machine (SVM) is a powerful classification algorithm. Unlike Logistic Regression, which outputs probabilities based on a curve, SVM draws a strict line (or plane) through the data to separate the classes. It specifically looks for the line that leaves the maximum possible "gap" (margin) between the classes.

## Why This Topic Matters
Before Deep Learning became dominant, SVMs with non-linear kernels were considered the state-of-the-art algorithm for many classification tasks. They are highly effective in high-dimensional spaces (e.g., text categorization, image recognition) and are still widely used today for complex tabular data.

## Core Intuition
Imagine a table with red apples and green apples mixed together. You need to put a straight stick on the table to separate them. You could put the stick in many places, but the *best* place is right in the middle, as far away from both the red apples and the green apples as possible. That stick is the **hyperplane**, and the distance from the stick to the closest apples is the **margin**. The closest apples that "support" this margin are the **Support Vectors**.

## Key Concepts
- **Hyperplane**: The decision boundary. In 2D, it's a line. In 3D, it's a flat plane. In $N$-dimensions, it's an $N-1$ dimensional hyperplane.
- **Margin**: The distance between the hyperplane and the closest data points from either class.
- **Support Vectors**: The data points that lie exactly on the edge of the margin. *These are the only points the algorithm cares about.* If you remove all other data points, the hyperplane wouldn't change.
- **Hard Margin vs. Soft Margin**: A hard margin allows zero misclassifications (requires perfectly separable data). A soft margin allows some points to cross the line to prevent overfitting.
- **The Kernel Trick**: If data is not linearly separable in 2D, a Kernel function mathematically projects the data into 3D (or higher) where it *is* linearly separable, without actually doing the heavy computation.

## Important Parameters
- **`C` (Regularization parameter)**: Controls the trade-off between a wide margin and misclassification.
  - **High `C`**: Strict. "I don't want any misclassifications!" (Smaller margin, risks overfitting).
  - **Low `C`**: Relaxed. "I want a wide margin, even if I get a few points wrong." (Larger margin, risks underfitting).
- **`kernel`**: The function used to transform the data (`linear`, `poly`, `rbf`). `rbf` (Radial Basis Function) is the default and most popular for non-linear data.
- **`gamma`**: (Used in RBF kernel). Controls the "influence" of a single training point. High gamma means points only influence things very close to them (bumpy decision boundary). Low gamma means points have a far-reaching influence (smooth decision boundary).

## Advantages
- Extremely effective in high dimensional spaces.
- Memory efficient (only uses a subset of training points—the support vectors).
- Versatile, due to the ability to use different Kernel functions.

## Limitations
- **Does not scale well to large datasets** (training time is $O(n^2)$ to $O(n^3)$). Do not use on datasets with $>100,000$ rows.
- Highly sensitive to unscaled data.
- It does not directly provide probability estimates (though they can be calculated using expensive cross-validation via `probability=True`).
- Hard to interpret compared to Decision Trees or Logistic Regression.

## Related Methods
- Support Vector Regression (SVR) for continuous targets.
- Logistic Regression (similar to a linear SVM).

## Code References
- `code/example-01-basic-svm-linear.py`
- `code/example-02-svm-kernels.py`
- `code/example-03-real-world-medical.py`
