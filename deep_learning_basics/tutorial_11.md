# Tutorial 11: Loss & Cost Functions

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=E1yyaLRUnLo)

## Executive Summary

The loss function measures the "pain" of a single error, while the cost function measures the "pain" of the entire model. We explore **MSE, Log Loss, and Categorical Cross-Entropy**, and why choosing the right one is critical for gradient stability.

## Technical Deep Dive: Mathematical Optimization

### 1. Mean Squared Error (MSE)

MSE is the gold standard for regression. By squaring the error, we ensure the function is **convex** (bowl-shaped), which guarantees that Gradient Descent will eventually find the global minimum.
J(w, b) = (1 / 2m) Σi=1..m (ŷ(i) - y(i))2

### 2. Binary Cross-Entropy (Log Loss)

Why not use MSE for classification? Because if we use a Sigmoid activation, MSE results in a **non-convex** cost function with many local minima. Log Loss restores convexity for classification.
Loss = -(y · log(ŷ) + (1 - y) · log(1 - ŷ))

### 3. Categorical Cross-Entropy

Used for multi-class problems (like MNIST). It compares the predicted probability distribution against the one-hot encoded true labels.

## Numerical Stability in Code

```python
# Import the numpy library for mathematical array operations
import numpy as np

# Create a sample predictions array containing perfect 1.0 and 0.0 values (which can cause mathematical errors in log operations)
y_pred = np.array([1.0, 0.0, 0.9])
# Set a tiny threshold value epsilon to prevent taking log(0) which is undefined
epsilon = 1e-15

# Clip the predictions between epsilon and (1 - epsilon) so they are strictly between 0 and 1
y_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

# Define a function to calculate the Mean Squared Error (MSE) between true and predicted arrays
def mse(y_true, y_pred):
   # Calculate the average of the squared differences between the true labels and model predictions
   return np.mean(np.square(y_true - y_pred))
```

### 💡 Beginner's Blueprint: Choosing your Metric

It is easy to get confused about the differences between terms. Here is a simple breakdown:

* **Loss vs. Cost**:
  * **Loss Function**: Measures the error of a **single** training example (e.g., comparing the predicted price of one house to its actual price).
  * **Cost Function**: Measures the **average** error across the **entire** training set (the sum of all individual losses divided by the number of examples). We use the cost function to judge if the model is getting better overall.

* **Choosing the Right Tool**:
  * **Regression (Predicting a continuous number, like prices)**: Use **Mean Squared Error (MSE)**. It heavily penalizes large errors.
  * **Binary Classification (Yes/No, Cat/Dog)**: Use **Binary Cross-Entropy (Log Loss)**. It measures how confident the model is in its binary prediction.
  * **Multi-class Classification (MNIST, 10 categories)**: Use **Categorical Cross-Entropy**. It measures how well the predicted probability distribution matches the true distribution.

---

### 💡 Supplementary Notes

* **MAE vs. MSE Robustness**: Mean Absolute Error (MAE / L1 loss) is more robust to outliers than Mean Squared Error (MSE / L2 loss) because it doesn't square the errors. However, MAE has a non-differentiable point at 0, which can make convergence slower near the optimum.

## Active Recall Checkpoint

Convexity

Why is it dangerous to have a non-convex cost function during training?

Penalty Magnitude

If a model is 99% confident that a sample is 'Class A' but it is actually 'Class B', which function penalizes it more: MAE or Log Loss?