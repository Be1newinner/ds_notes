# Tutorial 13: ANN Implementation from Scratch

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=PQCE9ChuIDY)

## Executive Summary

This guide moves beyond simple functions to **Object-Oriented Programming (OOP)** for Deep Learning. We build a `myNN` class that encapsulates state (weights/biases) and behavior (training/prediction), mirroring the industry-standard `.fit()` and `.predict()` workflow.

## Technical Deep Dive: The Engine Under the Hood

### 1. Smart Weight Initialization

Setting weights to 1 or 0 is dangerous (symmetry breaking). For deeper networks, we use **Xavier/Glorot Initialization**: drawing weights from a distribution with variance related to the number of inputs.
self.weights = np.random.randn(input_size) * np.sqrt(1/input_size)

### 2. Vectorized Backpropagation

Instead of calculating gradients per weight using loops, we calculate the entire gradient vector at once using Matrix Transposition.
# Gradient of weights = (1/m) * (X_transposed dot error)
dw = (1/m) * np.dot(X.T, (y_pred - y))

## Class Implementation

```python
# Define a custom neural network class from scratch
class myNN:
   # Initialize the class with input_size representing the number of input features
   def __init__(self, input_size):
       # Initialize the weights array with random values from a standard normal distribution
       self.w = np.random.randn(input_size)
       # Initialize the bias parameter to a baseline value of 0
       self.b = 0

   # Define the training fit method using training features, labels, epoch iterations, and learning rate
   def fit(self, X, y, epochs, lr):
       # Iterate through the specified number of training epochs
       for _ in range(epochs):
           # Compute current predictions for all inputs in the matrix (forward pass)
           y_hat = self.predict(X)
           
           # Calculate gradient of loss with respect to weights by dotting transposed inputs with errors (backward pass)
           dw = (1/len(y)) * np.dot(X.T, (y_hat - y))
           
           # Calculate gradient of loss with respect to bias by taking the mean prediction error
           db = np.mean(y_hat - y)
           
           # Update weights by subtracting the weight gradient scaled by the learning rate
           self.w -= lr * dw
           # Update bias by subtracting the bias gradient scaled by the learning rate
           self.b -= lr * db

   # Define prediction method to estimate classification outputs
   def predict(self, X):
       # Calculate dot product of inputs and weights, add bias, and apply Sigmoid squashing function
       return 1 / (1 + np.exp(-(np.dot(X, self.w) + self.b)))
```

### 💡 Beginner's Blueprint: Demystifying the Math

If you look at the line `dw = (1/len(y)) * np.dot(X.T, (y_hat - y))`, here is what is happening under the hood:
1. `(y_hat - y)` is the prediction error vector. It tells us how far off each prediction was.
2. `X.T` is the transposed input matrix.
3. `np.dot(X.T, ...)` matches each feature's values with its corresponding error across all data samples. If a feature is consistently high when the error is positive, its weight gets pushed down; if a feature is high when the error is negative, its weight gets pushed up.
4. `self.w -= lr * dw` applies the learning rate adjustment.

---

### 💡 Supplementary Notes

* **Vectorized Forward/Backward Pass**: Doing matrix calculations in batch mode ($Y = X \cdot W + b$) speeds up training by leveraging hardware parallelism and vector-unit caches rather than executing individual calculations per training sample.

## Active Recall Checkpoint

State Management

Why do we store 'w' and 'b' as 'self' attributes rather than just returning them from the fit function?

The Abstraction Layer

Compare this class to Keras. Which Keras layer does this specific implementation replicate? (e.g., Dense, Conv2D, Dropout?)