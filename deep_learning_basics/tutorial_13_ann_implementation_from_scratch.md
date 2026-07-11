# Tutorial 13: ANN Implementation from Scratch

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=PQCE9ChuIDY)

## Executive Summary

This guide moves beyond simple functions to **Object-Oriented Programming (OOP)** for Deep Learning. We build a `myNN` class that encapsulates state (weights/biases) and behavior (training/prediction), mirroring the industry-standard `.fit()` and `.predict()` workflow.

## Technical Deep Dive: The Engine Under the Hood

### 1. Smart Weight Initialization

Setting weights to 1 or 0 is dangerous (symmetry breaking). For deeper networks, we use **Xavier/Glorot Initialization**: drawing weights from a distribution with variance related to the number of inputs.
self.weights = np.random.randn(input_size) \* np.sqrt(1/input_size)

### 2. Vectorized Backpropagation

Instead of calculating gradients per weight using loops, we calculate the entire gradient vector at once using Matrix Transposition.

# Gradient of weights = (1/m) \* (X_transposed dot error)

dw = (1/m) \* np.dot(X.T, (y_pred - y))

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

## ANN Structure

```md
                Artificial Neural Network (ANN)

                  Input Layer
         +--------+--------+--------+
         |   x1   |   x2   |   x3   |
         +--------+--------+--------+
                  │
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼

                Hidden Layer
      +--------+  +--------+  +--------+  +--------+
      |   h1   |  |   h2   |  |   h3   |  |   h4   |
      +--------+  +--------+  +--------+  +--------+
         │  ╲        │  ╲        │  ╲        │
         │   ╲       │   ╲       │   ╲       │
         ▼    ▼      ▼    ▼      ▼    ▼      ▼

                Output Layer
             +--------+--------+
             |   y1   |   y2   |
             +--------+--------+
```

## Arc Implementation

```md
Dataset
│
▼
Initialize Weights & Biases
│
▼
For epoch in range(EPOCHS):

      For every training example:

            1. Forward Pass
                   │
                   ▼
            2. Prediction
                   │
                   ▼
            3. Calculate Loss
                   │
                   ▼
            4. Backpropagation
                   │
                   ▼
            5. Compute Gradients
                   │
                   ▼
            6. Update Weights
                   │
                   ▼
      End Sample Loop

Print Loss

End Epoch Loop

Save Model
```

This is probably the **smallest ANN** that still demonstrates **real backpropagation**.

Architecture:

```text
2 Inputs → 2 Hidden Neurons → 1 Output
```

```
      x1
        \
         \
          h1 ----\
         /        \
x2 -----/          ---> y
         \        /
          h2 ----/
```

---

# Step 1: Initialize Weights

```python
import numpy as np

np.random.seed(42)

# Input -> Hidden
W1 = np.random.randn(2,2)
b1 = np.zeros((1,2))

# Hidden -> Output
W2 = np.random.randn(2,1)
b2 = np.zeros((1,1))
```

---

# Step 2: Sigmoid Function

```python
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(a):
    return a*(1-a)
```

Notice that the derivative is calculated using the activated value `a`.

---

# Step 3: Dataset

```python
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])
```

---

# Step 4: Training Loop

```python
learning_rate = 0.1

epochs = 10000

for epoch in range(epochs):

    # ==========================
    # FORWARD PASS
    # ==========================

    # Hidden Layer
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    # Output Layer
    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)
```

---

## What happened?

### Hidden layer

$$
z_1 = X W_1 + b_1
$$

↓

$$
a_1 = \sigma(z_1)
$$

---

### Output layer

$$
z_2 = a_1 W_2 + b_2
$$

↓

$$
\hat{y} = \sigma(z_2)
$$

---

# Step 5: Loss

```python
    loss = np.mean((y_hat-y)**2)
```

Mathematically

$$
L = \frac{1}{n} \sum (\hat{y} - y)^2
$$

---

# Step 6: Backpropagation

This is the important part.

## Output Layer Error

```python
    output_error = y_hat - y
```

Mathematically

$$
\hat{y} - y
$$

---

## Output Delta

```python
    output_delta = output_error * sigmoid_derivative(y_hat)
```

Mathematically

$$
\delta_2 = (\hat{y} - y) \sigma'(z_2)
$$

---

## Gradient for W2

```python
    dW2 = np.dot(a1.T, output_delta)
```

Formula

$$
\boxed{dW_2 = a_1^T \delta_2}
$$

---

## Bias Gradient

```python
    db2 = np.sum(output_delta, axis=0, keepdims=True)
```

Formula

$$
db_2 = \sum \delta_2
$$

---

# Here's the REAL Backpropagation

The error must travel backward to the hidden layer.

```python
    hidden_error = np.dot(output_delta, W2.T)
```

Formula

$$
\boxed{\text{HiddenError} = \delta_2 W_2^T}
$$

Notice:

The output error is sent backwards through **W2**.

This is the "back" in **backpropagation**.

---

## Hidden Delta

```python
    hidden_delta = hidden_error * sigmoid_derivative(a1)
```

Formula

$$
\boxed{\delta_1 = (\delta_2 W_2^T) \sigma'(z_1)}
$$

Now every hidden neuron knows how much it contributed to the error.

---

## Gradient of W1

```python
    dW1 = np.dot(X.T, hidden_delta)
```

Formula

$$
\boxed{dW_1 = X^T \delta_1}
$$

---

## Hidden Bias

```python
    db1 = np.sum(hidden_delta, axis=0, keepdims=True)
```

---

# Step 7: Update Weights

```python
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
```

Formula

$$
W = W - \alpha dW
$$

---

# Complete Code

```python
import numpy as np

np.random.seed(42)

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

W1 = np.random.randn(2,2)
b1 = np.zeros((1,2))

W2 = np.random.randn(2,1)
b2 = np.zeros((1,1))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(a):
    return a*(1-a)

lr = 0.1

for epoch in range(10000):

    # Forward
    z1 = np.dot(X,W1)+b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1,W2)+b2
    y_hat = sigmoid(z2)

    # Loss
    loss = np.mean((y_hat-y)**2)

    # Backpropagation
    output_error = y_hat-y
    output_delta = output_error * sigmoid_derivative(y_hat)

    dW2 = np.dot(a1.T, output_delta)
    db2 = np.sum(output_delta, axis=0, keepdims=True)

    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(a1)

    dW1 = np.dot(X.T, hidden_delta)
    db1 = np.sum(hidden_delta, axis=0, keepdims=True)

    # Update
    W2 -= lr*dW2
    b2 -= lr*db2

    W1 -= lr*dW1
    b1 -= lr*db1

    if epoch % 1000 == 0:
        print(epoch, loss)
```

## Visualizing the Backpropagation Flow

```text
                 FORWARD PASS

Input
  │
  ▼
z1 = XW1 + b1
  │
  ▼
a1 = sigmoid(z1)
  │
  ▼
z2 = a1W2 + b2
  │
  ▼
ŷ = sigmoid(z2)
  │
  ▼
Loss


           BACKPROPAGATION

Loss
 │
 ▼
δ₂ = (ŷ − y) · σ'(z₂)
 │
 ▼
dW₂ = a₁ᵀ · δ₂
 │
 ▼
Hidden Error = δ₂ · W₂ᵀ
 │
 ▼
δ₁ = Hidden Error · σ'(z₁)
 │
 ▼
dW₁ = Xᵀ · δ₁
 │
 ▼
Update W₂, b₂, W₁, b₁
```

The key line that distinguishes a neural network from logistic regression is:

```python
hidden_error = np.dot(output_delta, W2.T)
```

This is where the output layer's error is **propagated backward** through the network to compute how the hidden layer's weights should change. Without hidden layers, this step doesn't exist, which is why logistic regression doesn't require the full backpropagation algorithm.

### 💡 Beginner's Blueprint: Demystifying the Math

If you look at the line `dw = (1/len(y)) * np.dot(X.T, (y_hat - y))`, here is what is happening under the hood:

1. `(y_hat - y)` is the prediction error vector. It tells us how far off each prediction was.
2. `X.T` is the transposed input matrix.
3. `np.dot(X.T, ...)` matches each feature's values with its corresponding error across all data samples. If a feature is consistently high when the error is positive, its weight gets pushed down; if a feature is high when the error is negative, its weight gets pushed up.
4. `self.w -= lr * dw` applies the learning rate adjustment.

---

### 💡 Supplementary Notes

- **Vectorized Forward/Backward Pass**: Doing matrix calculations in batch mode ($Y = X \cdot W + b$) speeds up training by leveraging hardware parallelism and vector-unit caches rather than executing individual calculations per training sample.

## Active Recall Checkpoint

State Management

Why do we store 'w' and 'b' as 'self' attributes rather than just returning them from the fit function?

The Abstraction Layer

Compare this class to Keras. Which Keras layer does this specific implementation replicate? (e.g., Dense, Conv2D, Dropout?)

## IN FUTURE WE WILL DO

Episode 1 ✅
Implement one neuron (Logistic Regression)

↓

Episode 2
Implement one hidden layer manually
(2 → 3 → 1)

↓

Episode 3
Generalize it for any number of hidden layers

↓

Episode 4
Implement backpropagation from scratch

↓

Episode 5
Build a mini TensorFlow/PyTorch-like neural network library

```

```
