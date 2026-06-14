# Tutorial 8: Activation Functions Deep Dive

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=icZItWxw7AI)

## Executive Summary

This guide explores the fundamental role of activation functions in neural networks. Without them, a neural network is just a linear regression model. We cover **Sigmoid, tanh, ReLU, Leaky ReLU, and Softmax**, along with their mathematical derivatives which are crucial for backpropagation.

## Technical Deep Dive: Math & Gradients

#### Sigmoid (σ)

Maps any input value to a probability between 0 and 1. It acts as an "S-shaped" squashing function, pushing large positive values close to 1 and large negative values close to 0. While historically popular, it suffers from the **vanishing gradient** problem where learning halts if inputs are too extreme because the curve becomes flat.

**Formula:** σ(x) = 1 / (1 + e-x)

Derivative: σ(x)(1 - σ(x))

When to use:

*   **Output layer for binary classification:** Since it maps outputs to a strict probability range of $[0, 1]$, the result can be directly interpreted as the probability of the positive class.
*   **Avoid in hidden layers due to vanishing gradients:** The derivative of Sigmoid, $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, reaches its maximum value of only $0.25$ (at $x=0$). During backpropagation, the chain rule multiplies these derivatives across layers. In deep networks, repeatedly multiplying values $\le 0.25$ causes the gradient to shrink exponentially toward zero, which halts learning in early layers.

#### Tanh

Similar to Sigmoid, but it squashes inputs into a range between -1 and 1. This **zero-centered** nature usually makes optimization easier and helps networks converge faster than Sigmoid. However, it still falls prey to the vanishing gradient problem at extreme values.

**Formula:** tanh(x) = (ex - e-x) / (ex + e-x)

Derivative: 1 - tanh²(x)

When to use:

*   **Hidden layers in Recurrent Neural Networks (RNNs) and standard MLPs:**
    *   **Why:** Tanh is zero-centered, mapping outputs to a range of $[-1, 1]$. Because the mean of the outputs is close to $0$, it centers the data for the next layer. This zero-centering property prevents gradients from shifting systematically in one direction, making optimization more stable and leading to faster convergence than Sigmoid.
    *   **Application:** Used extensively in LSTM and GRU networks (for cell and hidden state updates) and in standard multi-layer perceptron hidden layers where negative activations are desired.

#### ReLU (Rectified Linear Unit)

The undisputed default for deep learning. It simply outputs 0 if the input is negative, and passes the input straight through if positive. This piecewise linearity allows for extremely fast computation and mitigates vanishing gradients for positive values. Its main drawback is the **"Dying ReLU"** problem.

**Formula:** f(x) = max(0, x)

Derivative: 1 if x > 0, else 0

When to use:

*   **Default choice for hidden layers in deep Feedforward and Convolutional Networks (CNNs):**
    *   **Why:** The derivative is a constant $1$ for all positive inputs ($x > 0$), meaning gradients flow backward without shrinking. This directly prevents the vanishing gradient problem in positive activation paths. Additionally, computing $max(0, x)$ requires a simple threshold operation, making it computationally trivial and significantly faster than exponential functions (like Sigmoid or Tanh).
    *   **Application:** Used as the standard activation function in hidden layers of almost all modern neural architectures (e.g., ResNet, VGG, MobileNet) and multi-layer perceptrons.

#### Leaky ReLU

A straightforward fix to the Dying ReLU problem. Instead of outputting a hard 0 for negative inputs, it allows a small, non-zero gradient (e.g., 0.01x). This ensures that neurons with negative inputs can still learn and update their weights during backpropagation, keeping the entire network actively participating.

**Formula:** f(x) = max(0.01x, x)

Derivative: 1 if x > 0, else 0.01

When to use:

*   **Replacing standard ReLU when experiencing the "Dying ReLU" problem or in Generative Adversarial Networks (GANs):**
    *   **Why:** Instead of outputting a flat $0$ for negative inputs, it outputs a tiny slope (like $0.01x$). This ensures that the gradient never becomes exactly zero ($\sigma'(x) = 0.01$ for $x < 0$). Weights continue to update, keeping the neurons active during training.
    *   **Application:** Commonly used in deep generative architectures (especially GAN discriminators) and deep models where traditional ReLU causes performance to plateau due to high numbers of dead neurons.

#### Softmax

Softmax takes a vector of raw prediction scores (logits) and normalizes them into a proper probability distribution. Every output value falls exactly between 0 and 1, and the entire set of outputs perfectly sums to 1.0. It heavily penalizes incorrect classes while exponentially rewarding the highest scoring class.

**Formula:** S(xi) = exi / Σj exj

Derivative: S(xi)(δij - S(xj))

When to use:

*   **Output layer for multi-class classification:**
    *   **Why:** It takes raw scores (logits) and normalizes them into a probability distribution where each value is in the range $(0, 1)$ and all values sum to exactly $1.0$. This treats class predictions as mutually exclusive probabilities.
    *   **Application:** Used as the final layer in multi-class classification models (e.g., MNIST digit classification, ImageNet object detection) to output prediction probabilities for each category.

## Technical Execution

### # Numpy Implementation (Under the Hood)

```python
# Import the numpy library for fast vector and matrix calculations
import numpy as np

# Define a function to compute the Sigmoid activation function
def sigmoid(x):
   # Calculate 1 divided by (1 + e^-x) to map inputs to a [0, 1] range
   return 1 / (1 + np.exp(-x))

# Define a function to compute the Rectified Linear Unit (ReLU) activation function
def relu(x):
   # Replace all negative values with 0, leaving positive values unchanged
   return np.maximum(0, x)

# Define a function to compute the Leaky ReLU activation function
def leaky_relu(x, alpha=0.01):
   # Keep x if it's positive; otherwise, multiply x by alpha to allow a tiny gradient for negative inputs
   return np.where(x > 0, x, alpha * x)

# Define a function to compute the Softmax activation function
def softmax(x):
   # Subtract the maximum value in x to prevent numerical overflow when computing exponents
   e_x = np.exp(x - np.max(x))
   # Divide each exponent by the sum of exponents along the last axis to obtain probabilities summing to 1.0
   return e_x / e_x.sum(axis=-1, keepdims=True)

# Create a sample NumPy array containing both positive and negative values to test
inputs = np.array([-10, -5, 0, 5, 10])
# Print the output of the relu function applied to our sample inputs array
print(f"ReLU Output: {relu(inputs)}")
```

### # Keras Implementation (Production)

```python
# Import the Dense layer and LeakyReLU layer classes from Keras layers
from keras.layers import Dense, LeakyReLU

# Create a dense output layer with 1 neuron and Sigmoid activation to predict binary class probabilities
output_layer = Dense(units=1, activation='sigmoid')

# Create a dense hidden layer with 64 units and Tanh activation (squashing outputs between -1 and 1)
rnn_dense = Dense(units=64, activation='tanh')

# Create a dense hidden layer with 128 units and ReLU activation as a fast, non-linear default layer
hidden_layer = Dense(units=128, activation='relu')

# Create a dense hidden layer with 128 units using Leaky ReLU activation specified via string alias (Keras 3)
leaky_layer = Dense(units=128, activation='leaky_relu')

# Create a base dense layer and apply it directly to the input tensor (functional API style)
dense_base = Dense(units=128)(inputs)
# Instantiate and apply a LeakyReLU activation layer to the outputs of the dense_base layer with alpha=0.01
leaky_out = LeakyReLU(alpha=0.01)(dense_base)

# Create a dense output layer with 10 neurons and Softmax activation to predict multi-class probabilities summing to 1.0
multiclass_output = Dense(units=10, activation='softmax')
```

### 💡 Beginner's Perspective: The "Line Bender"

Why do we even need activation functions? Why not just compute $z = w \cdot x + b$ and pass it directly to the next layer?

Let's look at some simple algebra:

- Imagine Layer 1 computes: $y_1 = 2x + 3$ (a straight line)
- Imagine Layer 2 computes: $y_2 = 3y_1 + 1$ (another straight line)

If we stack them together without any activation function, we get:
$$y_2 = 3(2x + 3) + 1 = 6x + 9 + 1 = 6x + 10$$

Notice that $y_2 = 6x + 10$ is **still just a straight line**! No matter how many layers you stack (even 1,000 layers), if you don't use activation functions, the entire network collapses into a single linear formula. A linear network can only separate data with a straight line, which fails on complex, real-world patterns.

An **activation function** introduces **non-linearity**. Functions like ReLU or Sigmoid "bend" the lines. By bending the lines at each layer, a deep network can warp and fold its decision boundaries, allowing it to classify incredibly complex shapes (like circles, waves, or spirals).

---

### 💡 Supplementary Notes

- **Dying ReLU Phenomenon**: Standard ReLU outputs exactly 0 for any negative input, meaning its gradient becomes 0. If a neuron is updated such that it never activates on any training data, it remains permanently inactive ('dead'). Leaky ReLU solves this by maintaining a small gradient (e.g., 0.01) for negative inputs.

## Active Recall Checkpoint

#### The Linear Trap

If you have 100 hidden layers but no activation function, how many effective layers do you have? Why?

#### Vanishing Gradients

Look at the derivative of Sigmoid. What is the maximum value it can take? How does this explain why training deep networks with Sigmoid is slow?
