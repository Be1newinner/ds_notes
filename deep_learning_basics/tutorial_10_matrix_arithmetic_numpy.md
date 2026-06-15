# Tutorial 10: Matrix Arithmetic & NumPy

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=Wibxjrxf5ko)

## Executive Summary

Matrix arithmetic is the engine of Deep Learning. This guide covers **Subtraction, Dot Products, and Transposition**, focusing on how these operations allow us to process millions of parameters in parallel using NumPy.

## Technical Deep Dive: Linear Algebra in Practice

### 1. Dot Product Compatibility

You cannot multiply just any two matrices. For the Dot Product $A \cdot B$ to work:
Matrix A (**M** x N) • Matrix B (N x **P**) = Result (**M** x **P**)

The **inner dimensions** (N) must match exactly. The **outer dimensions** (M and P) determine the shape of the resulting matrix.

### 2. Transposition ($A^T$)

Transposing a matrix means flipping it over its diagonal—switching rows with columns. This is often necessary in Neural Networks to make matrices compatible for multiplication.

Original (2x3)
[1, 2, 3]
[4, 5, 6]

Transposed (3x2)
[1, 4]
[2, 5]
[3, 6]

### 3. Vectorization: Why NumPy is Fast

Standard Python `for` loops are slow because they process one item at a time. NumPy uses **SIMD (Single Instruction, Multiple Data)** to perform operations on entire blocks of memory at once, often 100x-1000x faster than loops.

## Technical Execution

```python
# Import the numpy library for high-performance linear algebra operations
import numpy as np

# Create a 2D weights array of shape (2, 3) representing a layer with 2 neurons and 3 inputs each
weights = np.array([[0.2, 0.8, -0.5], [0.5, -0.9, 0.8]])
# Create a 1D inputs array of shape (3,) containing values for a single data sample
inputs = np.array([10, 2, 8])

# Compute the dot product between the weights and inputs to calculate the weighted sum of inputs for both neurons
output = np.dot(weights, inputs)
# Print the shape of the resulting output array to verify it is a 1D array of size 2 (one output per neuron)
print(f"Output shape: {output.shape}")

# Reshape the 1D inputs array into a 2D column vector of shape (3, 1) for matrix operations
inputs_reshaped = inputs.reshape(3, 1)
# Transpose the column vector into a row vector of shape (1, 3) using the .T attribute
inputs_T = inputs_reshaped.T
# Print the shape of the transposed row vector to confirm it was successfully swapped
print(f"Transposed shape: {inputs_T.shape}")
```

### 💡 Beginner's Sandbox: Dot Product Walkthrough

Why do we use matrices instead of simple numbers? Because a single neuron processes multiple inputs, and a layer contains multiple neurons. Writing them as matrix multiplications allows us to calculate everything in one step.

Let's do a simple walkthrough of a single neuron with $3$ inputs:
* **Inputs ($X$)**: $[10, 2, 8]$
* **Weights ($W$)**: $[0.2, 0.8, -0.5]$
* **Bias ($b$)**: $1.0$

The calculation for $z = W \cdot X + b$ is:
$$z = (10 \cdot 0.2) + (2 \cdot 0.8) + (8 \cdot -0.5) + 1.0$$
$$z = 2.0 + 1.6 - 4.0 + 1.0 = 0.6$$

In code, this is calculated as:
```python
# Import the numpy library for linear algebra calculations
import numpy as np
# Create a 1D NumPy array representing 3 input values
inputs = np.array([10, 2, 8])
# Create a 1D NumPy array representing the corresponding 3 weight values for a single neuron
weights = np.array([0.2, 0.8, -0.5])
# Set the bias value to shift the activation function trigger point
bias = 1.0
# Calculate the final neuron activation z = W.X + b by taking the dot product and adding the bias
z = np.dot(weights, inputs) + bias
```

Using matrices lets us run this math for a batch of 100 images and 1,000 neurons all at once instead of writing slow `for` loops.

---

### 💡 Supplementary Notes

* **NumPy Broadcasting Rules**: When operating on two arrays of different shapes, NumPy matches their shapes starting from the trailing dimensions. Two dimensions are compatible if they are equal or if one of them is 1, enabling efficient element-wise operations without copying data.

## Active Recall Checkpoint

Dimensional Match

If you have a Weight matrix of (128, 784) and an Input matrix of (64, 784), can you multiply them as-is? If not, what operation must you perform on the Input matrix?

Loop vs. Vector

Why is it said that "Loops are the enemy of Deep Learning"? Explain the computational difference.