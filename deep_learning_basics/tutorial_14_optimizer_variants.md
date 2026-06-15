# Tutorial 14: Optimizer Variants

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=IU5fuoYBTAM)

## Executive Summary

Choosing the right variant of Gradient Descent determines the speed and quality of training. We analyze the trade-offs between **Memory (Batch)**, **Noise (SGD)**, and **Hardware Utilization (Mini-Batch)**.

## Structural Comparison
Feature Batch GD Stochastic GD Mini-Batch GD Update Frequency Once per Epoch Every Sample Every Batch Memory Usage Extremely High Low Moderate (Configurable) Convergence Path Smooth & Direct Very Noisy Relatively Stable

## Hardware Utilization & Vectorization

### 1. Why Mini-Batch Wins

Modern CPUs and GPUs are designed for **Matrix Operations**, not scalar ones. SGD updates one sample at a time, meaning the hardware sits idle waiting for the next instruction. Batch GD is too big for memory. Mini-Batch fits into the **GPU Memory (VRAM)** and uses parallelization to process 32-512 samples simultaneously.
# The Sweet Spot: Batch size power of 2 (32, 64, 128)
# Replicates the best of both worlds.

### 💡 Beginner's Analogy: The Exam Study Methods

Imagine you have a huge exam coming up and need to study a textbook with 10,000 pages:

1. **Batch Gradient Descent**:
   You read the **entire 10,000-page book** cover-to-cover, summarize the information, and then write down your study plan revisions once.
   * *Pros*: Very accurate and smooth adjustments.
   * *Cons*: Takes an enormous amount of time and memory before you make even one single adjustment.

2. **Stochastic Gradient Descent (SGD)**:
   You read **one single sentence** from page 1, immediately update your study plan, then read one sentence from page 2, update again, etc.
   * *Pros*: Extremely fast and interactive.
   * *Cons*: Very noisy and erratic. Your updates will bounce all over the place (e.g., changing your mind 100 times in 5 minutes).

3. **Mini-Batch Gradient Descent**:
   You read **one chapter at a time** (e.g., 32 pages), summarize it, and update your study plan.
   * *Pros*: The best of both worlds. Fast, stable, and fits easily into your hardware's memory (VRAM). This is the industry standard!

---

## Deep Dive: Batch vs. SGD vs. Mini-Batch Gradient Descent

To truly understand how a model learns, we need to look at how we feed data into the training algorithm and calculate gradients. Here is a detailed, beginner-friendly breakdown of the three main methods.

### 1. Batch Gradient Descent
* **The Concept:** You calculate the prediction error (loss) for **every single item** in your entire dataset. Then, you average those errors to calculate the gradient and update your model's weights exactly **once per epoch**.
* **Beginner Note:** Think of this as taking a full survey of all citizens in a city before making any city council budget changes.
* **Pros:**
  - Moves in a very smooth, stable, and direct path toward the lowest loss (global minimum).
  - Works great for small datasets.
* **Cons:**
  - Extremely slow and computationally heavy. If you have 1 million items, you must calculate 1 million predictions just to make a single tiny weight update.
  - Can easily run out of memory (OOM) on large datasets.

### 2. Stochastic Gradient Descent (SGD)
* **The Concept:** "Stochastic" means random. Instead of waiting to look at all data, you look at **one single sample**, calculate its error, and update the model's weights **immediately**. You repeat this for every single sample in your dataset.
* **Beginner Note:** Think of this as asking one random citizen on the street, immediately updating the budget based on their opinion, walking to the next block, asking another citizen, and updating the budget again.
* **Pros:**
  - Updates weights instantly, allowing the model to start learning immediately.
  - The "noise" (wild bouncing in weights) can actually help the model jump out of bad local minima (traps where the loss looks low but isn't the absolute lowest).
* **Cons:**
  - The path to convergence is highly erratic and looks like a jagged zig-zag.
  - The model never truly settles down at the absolute lowest loss—it tends to keep bouncing around it.
  - It misses out on GPU speedups since it does not process data in parallel.

### 3. Mini-Batch Gradient Descent
* **The Concept:** The perfect middle ground. You divide your dataset into small, manageable groups called **batches** (typically sizes like 32, 64, or 128). You calculate the error for the batch, average them, and update the weights.
* **Beginner Note:** Think of this as surveying a group of 32 citizens at a time, summarizing their feedback, and updating the budget after each group.
* **Pros:**
  - Highly efficient because it harnesses GPU vectorization (processing multiple samples in parallel).
  - More stable than SGD because averaging reduces wild noise, yet fast enough to converge quickly.
  - **This is the industry standard** used to train almost all modern deep learning models!

---

## Plain Python Implementations (No Libraries Required)

Here is a complete, runnable Python script that implements all three gradient descent methods using simple lists and loop operations on a small sample dataset.

```python
# Sample Data: House sizes (in 1000s of sq ft) and their selling prices (in $100k)
# True underlying relationship: y = 2 * x + 1 (Weight w = 2.0, Bias b = 1.0)
X = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [3.0, 5.0, 7.0, 9.0, 11.0]

# Helper function to print results nicely
def print_separator(title):
    print("\n" + "=" * 50)
    print(f" {title} ")
    print("=" * 50)

# ----------------------------------------------------
# 1. Batch Gradient Descent
# ----------------------------------------------------
def batch_gradient_descent(X, y, epochs=100, lr=0.01):
    w = 0.0  # Initial weight guess
    b = 0.0  # Initial bias guess
    N = len(X)
    
    print_separator("Running Batch Gradient Descent")
    
    for epoch in range(epochs):
        dw_total = 0.0
        db_total = 0.0
        total_loss = 0.0
        
        # 1. Look at ALL data points first
        for i in range(N):
            x_val = X[i]
            y_val = y[i]
            
            # Predict & calculate squared loss
            y_pred = w * x_val + b
            loss = (y_pred - y_val) ** 2
            total_loss += loss
            
            # Accumulate gradients (derivatives of Mean Squared Error)
            error = y_pred - y_val
            dw_total += 2 * error * x_val
            db_total += 2 * error
            
        # 2. Average the gradients over the entire dataset
        dw = dw_total / N
        db = db_total / N
        
        # 3. Update weight and bias ONCE per epoch
        w = w - lr * dw
        b = b - lr * db
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / N
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:8.4f} | w = {w:6.4f}, b = {b:6.4f}")
            
    return w, b

# ----------------------------------------------------
# 2. Stochastic Gradient Descent (SGD)
# ----------------------------------------------------
def stochastic_gradient_descent(X, y, epochs=100, lr=0.01):
    w = 0.0  # Initial weight guess
    b = 0.0  # Initial bias guess
    N = len(X)
    
    print_separator("Running Stochastic Gradient Descent")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Loop through each data point individually
        for i in range(N):
            x_val = X[i]
            y_val = y[i]
            
            # Predict & calculate squared loss
            y_pred = w * x_val + b
            loss = (y_pred - y_val) ** 2
            total_loss += loss
            
            # Calculate gradient for this SINGLE point
            error = y_pred - y_val
            dw = 2 * error * x_val
            db = 2 * error
            
            # Update weight and bias IMMEDIATELY
            w = w - lr * dw
            b = b - lr * db
            
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / N
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:8.4f} | w = {w:6.4f}, b = {b:6.4f}")
            
    return w, b

# ----------------------------------------------------
# 3. Mini-Batch Gradient Descent
# ----------------------------------------------------
def mini_batch_gradient_descent(X, y, batch_size=2, epochs=100, lr=0.01):
    w = 0.0  # Initial weight guess
    b = 0.0  # Initial bias guess
    N = len(X)
    
    print_separator(f"Running Mini-Batch GD (Batch Size = {batch_size})")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Iterate over the dataset in batches
        for batch_start in range(0, N, batch_size):
            # Extract the current batch inputs and outputs
            batch_X = X[batch_start : batch_start + batch_size]
            batch_y = y[batch_start : batch_start + batch_size]
            B = len(batch_X)  # Size of current batch (can be smaller for last batch)
            
            dw_batch = 0.0
            db_batch = 0.0
            
            # Look at all data points in this current batch
            for i in range(B):
                x_val = batch_X[i]
                y_val = batch_y[i]
                
                # Predict & calculate loss
                y_pred = w * x_val + b
                loss = (y_pred - y_val) ** 2
                total_loss += loss
                
                # Accumulate gradients for the batch
                error = y_pred - y_val
                dw_batch += 2 * error * x_val
                db_batch += 2 * error
                
            # Average gradients for this batch
            dw = dw_batch / B
            db = db_batch / B
            
            # Update weight and bias after finishing the batch
            w = w - lr * dw
            b = b - lr * db
            
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / N
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:8.4f} | w = {w:6.4f}, b = {b:6.4f}")
            
    return w, b

# Execute all three
w_batch, b_batch = batch_gradient_descent(X, y, epochs=100, lr=0.01)
w_sgd, b_sgd     = stochastic_gradient_descent(X, y, epochs=100, lr=0.01)
w_mini, b_mini   = mini_batch_gradient_descent(X, y, batch_size=2, epochs=100, lr=0.01)
```

---

## Real-Life Framework Implementation

In production machine learning libraries like **Keras/TensorFlow** and **PyTorch**, you do not write these loops manually. Instead, the framework controls the gradient descent variant using the **batch size** configuration.

### 1. In Keras (TensorFlow)
In Keras, the optimization loop is controlled via the `batch_size` argument inside the `model.fit()` function.

```python
# 1. Batch Gradient Descent
# Set batch_size equal to the size of the whole training dataset
model.fit(x_train, y_train, batch_size=len(x_train), epochs=50)

# 2. Stochastic Gradient Descent (SGD)
# Set batch_size to 1 (calculates loss and updates weights after every single sample)
model.fit(x_train, y_train, batch_size=1, epochs=50)

# 3. Mini-Batch Gradient Descent (The Sweet Spot / Default)
# Set batch_size to standard powers of 2 (like 32, 64, 128, or 256)
model.fit(x_train, y_train, batch_size=32, epochs=50)
```

### 2. In PyTorch
In PyTorch, the batching is managed by the `DataLoader` utility class, which wraps around your dataset.

```python
from torch.utils.data import DataLoader, TensorDataset
import torch

# Create some dummy tensor data
x_data = torch.randn(1000, 20)
y_data = torch.randint(0, 2, (1000,))
dataset = TensorDataset(x_data, y_data)

# 1. Batch Gradient Descent
# Load the entire dataset at once
batch_loader = DataLoader(dataset, batch_size=1000, shuffle=True)

# 2. Stochastic Gradient Descent (SGD)
# Load one sample at a time
sgd_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 3. Mini-Batch Gradient Descent (Standard practice)
# Load in small groups (e.g., 32 samples per step)
minibatch_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

### 💡 Supplementary Notes

* **Adam Optimizer Momentum**: Adam combines the principles of **Momentum** (accumulating past gradients to smooth oscillations) and **RMSprop** (dividing the learning rate by an exponentially decaying average of squared gradients) to adaptively scale learning rates per parameter.

## Active Recall Checkpoint

Noise vs. Escape

Why is the "noise" in SGD sometimes considered a feature rather than a bug? (Hint: Think about Local Minima).

Numerical Scaling

If one feature is "Annual Income" (0-100,000) and another is "Years of Experience" (0-40), why will GD struggle to converge without normalization?