# Tutorial 15: Chain Rule & Backpropagation

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=5ogmEkujoqE)

## Executive Summary

Backpropagation is simply the application of the **Chain Rule** to find the gradient of the loss function with respect to every weight and bias in a multi-layered network. This allows us to "assign credit" (or blame) for error to specific parameters, enabling the network to learn by adjusting its parameters to minimize overall error.

---

## Technical Deep Dive: The Chain of Derivatives

### 1. Computational Graphs & Gradient Flow

Every neural network can be represented as a **Computational Graph**, where nodes represent mathematical operations (like addition, multiplication, or activation functions) and edges represent the variables flowing between them.

For a simple sequence $x \xrightarrow{\cdot w} z \xrightarrow{\sigma} a \xrightarrow{\text{Loss}} L$, we compute values from left to right during the **Forward Pass**, and calculate derivatives from right to left during the **Backward Pass**:

```
[ Input x ] ------> ( Multiply ) ----[ z ]----> ( Activation ) ----[ a ]----> ( Loss ) ----> [ Loss L ]
                       ^                                                         ^
                       |                                                         |
                   [ Weight w ]                                             [ Target y ]
```

During the backward pass, we want to find how the loss $L$ changes relative to our weight $w$ ($\frac{\partial L}{\partial w}$). According to the calculus **Chain Rule**, we do this by multiplying the derivatives along the path from $L$ back to $w$:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \times \frac{\partial a}{\partial z} \times \frac{\partial z}{\partial w}$$

### 2. Local vs. Global Gradients

A major strength of backpropagation is its **locality**. Each node in our graph only needs to compute its own **Local Gradient** (how its output changes relative to its immediate inputs). It does not need to know what happens in the rest of the network.

When an **Upstream Gradient** (the gradient of the loss with respect to this node's output) flows back to the node, the node multiplies it by its local gradient. This computes the **Global Gradient** which is passed further backward.

$$\text{Global Gradient} = \text{Upstream Gradient} \times \text{Local Gradient}$$

Let's break down the weight derivative:

$$\frac{\partial L}{\partial w} = \underbrace{\left(\frac{\partial L}{\partial a}\right)}_{\text{Upstream Gradient}} \times \underbrace{\left(\frac{\partial a}{\partial z}\right)}_{\text{Local Activation Gradient}} \times \underbrace{\left(\frac{\partial z}{\partial w}\right)}_{\text{Local Input Gradient}}$$

*   **Upstream Gradient ($\frac{\partial L}{\partial a}$):** How the final loss changes with respect to this layer's output activation.
*   **Local Activation Gradient ($\frac{\partial a}{\partial z}$):** How the activation function's output changes with respect to its pre-activation input $z$ (e.g., derivative of Sigmoid or ReLU).
*   **Local Input Gradient ($\frac{\partial z}{\partial w}$):** How the weighted sum $z = w \cdot x + b$ changes with respect to the weight $w$. This is simply the input $x$!

### 3. The Danger of Deep Chains (Vanishing & Exploding Gradients)

In a network with 100 layers, the gradient at the earliest layer is a product of 100 derivatives:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_{100}} \times \frac{\partial a_{100}}{\partial z_{100}} \times \dots \times \frac{\partial a_1}{\partial z_1} \times \frac{\partial z_1}{\partial w_1}$$

*   **Vanishing Gradient:** If these intermediate derivatives are small (like in Sigmoid where the maximum derivative is $0.25$), multiplying 100 of them together results in a product that becomes microscopic (close to 0). The early layers will stop learning.
*   **Exploding Gradient:** If the derivatives are large (e.g., weights initialized too large), multiplying them together causes the gradient to grow exponentially, making the weights update too drastically and destabilizing training.

This is why modern architectures use **ReLU** (whose derivative is $1$ for positive inputs) and **Skip Connections (ResNets)** to maintain healthy gradient flow.

---

## Step-by-Step Mathematical Walkthrough

Let's perform a complete manual backpropagation calculation on a simple 2-layer neural network with a hidden neuron and a sigmoid activation function.

### Network Structure:
*   **Input**: $x = 1.5$
*   **Layer 1 (Hidden)**: Weight $w_1 = 0.8$, Bias $b_1 = 0.1$. Pre-activation $z_1 = w_1 \cdot x + b_1$. Activation $h = \sigma(z_1)$ where $\sigma(z) = \frac{1}{1 + e^{-z}}$.
*   **Layer 2 (Output)**: Weight $w_2 = 1.2$, Bias $b_2 = -0.5$. Pre-activation $z_2 = w_2 \cdot h + b_2$. Prediction $\hat{y} = z_2$ (linear activation for simplicity).
*   **Loss**: Squared Error $L = \frac{1}{2}(\hat{y} - y_{true})^2$ where $y_{true} = 1.0$.

#### 1. Forward Pass (Calculations):
1.  **Hidden pre-activation $z_1$**: 
    $$z_1 = (0.8 \times 1.5) + 0.1 = 1.2 + 0.1 = 1.3$$
2.  **Hidden activation $h$**: 
    $$h = \sigma(1.3) = \frac{1}{1 + e^{-1.3}} \approx 0.7858$$
3.  **Output prediction $\hat{y}$**: 
    $$\hat{y} = z_2 = (1.2 \times 0.7858) - 0.5 = 0.9430 - 0.5 = 0.4430$$
4.  **Loss $L$**: 
    $$L = \frac{1}{2}(0.4430 - 1.0)^2 = \frac{1}{2}(-0.5570)^2 \approx 0.1551$$

#### 2. Backward Pass (Chain Rule & Derivatives):
We want to find the gradients of the loss with respect to all our parameters: $\frac{\partial L}{\partial w_2}$, $\frac{\partial L}{\partial b_2}$, $\frac{\partial L}{\partial w_1}$, and $\frac{\partial L}{\partial b_1}$.

1.  **Gradient of Loss w.r.t prediction $\hat{y}$**:
    $$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y_{true} = 0.4430 - 1.0 = -0.5570$$
    *(This is our initial upstream gradient $\delta_2 = -0.5570$)*

2.  **Gradients for Layer 2 parameters ($w_2$, $b_2$)**:
    *   $$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial w_2} = \delta_2 \times h = -0.5570 \times 0.7858 \approx -0.4377$$
    *   $$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial b_2} = \delta_2 \times 1 = -0.5570$$

3.  **Gradient of Loss w.r.t Hidden Activation $h$**:
    $$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial h} = \delta_2 \times w_2 = -0.5570 \times 1.2 = -0.6684$$

4.  **Gradient of Loss w.r.t Hidden Pre-activation $z_1$ ($\delta_1$)**:
    $$\delta_1 = \frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \times \frac{\partial h}{\partial z_1}$$
    *Note: The derivative of the sigmoid is $\sigma(z)(1 - \sigma(z)) = h(1 - h)$.*
    $$\frac{\partial h}{\partial z_1} = 0.7858 \times (1 - 0.7858) = 0.7858 \times 0.2142 \approx 0.1683$$
    $$\delta_1 = -0.6684 \times 0.1683 \approx -0.1125$$

5.  **Gradients for Layer 1 parameters ($w_1$, $b_1$)**:
    *   $$\frac{\partial L}{\partial w_1} = \delta_1 \times \frac{\partial z_1}{\partial w_1} = \delta_1 \times x = -0.1125 \times 1.5 \approx -0.1687$$
    *   $$\frac{\partial L}{\partial b_1} = \delta_1 \times \frac{\partial z_1}{\partial b_1} = \delta_1 \times 1 \approx -0.1125$$

If we update our parameters using a learning rate of $\eta = 0.1$, the new weights will be:
*   $w_1 \leftarrow w_1 - (\eta \times \frac{\partial L}{\partial w_1}) = 0.8 - (0.1 \times -0.1687) \approx 0.8169$
*   $w_2 \leftarrow w_2 - (\eta \times \frac{\partial L}{\partial w_2}) = 1.2 - (0.1 \times -0.4377) \approx 1.2438$

---

## Plain Python Implementations (No Libraries Required)

### 1. Simple Single-Neuron Backprop Walkthrough

This script demonstrates manual backpropagation calculation on a single input, single weight connection without activation.

```python
# Set initial values for input feature x, weight w, and ground-truth target label y_true
x, w, y_true = 10, 0.5, 7

# Forward Pass
# Calculate prediction y_hat by multiplying weight w and input x
y_hat = w * x
# Calculate the prediction error (difference between target and prediction)
error = y_true - y_hat
# Compute squared loss of the error term
loss = error**2

# Backward Pass (Chain Rule)
# Calculate local gradient of Loss with respect to error (d/de [error^2] = 2 * error)
dL_de = 2 * error
# Calculate local gradient of error with respect to prediction y_hat (d/dy_hat [y_true - y_hat] = -1)
de_dyhat = -1
# Calculate local gradient of prediction y_hat with respect to weight w (d/dw [w * x] = x)
dyhat_dw = x

# Final Gradient dL/dw
# Multiply the local gradients together using the chain rule to get the total gradient of Loss with respect to weight w
dL_dw = dL_de * de_dyhat * dyhat_dw
# Print the final calculated gradient value to check direction and magnitude
print(f"Gradient: {dL_dw}")
```

### 2. Multi-Layer Backprop Implementation from Scratch

This script implements the exact 2-layer neural network described in our mathematical walkthrough above.

```python
import math

# Sigmoid activation and its derivative
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(a):
    # 'a' is already the sigmoid output: h = sigmoid(z)
    return a * (1 - a)

class SimpleTwoLayerNN:
    def __init__(self):
        # Initialize weights and biases matching our manual walkthrough
        self.w1 = 0.8
        self.b1 = 0.1
        self.w2 = 1.2
        self.b2 = -0.5
        
    def forward(self, x):
        self.x = x
        # Layer 1 (Hidden Layer)
        self.z1 = self.w1 * self.x + self.b1
        self.h = sigmoid(self.z1)
        
        # Layer 2 (Output Layer)
        self.z2 = self.w2 * self.h + self.b2
        self.y_hat = self.z2
        return self.y_hat

    def backward(self, y_true):
        # 1. Output Layer upstream gradient (dL/dy_hat)
        # Loss L = 0.5 * (y_hat - y_true)^2 -> dL/dy_hat = y_hat - y_true
        dL_dyhat = self.y_hat - y_true
        
        # 2. Gradients for Layer 2 weights & biases
        self.dL_dw2 = dL_dyhat * self.h
        self.dL_db2 = dL_dyhat
        
        # 3. Propagate gradient back to hidden layer activation
        # dL/dh = dL/dy_hat * dy_hat/dh = dL/dy_hat * w2
        dL_dh = dL_dyhat * self.w2
        
        # 4. Hidden Layer local gradient
        # dL/dz1 = dL/dh * dh/dz1 = dL/dh * sigmoid_derivative(h)
        dL_dz1 = dL_dh * sigmoid_derivative(self.h)
        
        # 5. Gradients for Layer 1 weights & biases
        self.dL_dw1 = dL_dz1 * self.x
        self.dL_db1 = dL_dz1

    def update_weights(self, lr):
        self.w1 -= lr * self.dL_dw1
        self.b1 -= lr * self.dL_db1
        self.w2 -= lr * self.dL_dw2
        self.b2 -= lr * self.dL_db2

# Execute the step and print results
nn = SimpleTwoLayerNN()
prediction = nn.forward(1.5)
nn.backward(1.0)

print("\n--- 2-Layer Neural Network Manual Backpropagation Verification ---")
print(f"Forward Pass Prediction (y_hat): {prediction:.4f}")
print(f"Gradient dL_dw2: {nn.dL_dw2:.4f} (Expected: -0.4377)")
print(f"Gradient dL_db2: {nn.dL_db2:.4f} (Expected: -0.5570)")
print(f"Gradient dL_dw1: {nn.dL_dw1:.4f} (Expected: -0.1687)")
print(f"Gradient dL_db1: {nn.dL_db1:.4f} (Expected: -0.1125)")

# Perform single update step
nn.update_weights(0.1)
print("Updated parameters:")
print(f"  w1: {nn.w1:.4f} (Expected: 0.8169)")
print(f"  w2: {nn.w2:.4f} (Expected: 1.2438)")
```

---

## Real-Life Framework Implementation

In production machine learning libraries, we write the forward pass and let the framework construct the computation graph and compute gradients using **Automatic Differentiation (Autodiff)**.

### 1. In PyTorch (Dynamic Graph & `backward()`)

PyTorch uses a **Define-by-Run** (eager) paradigm. It builds the graph dynamically during the forward pass. Setting `requires_grad=True` triggers tracking.

```python
import torch

# Define inputs and target
x = torch.tensor(1.5)
y_true = torch.tensor(1.0)

# Initialize weights and biases with gradient tracking enabled
w1 = torch.tensor(0.8, requires_grad=True)
b1 = torch.tensor(0.1, requires_grad=True)
w2 = torch.tensor(1.2, requires_grad=True)
b2 = torch.tensor(-0.5, requires_grad=True)

# Forward Pass (operations are automatically added to the graph)
z1 = w1 * x + b1
h = torch.sigmoid(z1)
y_hat = w2 * h + b2
loss = 0.5 * (y_hat - y_true)**2

# Backward Pass (PyTorch traces the graph in reverse to calculate gradients)
loss.backward()

# Access computed gradients
print("--- PyTorch Autodiff Gradients ---")
print(f"w1 gradient (dL_dw1): {w1.grad.item():.4f}")
print(f"w2 gradient (dL_dw2): {w2.grad.item():.4f}")
```

### 2. In TensorFlow / Keras (Using `tf.GradientTape`)

TensorFlow uses a context manager called `tf.GradientTape` to record mathematical operations on variables and calculate gradients.

```python
import tensorflow as tf

# Define inputs and target
x = tf.constant(1.5)
y_true = tf.constant(1.0)

# Initialize variables
w1 = tf.Variable(0.8)
b1 = tf.Variable(0.1)
w2 = tf.Variable(1.2)
b2 = tf.Variable(-0.5)

# Record operations on the tape
with tf.GradientTape() as tape:
    z1 = w1 * x + b1
    h = tf.sigmoid(z1)
    y_hat = w2 * h + b2
    loss = 0.5 * (y_hat - y_true)**2

# Compute gradients of loss with respect to variables w1 and w2
gradients = tape.gradient(loss, [w1, w2])

print("--- TensorFlow GradientTape Gradients ---")
print(f"w1 gradient (dL_dw1): {gradients[0].numpy():.4f}")
print(f"w2 gradient (dL_dw2): {gradients[1].numpy():.4f}")
```

---

### 💡 Beginner's Analogy: The Corporate Blame Chain

How does a neural network actually backpropagate errors? Think of it like a **large corporation** working on a big project that fails:

1.  **The Mistake (Output Layer Error):**
    The final product has a major bug, causing a $4,000 loss. The CEO (the **Loss Function**) is furious and demands to know who is responsible.
    *   *Math Connection*: The final loss $L$ and error vector $\delta_2$ determine the size and direction of the initial upstream gradient.

2.  **Tracing the Blame (Upstream Gradient):**
    The CEO asks the VP of Product (the **Output Layer**). The VP says, "We just packaged what the Engineering Director gave us. Because they handed us bad code, they are 80% responsible for the bug."
    *   *Math Connection*: The VP passes blame upstream to the Director by multiplying the error by the weight connecting them: $\frac{\partial L}{\partial h} = \delta_2 \times w_2$.

3.  **Multiplying Local Errors (The Chain Rule):**
    The CEO walks down to the Engineering Department and talks to the Engineering Director (the **Hidden Layer**). The Director says, "I was just executing the instructions from the Senior Developer. Because their instruction code had a major logic error, it multiplied my department's mistake by a factor of 10."
    *   *Math Connection*: To find the Senior Developer's total responsibility (weight gradient $\frac{\partial L}{\partial w_1}$), we **multiply** the CEO's frustration ($\delta_2 \times w_2$) by the Director's instruction multiplier ($\frac{\partial h}{\partial z_1}$) by the Developer's local contribution (input $x$).
    
    $$\frac{\partial L}{\partial w_1} = \underbrace{(\delta_2 \times w_2)}_{\text{Upstream Blame}} \times \underbrace{\frac{\partial h}{\partial z_1}}_{\text{Local Activation Gradient}} \times \underbrace{x}_{\text{Local Input Gradient}}$$

By using this chain of blame, the company (network) can trace the final error all the way back to the exact employee (the weights in early layers) who made a typo, and adjust their work (update weights) accordingly.

---

### 💡 Supplementary Notes

*   **Automatic Differentiation (Autodiff):** Modern frameworks build a computational graph of all operations during the forward pass. Backpropagation is a reverse-mode automatic differentiation that computes derivatives from the final output node back to the inputs in a single traversal, which is computationally highly efficient when there are many parameters.

---

## Active Recall Checkpoint

### The Gradient Flow

If one derivative in the chain is zero, what happens to the gradients of all layers closer to the input?

<details>
<summary><b>Reveal Answer</b></summary>

If any derivative in the chain is zero, the entire product becomes **zero** (since anything multiplied by zero is zero). 

This completely blocks the gradient from flowing further backward. Consequently:
1. All layers closer to the input (upstream of the zero derivative) will receive a gradient of zero.
2. Their weights and biases will **not be updated** during gradient descent.
3. This is the core issue behind **"Dead Neurons"** (e.g., when a ReLU neuron outputs zero for all inputs, its gradient becomes permanently zero) and is an extreme case of the **Vanishing Gradient** problem.
</details>

### Calculus Intuition

In the chain rule $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$, which part represents the "hidden layer" and which part represents the "output"?

<details>
<summary><b>Reveal Answer</b></summary>

*   **$y$ represents the Output / Loss:** It is the final quantity we want to measure changes in (the error).
*   **$u$ represents the Hidden Layer Activation:** It is the intermediate variable that sits between the input $x$ and the output $y$.
*   **$x$ represents the Input / Weight:** It is the independent variable at the beginning of the chain that we want to differentiate against.

By multiplying the rate of change of the output with respect to the hidden state ($\frac{dy}{du}$) by the rate of change of the hidden state with respect to the input ($\frac{du}{dx}$), we bypass the hidden layer to find the direct impact of the input on the output ($\frac{dy}{dx}$).
</details>