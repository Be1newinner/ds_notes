# Tutorial 15: Chain Rule & Backpropagation

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=5ogmEkujoqE)

## Executive Summary

Backpropagation is simply the application of the **Chain Rule** to find the gradient of the loss function with respect to every weight in a multi-layered network. This allows us to "assign credit" for error to specific parameters.

## Technical Deep Dive: The Chain of Derivatives

### 1. Local vs. Global Gradients

Each neuron only needs to know its "Local Gradient" (how its output changes relative to its input). Backprop multiplies this local gradient by the "Upstream Gradient" flowing back from the error.
∂Loss / ∂w = (∂Loss / ∂y)Upstream × (∂y / ∂z)Activation × (∂z / ∂w)Local

### 2. The Danger of Deep Chains

In a network with 100 layers, we multiply 100 derivatives together. If these derivatives are small (like in Sigmoid), the product becomes microscopic (Vanishing Gradient). If they are large, the product explodes (Exploding Gradient).
This is why modern architectures use **ReLU** and **Skip Connections (ResNets)** to maintain gradient flow.

## Manual Backprop Walkthrough

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

### 💡 Beginner's Analogy: The Corporate Blame Chain

How does a neural network actually backpropagate errors? Think of it like a **large corporation** working on a big project that fails:

1. **The Mistake (Output Layer Error)**:
   The final product has a major bug, causing a $4,000 loss. The CEO (the Loss Function) is furious and demands to know who is responsible.

2. **Tracing the Blame (Upstream Gradient)**:
   The CEO asks the VP of Product (the Output Layer). The VP says, "We just packaged what the Engineering Director gave us. They are 80% responsible for the bug, and Design is 20% responsible." 
   * This division of responsibility represents the **Upstream Gradient** being distributed.

3. **Multiplying local errors (The Chain Rule)**:
   The CEO walks down to the Engineering Department and talks to the Engineering Director (Hidden Layer). The Director says, "I was just executing the instructions from the Senior Developer. Because their instruction code had a major logic error, it multiplied my department's mistake by a factor of 10."
   * To find the Senior Developer's total responsibility, we **multiply** the CEO's frustration $\times$ the Director's hand-off factor $\times$ the Developer's local mistake.
   * Mathematically: $\frac{\partial Loss}{\partial w_{dev}} = \text{Upstream Error} \times \text{Director's local gradient} \times \text{Developer's local gradient}$.

By using this chain of blame, the company (network) can trace the final error all the way back to the exact employee (the weights in early layers) who made a typo, and adjust their work (update weights) accordingly.

---

### 💡 Supplementary Notes

* **Automatic Differentiation (Autodiff)**: Frameworks build a computational graph of all operations during the forward pass. Backpropagation is a reverse-mode automatic differentiation that computes derivatives from the final output node back to the inputs in a single traversal.

## Active Recall Checkpoint

The Gradient Flow

If one derivative in the chain is zero, what happens to the gradients of all layers closer to the input?

Calculus Intuition

In the chain rule dy/dx = dy/du · du/dx, which part represents the "hidden layer" and which part represents the "output"?