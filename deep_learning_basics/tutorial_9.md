# Tutorial 9: Derivatives Intuition

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=cT4pQT5Da0Q)

## Executive Summary

Derivatives are the backbone of optimization. This guide explains how we use **Partial Derivatives** and the **Chain Rule** to navigate the complex multi-dimensional error landscapes of neural networks.

## Technical Deep Dive: Multi-Variable Calculus

### 1. The Gradient Vector (∇f)

In deep learning, we don't just have one derivative. We have a **Gradient Vector**, which is a vector containing the partial derivatives for every weight (w) and bias (b) in the network.
∇f = [∂f/∂w1, ∂f/∂w2, ..., ∂f/∂b]

The gradient points in the direction of the steepest ASCENT. We move in the opposite direction (Gradient Descent).

### 2. The Chain Rule: Nested Dependencies

Neural networks are layers of nested functions: Loss(Activation(WeightedSum(Input))). To find the derivative of the Loss with respect to an early weight, we must multiply the derivatives of each step.

#### Example: y = sin(x2)

Let u = x2 and y = sin(u)

dy/dx = dy/du · du/dx

= cos(u) · 2x = 2x · cos(x2)

#### Deep Learning Mapping

∂Loss/∂w = ∂Loss/∂Output · ∂Output/∂w

This is the heart of Backpropagation.

## Numerical vs. Analytical Derivatives

```python
# Define a function to compute the exact analytical derivative of x^2 using the power rule (d/dx[x^2] = 2x)
def derivative_x2(x):
   # Return the exact rate of change, which is 2 multiplied by x
   return 2 * x

# Define a function to approximate the numerical derivative of function f at point x using a small increment h
def numerical_derivative(f, x, h=0.00001):
   # Calculate the limit quotient [f(x + h) - f(x)] / h to find the slope of the secant line
   return (f(x + h) - f(x)) / h

# Define a simple quadratic function to test our derivative calculations
def my_func(x):
   # Return the square of the input x
   return x**2

# Print the analytical derivative of x^2 at x = 5 (which evaluates to exactly 10)
print(f"Analytical at 5: {derivative_x2(5)}")
# Print the numerically approximated derivative of my_func at x = 5 using the small step h (evaluates to ~10.00001)
print(f"Numerical at 5: {numerical_derivative(my_func, 5)}")
```

### 💡 Beginner's Intuition: The Slope of the Hill

If calculus sounds scary, think of derivatives as a simple **slope** or **rate of change**:
* Imagine you are driving a car. Your position changes over time. The derivative of your position with respect to time is your **speed**. It tells you how fast and in what direction your position is changing.
* In a neural network, the derivative of the **Loss** with respect to a **Weight** ($\frac{\partial Loss}{\partial w}$) tells us how the error changes when we tweak that weight.

If $\frac{\partial Loss}{\partial w}$ is **positive** ($> 0$), it means increasing the weight increases the error. So, we should *decrease* the weight.
If $\frac{\partial Loss}{\partial w}$ is **negative** ($< 0$), it means increasing the weight decreases the error. So, we should *increase* the weight.

By calculating derivatives, the network learns exactly how to turn each "knob" (weight) to make the error as close to zero as possible.

---

### 💡 Supplementary Notes

* **Gradient Vectors**: In multi-variable optimization, the gradient vector points in the direction of the steepest rate of increase. In deep learning, we move in the opposite direction (negative gradient) to minimize the loss.

## Active Recall Checkpoint
1

If a function has 10,000 weights, how many partial derivatives are in its Gradient Vector?
2

Why can't we use a simple linear slope to find the minimum of a neural network's error surface?