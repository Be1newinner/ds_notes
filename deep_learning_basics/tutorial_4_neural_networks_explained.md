# Tutorial 4: Neural Networks Explained

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=ER2It2mIagI)

## Executive Summary

Neural Networks are universal function approximators. We explain this by moving from the **Biological Analogy** (neurons firing) to the **Mathematical Reality** (dot products and activation functions), showing how a "team" of nodes can solve non-linear problems that a single rule-based system cannot.

## Technical Deep Dive: The Logic of Layers

### 1. Why Hidden Layers? (Feature Hierarchy)

A single neuron can only learn a straight line (Linear Separability). Hidden layers allow the network to "fold" the coordinate space, creating complex decision boundaries.

**_ Lower Layers : Detect edges, blobs, and simple gradients. _**

**_ Higher Layers : Combine edges into shapes (eyes, tires, text)._**

### 2. The Learning Loop: Forward & Backward

Learning is a two-stroke engine:

- Forward: Prediction = σ(Σ wi xi + b). This is essentially a giant matrix multiplication followed by a non-linear "squashing" function.

- Backward: We use the **Chain Rule** to walk back from the error to every weight. The weight is updated by w = w - η · (∂Loss / ∂w).

### 💡 Beginner's Perspective: The Concert Analogy

To understand how a single neuron makes decisions, imagine you are deciding whether to go to an outdoor music concert. Your final decision (the **Output**) depends on three main factors (the **Inputs**):

1. **Weather ($x_1$)**: Is it sunny? (1 for Yes, 0 for No)
2. **Price ($x_2$)**: Is the ticket cheap? (1 for Yes, 0 for No)
3. **Friend ($x_3$)**: Is your best friend going? (1 for Yes, 0 for No)

However, these factors don't have equal importance:

- If you absolutely hate rain, the weather factor will have a very high **Weight ($w_1$)** (e.g., $6$).
- If you are wealthy, the ticket price might have a low **Weight ($w_2$)** (e.g., $2$).

So, the weighted sum of inputs is $z = (w_1 \cdot x_1) + (w_2 \cdot x_2) + (w_3 \cdot x_3)$.

Now, what is the **Bias ($b$)**? Think of it as your natural tendency to say "yes" or "no" to going out. If you are a homebody who rarely goes out, you have a negative bias (e.g., $b = -5$). To make you go, the weighted sum of inputs must overcome this negative threshold.

Finally, the **Activation Function** acts like a switch. If the final value $z + b$ is positive, it outputs "Yes, go!"; if negative, it outputs "No, stay home."

---

### 💡 Supplementary Notes

- **Universal Approximation Theorem Limits**: Although a feedforward neural network with a single hidden layer and non-linear activation functions can approximate any continuous function, it does not guarantee that gradient descent will find the optimal parameters. Furthermore, without non-linear activation functions, a network of any depth collapses into a simple linear model.

## Active Recall Checkpoint

Feature Autonomy

In traditional programming, you tell the computer "Check if pixel (10,10) is black." In Deep Learning, how does the computer "decide" which pixels to check?

Linearity vs. Non-Linearity

If we removed all activation functions (Sigmoid, ReLU) from a 100-layer network, would it be more powerful or exactly the same as a single-layer network?
