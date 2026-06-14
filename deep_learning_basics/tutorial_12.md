# Tutorial 12: Mastering Gradient Descent

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=pXGBHV3y8rs)

## Executive Summary

Gradient Descent is the algorithm that makes "Learning" possible. We move from high-level abstractions to the raw mathematics of **Partial Derivatives** and **Learning Rates**, exploring why Stochastic and Mini-Batch approaches are preferred for large datasets.

## What is Gradient Descent? (In-Depth Mechanics)

### 1. The Core Intuition: The Mountain Hiker Analogy

Imagine you are blindfolded at the top of a rugged, foggy mountain, and your goal is to reach the lowest point in the valley. Since you cannot see the overall landscape, you check the slope of the ground directly beneath your feet. You take a step in the direction where the ground descends the steepest. By repeating this process iteratively, you gradually work your way down to the bottom.

In Deep Learning, the **mountain landscape** represents the *Loss Function* (the error of our neural network), your **current coordinates** represent the network's *Weights and Biases*, and the **steepest downward slope** is found using the negative of the *Gradient*.

### 2. The Mathematical Engine

Mathematically, a gradient is a vector containing all the partial derivatives of the loss function with respect to each parameter. The update rule for any single parameter w is defined as:
`
                               wnew = wold - η · (∂L / ∂w)
                           `
w (Weight) The learnable parameter we want to optimize to minimize error.
η (Learning Rate) A hyperparameter controlling the step size. Too high causes divergence; too low causes slow training.
∂L / ∂w (Gradient) The direction and magnitude of the steepest ascent. We subtract it to move downward.

### 3. Step-by-Step Numerical Example

Let’s minimize a simple quadratic loss function: `L(w) = w²`.
The derivative (gradient) with respect to w is: `dL/dw = 2w`.
Assume we start at an initial weight `w₀ = 4` with a learning rate `η = 0.1`.
Iteration 1 Loss = 16.0

* • Current Weight: w₀ = 4

* • Gradient: dL/dw = 2 × 4 = 8

* • Update: w₁ = 4 - (0.1 × 8) = 3.2
Iteration 2 Loss = 10.24

* • Current Weight: w₁ = 3.2

* • Gradient: dL/dw = 2 × 3.2 = 6.4

* • Update: w₂ = 3.2 - (0.1 × 6.4) = 2.56
Iteration 3 Loss = 6.55

* • Current Weight: w₂ = 2.56

* • Gradient: dL/dw = 2 × 2.56 = 5.12

* • Update: w₃ = 2.56 - (0.1 × 5.12) = 2.048

Notice how the weight steadily approaches the optimal value of 0, and the loss decreases at each step. As the slope becomes flatter near the minimum, the updates automatically become smaller, ensuring smooth convergence.

## Technical Deep Dive: Variants & Geography

### 1. The Three Primary Variants

Batch GD

Uses ALL data samples to calculate one gradient. Stable but slow for big data.

Stochastic (SGD)

Uses ONE sample per update. Fast and noisy—helps jump out of local minima.

Mini-Batch

Uses a small subset (e.g., 32 samples). The industry standard—balances speed and stability.

### 2. Local Minima vs. Saddle Points

In high-dimensional space, we rarely get stuck in true local minima. Instead, we face **Saddle Points**—where the slope is zero in some directions but not others. Modern optimizers like **Adam** use Momentum to "push" past these points.

## Python Implementation: Gradient Descent

```python
# Define a prediction function based on inputs, weights (coef), and bias (intercept)
def prediction_function(age, affordability):
   # Calculate the weighted sum of features using predefined coefficients and intercept
   weigthed_sum = coef[0]*age + coef[1]*affordability + intercept
   # Return the sigmoid output to convert the weighted sum into a value between 0 and 1
   return sigmoid(weigthed_sum)

# Test the prediction function with sample values (0.47 for age, 1 for affordability)
prediction_function(.47, 1)

# Test the prediction function with another sample pair (0.18 for age, 1 for affordability)
prediction_function(.18, 1)

# Define a function to compute binary crossentropy loss (log loss) between actual and predicted vectors
def log_loss(y_true, y_predicted):
   # Set a tiny epsilon value to avoid calculating log(0)
   epsilon = 1e-15
   # Replace any value smaller than epsilon in predictions with epsilon
   y_predicted_new = [max(i,epsilon) for i in y_predicted]
   # Replace any value larger than 1-epsilon in predictions with 1-epsilon
   y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
   # Convert the clipped predictions list into a NumPy array for vectorized math
   y_predicted_new = np.array(y_predicted_new)
   # Calculate and return the average binary crossentropy loss across all predictions
   return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))

# Define a function to compute sigmoid element-wise on a NumPy array
def sigmoid_numpy(X):
   # Compute the mathematical sigmoid function 1 / (1 + e^-X)
   return 1/(1+np.exp(-X))

# Test the numpy sigmoid function with an input array [12, 0, 1]
sigmoid_numpy(np.array([12,0,1]))
```

```python
# Define a gradient descent optimization function for logistic regression
def gradient_descent(age, affordability, y_true, epochs, loss_thresold):
   # Initialize both weight parameters w1 and w2 to 1
   w1 = w2 = 1
   # Initialize the bias parameter to 0
   bias = 0
   # Set the learning rate parameter (step size) to 0.5
   rate = 0.5
   # Calculate the number of training samples in the dataset
   n = len(age)

   # Iterate through the specified number of training epochs
   for i in range(epochs):
       # Calculate the weighted sum of inputs plus bias for all samples
       weighted_sum = w1 * age + w2 * affordability + bias
       # Apply the sigmoid function to get prediction probabilities between 0 and 1
       y_predicted = sigmoid_numpy(weighted_sum)

       # Calculate the current binary crossentropy loss (log loss) for this epoch
       loss = log_loss(y_true, y_predicted)

       # Calculate the gradient (derivative of loss) with respect to weight w1 using the dot product
       w1d = (1/n)*np.dot(np.transpose(age),(y_predicted-y_true))
       # Calculate the gradient (derivative of loss) with respect to weight w2
       w2d = (1/n)*np.dot(np.transpose(affordability),(y_predicted-y_true))

       # Calculate the gradient (derivative of loss) with respect to bias by averaging prediction errors
       bias_d = np.mean(y_predicted-y_true)

       # Update weight w1 by stepping in the direction of the negative gradient scaled by learning rate
       w1 = w1 - rate * w1d
       # Update weight w2 by stepping in the direction of the negative gradient scaled by learning rate
       w2 = w2 - rate * w2d
       # Update bias by stepping in the direction of the negative gradient scaled by learning rate
       bias = bias - rate * bias_d

       # Print the progress details (epoch number, weights, bias, and loss)
       print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')

       # Check if the loss has dropped below the threshold to stop training early
       if loss<=loss_thresold:
           # Break out of the training loop early if the criteria is met
           break

   # Return the final optimized weights and bias
   return w1, w2, bias
```

### 💡 Beginner's Intuition: Tuning the Step Size (Learning Rate $\eta$)

Recall the blindfolded hiker trying to find the valley. The **Learning Rate** ($\eta$) controls how large of a step the hiker takes:

* **If the step size ($\eta$) is too small (e.g., $0.00001$)**:
  The hiker takes tiny, baby steps. They will eventually reach the bottom, but it might take them days. In code, your training will take forever (slow convergence).
  
* **If the step size ($\eta$) is too large (e.g., $10.0$)**:
  The hiker takes massive leaps. They might leap right over the valley and end up on a higher peak on the other side. This is called **divergence**—the loss will start increasing or bounce back and forth wildly, and the network will never learn.
  
* **The Sweet Spot (typically between $0.001$ and $0.1$)**:
  Allows the hiker to take steady, reasonable steps that converge efficiently to the lowest point.

---

### 💡 Supplementary Notes

* **Saddle Points vs. Local Minima**: In high-dimensional optimization landscapes of deep neural networks, local minima are rare. The optimizer is much more likely to encounter **saddle points**, where the gradient is zero but eigenvalues of the Hessian have mixed signs.

## Active Recall Checkpoint

The Directionality Question

If the derivative (∂L / ∂w) is POSITIVE, does that mean the current weight is too high or too low to reach the minimum?

Mini-Batch Advantage

Why is Mini-Batch GD technically better for hardware (GPU) utilization compared to Stochastic GD?