# Lesson 5: Backpropagation & Optimizers

## Introduction & The "Why"

In Lesson 4, we explored loss landscapes and saw how objective functions define high-dimensional geometric surfaces representing model error. However, defining a loss function is only half the battle. To train a neural network, we must have an efficient method to calculate the gradient of this loss with respect to every single parameter (weight and bias) in the network, and an optimization algorithm to navigate the landscape and find the optimal parameters.

If we attempted to estimate gradients numerically using finite differences:
$$\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w)}{\epsilon}$$
we would have to run a complete forward pass through the entire network for every single weight. For a modern network with millions of parameters, training would be impossibly slow.

To solve this, deep learning relies on **Backpropagation**, an algorithm that calculates exact analytical gradients in a single backward pass. Once these gradients are calculated, an **Optimizer** updates the weights. This lesson covers the mathematics of the chain rule in backpropagation, explores Stochastic Gradient Descent (SGD) with Momentum, and analyzes the inner workings of adaptive optimizers like RMSprop and Adam, which serve as the standard training engines for modern neural networks.

---

## Topic 1: The Chain Rule & Backpropagation: Tracing Errors Backwards

### Rationale and Mechanics
In classical machine learning, we optimize parameters by taking the derivative of the loss function directly. In multi-layer neural networks, weights are nested deep within composition functions:
$$f(\mathbf{x}) = f^{(L)}(f^{(L-1)}(\dots f^{(1)}(\mathbf{x})))$$

To calculate derivatives of these nested functions, we use the **Chain Rule** of calculus. Backpropagation is a dynamic programming algorithm that applies the chain rule recursively, propagating error signals backward from the output layer to the input layer.

Under the hood, let's trace the math for a single neuron $j$ in layer $l$. The pre-activation value $z_j^{(l)}$ is:
$$z_j^{(l)} = \sum_k w_{j,k}^{(l)} a_k^{(l-1)} + b_j^{(l)}$$
This is passed to the activation function to obtain the activation:
$$a_j^{(l)} = g(z_j^{(l)})$$

Let $L$ be the total loss. We want to find the gradient of the loss with respect to the weight $w_{j,k}^{(l)}$. Applying the chain rule:
$$\frac{\partial L}{\partial w_{j,k}^{(l)}} = \frac{\partial L}{\partial a_j^{(l)}} \cdot \frac{\partial a_j^{(l)}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{j,k}^{(l)}}$$

To simplify, we define the error term (or delta) $\delta_j^{(l)}$ as the partial derivative of the loss with respect to the pre-activation:
$$\delta_j^{(l)} = \frac{\partial L}{\partial z_j^{(l)}} = \frac{\partial L}{\partial a_j^{(l)}} \cdot g'(z_j^{(l)})$$

Since $\frac{\partial z_j^{(l)}}{\partial w_{j,k}^{(l)}} = a_k^{(l-1)}$, we can write the gradient for the weight as:
$$\frac{\partial L}{\partial w_{j,k}^{(l)}} = \delta_j^{(l)} a_k^{(l-1)}$$
Similarly, since $\frac{\partial z_j^{(l)}}{\partial b_j^{(l)}} = 1$, the gradient for the bias is:
$$\frac{\partial L}{\partial b_j^{(l)}} = \delta_j^{(l)}$$

To propagate this error backward to the previous layer $l-1$, we apply the chain rule again:
$$\delta_k^{(l-1)} = \frac{\partial L}{\partial z_k^{(l-1)}} = \sum_j \left( \frac{\partial L}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial a_k^{(l-1)}} \right) \cdot g'(z_k^{(l-1)})$$
$$\delta_k^{(l-1)} = \left( \sum_j \delta_j^{(l)} w_{j,k}^{(l)} \right) \cdot g'(z_k^{(l-1)})$$

This recursive relationship is the core of backpropagation. We start at the output layer $L$, calculate $\delta^{(L)}$, and then use the weight matrices to project these deltas backward, layer-by-layer, calculating all derivatives in a single backward pass. This pass scales linearly with the number of weights, $O(M)$, making training deep networks feasible.

```
       Forward Pass:  x ---> [Layer 1] ---> [Layer 2] ---> y_hat ---> Loss
       
       Backward Pass: dx <--- [Layer 1] <--- [Layer 2] <--- dy_hat <--- Loss
                                 |              |
                              dw1, db1       dw2, db2
```

### Trade-offs
Backpropagation is highly efficient, but it has a significant memory trade-off. To compute the weight gradient $\frac{\partial L}{\partial w_{j,k}^{(l)}} = \delta_j^{(l)} a_k^{(l-1)}$, we need the activation $a_k^{(l-1)}$ from the forward pass. 

Consequently, the network must cache (store in GPU memory) all intermediate activations calculated during the forward pass until the backward pass is complete. For deep networks with large batch sizes, this activation cache dominates GPU memory usage, often leading to "Out of Memory" (OOM) errors. To mitigate this, techniques like **Gradient Checkpointing** are used, which discard intermediate activations and recompute them on the fly when needed, trading compute time for memory space.

### Real-World Applications (Rule of 4)

1. **Example 1: Single Weight Gradient Calculation**
   - **Input/Scenario:** In a neural network, a hidden neuron outputs activation $a_k^{(l-1)} = 0.8$. The pre-activation derivative of the next layer is $\delta_j^{(l)} = -0.5$.
   - **Expected Output:** The gradient of the loss with respect to the connecting weight is $\frac{\partial L}{\partial w_{j,k}^{(l)}} = \delta_j^{(l)} a_k^{(l-1)} = -0.5 \cdot 0.8 = -0.4$. This tells the optimizer to increase this weight to reduce the loss.
2. **Example 2: Backpropagation of Error Deltas**
   - **Input/Scenario:** The error delta at layer $l$ is $\delta^{(l)} = [-0.6, 0.4]^T$. The connecting weight matrix is $\mathbf{W}^{(l)} = \begin{pmatrix} 0.5 & -0.2 \\ 0.1 & 0.3 \end{pmatrix}$. The pre-activation derivative of neuron $1$ in the previous layer is $g'(z_1^{(l-1)}) = 0.25$.
   - **Expected Output:** The propagated delta is:
     $$\delta_1^{(l-1)} = \left( \delta_1^{(l)} w_{1,1}^{(l)} + \delta_2^{(l)} w_{2,1}^{(l)} \right) \cdot g'(z_1^{(l-1)}) = (-0.6(0.5) + 0.4(0.1)) \cdot 0.25 = (-0.3 + 0.04) \cdot 0.25 = -0.065$$
3. **Example 3: Bias Gradient Calculation**
   - **Input/Scenario:** The calculated pre-activation error delta for a hidden neuron is $\delta_j^{(l)} = 2.3$.
   - **Expected Output:** The gradient of the loss with respect to the bias is $\frac{\partial L}{\partial b_j^{(l)}} = \delta_j^{(l)} = 2.3$. The bias is updated directly by this error signal.
4. **Example 4: Output Layer Delta (MSE case)**
   - **Input/Scenario:** A regression network output layer has a linear activation ($g'(z) = 1.0$) and uses MSE loss. The target value is $y = 5.0$ and the prediction is $\hat{y} = 4.0$.
   - **Expected Output:** The output delta is $\delta^{(L)} = \frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot g'(z) = -(y - \hat{y}) \cdot 1.0 = -(5.0 - 4.0) = -1.0$. This negative error delta is propagated backward to update the network's weights.

> **Metacognitive Checkpoint:** Why does backpropagation require storing activations during the forward pass? If we ran out of GPU memory during a large training job, how would reducing the batch size or implementing gradient checkpointing affect the memory vs compute trade-off?

---

## Topic 2: Stochastic Gradient Descent (SGD) with Momentum

### Rationale and Mechanics
In classical optimization, we calculate gradients over the entire dataset (Batch Gradient Descent). For large datasets, this is computationally impossible. In deep learning, we estimate the gradient using a small, randomly selected subset of data called a mini-batch (Stochastic Gradient Descent).

Under the hood, the standard SGD update rule is:
$$\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$$
where $\eta$ is the learning rate. Because the gradient is calculated over a mini-batch, it contains noise, causing the optimization path to oscillate wildly.

To stabilize training, we introduce **Momentum**. This technique simulates a physical ball rolling down a hill: the ball accumulates velocity from gravity, allowing it to roll through flat areas and escape shallow valleys.

Mathematically, we maintain a velocity vector $\mathbf{v}$ that acts as a moving average of the gradients:
$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) \nabla_\theta L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \mathbf{v}_t$$
where $\beta \in [0, 1)$ is the momentum decay factor (typically set to $0.9$).

```
       SGD:          Oscillates wildly in narrow valleys
                     \   /\   /\   /
                      \/  \/  \/  / ---> Optimal
                      
       Momentum:     Dampens oscillations and speeds straight down
                     ------------------------------> Optimal
```

When gradients point in consistent directions, the velocity vector grows, accelerating training. When gradients oscillate back and forth (e.g., across the walls of a narrow valley), the momentum term averages them out, dampening the oscillations and focusing the updates down the center of the valley.

### Trade-offs
SGD with Momentum is simple, memory-efficient, and often yields excellent generalization. Research shows that SGD can converge to flatter minima than more complex optimizers, resulting in better test accuracy.

The trade-off is that SGD is highly sensitive to the learning rate $\eta$ and momentum factor $\beta$. If the learning rate is too high, the model will diverge. If it is too low, training will take too long. Additionally, SGD uses a single, global learning rate for all parameters in the network. If some weights require large updates (e.g., sparse features) while others require small updates, SGD will struggle to optimize them simultaneously. This sensitivity requires extensive hyperparameter tuning and learning rate scheduling.

### Real-World Applications (Rule of 4)

1. **Example 1: Gradient Acceleration**
   - **Input/Scenario:** An optimizer descends a consistent slope where the gradient is $\nabla_\theta L = -2.0$ at every step. The learning rate is $\eta = 0.1$, and momentum is $\beta = 0.9$. Initial velocity is $v_0 = 0$.
   - **Expected Output:**
     - Step 1: $v_1 = 0.9(0) + 0.1(-2.0) = -0.2$. Weight update is $- \eta v_1 = +0.02$.
     - Step 2: $v_2 = 0.9(-0.2) + 0.1(-2.0) = -0.38$. Weight update is $- \eta v_2 = +0.038$.
     The velocity accumulates, accelerating the step size along the consistent gradient direction.
2. **Example 3: Dampening Oscillations in Valleys**
   - **Input/Scenario:** An optimizer oscillates across a valley. The gradient along the oscillation axis is $+5.0$ at step 1 and $-5.0$ at step 2. Momentum is $\beta = 0.9$ and initial velocity is $v_0 = 0$.
   - **Expected Output:**
     - Step 1: $v_1 = 0.9(0) + 0.1(5.0) = 0.5$.
     - Step 2: $v_2 = 0.9(0.5) + 0.1(-5.0) = 0.45 - 0.5 = -0.05$.
     The velocity vector has dropped close to zero, preventing the optimizer from oscillating wildly across the valley walls.
3. **Example 3: Rolling Over a Local Maximum**
   - **Input/Scenario:** A model rolls down a hill and encounters a small bump (local maximum) of width $1.0$ where the gradient temporarily opposes its direction of motion.
   - **Expected Output:** Because the velocity vector is large, the momentum term carries the parameters up and over the bump, escaping a potential stall that would stop standard SGD.
4. **Example 4: Escaping Saddle Point Plateau**
   - **Input/Scenario:** An optimizer enters a flat saddle point region where the gradient magnitude drops to zero.
   - **Expected Output:** Standard SGD stops moving. SGD with Momentum uses its accumulated velocity to roll through the flat region until it finds the exit slope, continuing optimization.

> **Metacognitive Checkpoint:** How does the momentum parameter $\beta$ act as a low-pass filter on the gradient steps? Express the update rule as an infinite sum of historical gradients to show how older gradients are exponentially discounted.

---

## Topic 3: Adaptive Optimizers: RMSprop & Adam

### Rationale and Mechanics
In deep neural networks, different features have different frequencies and scales. A global learning rate is inefficient: sparse, infrequent features need larger steps to learn, while frequent features need smaller, more stable updates. To address this, we use adaptive learning rate optimizers like **RMSprop** and **Adam** (Adaptive Moment Estimation), which adjust the learning rate for each individual parameter based on its historical gradients.

Under the hood, **RMSprop** keeps a moving average of the *squared* gradients for each parameter:
$$\mathbf{s}_t = \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$
where $\mathbf{g}_t = \nabla_\theta L(\theta_t)$ is the current gradient, and $\beta_2$ is the decay rate (typically $0.99$). 

When updating the weights, the gradient is scaled by the square root of this moving average:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon} \mathbf{g}_t$$
where $\epsilon$ is a tiny constant (e.g., $10^{-8}$) to prevent division by zero. If a parameter has large, frequent gradients, its corresponding $s_t$ is large, which scales down its learning rate. If a parameter has small, sparse gradients, its $s_t$ is small, which increases its learning rate.

**Adam** combines the principles of Momentum and RMSprop. It tracks both the first moment $\mathbf{m}_t$ (moving average of gradients) and the second moment $\mathbf{s}_t$ (moving average of squared gradients):
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$
$$\mathbf{s}_t = \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$

Because $\mathbf{m}_t$ and $\mathbf{s}_t$ are initialized as zero vectors, they are biased toward zero, especially during early training steps. To correct this, we compute bias-corrected estimates:
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$$
$$\hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}$$
where $t$ is the current training step. The final Adam parameter update is:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon} \hat{\mathbf{m}}_t$$
Standard defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$.

### Trade-offs
Adam is the default optimizer for most deep learning architectures (including Transformers and CNNs) because it is highly robust and requires very little hyperparameter tuning. It handles noisy gradients, sparse data, and non-stationary objectives extremely well.

The primary trade-off is memory footprint. For every parameter in the network, Adam must store both the first moment $\mathbf{m}_t$ and the second moment $\mathbf{s}_t$ in GPU memory. This triples the memory requirement for model parameters:
- A model with 100 million parameters requires 400 MB of memory to store its weights (in 32-bit float).
- Storing the gradients, first moments, and second moments requires an additional 1.2 GB of GPU memory, significantly reducing the maximum batch size that can be trained on a single GPU.

### Real-World Applications (Rule of 4)

1. **Example 1: RMSprop Parameter-Specific Scaling**
   - **Input/Scenario:** Two weights in a network have gradients $g_1 = 10.0$ (highly active) and $g_2 = 0.1$ (inactive). The running averages of squared gradients are $s_1 = 100.0$ and $s_2 = 0.01$. The learning rate is $\eta = 0.001$.
   - **Expected Output:** The effective learning rates are:
     - Weight 1: $\frac{\eta}{\sqrt{s_1}} = \frac{0.001}{10.0} = 0.0001$. The update step is $-0.0001 \cdot 10.0 = -0.001$.
     - Weight 2: $\frac{\eta}{\sqrt{s_2}} = \frac{0.001}{0.1} = 0.01$. The update step is $-0.01 \cdot 0.1 = -0.001$.
     The step sizes are balanced automatically, preventing Weight 1 from diverging and ensuring Weight 2 continues to learn.
2. **Example 2: Adam Bias Correction in Step 1**
   - **Input/Scenario:** In the first training step ($t=1$), the gradient is $g_1 = 1.0$. The parameters are $\beta_1 = 0.9$ and $\beta_2 = 0.999$. Initial moments are $m_0 = 0, s_0 = 0$.
   - **Expected Output:** The uncorrected moments are $m_1 = 0.1(1.0) = 0.1$ and $s_1 = 0.001(1.0^2) = 0.001$.
     The bias-corrected moments are:
     $$\hat{m}_1 = \frac{0.1}{1 - 0.9^1} = \frac{0.1}{0.1} = 1.0$$
     $$\hat{s}_1 = \frac{0.001}{1 - 0.999^1} = \frac{0.001}{0.001} = 1.0$$
     Without bias correction, the update step would be scaled down by a factor of 100, stalling the early phase of training.
3. **Example 3: Natural Language Processing (Sparse Features)**
   - **Input/Scenario:** A language model processes rare vocabulary words that appear in only 1 out of 10,000 batches.
   - **Expected Output:** While SGD would make negligible updates to these sparse word embeddings, Adam scales up their learning rate (via a small $s_t$), ensuring the model learns representational details for rare words whenever they appear.
4. **Example 4: Escaping Pathological Curvature with Adam**
   - **Input/Scenario:** A model enters a narrow canyon landscape where the gradient is steep along the walls and flat along the canyon floor.
   - **Expected Output:** Adam's second moment term $\mathbf{s}_t$ scales down the updates along the steep wall directions, while the momentum term $\mathbf{m}_t$ accumulates velocity along the floor direction, allowing the optimizer to glide smoothly down the canyon floor.

> **Metacognitive Checkpoint:** Why does Adam require a bias correction step? Write down what would happen to the effective learning rate during the first few training steps if we omitted the bias correction terms $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{s}}_t$.

---

## Summary & Next Steps

- **Backpropagation Uses the Chain Rule:** It calculates analytical gradients in a single backward pass, saving computational time but requiring intermediate activations to be cached in memory.
- **Momentum Prevents Oscillations:** SGD with Momentum dampens noisy updates and accumulates velocity to slide through saddle points and flat plateaus.
- **Adaptive Optimizers Adjust per Parameter:** RMSprop and Adam adjust learning rates individually for each weight based on historical gradients, accelerating convergence at the cost of a higher memory footprint.

In the next lesson, we will explore **The Keras Sequential API**, shifting from mathematical foundations to practical code, building our first Multi-Layer Perceptron, and learning how to translate data preprocessing into tensor structures.
