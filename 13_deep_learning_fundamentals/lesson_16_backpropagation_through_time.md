# Lesson 16: Backpropagation Through Time (BPTT)

## Introduction & The "Why"

In Lesson 15, we explored the architecture of Vanilla Recurrent Neural Networks (RNNs) and saw how they process sequences sequentially using shared weight matrices. However, understanding the forward pass is only the first step. To train these networks, we must compute gradients and update their parameters.

To compute gradients in a recurrent network, we use **Backpropagation Through Time (BPTT)**. BPTT unrolls the recurrent connections over the temporal dimension, transforming the recurrent loop into a deep feedforward network where each layer represents a single timestep. 

While BPTT is mathematically elegant, it exposes a fundamental flaw in Vanilla RNNs. Stacking weights across time is mathematically equivalent to multiplying the same weight matrix repeatedly. In deep sequences, this repeated multiplication causes gradients to either explode to infinity or vanish to zero. This makes Vanilla RNNs unable to learn long-term dependencies—the model "forgets" early inputs after only a few timesteps. This lesson details the mathematics of BPTT, explains the causes of vanishing and exploding gradients, and introduces Gradient Clipping as a solution for exploding gradients.

---

## Topic 1: Unrolling the Computation Graph and BPTT

### Rationale and Mechanics
To calculate gradients in an RNN, we must map its temporal loops to a static structure. We achieve this by **Unrolling** (or unfolding) the computational graph. An unrolled RNN is a feedforward network where the number of layers corresponds to the sequence length $T$.

```
       Recurrent Representation:                  Unrolled Representation:
       
                                             y_1           y_2           y_T
                                              ^             ^             ^
                                              |             |             |
            [ RNN Layer ] <----\        h_0 ->[ h_1 ]------->[ h_2 ]------->[ h_T ]
              ^              |                ^             ^             ^
              |              |                |             |             |
             x_t ------------/               x_1           x_2           x_T
```

Once the graph is unrolled, we calculate gradients using **Backpropagation Through Time (BPTT)**. Let's analyze the math.

Under the hood, let the total loss over the sequence be the sum of losses at each timestep:
$$L = \sum_{t=1}^T L_t$$

We want to calculate the gradient of the loss with respect to the recurrent weight matrix $\mathbf{W}_{hh}$. Because $\mathbf{W}_{hh}$ is shared across all timesteps, its gradient is the sum of its contributions at each step:
$$\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \frac{\partial L_t}{\partial \mathbf{W}_{hh}}$$

For a specific loss $L_t$ at timestep $t$, the hidden state $\mathbf{h}_t$ depends on the current input and the previous hidden state $\mathbf{h}_{t-1}$, which in turn depends on $\mathbf{h}_{t-2}$, and so on. Applying the multivariate chain rule:
$$\frac{\partial L_t}{\partial \mathbf{W}_{hh}} = \sum_{k=1}^t \frac{\partial L_t}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}$$

The middle term $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}$ measures how a change in the hidden state at timestep $k$ affects the hidden state at a later timestep $t$. It is computed as a product of partial derivative matrices (Jacobians) for each step:
$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{j=k+1}^t \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}$$

This product of Jacobians is the source of the optimization problems in recurrent networks.

### Trade-offs
BPTT calculates the exact analytical gradients for all shared weights.

The trade-off is memory. To perform BPTT over a sequence of length $T$, we must cache the hidden states $\mathbf{h}_t$ and inputs $\mathbf{x}_t$ for all $T$ timesteps in memory. For long sequences (e.g., $T > 1,000$), storing this history consumes massive GPU memory. To mitigate this, we use **Truncated BPTT (TBPTT)**: we split the sequence into smaller chunks (e.g., length 50) and only propagate gradients back within the current chunk, sacrificing long-term updates to save memory.

### Real-World Applications (Rule of 4)

1. **Example 1: Unrolled Sequence of Length 3**
   - **Input/Scenario:** An RNN processes a sequence of length $T=3$. We calculate the gradient of the loss $L_3$ at timestep 3 with respect to the recurrent weights $\mathbf{W}_{hh}$.
   - **Expected Output:** The unrolled gradient is:
     $$\frac{\partial L_3}{\partial \mathbf{W}_{hh}} = \frac{\partial L_3}{\partial \mathbf{h}_3}\frac{\partial \mathbf{h}_3}{\partial \mathbf{W}_{hh}} + \frac{\partial L_3}{\partial \mathbf{h}_3}\frac{\partial \mathbf{h}_3}{\partial \mathbf{h}_2}\frac{\partial \mathbf{h}_2}{\partial \mathbf{W}_{hh}} + \frac{\partial L_3}{\partial \mathbf{h}_3}\frac{\partial \mathbf{h}_3}{\partial \mathbf{h}_2}\frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1}\frac{\partial \mathbf{h}_1}{\partial \mathbf{W}_{hh}}$$
2. **Example 2: Memory Footprint under BPTT**
   - **Input/Scenario:** A batch of 32 sequences, each of length $T=1000$ with $D=100$ features, is trained. The hidden state dimension is $H=256$.
   - **Expected Output:** The network must cache $32 \times 1000 \times 256 = 8,192,000$ floating-point values in memory just for the hidden states, illustrating why long sequences lead to high GPU memory consumption.
3. **Example 3: Truncated BPTT in Action**
   - **Input/Scenario:** A model predicts sentence endings on a text document with 5,000 words. We configure Truncated BPTT with a window of 50 steps.
   - **Expected Output:** The model runs a forward pass over 5,000 steps but halts gradient propagation after 50 steps, keeping memory usage constant and enabling training.
4. **Example 4: Code Implementation of Gradient Truncation**
   - **Input/Scenario:** A developer configures a recurrent layer in Keras.
   - **Expected Output:** Keras does not have a direct TBPTT parameter; instead, the developer structures their training dataset into input sequences of fixed length (e.g., shape `(N, 50, D)`), forcing the computational graph to truncate BPTT to 50 steps.

> **Metacognitive Checkpoint:** Why is the gradient of the loss with respect to the recurrent weights $\mathbf{W}_{hh}$ computed as a sum of gradients across all timesteps? Relate this to the concept of parameter sharing over time.

---

## Topic 2: The Mathematical Cause of Vanishing & Exploding Gradients in RNNs

### Rationale and Mechanics
The product of Jacobians in BPTT is:
$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{j=k+1}^t \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}$$

Under the hood, let's analyze a single Jacobian matrix $\frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}$ for a Vanilla RNN using a Tanh activation function. The state equation is:
$$\mathbf{h}_j = \tanh\left( \mathbf{W}_{hh} \mathbf{h}_{j-1} + \mathbf{W}_{xh} \mathbf{x}_j + \mathbf{b}_h \right)$$

Taking the partial derivative yields:
$$\frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \text{diag}\left( 1 - \mathbf{h}_j^2 \right) \mathbf{W}_{hh}^T$$
where $\text{diag}\left( 1 - \mathbf{h}_j^2 \right)$ is a diagonal matrix containing the derivative of the Tanh function for each hidden unit.

Now, let's substitute this back into the product of Jacobians:
$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{j=k+1}^t \text{diag}\left( 1 - \mathbf{h}_j^2 \right) \mathbf{W}_{hh}^T$$

Let the largest eigenvalue (spectral radius) of the weight matrix $\mathbf{W}_{hh}$ be $\lambda_{\text{max}}$. The behavior of this product over many timesteps depends on the value of $\lambda_{\text{max}}$:
- **Vanishing Gradients ($\lambda_{\text{max}} < 1$):** Since the derivative of Tanh is bounded by $1.0$ ($\|1 - \mathbf{h}_j^2\| \le 1.0$), if the largest eigenvalue of the weight matrix is less than 1, the product will decay exponentially toward zero as the temporal distance $t - k$ increases. The gradient vanishes, meaning the loss at timestep $t$ cannot update the weights for inputs at early timestep $k$.
- **Exploding Gradients ($\lambda_{\text{max}} > 1$):** If the largest eigenvalue is greater than 1, and the activations are in their linear regions (so the Tanh derivative is close to 1.0), the product will grow exponentially, causing the gradients to explode to infinity.

```
       Temporal Distance (t - k): 1 -------> 10 -------> 50 -------> 100
       
       Vanishing Gradient (0.9^d): 0.9 ------> 0.34 -----> 0.005 ----> 0.00002 (Signal lost)
       
       Exploding Gradient (1.1^d): 1.1 ------> 2.59 -----> 117.3 ----> 13780.6 (Numerical overflow)
```

In Vanilla RNNs, vanishing gradients are the dominant failure mode. This math proves that Vanilla RNNs cannot capture dependencies longer than 10 to 20 timesteps: the network is mathematically incapable of retaining long-term memory.

### Trade-offs
The vanishing gradient problem is a systemic flaw of the Vanilla RNN architecture. No amount of regularization or learning rate tuning can solve it because the decay is built into the matrix multiplication. To solve vanishing gradients, we must change the architecture, transitioning to gated cells like LSTMs or GRUs.

### Real-World Applications (Rule of 4)

1. **Example 1: Exponential Decay Calculation**
   - **Input/Scenario:** A sequence has length 50. The effective scale of the Jacobian is constant at $0.9$.
   - **Expected Output:** The gradient is scaled by $0.9^{50} \approx 0.005$. The error signal from step 50 is reduced by $99.5\%$ by the time it reaches step 1, preventing the model from learning relationships between step 1 and step 50.
2. **Example 2: Exponential Explosion Calculation**
   - **Input/Scenario:** The effective scale of the Jacobian is constant at $1.15$.
   - **Expected Output:** The gradient is scaled by $1.15^{50} \approx 1083.6$. The gradient is amplified $1,000\times$, causing the weight updates to become unstable and leading to numerical overflow (NaN).
3. **Example 3: Long-Text Sentiment Analysis Failure**
   - **Input/Scenario:** A movie review starts with "This movie is terrible..." followed by 200 words of plot summary, ending with "...but I liked it."
   - **Expected Output:** A Vanilla RNN fails to connect the positive ending to the negative beginning because the gradient vanishes over the 200-word gap, classifying the review based only on the final sentences.
4. **Example 4: Time-Series Trend Drift**
   - **Input/Scenario:** A sales prediction model forecasts December sales using hourly records from January to November.
   - **Expected Output:** The Vanilla RNN cannot retain the seasonal trends from early months, behaving like a short-term moving average model.

> **Metacognitive Checkpoint:** Why does the derivative of the Tanh activation function ($\le 1.0$) accelerate the vanishing gradient problem in Vanilla RNNs? What would happen if we used a linear activation function instead?

---

## Topic 3: Gradient Clipping: Taming the Exploding Gradients

### Rationale and Mechanics
While vanishing gradients prevent learning, exploding gradients cause numerical instability. The weights undergo massive updates, causing the model to oscillate wildly or crash with `NaN` (Not a Number) values as weights exceed floating-point limits.

To solve this, we use **Gradient Clipping**. This technique inspects the magnitude of the gradient vector before updating the parameters. If the magnitude exceeds a threshold, we scale it down.

Under the hood, let $\mathbf{g}$ represent the gradient vector containing the partial derivatives of the loss with respect to all model parameters $\theta$:
$$\mathbf{g} = \nabla_\theta L$$

There are two primary methods for gradient clipping:
1. **Clip by Value:** We clip each element $g_i$ of the gradient vector independently to a range $[-c, c]$:
   $$g_i \leftarrow \max(-c, \min(c, g_i))$$
2. **Clip by Norm (Global Clipping):** We calculate the Euclidean norm (length) of the entire gradient vector: $\|\mathbf{g}\| = \sqrt{\sum_i g_i^2}$. If the norm exceeds a threshold $c$, we scale the entire vector:
   $$\mathbf{g} \leftarrow c \times \frac{\mathbf{g}}{\|\mathbf{g}\|}$$

```
                          Clip by Norm: Preserves Direction
                          
                                 y
                                 ^       / (Original Gradient vector)
                                 |     / 
                                 |   /   
                                 | /     (Clipped Vector: same angle,
                                 +----->   shorter length)
                                 |
                                 +------------------------> x
```

Clip by Norm is preferred because it preserves the **direction** of the gradient vector in parameter space, only scaling down the step size. Clip by Value changes the direction of the vector because it clips some coordinates while leaving others unchanged.

### Trade-offs
- **Advantages:** Gradient clipping is simple, computationally cheap, and highly effective at preventing exploding gradients. It ensures training stability even in very deep recurrent networks.
- **Disadvantages:** Gradient clipping is a heuristic. It does not solve the underlying cause of gradient instability, and it does not solve the vanishing gradient problem. It only prevents the model from crashing, leaving the model unable to learn long-term dependencies.

### Real-World Applications (Rule of 4)

1. **Example 1: Clip by Value Calculation**
   - **Input/Scenario:** A gradient vector is $\mathbf{g} = [5.0, 0.2, -10.0]^T$. We configure Clip by Value with threshold $c = 2.0$.
   - **Expected Output:** Each element is clipped independently: $\mathbf{g}_{\text{clipped}} = [2.0, 0.2, -2.0]^T$. The direction of the vector has changed.
2. **Example 2: Clip by Norm Calculation**
   - **Input/Scenario:** The same gradient vector $\mathbf{g} = [5.0, 0.2, -10.0]^T$ is processed using Clip by Norm with threshold $c = 2.0$. The norm is $\|\mathbf{g}\| = \sqrt{5.0^2 + 0.2^2 + (-10.0)^2} = \sqrt{25 + 0.04 + 100} \approx 11.18$.
   - **Expected Output:** The scaling factor is $\frac{2.0}{11.18} \approx 0.179$. The clipped vector is:
     $$\mathbf{g}_{\text{clipped}} = 0.179 \times [5.0, 0.2, -10.0]^T \approx [0.895, 0.036, -1.790]^T$$
     The direction (ratio between coordinates) is preserved.
3. **Example 3: Keras Compile Implementation**
   - **Input/Scenario:** A developer compiles an RNN model in Keras and wants to prevent numerical overflow during training.
   - **Expected Output:**
     ```python
     optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
     model.compile(optimizer=optimizer, loss='mse')
     ```
     The Keras runtime automatically scales the gradients of all layers using global norm clipping before applying weight updates.
4. **Example 4: Preventing NaN Crashes**
   - **Input/Scenario:** A sequence model regularly crashes with loss outputting `NaN` after 10 epochs.
   - **Expected Output:** Adding `clipvalue=0.5` stabilizes the loss trajectory, allowing the model to complete 100 training epochs successfully.

> **Metacognitive Checkpoint:** Why is clipping gradients by norm generally preferred over clipping by value? Explain how clipping coordinates independently alters the direction of parameter updates in the loss landscape.

---

## Summary & Next Steps

- **BPTT Unrolls Networks:** Backpropagation Through Time treats recurrent networks as deep feedforward structures, calculating gradients across unrolled timesteps.
- **Vanishing Gradients Limit Memory:** Vanilla RNNs multiply weight matrices repeatedly across time, causing gradients to decay exponentially and making the network unable to learn long-term dependencies.
- **Clipping Tames Exploding Gradients:** Gradient clipping scales down large gradients to prevent numerical overflow, stabilizing training without solving the vanishing gradient issue.

In the next lesson, we will explore **Long Short-Term Memory (LSTM) Networks**, learning how gated architectures introduce cell states and constant error carousels to solve the vanishing gradient problem.
