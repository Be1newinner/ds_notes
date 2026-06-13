# Lesson 8: Internal Covariate Shift & Normalization

## Introduction & The "Why"

In classical machine learning, we scale our input features (e.g., using Scikit-Learn's `StandardScaler`) to ensure they all have zero mean and unit variance. This preprocessing step ensures that the optimization landscape is balanced, preventing features with large scales from dominating the gradient updates. However, this only solves the problem for the first layer.

As activations flow deeper into a multi-layer neural network, they are multiplied by weight matrices at each layer. As training progresses and weights are updated, the distribution of inputs to deeper layers changes constantly. This shifting of intermediate distributions is called **Internal Covariate Shift**. It forces downstream layers to continuously adapt to new distributions, stalling training and requiring tiny learning rates.

To solve this, deep learning introduces **Normalization Layers**. Instead of only scaling the inputs to the first layer, we normalize the activations at *every* layer in the network. This lesson covers the mechanics of Internal Covariate Shift, details Batch Normalization (BatchNorm) for standardizing across batches, and explains Layer Normalization (LayerNorm), which standardizes across features and serves as the foundation for modern Transformer models.

---

## Topic 1: Internal Covariate Shift: The Drifting Distributions

### Rationale and Mechanics
In classical machine learning, **Covariate Shift** refers to a change in the distribution of input data between training and testing. This shift degrades model performance because the patterns learned on the training distribution no longer apply to the test distribution.

In deep neural networks, a similar phenomenon occurs *internally* during training. Each layer's output serves as the input to the next layer. Let's analyze the pre-activation of layer $l$:
$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$

As the optimizer updates the parameters of the early layers ($\mathbf{W}^{(1)}, \mathbf{W}^{(2)}, \dots$), the distribution of activations $\mathbf{a}^{(l-1)}$ shifts. This means the downstream layer $l$ is constantly chasing a moving target: it must adapt its weights $\mathbf{W}^{(l)}$ to classify features whose mean, variance, and range are continuously drifting.

This distribution drift scales exponentially with network depth. A tiny change in the weights of Layer 1 can cause massive shifts in Layer 10. This drift is particularly problematic when using saturating activation functions like Sigmoid or Tanh: if the distribution shifts into the flat regions ($|z| > 5$), the gradients vanish, and training stalls completely.

### Python Code Implementation
Here is a Python script using NumPy to simulate how activation distributions drift and scale out of control as they propagate through multiple layers without normalization:

```python
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Simulate 100 samples with 10 features (mean=0, var=1)
X = np.random.randn(100, 10)
print(f"Layer 0 (Input)  | Mean: {X.mean():7.4f} | Var: {X.var():7.4f}")

# Propagate through 3 layers with unnormalized weight transformations and ReLU
a = X
for layer in range(1, 4):
    # Initialize weights with a slight positive bias and variance > 1
    W = np.random.randn(10, 10) * 1.2 + 0.1
    b = np.random.randn(10) * 0.2
    
    # Linear projection
    z = np.dot(a, W) + b
    # ReLU activation
    a = np.maximum(0, z)
    
    print(f"Layer {layer:1d} (Outputs) | Mean: {a.mean():7.4f} | Var: {a.var():7.4f}")
```

### Trade-offs
Preventing internal covariate shift is essential for training deep architectures. By standardizing intermediate distributions, we decouple the learning of different layers. Each layer can learn independently of changes in the distributions of earlier layers.

The trade-off of not normalizing is that we are forced to use very small learning rates and precise weight initializations. Small learning rates ensure that early layers update slowly, reducing the rate of distribution shift but making training extremely slow. Implementing normalization layers allows us to use much larger learning rates, accelerating convergence by orders of magnitude.

### Real-World Applications (Rule of 4)

1. **Example 1: Gradient Flatlining in Deep Networks**
   - **Input/Scenario:** A 10-layer network uses Tanh activations without normalization. During early training, a weight update in Layer 1 shifts the mean pre-activation of Layer 8 from $0.0$ to $6.0$.
   - **Expected Output:** The activations of Layer 8 saturate near $1.0$ ($\tanh(6.0) \approx 0.9999$). The derivatives of the activations collapse to zero ($\tanh'(6.0) \approx 0.0001$), causing backpropagation gradients to vanish and stalling training.
2. **Example 2: Decoupled Layer Learning**
   - **Input/Scenario:** A normalization layer is inserted between Layer 1 and Layer 2. Layer 1's weights are updated, shifting its raw output mean from $10.0$ to $50.0$.
   - **Expected Output:** The normalization layer standardizes the output back to a mean of $0.0$ and variance of $1.0$ before it reaches Layer 2. Layer 2's weights do not need to adjust to the new scale, keeping training stable.
3. **Example 3: Learning Rate Acceleration**
   - **Input/Scenario:** We train a deep CNN. Without normalization, we must set the learning rate to $\eta = 10^{-4}$ to prevent gradients from exploding due to distribution shift.
   - **Expected Output:** Adding normalization layers stabilizes the intermediate distributions, allowing us to increase the learning rate to $\eta = 10^{-2}$ ($100\times$ larger), accelerating training speed.
4. **Example 4: Initialization Sensitivity**
   - **Input/Scenario:** A model is initialized with random weights. Without normalization, if weights are slightly too large, activations scale up layer-by-layer, leading to numerical overflow (NaN).
   - **Expected Output:** Normalization layers reset the scale at each layer, preventing exponential scaling and allowing the network to train successfully even with sub-optimal weight initialization.

> **Metacognitive Checkpoint:** How does internal covariate shift explain why deep neural networks without normalization are highly sensitive to weight initialization? Connect this to the exponential scaling of activations across layers.

---

## Topic 2: Batch Normalization (BatchNorm): Standardizing Across the Batch

### Rationale and Mechanics
To prevent internal covariate shift, Sergey Ioffe and Christian Szegedy introduced **Batch Normalization (BatchNorm)** in 2015. BatchNorm standardizes the activations of each neuron across the mini-batch dimension.

Under the hood, let's consider a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$ representing the pre-activations of a single neuron for $m$ samples. BatchNorm normalizes these values using the following steps:
1. **Compute Batch Mean:**
   $$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^m x_i$$
2. **Compute Batch Variance:**
   $$\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2$$
3. **Normalize:**
   $$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$
   where $\epsilon$ is a tiny constant (e.g., $10^{-5}$) to prevent division by zero.
4. **Scale and Shift:**
   $$y_i = \gamma \hat{x}_i + \beta$$
   where $\gamma$ and $\beta$ are **learnable parameters** unique to each neuron.

The scale ($\gamma$) and shift ($\beta$) parameters are critical: if standardizing the activations to zero mean and unit variance reduces the model's capacity (for example, by forcing inputs into the linear region of a Sigmoid function), the network can learn to set $\gamma = \sigma$ and $\beta = \mu$ to restore the original distribution. These parameters are optimized during training via backpropagation.

During inference, we cannot calculate batch statistics because we might make predictions on a single sample ($m = 1$). To handle this, BatchNorm tracks running averages of the mean and variance during training:
$$\mu_{\text{running}} \leftarrow \alpha \mu_{\text{running}} + (1 - \alpha) \mu_{\mathcal{B}}$$
$$\sigma^2_{\text{running}} \leftarrow \alpha \sigma^2_{\text{running}} + (1 - \alpha) \sigma^2_{\mathcal{B}}$$
where $\alpha \approx 0.99$ is a momentum hyperparameter. These running statistics are used to normalize inputs during testing.

### Python Code Implementation
Here is a Python class implementing Batch Normalization forward propagation from scratch, covering both training and inference modes:

```python
import numpy as np

class BatchNormScratch:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.momentum = momentum
        self.eps = eps
        
        # Learnable scale (gamma) and shift (beta) parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, X, training=True):
        if training:
            # Mean and variance across the batch dimension (axis 0)
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            
            # Normalize inputs
            X_norm = (X - mean) / np.sqrt(var + self.eps)
            
            # Update running stats using momentum
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        else:
            # Normalize using tracked running statistics during test time
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            
        # Apply learnable scale and shift
        return self.gamma * X_norm + self.beta

# Test BatchNorm on a batch of 3 samples and 2 features
batch_data = np.array([[2.0, 10.0],
                       [4.0, 20.0],
                       [6.0, 30.0]])

bn = BatchNormScratch(num_features=2, momentum=0.9)
print("--- Training Mode Output ---")
print(bn.forward(batch_data, training=True))
print("Tracked Running Mean:", bn.running_mean)

print("\n--- Inference Mode Output (Single Sample) ---")
test_sample = np.array([[3.0, 15.0]])
print(bn.forward(test_sample, training=False))
```

### Trade-offs
BatchNorm stabilizes training, allows for higher learning rates, and acts as a regularizer. Because $\mu_{\mathcal{B}}$ and $\sigma_{\mathcal{B}}^2$ are calculated over a random mini-batch, they contain stochastic noise. This noise is passed to the activations, acting as a form of regularization that reduces overfitting.

The primary trade-off is dependence on batch size. If the training batch size is very small (e.g., 2 or 4), the calculated batch statistics will have high variance, leading to poor training performance. Additionally, BatchNorm is difficult to use in Recurrent Neural Networks (RNNs) because sequence lengths vary, and storing running statistics for every time step is complex.

### Real-World Applications (Rule of 4)

1. **Example 1: BatchNorm Step-by-Step Calculation**
   - **Input/Scenario:** A mini-batch of 3 samples yields pre-activations for a neuron: $\mathcal{B} = [2.0, 5.0, 5.0]$. The learnable parameters are initialized to $\gamma = 1.0, \beta = 0.0$. We set $\epsilon = 0.0$.
   - **Expected Output:**
     - $\mu_{\mathcal{B}} = \frac{2+5+5}{3} = 4.0$
     - $\sigma_{\mathcal{B}}^2 = \frac{(2-4)^2 + (5-4)^2 + (5-4)^2}{3} = \frac{4 + 1 + 1}{3} = 2.0$
     - Normalized values: $\hat{x}_1 = \frac{2-4}{\sqrt{2}} \approx -1.414$, $\hat{x}_2 = \frac{5-4}{\sqrt{2}} \approx 0.707$, $\hat{x}_3 = \frac{5-4}{\sqrt{2}} \approx 0.707$.
     - Output activations: $y = [-1.414, 0.707, 0.707]$. The mean is exactly $0.0$ and variance is $1.0$.
2. **Example 2: Restoring the Original Distribution**
   - **Input/Scenario:** A network learns that a neuron performs best when its pre-activations are not normalized, requiring a mean of $4.0$ and standard deviation of $1.414$.
   - **Expected Output:** During backpropagation, the optimizer updates the parameters to $\gamma = 1.414$ and $\beta = 4.0$. The scale and shift step output $y_i = 1.414 \hat{x}_i + 4.0$, restoring the preferred distribution.
3. **Example 3: BatchNorm as Regularizer**
   - **Input/Scenario:** An image classifier is overfitting. We add `BatchNormalization()` layers after every convolutional layer.
   - **Expected Output:** The noise introduced by the mini-batch statistics acts as a regularizer, reducing the need for heavy dropout and improving validation accuracy.
4. **Example 4: Batch Size Sensitivity Failure**
   - **Input/Scenario:** A developer trains a segmentation model on a GPU with high-resolution images, forcing them to use a batch size of $2$.
   - **Expected Output:** The batch statistics are highly unstable, leading to training divergence. The developer must replace BatchNorm with Group Normalization or Layer Normalization to stabilize training.

> **Metacognitive Checkpoint:** Why does Batch Normalization behave differently during training versus inference? Explain how the running mean and variance are used during testing to make predictions on single samples.

---

## Topic 3: Layer Normalization (LayerNorm): Standardizing Across Features

### Rationale and Mechanics
While Batch Normalization normalizes activations across the batch dimension, **Layer Normalization (LayerNorm)**, introduced by Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey Hinton in 2016, normalizes the activations of a single sample across the feature dimension.

Under the hood, let's consider a layer with $H$ hidden units (features). For a single training sample, the activations of this layer form a vector $\mathbf{h} = [h_1, h_2, \dots, h_H]^T$. LayerNorm normalizes these features using the following steps:
1. **Compute Layer Mean:**
   $$\mu = \frac{1}{H} \sum_{j=1}^H h_j$$
2. **Compute Layer Variance:**
   $$\sigma^2 = \frac{1}{H} \sum_{j=1}^H (h_j - \mu)^2$$
3. **Normalize:**
   $$\hat{h}_j = \frac{h_j - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
4. **Scale and Shift:**
   $$y_j = \gamma_j \hat{h}_j + \beta_j$$
   where $\gamma_j$ and $\beta_j$ are learnable parameters unique to feature coordinate $j$.

Crucially, LayerNorm performs its calculations independently for each sample. This means the mean and variance are calculated along the rows (features) rather than down the columns (samples).

### Python Code Implementation
Here is a Python function implementing Layer Normalization, showing how it normalizes along the feature axis for each sample independently:

```python
import numpy as np

def layer_normalization(X, gamma, beta, eps=1e-5):
    # Calculate mean and variance across features (axis 1)
    # keepdims=True is necessary for correct broadcasting shape (N, 1)
    mean = np.mean(X, axis=1, keepdims=True)
    var = np.var(X, axis=1, keepdims=True)
    
    # Normalize features
    X_norm = (X - mean) / np.sqrt(var + eps)
    
    # Scale and shift (gamma and beta are vectors of shape (features,))
    return gamma * X_norm + beta

# Simulating a batch of 3 samples with 4 features each
batch_data = np.array([[1.0, 2.0, 5.0, 8.0],
                       [10.0, 20.0, 30.0, 40.0],
                       [-1.0, 0.0, 1.0, 2.0]])

# Initialize learnable parameters
gamma = np.ones(4)
beta = np.zeros(4)

normalized_data = layer_normalization(batch_data, gamma, beta)
print("Original Batch Data:\n", batch_data)
print("\nLayerNorm Normalized Output:\n", np.round(normalized_data, 4))
# Check that the mean of each row is 0.0 and variance is 1.0
print("\nRow Means:", np.round(normalized_data.mean(axis=1), 4))
print("Row Vars :", np.round(normalized_data.var(axis=1), 4))
```

### Trade-offs
LayerNorm offers significant advantages for sequential and transformer models:
- **Batch Size Independence:** Since LayerNorm does not use batch statistics, its behavior is identical during training and testing. It works perfectly with batch sizes of 1, making it ideal for online learning.
- **Dynamic Sequence Lengths:** In Natural Language Processing, sentences have different lengths. BatchNorm struggles with varying sequence lengths, while LayerNorm normalizes across the features of each token individually, regardless of sequence length.

The trade-off is that LayerNorm does not introduce the regularizing noise of BatchNorm because it does not mix statistics across samples. In convolutional networks (CNNs), LayerNorm is generally less effective than BatchNorm because spatial statistics are highly localized, and mixing features across channels can wash out spatial relationships.

### Real-World Applications (Rule of 4)

1. **Example 1: LayerNorm Step-by-Step Calculation**
   - **Input/Scenario:** A single sample passes through a layer with 4 features, yielding activations $\mathbf{h} = [1.0, 2.0, 5.0, 8.0]^T$. The learnable scale and shift parameters are initialized to $\gamma_j = 1.0, \beta_j = 0.0$. We set $\epsilon = 0.0$.
   - **Expected Output:**
     - $\mu = \frac{1+2+5+8}{4} = 4.0$
     - $\sigma^2 = \frac{(1-4)^2 + (2-4)^2 + (5-4)^2 + (8-4)^2}{4} = \frac{9 + 4 + 1 + 16}{4} = 7.5$
     - Normalized vector: $\hat{\mathbf{h}} = \left[\frac{1-4}{\sqrt{7.5}}, \frac{2-4}{\sqrt{7.5}}, \frac{5-4}{\sqrt{7.5}}, \frac{8-4}{\sqrt{7.5}}\right]^T \approx [-1.095, -0.730, 0.365, 1.461]^T$.
     - Output activations: $y = [-1.095, -0.730, 0.365, 1.461]^T$.
2. **Example 2: Transformer Layer (Self-Attention)**
   - **Input/Scenario:** A Transformer model processes a sentence of length 10. Each token is represented by a 512-dimensional vector. We apply LayerNorm.
   - **Expected Output:** LayerNorm calculates 10 distinct means and variances—one for each token's 512 features. The normalization is independent of batch size and sequence length, stabilizing the inputs to the attention layers.
3. **Example 3: Recurrent Neural Network (LSTM)**
   - **Input/Scenario:** We train a sequence model for time-series forecasting. The sequence length is dynamic.
   - **Expected Output:** Using LayerNorm at each time step stabilizes the recurrent dynamics without needing to store running statistics for different sequence lengths, preventing exploding gradients.
4. **Example 4: Comparison in CNNs**
   - **Input/Scenario:** A developer replaces BatchNorm with LayerNorm in a ResNet classifier trained on ImageNet.
   - **Expected Output:** The validation accuracy drops. Because LayerNorm normalizes across channels within the same pixel location, it discards channel-specific contrast information that is useful for image classification.

> **Metacognitive Checkpoint:** Why is Layer Normalization better suited for Recurrent Neural Networks (RNNs) and Transformers than Batch Normalization? Explain in terms of batch size independence and varying sequence lengths.

---

## Summary & Next Steps

- **Internal Covariate Shift Stalls Training:** Updates to early layers shift the distribution of inputs to deeper layers, requiring downstream units to constantly readapt.
- **BatchNorm Standardizes Across Batches:** BatchNorm normalizes activations down the batch column. It acts as a regularizer but depends on batch size and is difficult to apply to sequence models.
- **LayerNorm Standardizes Across Features:** LayerNorm normalizes activations along the feature row for each sample. It is batch-size independent and serves as the standard for RNNs and Transformers.

In the next lesson, we will explore **The Keras Functional API**, moving beyond linear layer stacks to build complex architectures with multiple inputs, multiple outputs, and skip connections.
