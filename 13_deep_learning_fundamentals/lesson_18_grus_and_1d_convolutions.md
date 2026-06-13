# Lesson 18: Gated Recurrent Units (GRUs) & 1D Convolutions

## Introduction & The "Why"

In Lesson 17, we explored LSTMs and saw how they solve the vanishing gradient problem using a cell state conveyor belt regulated by three gates. While LSTMs are highly effective, their design is computationally expensive: they require four distinct linear transformations at each timestep. This complexity makes LSTMs slow to train, memory-intensive, and prone to overfitting on smaller datasets.

To improve efficiency, deep learning researchers developed two main alternatives:
1. **Gated Recurrent Units (GRUs):** A simplified, lightweight recurrent cell that reduces the number of gates to two and merges the hidden and cell states, cutting parameters by $25\%$.
2. **1D Convolutional Layers (Conv1D):** A non-recurrent approach that slides a kernel along the temporal dimension. Because Conv1D does not process data step-by-step sequentially, it can be computed in parallel, making it orders of magnitude faster on GPUs than RNNs.

This lesson covers the mathematics of GRU gating, explores the mechanics of 1D convolutions for sequence processing, and demonstrates how to build hybrid Conv1D-RNN architectures to combine local feature extraction with long-term memory.

---

## Topic 1: Gated Recurrent Units (GRUs): Simplifying the Gates

### Rationale and Mechanics
The Gated Recurrent Unit (GRU), designed by Kyunghyun Cho et al. in 2014, simplifies the LSTM architecture. Instead of maintaining separate hidden and cell states, a GRU merges them into a single **Hidden State** $\mathbf{h}_t$. It also reduces the number of gates from three to two: the **Update Gate** and the **Reset Gate**.

Under the hood, at each timestep $t$:
1. **The Update Gate ($\mathbf{z}_t$):** Determines how much of the past hidden state to keep and how much new information to write. It acts as both the forget and input gates combined:
   $$\mathbf{z}_t = \sigma\left( \mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z \right)$$
2. **The Reset Gate ($\mathbf{r}_t$):** Determines how much of the past state to forget when generating the new candidate hidden state:
   $$\mathbf{r}_t = \sigma\left( \mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r \right)$$
3. **The Candidate Hidden State ($\tilde{\mathbf{h}}_t$):** The new information candidate, computed using the reset-gated past hidden state:
   $$\tilde{\mathbf{h}}_t = \tanh\left( \mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h \right)$$
4. **Update Hidden State ($\mathbf{h}_t$):** The new hidden state is calculated as a linear interpolation between the past state and the candidate state, controlled by the update gate $\mathbf{z}_t$:
   $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

Because a GRU only has 3 internal sub-layers (compared to 4 in an LSTM), the parameter count is significantly reduced:
$$\text{Parameters}_{\text{GRU}} = 3 \times H \times (H + D + 1)$$
where $H$ is the hidden state size and $D$ is the input feature dimension.

### Python Code Implementation
Here is a Python function implementing a single forward pass of a GRU cell from scratch using NumPy, showing how the update and reset gates control the state transitions:

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def gru_cell_forward(x_t, h_prev, weights, biases):
    # Unpack weights and biases
    W_z, W_r, W_h = weights
    b_z, b_r, b_h = biases
    
    # Concatenate inputs
    concat = np.concatenate([h_prev, x_t])
    
    # 1. Update Gate (z_t)
    z_t = sigmoid(np.dot(W_z, concat) + b_z)
    
    # 2. Reset Gate (r_t)
    r_t = sigmoid(np.dot(W_r, concat) + b_r)
    
    # 3. Candidate Hidden State (h_tilde)
    # The reset gate scales the previous hidden state
    gated_h_prev = r_t * h_prev
    concat_candidate = np.concatenate([gated_h_prev, x_t])
    h_tilde = np.tanh(np.dot(W_h, concat_candidate) + b_h)
    
    # 4. Final Hidden State update
    h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
    
    return h_t, (z_t, r_t, h_tilde)

# Setup shapes: Input D=2, Hidden H=3
D = 2
H = 3

# Initialize parameters randomly
np.random.seed(42)
weights = [np.random.randn(H, H + D) for _ in range(3)]
biases = [np.zeros(H) for _ in range(3)]

# Inputs
x_t = np.array([1.5, -0.2])
h_prev = np.array([0.5, 0.5, 0.5])

# Forward step
h_t, gates = gru_cell_forward(x_t, h_prev, weights, biases)
z_t, r_t, h_tilde = gates

print("--- GRU Gate Activations ---")
print("Update Gate (z_t):", np.round(z_t, 4))
print("Reset Gate  (r_t):", np.round(r_t, 4))
print("\n--- Updated Hidden State ---")
print("New Hidden State (h_t):", np.round(h_t, 4))
```

### Trade-offs
- **Advantages:** GRUs require $25\%$ fewer parameters than LSTMs, leading to faster training times, lower memory consumption, and a reduced risk of overfitting on small datasets.
- **Disadvantages:** Because the update gate controls both writing and erasing simultaneously, the GRU has slightly less representational capacity than an LSTM. On very large, complex datasets (e.g., massive language models), an LSTM can outperform a GRU, though they perform similarly on most tasks.

### Real-World Applications (Rule of 4)

1. **Example 1: GRU Parameter Count Calculation**
   - **Input/Scenario:** A GRU layer has 128 units ($H=128$) and receives inputs of size $D=50$.
   - **Expected Output:**
     $$\text{Params} = 3 \times 128 \times (128 + 50 + 1) = 384 \times 179 = 68,736 \text{ parameters}$$
     This is $22,912$ fewer parameters than the equivalent LSTM ($91,648$).
2. **Example 2: Reset Gate Action**
   - **Input/Scenario:** An NLP model reads a punctuation mark, indicating a change in context. The reset gate output is calculated as $\mathbf{r}_t \approx 0.0$.
   - **Expected Output:** The past hidden state contribution is zeroed out: $\mathbf{r}_t \odot \mathbf{h}_{t-1} = 0.0$. The candidate state $\tilde{\mathbf{h}}_t$ is computed using only the current input $\mathbf{x}_t$, clearing the historical context.
3. **Example 3: Update Gate Preservation**
   - **Input/Scenario:** A time-series model identifies a strong seasonal trend. The update gate vector is calculated as $\mathbf{z}_t \approx 0.0$.
   - **Expected Output:** The update equation simplifies to:
     $$\mathbf{h}_t \approx (1.0) \odot \mathbf{h}_{t-1} + (0.0) \odot \tilde{\mathbf{h}}_t = \mathbf{h}_{t-1}$$
     The historical state is passed forward unchanged, protecting the long-term trend memory.
4. **Example 4: Code Implementation**
   - **Input/Scenario:** A developer instantiates a GRU layer in Keras.
   - **Expected Output:**
     ```python
     model = keras.Sequential([
         keras.layers.GRU(64, input_shape=(None, 10))
     ])
     ```
     Keras initializes and compiles the GRU graph, managing the update and reset gating operations automatically.

> **Metacognitive Checkpoint:** How does the GRU combine the functions of the LSTM's Forget and Input gates into a single Update Gate? Explain this by analyzing the GRU state update equation: $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$.

---

## Topic 2: 1D Convolutions (Conv1D) for Sequence Processing

### Rationale and Mechanics
In computer vision, 2D convolutions slide a kernel across height and width to extract spatial patterns. For sequential data, we use **1D Convolutions (Conv1D)**. A Conv1D layer slides a kernel along the temporal dimension (timesteps) of a sequence.

Under the hood, let the input sequence $X$ have shape $(T, D)$ (Timesteps, Features). A 1D kernel $K$ of size $k$ has shape $(k, D)$. The convolution operation at timestep $t$ calculates the dot product across the window:
$$Z(t) = \sum_{m=1}^k \sum_{d=1}^D X(t + m - 1, d) K(m, d) + b$$

Crucially, **Conv1D avoids the sequential bottleneck**. Unlike RNNs, which must calculate steps one-by-one ($\mathbf{h}_{t-1} \to \mathbf{h}_t$), Conv1D calculates the outputs at all timesteps in parallel. There are no temporal state dependencies in the forward pass, allowing GPUs to parallelize the operations and train models orders of magnitude faster than RNNs.

### Python Code Implementation
Here is a Python function implementing a 1D convolution operation from scratch using NumPy, showing how a temporal kernel slides along sequence timesteps to extract local patterns:

```python
import numpy as np

def convolve1d(sequence, kernel, bias=0.0):
    seq_len, num_features = sequence.shape
    k_size, _ = kernel.shape
    
    # Output length with Valid padding (P=0) and Stride=1
    out_len = seq_len - k_size + 1
    output = np.zeros(out_len)
    
    # Perform sliding window convolution
    for t in range(out_len):
        # Extract sequence patch matching the kernel size
        patch = sequence[t : t + k_size, :]
        # Element-wise multiply and sum
        output[t] = np.sum(patch * kernel) + bias
        
    return output

# Synthetic sequence of length T=6, features D=2
# Representing closing price and volume indicators
sequence = np.array([
    [100.0, 1.0],
    [102.0, 1.2],
    [101.0, 0.9],
    [104.0, 1.5],
    [105.0, 1.6],
    [103.0, 1.1]
])

# 1D kernel of size k=3, D=2
kernel = np.array([
    [0.1, 0.5],
    [0.2, 0.5],
    [0.7, 1.0]
])

out = convolve1d(sequence, kernel)
print("Input Sequence:\n", sequence)
print("\nKernel:\n", kernel)
print("\n1D Convolution Output Feature Map:", out)
```

### Trade-offs
- **Advantages:** Extremely fast training speeds and excellent extraction of local temporal features (e.g., 3-day stock price dips, audio wave patterns).
- **Disadvantages:** Limited **Receptive Field**. A single Conv1D layer with kernel size $k$ can only capture patterns spanning $k$ timesteps. It cannot capture long-term context (e.g., connecting word 1 to word 200). To capture long-term context, we must stack multiple dilated Conv1D layers or combine them with recurrent layers.

### Real-World Applications (Rule of 4)

1. **Example 1: Parameter Count Calculation (Conv1D)**
   - **Input/Scenario:** A Conv1D layer has 64 filters of kernel size $k=5$ receiving inputs with $D=10$ features.
   - **Expected Output:**
     $$\text{Params} = F \times (k \times D + 1) = 64 \times (5 \times 10 + 1) = 3,264 \text{ parameters}$$
     The parameter count is independent of sequence length.
2. **Example 2: Audio Wave Analysis**
   - **Input/Scenario:** An audio classification model processes high-frequency sound waves (16,000 samples per second).
   - **Expected Output:** Running an RNN is too slow due to the sequence length (16,000 timesteps). Conv1D layers slide along the wave, extracting frequency features in parallel, enabling real-time processing.
3. **Example 3: Receptive Field Limit**
   - **Input/Scenario:** A developer uses a single Conv1D layer with $k=3$ to predict stock prices.
   - **Expected Output:** The model can only capture relationships spanning 3 days. It cannot detect weekly or monthly cyclical patterns unless the kernel size is expanded or layers are stacked.
4. **Example 4: Code Implementation**
   - **Input/Scenario:** A developer implements a Conv1D model in Keras.
   - **Expected Output:**
     ```python
     model = keras.Sequential([
         keras.layers.Input(shape=(100, 10)),
         keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
         keras.layers.Dense(1)
     ])
     ```
     This slides 32 filters of size 3 along the 100 timesteps, extracting local features.

> **Metacognitive Checkpoint:** Why is training a Conv1D layer on a GPU significantly faster than training a GRU layer? Explain this in terms of temporal dependencies and parallel computation.

---

## Topic 3: Hybrid Architectures: Conv1D + LSTM / GRU

### Rationale and Mechanics
For very long sequences (e.g., hourly weather records over months), we want to capture both local patterns and long-term context. Using a pure RNN is too slow, and a pure Conv1D cannot model long-term sequence dependencies.

To solve this, we build a **Hybrid Conv1D-RNN Model**. We place Conv1D layers at the beginning of the network to act as feature extractors and downsamplers, and pass the compressed features to a downstream GRU or LSTM layer.

Under the hood:
1. The raw sequence of shape $(B, T, D)$ is passed to a Conv1D layer with a stride $S \ge 2$:
   ```python
   keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same')
   ```
2. The Conv1D layer downsamples the sequence. For a stride $S=2$, the temporal size is halved:
   $$\text{Output Shape} = \left( B, \frac{T}{2}, 64 \right)$$
3. This compressed sequence is passed to a GRU layer:
   ```python
   keras.layers.GRU(32)
   ```

By reducing the sequence length from $T$ to $T/2$, we halve the number of unrolled timesteps that the GRU has to process, speeding up BPTT and reducing memory consumption by $50\%$.

### Python Code Implementation
Here is a Python script using Keras to design a Hybrid Conv1D-GRU model and inspect how the shapes of tensors transition across the network:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Design the Hybrid model
model = keras.Sequential([
    layers.Input(shape=(200, 10), name='input_sequence'),  # 200 steps, 10 features
    layers.Conv1D(
        filters=32, 
        kernel_size=3, 
        strides=2, 
        padding='same', 
        activation='relu', 
        name='conv1d_compressor'
    ),
    layers.GRU(16, return_sequences=False, name='gru_summarizer'),
    layers.Dense(1, name='prediction')
])

# Verify output shapes at each stage of the network
print("--- Hybrid Model Summary ---")
model.summary()

# Trace a dummy batch through the model layers
dummy_batch = tf.random.normal((1, 200, 10))
print("\n--- Layer Output Shapes Trace ---")
x = dummy_input = dummy_batch
for layer in model.layers:
    x = layer(x)
    print(f"Layer: {layer.name:20s} | Output Shape: {x.shape}")
```

### Trade-offs
- **Advantages:** Best of both worlds: Conv1D extracts clean local features and shrinks the sequence, while the RNN tracks long-term context over the compressed sequence.
- **Disadvantages:** The architecture is more complex to design and tune. If the Conv1D downsampling is too aggressive (e.g., using a stride of 5), key temporal details can be discarded, preventing the RNN from learning accurate relationships.

### Real-World Applications (Rule of 4)

1. **Example 1: Dimension Shrinkage Calculation**
   - **Input/Scenario:** A sequence of length $T = 500$ with $D = 10$ is processed by a Conv1D layer with $k=3$, stride $S=2$, and same padding, followed by an LSTM.
   - **Expected Output:** The Conv1D layer outputs a tensor of shape $(B, 250, 64)$. The LSTM unrolls for only 250 steps instead of 500, reducing memory usage during BPTT by half.
2. **Example 2: Speech Emotion Recognition**
   - **Input/Scenario:** A model predicts speaker emotion from raw audio waveforms.
   - **Expected Output:** The Conv1D layer extracts local phoneme frequencies, and the GRU tracks the emotional arc of the sentence over the course of the recording, yielding a robust classifier.
3. **Example 3: Sensor Anomaly Detection**
   - **Input/Scenario:** A manufacturing turbine records vibrations 100 times per second. An anomaly is defined by a specific high-frequency vibration sequence lasting 0.1 seconds, followed by a slow temperature rise over 10 seconds.
   - **Expected Output:** The Conv1D layer detects the rapid vibration signature, and the LSTM tracks the slow temperature trend, identifying the anomaly successfully.
4. **Example 4: Code Implementation**
   - **Input/Scenario:** A developer implements a hybrid model in Keras.
   - **Expected Output:**
     ```python
     model = keras.Sequential([
         keras.layers.Input(shape=(200, 10)),
         keras.layers.Conv1D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
         keras.layers.GRU(32),
         keras.layers.Dense(1)
     ])
     ```
     The model compiles successfully, downsampling the input sequence before passing it to the recurrent layer.

> **Metacognitive Checkpoint:** How does using a strided Conv1D layer before a GRU layer reduce the computational memory required during backpropagation? Analyze the change in unrolled graph depth.

---

## Summary & Next Steps

- **GRUs Simplify RNN Gates:** GRUs merge the cell and hidden states and use only two gates (update and reset), reducing parameters by $25\%$ to accelerate training.
- **Conv1D Slides Along Time:** 1D convolutions process sequence steps in parallel, bypassing the sequential bottleneck of RNNs at the cost of a limited receptive field.
- **Hybrid Models Combine Strengths:** Combining Conv1D downsampling with recurrent layers allows us to extract local features efficiently while retaining long-term memory.

In the next module, we will explore **Module 5: Text, Semantics, and Transformers**, starting with **Lesson 19: Vectorizing Semantics** to learn tokenization and embedding layers.
