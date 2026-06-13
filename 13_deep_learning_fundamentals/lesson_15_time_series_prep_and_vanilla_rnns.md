# Lesson 15: Time-Series Preparation & Vanilla RNNs

## Introduction & The "Why"

In the previous module, we explored computer vision and saw how CNNs utilize spatial hierarchies in grid-like data (images). However, many real-world datasets are not spatial; they are chronological. In time-series forecasting, speech recognition, and natural language processing, the data is sequential: the meaning of a data point depends on the sequence of data points that came before it.

Standard feedforward networks (MLPs) and CNNs assume that inputs are independent and identically distributed. If you feed a time-series dataset to an MLP, it treats each timestep as an independent feature, ignoring the temporal order. Furthermore, dense networks require fixed-length inputs, making them unable to process sequences of varying lengths (such as sentences of different word counts).

To solve these sequential challenges, deep learning uses **Recurrent Neural Networks (RNNs)**. RNNs process sequences step-by-step, maintaining an internal memory called a **hidden state** that accumulates information over time. This lesson covers how to structure sequential data into 3D tensors, explains the mathematical architecture of a Vanilla RNN, and details the different output topologies used to solve diverse business tasks.

---

## Topic 1: Structuring Sequential Data: The 3D Tensor Convention

### Rationale and Mechanics
In classical machine learning with tabular data, we use 2D matrices of shape `(Samples, Features)`. When working with time-series in algorithms like ARIMA, we manually engineer lagged columns (e.g., $t-1$, $t-2$) to capture historical patterns, which keeps the input 2D but increases feature dimensions.

In deep learning, we represent sequential data using a **3D Tensor** convention. This structure preserves the temporal ordering of features without needing manual lag column engineering.

Under the hood, a sequential dataset is organized into a tensor of shape:
$$\text{Tensor Shape} = (\text{Samples}, \text{Timesteps}, \text{Features})$$

Where:
- **Samples ($N$):** The number of independent sequences in the batch (e.g., 32 patient records, 128 stock charts).
- **Timesteps ($T$):** The length of the sequence (e.g., 24 hourly temperature readings, 50 words in a sentence).
- **Features ($D$):** The number of variables recorded at each timestep (e.g., temperature and humidity = 2 features; a word embedding = 300 features).

At each individual timestep $t \in \{1, \dots, T\}$, the input is a feature vector $\mathbf{x}_t \in \mathbb{R}^D$.

```
       3D Tensor Structure: (Samples, Timesteps, Features)
       
       Sample 1:
       [ x_11  x_12  ...  x_1D ] - Timestep 1
       [ x_21  x_22  ...  x_2D ] - Timestep 2
       [ ...   ...   ...  ...  ]
       [ x_T1  x_T2  ...  x_TD ] - Timestep T
       
       Stacked for N samples along the 3rd axis.
```

To prepare tabular time-series data for an RNN, we apply a sliding window function:
1. Choose a sequence length $T$ (lookback window).
2. Slide the window across the chronological series, extracting $T$ historical steps as the input matrix and the step $T+1$ as the target label.
3. Stack these windows to form the $(N, T, D)$ input tensor.

### Trade-offs
The 3D tensor convention allows Keras RNN layers to dynamically process sequences of varying lengths: Keras input layers can be defined as `Input(shape=(None, D))`, where `None` indicates that the sequence length $T$ is determined at runtime.

The trade-off is memory footprint. Converting a single long time-series into overlapping sliding windows duplicates data, significantly increasing RAM requirements. For massive datasets, we must use generator pipelines (like Keras's `TimeseriesGenerator`) to construct the 3D tensors in memory on the fly during training.

### Real-World Applications (Rule of 4)

1. **Example 1: Weather Forecasting Windowing**
   - **Input/Scenario:** A weather station records `temperature`, `humidity`, and `wind_speed` (3 features) every hour. We want to use a 24-hour lookback window to predict the next hour's temperature. The raw dataset has 1,000 hours of continuous records.
   - **Expected Output:** Using a sliding window yields a 3D training tensor of shape $(976, 24, 3)$. Each sample contains a $24\times3$ matrix representing one day of history.
2. **Example 2: NLP Word Embedding Shapes**
   - **Input/Scenario:** A sentiment classifier processes sentences containing up to 50 words. Each word is represented by a 300-dimensional embedding vector. The batch size is 32.
   - **Expected Output:** The input tensor passed to the model has shape $(32, 50, 300)$. The feature dimension represents the semantic word space.
3. **Example 3: Stock Market Tick Data**
   - **Input/Scenario:** A high-frequency trading algorithm analyzes the last 100 stock price ticks. There is only 1 feature: `closing_price`.
   - **Expected Output:** The input tensor shape is $(B, 100, 1)$. The feature dimension is 1 because we are only tracking a single univariate time-series.
4. **Example 4: TimeseriesGenerator Integration**
   - **Input/Scenario:** A developer uses Keras's `TimeseriesGenerator` to train a model on a large array of shape $(10000, 5)$. They specify `length=50` and `batch_size=32`.
   - **Expected Output:** The generator outputs batches of shape $(32, 50, 5)$. The developer fits the model using `model.fit(generator)`.

> **Metacognitive Checkpoint:** If you have a univariate time-series (1 feature) with 1,000 chronological steps, and you extract sliding windows with a lookback length of 10 timesteps, what is the exact shape of the resulting 3D tensor? How many training samples ($N$) are generated?

---

## Topic 2: The Mathematical Architecture of a Vanilla RNN

### Rationale and Mechanics
In classical autoregressive models (like AR or ARMA), the prediction at time $t$ is calculated as a linear combination of a fixed number of past values. In deep learning, a Recurrent Neural Network (RNN) processes sequence steps sequentially, maintaining an internal vector called the **Hidden State** $\mathbf{h}_t$. The hidden state acts as a dynamic memory buffer, accumulating information from all prior timesteps.

Under the hood, at each timestep $t \in \{1, \dots, T\}$:
1. The network receives the current input vector $\mathbf{x}_t \in \mathbb{R}^D$ and the previous hidden state vector $\mathbf{h}_{t-1} \in \mathbb{R}^H$.
2. It combines these inputs linearly using two weight matrices:
   - $\mathbf{W}_{xh}$ (Input-to-Hidden weight matrix, shape: $H \times D$)
   - $\mathbf{W}_{hh}$ (Hidden-to-Hidden weight matrix, shape: $H \times H$)
3. It adds a bias vector $\mathbf{b}_h$ and applies a non-linear activation function (typically Tanh) to compute the new hidden state $\mathbf{h}_t$:
   $$\mathbf{h}_t = \tanh\left( \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h \right)$$
4. If the model outputs a prediction at this timestep, the output vector $\mathbf{y}_t$ is calculated as:
   $$\mathbf{y}_t = g\left( \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y \right)$$
   where $\mathbf{W}_{hy}$ is the Hidden-to-Output weight matrix (shape: $O \times H$), $\mathbf{b}_y$ is the output bias, and $g(\cdot)$ is the output activation (e.g., Softmax or Linear).

```
                 h_t-1 -------\
                              v
       x_t ---> [ W_xh ] ---> [ Sum ] ---> Tanh ---> h_t ---> [ W_hy ] ---> y_t
                               ^
       b_h -------------------/
```

Crucially, the weight matrices $\mathbf{W}_{xh}$, $\mathbf{W}_{hh}$, and $\mathbf{W}_{hy}$ are **shared across time**. The exact same parameters are applied at timestep 1, timestep 2, and timestep $T$. This parameter sharing allows the network to generalize patterns across different temporal positions and process sequences of any length.

### Trade-offs
- **Advantages:** The parameter sharing design keeps the network size independent of sequence length, restricting parameter counts and preventing overfitting.
- **Disadvantages:** Sequential computation is a major bottleneck. To calculate the hidden state $\mathbf{h}_t$, we must first calculate $\mathbf{h}_{t-1}$. This sequential dependency means we cannot parallelize the training process over the time dimension, making RNN training much slower on GPUs than CNN training.

### Real-World Applications (Rule of 4)

1. **Example 1: Parameter Count Calculation**
   - **Input/Scenario:** An RNN layer has 64 hidden units ($H=64$) and receives inputs with 10 features ($D=10$).
   - **Expected Output:** The layer contains:
     - $\mathbf{W}_{xh}$ weights: $64 \times 10 = 640$
     - $\mathbf{W}_{hh}$ weights: $64 \times 64 = 4096$
     - Bias vector $\mathbf{b}_h$: $64$
     - Total parameters: $640 + 4096 + 64 = 4,800$ parameters. This parameter count remains constant regardless of sequence length.
2. **Example 2: First Timestep Initialization**
   - **Input/Scenario:** An RNN begins processing a sequence at $t=1$. The initial hidden state $\mathbf{h}_0$ is required.
   - **Expected Output:** Keras initializes $\mathbf{h}_0$ as a vector of zeros ($[0, \dots, 0]^T$). The calculation at step 1 simplifies to $\mathbf{h}_1 = \tanh(\mathbf{W}_{xh}\mathbf{x}_1 + \mathbf{b}_h)$.
3. **Example 3: Time-Series State Propagation**
   - **Input/Scenario:** We predict weather. At $t=1$, input $\mathbf{x}_1$ indicates rain, setting $\mathbf{h}_1$ to represent "wet soil." At $t=2$, input $\mathbf{x}_2$ indicates sun.
   - **Expected Output:** The hidden state update $\mathbf{h}_2$ combines the current sun input with the historical "wet soil" memory state, allowing the output prediction to factor in humidity evaporation.
4. **Example 4: RNN Character Generation**
   - **Input/Scenario:** An RNN learns to generate text character-by-character. It receives input character 'h' at $t=1$, and updates its hidden state. At $t=2$, it receives 'e'.
   - **Expected Output:** The output prediction at $t=2$ uses $\mathbf{h}_2$ (containing the sequence history 'h' -> 'e') to output a high probability for the next character 'l' (predicting the word "hello").

> **Metacognitive Checkpoint:** Why are the weight matrices in an RNN shared across all timesteps? What would happen to the parameter count of the model if we used unique weights for each timestep?

---

## Topic 3: RNN Output Topologies: Many-to-One, Many-to-Many, and One-to-Many

### Rationale and Mechanics
Not all sequential tasks are identical. Some require predicting a single label from a sequence (e.g., classifying a movie review as positive), while others require mapping a sequence to another sequence (e.g., translating English to Spanish).

We configure these topologies in Keras using the `return_sequences` hyperparameter.

Under the hood:
- **Many-to-One (return_sequences=False):** The model processes a sequence of inputs $[\mathbf{x}_1, \dots, \mathbf{x}_T]$ and only outputs the final hidden state $\mathbf{h}_T$ at the end of the sequence. This final vector represents the summary of the entire sequence.
- **Many-to-Many (return_sequences=True):** The model outputs the hidden state $\mathbf{h}_t$ at *every* timestep, returning a sequence of hidden states $[\mathbf{h}_1, \dots, \mathbf{h}_T]$. This is used when the output must align with each input timestep (e.g., POS tagging).
- **One-to-Many:** The network receives a single input vector $\mathbf{x}$ and generates a sequence of outputs $[\mathbf{y}_1, \dots, \mathbf{y}_T]$ (e.g., generating a caption from an image).

```
       Many-to-One:                             Many-to-Many:
       
       x1 ---> RNN ---> h1                      x1 ---> RNN ---> h1 ---> y1
       x2 ---> RNN ---> h2                      x2 ---> RNN ---> h2 ---> y2
       x3 ---> RNN ---> h3 ---> y3              x3 ---> RNN ---> h3 ---> y3
```

In Keras, nesting RNN layers requires setting `return_sequences=True` on all hidden RNN layers, as a downstream RNN layer requires a 3D tensor input:
```python
model = keras.Sequential([
    keras.layers.SimpleRNN(64, return_sequences=True, input_shape=(None, 10)),
    keras.layers.SimpleRNN(32, return_sequences=False),
    keras.layers.Dense(1)
])
```

### Trade-offs
- **Many-to-One Trade-off:** Simple and clean for classification tasks. However, it forces the network to compress all historical details into a single vector $\mathbf{h}_T$. For very long sequences, early information is lost before reaching the output, a limitation addressed by attention mechanisms.
- **Many-to-Many Trade-off:** Allows token-level predictions. However, during training, it requires calculating losses at every timestep, which can increase computational overhead and require target sequences to be aligned and padded.

### Real-World Applications (Rule of 4)

1. **Example 1: Sentiment Analysis (Many-to-One)**
   - **Input/Scenario:** A model classifies movie review text. The input sequence is 50 words.
   - **Expected Output:** The RNN layer is compiled with `return_sequences=False`. It outputs a single vector $\mathbf{h}_{50}$ of shape $(32,)$ to a dense output neuron, which outputs the probability of positive sentiment.
2. **Example 2: Part-of-Speech Tagging (Many-to-Many)**
   - **Input/Scenario:** A model labels the grammatical role (noun, verb, adjective) of each word in a sentence.
   - **Expected Output:** The RNN uses `return_sequences=True`. It outputs 50 hidden states. A TimeDistributed dense layer processes each state to predict the POS tag for each word, yielding 50 output predictions.
3. **Example 3: Image Captioning (One-to-Many)**
   - **Input/Scenario:** A caption generator receives a 512-dimensional CNN image feature vector.
   - **Expected Output:** The network receives the image feature vector at $t=1$, and recurrently outputs a sequence of words ("a," "dog," "playing," "catch") until a stop token is generated.
4. **Example 4: Hidden Layer Nesting Failure**
   - **Input/Scenario:** A developer stacks two RNN layers: `SimpleRNN(64)` followed by `SimpleRNN(32)` without specifying `return_sequences=True` on the first layer.
   - **Expected Output:** Keras raises a `ValueError: Input 0 of layer simple_rnn_1 is incompatible with the layer: expected ndim=3, found ndim=2`. The developer must add `return_sequences=True` to the first layer to pass a 3D tensor to the second.

> **Metacognitive Checkpoint:** Why does a downstream RNN layer require the upstream RNN layer to set `return_sequences=True`? Explain in terms of tensor dimensions and the difference between returning a single vector versus a sequence of vectors.

---

## Summary & Next Steps

- **Sequences Require 3D Tensors:** Chronological data is structured as tensors of shape `(Samples, Timesteps, Features)`. The sliding window method is used to prepare inputs.
- **RNNs Maintain Hidden States:** Recurrent neurons share weights across time, updating a hidden state vector at each step using the input and previous state.
- **Topologies Adapt to Tasks:** Using `return_sequences` allows us to configure RNNs for Many-to-One classification or Many-to-Many sequence-to-sequence labeling.

In the next lesson, we will explore **Backpropagation Through Time (BPTT)**, analyzing the mathematical limits of Vanilla RNNs and understanding why they struggle to retain long-term dependencies.
