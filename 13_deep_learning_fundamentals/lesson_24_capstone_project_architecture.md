# Lesson 24: Capstone Project Part 1 (Architecture & Training)

## Introduction & The "Why"

Throughout this course, we explored the mathematical foundations and code implementations of deep learning. We traced neural networks from the single Perceptron to multi-layer feedforward networks, convolutional neural networks for vision, recurrent architectures (LSTMs and GRUs) for sequences, self-attention mechanisms, and unsupervised autoencoders.

However, in professional data science, models are not built on clean, pre-packaged datasets. Real-world data is messy, incomplete, and unstructured. To prove your competence as a deep learning engineer, you must be able to design and execute an end-to-end deep learning project from scratch.

This lesson serves as Part 1 of your Capstone Project. You will select a complex problem domain, design a data preprocessing pipeline to convert raw features into optimized 3D or 2D tensors, choose the correct architectural inductive bias (CNN, LSTM, or Functional ensemble), and configure a regularized training loop using callbacks to prevent overfitting.

---

## Topic 1: Selecting the Problem Domain and Preprocessing Pipelines

### Rationale and Mechanics
Before designing an architecture, you must scope the problem and prepare the input pipeline. Deep learning models require inputs to be formatted as structured, normalized tensors.

Under the hood, depending on your chosen domain, your preprocessing pipeline must execute the following operations:
1.  **Computer Vision (Spatial Domain):**
    *   **Resolution Standardization:** Resize raw images to a consistent target shape (e.g., $224\times224$).
    *   **Channel Normalization:** Scale pixel intensities from the integer range $[0, 255]$ to the floating-point range $[0, 1]$ or $[-1, 1]$:
        $$x_{\text{norm}} = \frac{x}{255.0} \quad \text{or} \quad x_{\text{norm}} = \frac{x}{127.5} - 1.0$$
2.  **Natural Language Processing (Sequential Text Domain):**
    *   **Tokenization:** Convert raw text strings into sequences of subword integers using a tokenizer.
    *   **Padding and Truncation:** Pad shorter sequences with zero tokens and truncate longer sequences to a target length $T$.
        *   **Pre-padding:** Adds zeros to the beginning of the sequence. LSTMs generally perform better with pre-padding because the active information is situated close to the end of the sequence, near the final hidden state classification decision.
3.  **Time-Series Forecasting (Sequential Continuous Domain):**
    *   **Feature Scaling:** Scale continuous features (e.g. price, temperature) using `StandardScaler` or `MinMaxScaler` based on training statistics.
    *   **Windowing:** Convert the chronological array into a 3D tensor of shape `(Samples, Timesteps, Features)` using sliding windows.

```
       Raw Data (Messy) ---> [ Domain Preprocessor ] ---> Normalized Tensors
                                                               |
                                   - Image: Scale to [0,1], shape (H, W, C)
                                   - Text: Tokenize & Pad, shape (T,)
                                   - Time-Series: Sliding Window, shape (T, D)
```

### Python Code Implementation
The following code demonstrates how to implement input preprocessing pipelines for all three domains (Vision, NLP, and Time-Series) using NumPy and TensorFlow/Keras.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Computer Vision: Resizing & Normalizing an image batch
# Simulate raw image batch of shape (2, 400, 400, 3) with pixel values in [0, 255]
np.random.seed(42)
raw_images = np.random.randint(0, 256, size=(2, 400, 400, 3)).astype("float32")

image_pipeline = keras.Sequential([
    keras.layers.Resizing(224, 224),
    keras.layers.Rescaling(1.0 / 255.0)  # Scale to [0.0, 1.0]
])
processed_images = image_pipeline(raw_images)
print("Vision Pipeline:")
print(" - Raw shape:", raw_images.shape)
print(" - Processed shape:", processed_images.shape)
print(" - Min/Max pixel values:", np.min(processed_images), np.max(processed_images))

# 2. NLP: Padding integer sequences (pre-padding for LSTM)
sequences = [[14, 256, 32], [1024, 33]] # Varying lengths
# Pad to max length of 4
padded_seqs = pad_sequences(sequences, maxlen=4, padding="pre", value=0)
print("\nNLP Pipeline:")
print(" - Padded Sequences (pre-padded):\n", padded_seqs)

# 3. Time-Series: Create 3D Sliding Windows from a 1D continuous sequence
def create_sliding_windows(data, window_size):
    num_samples = len(data) - window_size + 1
    windows = []
    for i in range(num_samples):
        windows.append(data[i:i+window_size])
    return np.array(windows)

# Continuous timeline of stock prices (1D array of 7 days)
prices = np.array([100.5, 101.2, 99.8, 102.1, 103.5, 102.8, 104.2])
# Create windows of size 3
series_windows = create_sliding_windows(prices, window_size=3)
print("\nTime-Series Pipeline:")
print(" - Raw prices:", prices)
print(" - Sliding Windows (size 3) shape:", series_windows.shape)
print(" - Sliding Windows:\n", series_windows)
```

### Trade-offs
*   **Advantages:** Tensor normalization prevents early gradients from exploding, while standardization ensures consistent shape and scale across batches.
*   **Disadvantages:** Preprocessing can introduce CPU bottlenecks if computed dynamically during training. Pre-computing and saving the dataset to disk is faster but inflexible if you decide to change image sizes or tokenizers.

### Real-World Applications (Rule of 4)
1.  **Example 1: Image Pixel Scaling**
    *   **Input/Scenario:** An RGB image has pixel intensities $I(x, y) = [255, 128, 0]^T$ (orange). We scale it to the range $[-1, 1]$.
    *   **Expected Output:** The scaled pixel values are $I_{\text{scaled}} \approx [1.0, 0.004, -1.0]^T$.
2.  **Example 2: Text Padding Alignment (LSTM)**
    *   **Input/Scenario:** A sentence has 3 words: `["I", "love", "DL"]`, tokenized as `[12, 85, 302]`. The target sequence length is $T = 5$.
    *   **Expected Output:** Pre-padding produces `[0, 0, 12, 85, 302]`. This keeps active embeddings close to the final hidden state output.
3.  **Example 3: Tabular Time-Series Windowing**
    *   **Input/Scenario:** A developer scales stock volume ($0$ to $10,000,000$ shares) using `MinMaxScaler` to fit the range $[0, 1]$.
    *   **Expected Output:** The scaled values prevent the volume magnitude from dominating stock price variations (range $100$ to $200$).
4.  **Example 4: Runtime Image Augmentation**
    *   **Input/Scenario:** A training pipeline applies random rotations and flips on-the-fly to a batch of images on the CPU while the GPU trains.
    *   **Expected Output:** The GPU receives a continuous stream of augmented images, preventing disk space explosion.

> **Metacognitive Checkpoint:** Why is pre-padding generally preferred over post-padding when preparing text sequences for Recurrent Neural Networks (LSTMs/GRUs)? Explain in terms of state decay over padded zeros.

---

## Topic 2: Architectural Design: Choosing CNN, LSTM, or Functional Ensembles

### Rationale and Mechanics
Once the input tensors are structured, you must select the appropriate architectural inductive bias. The architecture must match the geometry of the data:
*   **Spatial Data (Images):** Requires 2D Convolutions (`Conv2D` + `MaxPooling2D`) to capture spatial hierarchies and translation invariance.
*   **Sequential Data (Text/Time-Series):** Requires recurrent layers (`LSTM` or `GRU`) to capture temporal order, or `Conv1D` for fast local feature extraction.
*   **Heterogeneous Data (Multimodal):** Requires the Keras Functional API to branch the network, processing each input separately before merging.

Under the hood, when designing your network, follow these guidelines:
1.  **Start Simple:** Build a small baseline model first. For a CNN, start with 2 convolutional layers and a single dense layer. For an LSTM, start with a single layer of 32 units.
2.  **Check Capacity:** Evaluate training performance. If the baseline model cannot fit the training data (high training loss), increase model capacity by adding layers (depth) or neurons (width).
3.  **Transition to Functional if Branching:** If the task requires skip connections or multiple inputs/outputs, transition immediately from the Sequential API to the Functional API.

```
       Identify Data Geometry:
       - 2D Grid (Image) --------> Use Conv2D Stack
       - 1D Sequence (Time) -----> Use LSTM or GRU Stack
       - Multi-Modal ------------> Use Functional API Concatenation
```

### Python Code Implementation
The following code demonstrates how to use the Keras Functional API to build a multi-modal (multimodal) ensemble network. This model processes a time-series input sequence via an LSTM branch and a tabular metadata input via a Dense branch, concatenates the features, and outputs a single binary prediction.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_multimodal_model(seq_shape, tabular_shape):
    # 1. Time-Series Branch (Input shape: [Timesteps, Features])
    seq_input = keras.Input(shape=seq_shape, name="time_series_input")
    seq_x = layers.LSTM(32, return_sequences=False, name="lstm_extractor")(seq_input)
    seq_branch = layers.Dense(16, activation="relu", name="seq_features")(seq_x)
    
    # 2. Tabular/Metadata Branch (Input shape: [Features])
    tab_input = keras.Input(shape=tabular_shape, name="tabular_input")
    tab_x = layers.Dense(16, activation="relu", name="dense_extractor_1")(tab_input)
    tab_branch = layers.Dense(16, activation="relu", name="tab_features")(tab_x)
    
    # 3. Concatenate Features from both branches
    merged = layers.concatenate([seq_branch, tab_branch], name="merged_features")
    
    # 4. Dense Classifier Output
    classifier = layers.Dense(16, activation="relu", name="joint_dense")(merged)
    output = layers.Dense(1, activation="sigmoid", name="binary_output")(classifier)
    
    # Define and Compile Model
    model = keras.Model(inputs=[seq_input, tab_input], outputs=output, name="multimodal_ensemble")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Shape definitions: 10 timesteps with 3 features; 5 tabular metadata features
model = build_multimodal_model(seq_shape=(10, 3), tabular_shape=(5,))
print("Multimodal Model Architecture:")
model.summary()
```

### Trade-offs
*   **Advantages:** Tailoring the network design to the spatial or temporal nature of the data reduces the parameter count and speeds up training.
*   **Disadvantages:** Complex architectures (like functional ensembles) are harder to debug. A mismatch in merged layer shapes can lead to compilation errors that are difficult to trace.

### Real-World Applications (Rule of 4)
1.  **Example 1: CNN Classifier Design (Sequential)**
    *   **Input/Scenario:** A student designs a custom CNN to classify 10 types of waste from images.
    *   **Expected Output:** Stacking `Conv2D` and `MaxPooling2D` layers extracts spatial features before a final Softmax layer predicts class probabilities.
2.  **Example 2: LSTM Sequence Classifier Design**
    *   **Input/Scenario:** A student designs a network to predict machinery failure using 5 sensor inputs over 100 timesteps.
    *   **Expected Output:** Stacking LSTM layers extracts temporal sequences, outputting a binary failure probability.
3.  **Example 3: Multimodal Loan Default Predictor**
    *   **Input/Scenario:** A financial model predicts loan default using both user text descriptions and numerical transaction history.
    *   **Expected Output:** The network branches process text via an LSTM and tabular data via Dense layers, merging them before classification.
4.  **Example 4: Receptive Field Alignment (Conv1D)**
    *   **Input/Scenario:** A student processes audio clips using a Conv1D layer where local features span 10 timesteps.
    *   **Expected Output:** The student sets `kernel_size=11` to match the temporal receptive field to the target feature size.

> **Metacognitive Checkpoint:** Why should you start your deep learning project with a simple baseline model before building a complex architecture? Explain how this baseline helps debug optimization issues.

---

## Topic 3: Implementing a Strict Regularization and Training Configuration

### Rationale and Mechanics
After designing the architecture, you must configure the training loop to prevent overfitting. This configuration consists of:
*   **Optimizer:** Adam is the default choice for general convergence.
*   **Loss:** Match the loss to the target format (e.g. `binary_crossentropy` for binary classification, `sparse_categorical_crossentropy` for integer targets).
*   **Regularization:** Insert `Dropout` layers (typically rate $0.2$ to $0.5$) after large dense layers and add L2 weight decay to key weights.
*   **Callbacks:** Use Keras Callbacks to automate training adjustments.

Under the hood, two critical callbacks must be configured:
1.  **EarlyStopping:** Monitors validation loss and stops training once overfitting starts, restoring the best weights:
    ```python
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    ```
2.  **ReduceLROnPlateau (Learning Rate Decay):** During training, a constant learning rate can cause the optimizer to overshoot narrow minima. This callback monitors validation loss and reduces the learning rate by a factor (e.g. halving it) when improvement stalls, allowing the optimizer to settle:
    ```python
    lr_decay = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    ```

```
        Training Loop with Callbacks:
        Epoch starts ---> Batch updates ---> Calculate Val Loss ---> Callbacks Check:
                                                                       - Val Loss improved? Save best weights.
                                                                       - Val Loss stalled for 3 epochs? Half learning rate.
                                                                       - Val Loss stalled for 5 epochs? Stop training.
```

### Python Code Implementation
The following code builds a regularized feedforward network, creates synthetic binary classification data, and runs a training loop using both `EarlyStopping` and `ReduceLROnPlateau` callbacks to show how training is regulated.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Generate synthetic data for binary classification (1000 samples, 20 features)
np.random.seed(42)
X_train = np.random.randn(800, 20)
y_train = np.random.randint(0, 2, size=(800, 1))
X_val = np.random.randn(200, 20)
y_val = np.random.randint(0, 2, size=(200, 1))

# 2. Build model with L2 regularization and Dropout
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    # Add Dense layer with L2 weight regularization
    layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),  # Prevent overfitting
    layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

# 3. Configure optimizer and compile
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# 4. Set up Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )
]

# 5. Fit the model
print("Fitting model with Callbacks...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)
```

### Trade-offs
*   **Advantages:** Automating adjustments via callbacks eliminates the need to guess the exact number of epochs or tune learning rate schedules manually.
*   **Disadvantages:** Introduces extra hyperparameters (patience, decay factor). If decay patience is too short, the learning rate will drop to its minimum before the optimizer has had time to escape early saddle points, stalling training.

### Real-World Applications (Rule of 4)
1.  **Example 1: Compiler Configuration**
    *   **Input/Scenario:** We classify images into 3 classes where targets are integers.
    *   **Expected Output:** Compiling with `sparse_categorical_crossentropy` matches the integer target format without manual one-hot conversions.
2.  **Example 2: EarlyStopping in Action**
    *   **Input/Scenario:** A model's validation loss stops improving after epoch 15 and begins to rise. Patience is set to 5.
    *   **Expected Output:** The callback halts training at epoch 20 and restores the weights from epoch 15, preventing overfitting.
3.  **Example 3: Learning Rate Decay Action**
    *   **Input/Scenario:** The validation loss stops improving at epoch 12. Patience is 3.
    *   **Expected Output:** At epoch 15, `ReduceLROnPlateau` halving triggers, dropping the learning rate from $0.01$ to $0.005$.
4.  **Example 4: Regularizing Image Classification**
    *   **Input/Scenario:** A CNN is trained on a small dataset of 500 images.
    *   **Expected Output:** Adding $L2$ decay ($0.001$) and a dropout layer ($0.4$) restricts weight growth, keeping validation accuracy high.

> **Metacognitive Checkpoint:** Why is the `ReduceLROnPlateau` callback effective at optimizing convergence in non-convex loss landscapes? Explain how reducing the learning rate helps the optimizer settle into narrow basins.

---

## Summary & Next Steps

*   **Pipelines Format Input:** Preprocessing must scale pixels, pad sequences, or extract sliding windows to structure data as clean, normalized input tensors.
*   **Architectures Match Data:** CNNs capture spatial features, LSTMs model temporal dependencies, and the Functional API merges heterogeneous inputs.
*   **Callbacks Control Training:** Early Stopping halts training before overfitting occurs, and learning rate decay reduces the step size to optimize convergence.

In the next lesson, we will cover **Capstone Project Part 2 (Evaluation & Defense)**, learning how to analyze training trajectories, compare neural models against classical baselines, and defend your architectural choices.
