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
1. **Computer Vision (Spatial Domain):**
   - **Resolution Standardization:** Resize raw images to a consistent target shape (e.g., $224\times224$).
   - **Channel Normalization:** Scale pixel intensities from the integer range $[0, 255]$ to the floating-point range $[0, 1]$ or $[-1, 1]$:
     $$x_{\text{norm}} = \frac{x}{255.0} \quad \text{or} \quad x_{\text{norm}} = \frac{x}{127.5} - 1.0$$
     This normalization ensures that input features have similar scales, preventing early gradients from exploding.
2. **Natural Language Processing (Sequential Text Domain):**
   - **Tokenization:** Convert raw text strings into sequences of subword integers using a BPE or WordPiece tokenizer.
   - **Padding and Truncation:** Sentences have different lengths. To compile batches, we must pad shorter sequences with zero tokens and truncate longer sequences to a target length $T$.
     - **Pre-padding:** Adds zeros to the beginning of the sequence. LSTMs generally perform better with pre-padding because the active information is situated close to the end of the sequence, near the classification decision.
     - **Post-padding:** Adds zeros to the end of the sequence.
3. **Time-Series Forecasting (Sequential Continuous Domain):**
   - **Feature Scaling:** Scale continuous features (price, temperature) using `StandardScaler` or `MinMaxScaler` based on training set statistics.
   - **Windowing:** Convert the chronological array into a 3D tensor of shape `(Samples, Timesteps, Features)` using sliding windows.

```
       Raw Data (Messy) ---> [ Domain Preprocessor ] ---> Normalized Tensors
                                                               |
                                   - Image: Scale to [0,1], shape (H, W, C)
                                   - Text: Tokenize & Pad, shape (T,)
                                   - Time-Series: Sliding Window, shape (T, D)
```

### Trade-offs
A key trade-off in preprocessing is between CPU pre-computation and GPU runtime pipeline mapping.
- **Pre-computation:** Preprocessing the entire dataset beforehand and saving it to disk is simple but inflexible: if you decide to change the image size or tokenization, you must recalculate the entire database.
- **Runtime Pipelines:** Using Keras preprocessing layers or the `tf.data` API processes data dynamically in memory. This is highly flexible and integrates preprocessing into the model graph, but can introduce CPU bottlenecks if not optimized.

### Real-World Applications (Rule of 4)

1. **Example 1: Image Pixel Scaling**
   - **Input/Scenario:** An RGB image has pixel intensities $I(x, y) = [255, 128, 0]^T$ (orange). We normalize it to the range $[-1, 1]$.
   - **Expected Output:** The scaled pixel values are:
     $$I_{\text{scaled}} = \left[ \frac{255}{127.5}-1, \frac{128}{127.5}-1, \frac{0}{127.5}-1 \right]^T \approx [1.0, 0.004, -1.0]^T$$
2. **Example 2: Text Padding Alignment (LSTM)**
   - **Input/Scenario:** A sentence has 3 words: `["I", "love", "DL"]`, tokenized as `[12, 85, 302]`. The target sequence length is $T = 5$.
   - **Expected Output:**
     - Pre-padding: The input tensor is `[0, 0, 12, 85, 302]`.
     - Post-padding: The input tensor is `[12, 85, 302, 0, 0]`.
     For LSTMs, pre-padding is selected to ensure the active embeddings are close to the final hidden state output.
3. **Example 3: Tabular Time-Series Windowing**
   - **Input/Scenario:** A developer scales stock volume ($0$ to $10,000,000$ shares) using `MinMaxScaler` to fit the range $[0, 1]$.
   - **Expected Output:** The massive volumes are scaled down, preventing the network weights from being dominated by volume scales compared to stock prices (range $100$ to $200$).
4. **Example 4: Code Implementation of Image Pipeline**
   - **Input/Scenario:** A student sets up an image input pipeline in Keras.
   - **Expected Output:**
     ```python
     preprocess = keras.Sequential([
         keras.layers.Resizing(224, 224),
         keras.layers.Rescaling(1.0 / 255.0)
     ])
     ```
     The pipeline resizes and scales batches of images automatically.

> **Metacognitive Checkpoint:** Why is pre-padding generally preferred over post-padding when preparing text sequences for Recurrent Neural Networks (LSTMs/GRUs)? Explain in terms of state decay over padded zeros.

---

## Topic 2: Architectural Design: Choosing CNN, LSTM, or Functional Ensembles

### Rationale and Mechanics
Once the input tensors are structured, you must select the appropriate architectural inductive bias. The architecture must match the geometry of the data:
- **Spatial Data (Images):** Requires 2D Convolutions (`Conv2D` + `MaxPooling2D`) to capture spatial hierarchies and translation invariance.
- **Sequential Data (Text/Time-Series):** Requires recurrent layers (`LSTM` or `GRU`) to capture temporal order, or `Conv1D` for fast local feature extraction.
- **Heterogeneous Data (Multimodal):** Requires the Keras Functional API to branch the network, processing each input separately before merging.

Under the hood, when designing your network, follow these guidelines:
1. **Start Simple:** Build a small baseline model first. For a CNN, start with 2 convolutional layers and a single dense layer. For an LSTM, start with a single layer of 32 units.
2. **Check Capacity:** Evaluate training performance. If the baseline model cannot fit the training data (high training loss), increase model capacity by adding layers (depth) or neurons (width).
3. **Transition to Functional if Branching:** If the task requires skip connections (to train deep networks without vanishing gradients) or multiple inputs/outputs, transition immediately from the Sequential API to the Functional API.

```
       Identify Data Geometry:
       - 2D Grid (Image) --------> Use Conv2D Stack
       - 1D Sequence (Time) -----> Use LSTM or GRU Stack
       - Multi-Modal ------------> Use Functional API Concatenation
```

### Trade-offs
The primary trade-off in design is **Expressive Capacity** vs. **Optimization Difficulty**:
- Deep, wide networks can fit complex, non-linear relationships.
- However, they contain more parameters, making them prone to overfitting, slow to train, and sensitive to vanishing or exploding gradients.
You must balance capacity and regularization: only add layers if the validation loss continues to decrease along with the training loss.

### Real-World Applications (Rule of 4)

1. **Example 1: CNN Classifier Design (Sequential)**
   - **Input/Scenario:** A student designs a custom CNN to classify 10 types of waste from images.
   - **Expected Output:**
     ```python
     model = keras.Sequential([
         keras.layers.Input(shape=(128, 128, 3)),
         keras.layers.Conv2D(32, (3, 3), activation='relu'),
         keras.layers.MaxPooling2D((2, 2)),
         keras.layers.Conv2D(64, (3, 3), activation='relu'),
         keras.layers.MaxPooling2D((2, 2)),
         keras.layers.GlobalAveragePooling2D(),
         keras.layers.Dense(10, activation='softmax')
     ])
     ```
     The architecture extracts spatial features and downsamples them, outputting 10 probabilities.
2. **Example 2: LSTM Sequence Classifier Design**
   - **Input/Scenario:** A student designs a network to predict machinery failure using 5 sensor inputs over 100 timesteps.
   - **Expected Output:**
     ```python
     model = keras.Sequential([
         keras.layers.Input(shape=(100, 5)),
         keras.layers.LSTM(64, return_sequences=True),
         keras.layers.LSTM(32),
         keras.layers.Dense(1, activation='sigmoid')
     ])
     ```
     The stacked LSTM processes the sequence, outputting a binary failure probability.
3. **Example 3: Multimodal Functional Merger**
   - **Input/Scenario:** A student predicts loan default using both user text descriptions and numerical transaction history.
   - **Expected Output:** The student uses the Functional API to build a text LSTM branch and a tabular Dense branch, concatenating them using `keras.layers.concatenate` to make the final prediction.
4. **Example 4: Receptive Field Alignment (Conv1D)**
   - **Input/Scenario:** A student processes audio clips using a Conv1D layer. The sample rate is high, and features span 10 timesteps.
   - **Expected Output:** The student sets `kernel_size=11` to match the temporal receptive field to the target feature size.

> **Metacognitive Checkpoint:** Why should you start your deep learning project with a simple baseline model before building a complex architecture? Explain how this baseline helps debug optimization issues.

---

## Topic 3: Implementing a Strict Regularization and Training Configuration

### Rationale and Mechanics
After designing the architecture, you must configure the training loop to prevent overfitting. This configuration consists of:
- **Optimizer:** Adam is the default choice for general convergence.
- **Loss:** Match the loss to the target format (e.g., `binary_crossentropy` for binary classification, `sparse_categorical_crossentropy` for integer multi-class targets, and `mse` for regression).
- **Regularization:** Insert `Dropout` layers (typically rate $0.2$ to $0.5$) after large dense layers and add L2 weight decay to key convolutional or dense weights.
- **Callbacks:** Use Keras Callbacks to automate training adjustments.

Under the hood, two critical callbacks must be configured:
1. **EarlyStopping:** Monitors validation loss and stops training once overfitting starts, restoring the best weights:
   ```python
   early_stop = keras.callbacks.EarlyStopping(
       monitor='val_loss',
       patience=5,
       restore_best_weights=True
   )
   ```
2. **ReduceLROnPlateau (Learning Rate Decay):** During training, a constant learning rate (e.g., $\eta = 10^{-3}$) can cause the optimizer to overshoot narrow minima in the loss landscape. This callback monitors validation loss and reduces the learning rate by a factor (e.g., halving it) when improvement stalls, allowing the optimizer to settle:
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

### Trade-offs
Adding adaptive learning rate schedulers and early stopping makes training highly robust. It eliminates the need to guess the exact number of epochs or tune learning rate schedules manually.

The trade-off is the addition of hyperparameters: you must choose the patience, decay factor, and monitoring metric. If the decay patience is too small, the learning rate will drop to its minimum before the model has had time to escape early saddle points, stalling training.

### Real-World Applications (Rule of 4)

1. **Example 1: Compiler Configuration**
   - **Input/Scenario:** We classify images into 3 classes where targets are integers: `[0, 1, 2]`. We compile the model.
   - **Expected Output:**
     ```python
     model.compile(
         optimizer=keras.optimizers.Adam(learning_rate=0.001),
         loss='sparse_categorical_crossentropy',
         metrics=['accuracy']
     )
     ```
2. **Example 2: Complete Fit Loop with Callbacks**
   - **Input/Scenario:** A student trains a model for 100 epochs, integrating early stopping and learning rate decay.
   - **Expected Output:**
     ```python
     callbacks = [
         keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
         keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
     ]
     model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks)
     ```
3. **Example 3: Learning Rate Decay Action**
   - **Input/Scenario:** The validation loss stops improving at epoch 12. Patience is 3.
   - **Expected Output:** At epoch 15, the `ReduceLROnPlateau` callback triggers, reducing the learning rate from $0.001$ to $0.0005$, allowing the optimizer to make finer adjustments.
4. **Example 4: Dropout and L2 Regularization in Layers**
   - **Input/Scenario:** A student adds regularization to a Dense layer.
   - **Expected Output:**
     ```python
     keras.layers.Dense(
         128, 
         activation='relu', 
         kernel_regularizer=keras.regularizers.l2(0.01)
     )
     keras.layers.Dropout(0.3)
     ```
     The weights are penalized, and 30% of activations are randomly zeroed out during training.

> **Metacognitive Checkpoint:** Why is the `ReduceLROnPlateau` callback effective at optimizing convergence in non-convex loss landscapes? Explain how reducing the learning rate helps the optimizer settle into narrow basins.

---

## Summary & Next Steps

- **Pipelines Format Input:** Preprocessing must scale pixels, pad sequences, or extract sliding windows to structure data as clean, normalized input tensors.
- **Architectures Match Data:** CNNs capture spatial features, LSTMs model temporal dependencies, and the Functional API merges heterogeneous inputs.
- **Callbacks Control Training:** Early Stopping halts training before overfitting occurs, and learning rate decay reduces the step size to optimize convergence.

In the next lesson, we will cover **Capstone Project Part 2 (Evaluation & Defense)**, learning how to analyze training trajectories, compare neural models against classical baselines, and defend your architectural choices.
