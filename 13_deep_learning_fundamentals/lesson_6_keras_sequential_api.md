# Lesson 6: The Keras Sequential API

## Introduction & The "Why"

In classical machine learning with Scikit-Learn, training a model is straightforward: you instantiate a classifier and call `.fit(X, y)`. The library manages weight initialization and optimization under the hood. In deep learning, we design custom architectures by specifying the number of layers, their dimensions, and how they connect. To do this efficiently, we use Keras, a high-level deep learning API that runs on top of TensorFlow.

Before building a model, we must translate our raw data into multi-dimensional arrays called **tensors** and define their shapes. Keras provides two main interfaces for building models: the Sequential API and the Functional API. This lesson covers the **Keras Sequential API**, which lets you build feedforward neural networks layer-by-layer in a straight line. We will learn how to map Scikit-Learn preprocessing outputs to Keras tensor shapes, design a Multi-Layer Perceptron, compile it with an optimizer and loss function, and execute the training loop using `model.fit()`.

---

## Topic 1: Translating Tabular Data to Tensor Shapes

### Rationale and Mechanics
In classical machine learning, we organize tabular data as a 2D matrix (samples $\times$ features) in a Pandas DataFrame or a NumPy array. In deep learning, all data is represented as multi-dimensional arrays called **Tensors**. A tensor is a generalization of vectors and matrices to an arbitrary number of dimensions (ranks).

Before passing tabular data to a neural network, we must preprocess it using standard techniques (like Scikit-Learn's `StandardScaler` or `OneHotEncoder`). Once preprocessed, we must map these matrices to the expected tensor shapes of our network's input layer.

Under the hood, if we have a tabular dataset with $N$ samples and $D$ features, the complete dataset forms a tensor of shape:
$$\text{Shape} = (N, D)$$

When defining the input layer in Keras, we use the `Input` layer or `input_shape` argument. Crucially, we specify the shape of a **single sample**, excluding the batch dimension:
$$\text{Input Shape} = (D,)$$

```
        Raw Data Matrix (N samples, D features)
        
        [  f1   f2   f3  ...  fD  ] - Sample 1
        [  f1   f2   f3  ...  fD  ] - Sample 2
        [  ...  ...  ...  ...  ... ]
        [  f1   f2   f3  ...  fD  ] - Sample N
        
        Mapped to Keras: Input(shape=(D,))  <-- Batch dimension is dynamic
```

We omit the batch dimension $N$ because it is dynamic: the network can process a single sample during inference, or batches of 32, 64, or 128 samples during training. Keras handles the batch dimension automatically at runtime, representing a batch of size $B$ as a tensor of shape $(B, D)$.

### Trade-offs
Defining input shapes explicitly is a best practice in modern Keras. It allows the framework to construct and initialize all weight matrices immediately. If you omit the input shape, Keras defers weight creation until you pass the first batch of data. This delayed initialization prevents you from inspecting the model summary or saving the model before training.

The trade-off of using standard NumPy/Scikit-Learn matrices directly is that they are loaded entirely into CPU memory. For large datasets that exceed system RAM, this causes crashes. In such cases, we must replace NumPy arrays with tf.data.Dataset generators, which stream batches from disk to GPU memory on the fly.

### Real-World Applications (Rule of 4)

1. **Example 1: Credit Card Default Classification**
   - **Input/Scenario:** A tabular dataset contains 15 continuous features (credit limit, payment history, etc.) and 3 one-hot encoded categorical features. The training set has 10,000 samples.
   - **Expected Output:** The preprocessed training matrix has shape $(10000, 18)$. The input layer in Keras is defined as `keras.layers.Input(shape=(18,))`. When training with a batch size of 32, the input tensor shape at runtime is $(32, 18)$.
2. **Example 2: House Price Regression**
   - **Input/Scenario:** A regression dataset contains 8 continuous features (square footage, bedrooms, age, etc.). We preprocess the features using `StandardScaler`.
   - **Expected Output:** The preprocessed input matrix has shape $(N, 8)$. The Keras input layer is instantiated as `Input(shape=(8,))`. A single prediction query passes a tensor of shape $(1, 8)$.
3. **Example 3: Multi-Class Sensor Classification**
   - **Input/Scenario:** An IoT sensor dataset records 120 features from accelerometers. The training set has 50,000 samples.
   - **Expected Output:** The input data matrix has shape $(50000, 120)$. The input layer is defined with `shape=(120,)`. During training with batch size 128, the input batch tensor has shape $(128, 120)$.
4. **Example 4: Dimension Mismatch Error**
   - **Input/Scenario:** A developer defines a model with `Input(shape=(10,))` but passes a preprocessed matrix of shape $(100, 11)$ due to an unaligned column in the dataset.
   - **Expected Output:** Keras raises a `ValueError: Input 0 of layer dense is incompatible with the layer: expected axis -1 of input shape to have value 10, but received input with shape (100, 11)`. This forces the developer to align the feature count.

> **Metacognitive Checkpoint:** Why do we exclude the batch dimension when defining the input shape of a Keras model? Explain how a model trained on a batch size of 32 can make predictions on a single sample during production deployment.

---

## Topic 2: Designing the Architecture with Keras Sequential

### Rationale and Mechanics
In classical machine learning, we instantiate classifiers as single units. In deep learning, we design architectures by stacking layers. The **Keras Sequential API** allows us to build neural networks by stacking layers in a linear sequence, where the output of each layer becomes the input of the next.

Under the hood, a typical feedforward neural network (Multi-Layer Perceptron) is built using the `keras.Sequential` class and `keras.layers.Dense` layers:
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(D,)),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])
```

Let's analyze the shapes of the parameters (weights and biases) initialized by this model:
- **Input Layer:** Does not contain parameters. It simply passes the input tensor of shape $(B, D)$ forward.
- **First Hidden Layer (64 units):** Connects $D$ inputs to 64 outputs. 
  - Weight matrix shape: $(D, 64)$
  - Bias vector shape: $(64,)$
  - Total parameters: $D \times 64 + 64$
- **Second Hidden Layer (32 units):** Connects 64 inputs to 32 outputs.
  - Weight matrix shape: $(64, 32)$
  - Bias vector shape: $(32,)$
  - Total parameters: $64 \times 32 + 32 = 2080$
- **Output Layer (1 unit):** Connects 32 inputs to 1 output.
  - Weight matrix shape: $(32, 1)$
  - Bias vector shape: $(1,)$
  - Total parameters: $32 \times 1 + 1 = 33$

The activation argument (e.g., `'relu'`, `'sigmoid'`) determines the non-linear function applied to the pre-activations of each layer.

### Trade-offs
The Sequential API is highly readable and easy to debug, making it the default choice for standard feedforward and simple convolutional networks.

The trade-off is structural. The Sequential API is strictly limited to architectures with a **single input tensor** and a **single output tensor** connected in a straight line. It cannot represent:
- Multi-input models (e.g., combining tabular data and an image).
- Multi-output models (e.g., predicting both house price and property type).
- Shared layers (e.g., Siamese networks).
- Skip connections or residual paths (e.g., ResNet blocks), where the output of a layer is added to a downstream layer.
For these complex architectures, you must transition to the Functional API.

### Real-World Applications (Rule of 4)

1. **Example 1: Binary Classifier (Churn Prediction)**
   - **Input/Scenario:** A dataset contains 20 preprocessed features. We design an MLP with two hidden layers (128 and 64 units) and a binary output.
   - **Expected Output:** The code is:
     ```python
     model = keras.Sequential([
         keras.layers.Input(shape=(20,)),
         keras.layers.Dense(128, activation='relu'),
         keras.layers.Dense(64, activation='relu'),
         keras.layers.Dense(1, activation='sigmoid')
     ])
     ```
     This initializes $(20 \cdot 128 + 128) + (128 \cdot 64 + 64) + (64 \cdot 1 + 1) = 2688 + 8256 + 65 = 11,009$ trainable parameters.
2. **Example 2: Multi-Class Classifier (Product Categorization)**
   - **Input/Scenario:** We classify products into 5 categories based on 50 features. We use a hidden layer of 128 units.
   - **Expected Output:** The architecture is:
     ```python
     model = keras.Sequential([
         keras.layers.Input(shape=(50,)),
         keras.layers.Dense(128, activation='relu'),
         keras.layers.Dense(5, activation='softmax')
     ])
     ```
     This outputs 5 probabilities summing to $1.0$ using the Softmax activation.
3. **Example 3: Regression Model (House Prices)**
   - **Input/Scenario:** We predict house price based on 10 features. We use hidden layers of 32 and 16 units.
   - **Expected Output:** The code is:
     ```python
     model = keras.Sequential([
         keras.layers.Input(shape=(10,)),
         keras.layers.Dense(32, activation='relu'),
         keras.layers.Dense(16, activation='relu'),
         keras.layers.Dense(1) # Linear activation (default)
     ])
     ```
     The output layer contains 1 unit with linear activation, allowing it to output unbounded continuous values.
4. **Example 4: Inspecting Model Topology**
   - **Input/Scenario:** A developer builds a model and calls `model.summary()`.
   - **Expected Output:** Keras prints a text table detailing each layer's name, output shape, and parameter count. This allows the developer to verify that the layer shapes match their design.

> **Metacognitive Checkpoint:** Given a Sequential model with input shape `(100,)`, a hidden layer with `50` units, and an output layer with `10` units, calculate the exact number of weights and biases in the model. Show your calculations step-by-step.

---

## Topic 3: Compiling and Fitting the Model

### Rationale and Mechanics
After defining the model architecture, we must configure its training settings. This configuration step is called **Compilation**. We specify the optimization algorithm, the loss function, and the evaluation metrics:
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Once compiled, we train the model parameters by calling `model.fit()`:
```python
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)
```

Under the hood, `model.fit()` executes the training loop over the specified number of **epochs**. In each epoch:
1. The training dataset is shuffled and partitioned into mini-batches of size `batch_size`.
2. For each mini-batch, the data is passed forward through the network to calculate predictions and loss.
3. Backpropagation calculates the gradients, and the optimizer updates the weights.
4. This batch update is repeated $N / B$ times (where $N$ is the dataset size and $B$ is the batch size) to complete one epoch.
5. At the end of the epoch, the validation dataset is passed forward to calculate the validation loss and metrics. Crucially, validation data is only used for evaluation; it does not update the weights.

The training history (losses and metrics for both training and validation sets) is returned in the `history` object, which can be plotted to monitor convergence.

### Trade-offs
Hyperparameter selection in `model.fit()` involves significant trade-offs:
- **Batch Size:** A small batch size (e.g., 16 or 32) introduces stochastic noise into the gradients. This noise acts as a regularizer, helping the optimizer escape saddle points, but is slower because it does not fully utilize GPU parallelization. A large batch size (e.g., 256 or 512) trains faster on GPUs but can lead to poorer generalization.
- **Epochs:** Training for too few epochs results in **underfitting** (the model has not fully learned the patterns). Training for too many epochs leads to **overfitting** (the model memorizes the training data, causing validation performance to degrade).

### Real-World Applications (Rule of 4)

1. **Example 1: Binary Classification Compile & Fit**
   - **Input/Scenario:** A model predicts customer churn. We compile it with Adam optimizer, binary cross-entropy, and accuracy metric, then train it for 15 epochs with a batch size of 64.
   - **Expected Output:**
     ```python
     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.1)
     ```
     Keras outputs training progress logs showing loss and accuracy decreasing and increasing, respectively, for both sets.
2. **Example 2: Multi-Class Classification Compile & Fit**
   - **Input/Scenario:** We classify images of handwritten digits (10 classes). The targets are one-hot encoded. We compile with categorical cross-entropy.
   - **Expected Output:**
     ```python
     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
     ```
     *(Note: If the labels were integers rather than one-hot encoded, we would compile with `loss='sparse_categorical_crossentropy'` to avoid manual encoding).*
3. **Example 3: Regression Compile & Fit**
   - **Input/Scenario:** We predict house prices. We compile the model using the RMSprop optimizer, Mean Squared Error loss, and Mean Absolute Error metric.
   - **Expected Output:**
     ```python
     model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
     model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2)
     ```
     The model optimizes parameters to minimize MSE, tracking progress in USD error using MAE.
4. **Example 4: Monitoring Overfitting via History**
   - **Input/Scenario:** A model is trained for 100 epochs. The training history is captured in `history = model.fit(...)`.
   - **Expected Output:** Plotting the training history shows the training loss decreasing steadily toward zero. The validation loss decreases until epoch 20, after which it begins to increase, indicating overfitting. This tells the developer to stop training at epoch 20.

> **Metacognitive Checkpoint:** If your training dataset contains 10,000 samples and you call `model.fit(..., batch_size=128, epochs=10)`, how many weight update steps (parameter updates) occur in total during the entire training process? Show your math.

---

## Summary & Next Steps

- **Tensors Represent Data:** Tabular datasets of shape $(N, D)$ map to input layers defined by `shape=(D,)`. The batch dimension is handled dynamically at runtime.
- **Sequential API Connects Layers:** `keras.Sequential` allows layers to be stacked linearly. Each `Dense` layer contains weights of shape $(inputs, units)$ and biases of shape $(units,)$.
- **Compile and Fit Configure Training:** Compilation sets the optimizer, loss, and metrics. Calling `model.fit()` executes mini-batch gradient descent over the specified epochs, monitoring progress on a validation set.

In the next lesson, we will explore **The Overparameterization Problem (Regularization)**, learning why neural networks memorize training data and how to prevent overfitting using Dropout, L1/L2 regularization, and Early Stopping.
