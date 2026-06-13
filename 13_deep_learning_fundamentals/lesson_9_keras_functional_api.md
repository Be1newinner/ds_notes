# Lesson 9: The Keras Functional API

## Introduction & The "Why"

In Lesson 6, we used the Keras Sequential API to build neural networks. While simple and elegant, the Sequential API is structurally limited: it assumes the network is a single, straight line of layers with one input and one output. In real-world applications, neural network architectures are rarely straight lines.

Many advanced deep learning models require complex topologies. For example, to predict a house's value, we might want to use both tabular features (square footage, bedrooms) and an image of the house. This requires a **multi-input** model. Similarly, we might want to predict both the house's price (a regression task) and whether it will sell in under a week (a classification task). This requires a **multi-output** model. Furthermore, modern deep architectures like ResNet rely on **skip connections**, where a layer's input is added directly to its output to assist gradient flow.

To build these non-linear architectures, Keras provides the **Functional API**. This API treats layers as mathematical functions that accept and return symbolic tensors, allowing us to build networks as Directed Acyclic Graphs (DAGs). This lesson covers the mechanics of the functional paradigm, demonstrates how to build multi-input and multi-output models, and explains the implementation of skip connections.

---

## Topic 1: The Functional Paradigm: Symbolic Tensors and Directed Acyclic Graphs (DAGs)

### Rationale and Mechanics
The Keras Functional API is built on a functional programming paradigm. Instead of instantiating a model container and appending layers to it, we define layers as functions and explicitly pass tensors through them.

Under the hood, we begin by instantiating an `Input` placeholder. This is a **Symbolic Tensor**—it does not contain actual data, but defines the shape and data type of the tensors that will flow through the network at runtime:
```python
from tensorflow import keras

inputs = keras.layers.Input(shape=(D,))
```

We then define layers and call them as functions, passing the symbolic tensors through them. Each function call returns a new symbolic tensor:
```python
x = keras.layers.Dense(64, activation='relu')(inputs)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
```

Finally, we instantiate the `Model` object by explicitly specifying the input and output nodes of our computational graph:
```python
model = keras.Model(inputs=inputs, outputs=outputs)
```

Keras builds the network topology by tracing the symbolic connections from the specified `inputs` to the `outputs`. This network is represented as a **Directed Acyclic Graph (DAG)**, where nodes represent operations (layers) and edges represent the flow of tensors.

### Python Code Implementation
Here is a Python script illustrating the architectural equivalence between the Sequential API and the Functional API, verifying they build models with the exact same parameter structure:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Build a simple feedforward network using Sequential API
seq_model = keras.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 2. Build the equivalent network using Functional API
inputs = layers.Input(shape=(10,))
x = layers.Dense(32, activation='relu')(inputs)
outputs = layers.Dense(1, activation='sigmoid')(x)
func_model = keras.Model(inputs=inputs, outputs=outputs)

# Output summary comparison
print("--- Sequential API Parameter Count ---")
print("Total Params:", seq_model.count_params())

print("\n--- Functional API Parameter Count ---")
print("Total Params:", func_model.count_params())

# Verify they match
assert seq_model.count_params() == func_model.count_params(), "Parameters do not match!"
print("\nValidation Succeeded: Both models are structurally identical.")
```

### Trade-offs
The Functional API is slightly more verbose than the Sequential API because we must explicitly define the input layer and assign variables to every intermediate tensor.

The trade-off is absolute architectural flexibility. Because we handle tensors directly, we can reuse layers (e.g., passing two different input tensors through the same weight matrix) and branch the network in any configuration. As long as there are no cycles (feedback loops) in the graph, Keras can compile and train the resulting DAG.

### Real-World Applications (Rule of 4)

1. **Example 1: Functional Equivalence to Sequential**
   - **Input/Scenario:** We build a simple feedforward network with 10 inputs, a hidden layer of 32 units, and a single sigmoid output using the Functional API.
   - **Expected Output:** The code is:
     ```python
     inputs = keras.layers.Input(shape=(10,))
     x = keras.layers.Dense(32, activation='relu')(inputs)
     outputs = keras.layers.Dense(1, activation='sigmoid')(x)
     model = keras.Model(inputs=inputs, outputs=outputs)
     ```
     This creates a model structurally identical to a Sequential stack but represented as a compiled graph.
2. **Example 2: Reusing a Hidden Layer (Shared Weights)**
   - **Input/Scenario:** We build a model that processes two different feature vectors (e.g., Features A and Features B) using the exact same feature extraction weights.
   - **Expected Output:**
     ```python
     shared_dense = keras.layers.Dense(32, activation='relu')
     input_a = keras.layers.Input(shape=(10,))
     input_b = keras.layers.Input(shape=(10,))
     features_a = shared_dense(input_a)
     features_b = shared_dense(input_b)
     ```
     Both branches share the same weight parameters, forcing the model to learn a unified representation for both input channels.
3. **Example 3: Direct Access to Intermediate Activations**
   - **Input/Scenario:** A developer wants to extract the features learned by a hidden layer in an existing model to use them for visualization.
   - **Expected Output:**
     ```python
     feature_extractor = keras.Model(inputs=model.input, outputs=model.layers[1].output)
     latent_features = feature_extractor.predict(X_val)
     ```
     The Functional API allows the developer to instantiate a new model using any internal nodes of the original graph.
4. **Example 4: Graph Compilation and Validation**
   - **Input/Scenario:** A developer builds a model where a layer's output is passed back into a prior layer, creating a cyclic loop.
   - **Expected Output:** During model instantiation, Keras raises a `ValueError` or enters an infinite loop, indicating that the graph is not acyclic. This forces the developer to maintain a feedforward structure.

> **Metacognitive Checkpoint:** What is a symbolic tensor? How does it differ from a standard NumPy array or TensorFlow constant during model definition and training?

---

## Topic 2: Multi-Input Architectures: Joining Tabular and Image Data

### Rationale and Mechanics
In many industrial classification and regression tasks, a single data source is insufficient. For instance, predicting patient health outcomes is more accurate if we combine tabular medical records (age, blood pressure) with an medical image (X-ray).

To handle this, we build a **Multi-Input Model** using the Functional API. We design separate branches to extract features from each input type, fuse the extracted representations, and pass the combined feature vector to downstream layers.

Under the hood, let's examine the fusion step. We define two input branches:
1. **Tabular Branch:** Processes a vector of shape $(B, D)$ through dense layers.
2. **Image Branch:** Processes an image of shape $(B, H, W, C)$ through convolutional layers, outputting a flattened feature vector of shape $(B, F)$.

To combine these representations, we use a `Concatenate` layer:
```python
merged_features = keras.layers.concatenate([tabular_features, image_features])
```

The `concatenate` operation joins tensors along a specified axis. For two 2D tensors of shapes $(B, D)$ and $(B, F)$, the concatenation along the feature axis (axis -1) yields a single tensor of shape:
$$\text{Output Shape} = (B, D + F)$$

This fused representation is then passed to downstream dense layers to make the final prediction.

### Python Code Implementation
Here is a Python script demonstrating how to construct, compile, and train a Multi-Input model in Keras using synthetic tabular and image-like data:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Generate synthetic data
# 100 samples, 5 tabular features
X_tabular = np.random.randn(100, 5)
# 100 samples, 16x16 grayscale images
X_images = np.random.randn(100, 16, 16, 1)
# Target: binary labels
y = np.random.randint(0, 2, size=(100, 1))

# 2. Design the Multi-Input model
tabular_input = layers.Input(shape=(5,), name='tabular_input')
image_input = layers.Input(shape=(16, 16, 1), name='image_input')

# Tabular processing branch
x_tab = layers.Dense(16, activation='relu')(tabular_input)

# Image processing branch (Simple CNN representation)
x_img = layers.Conv2D(8, kernel_size=(3, 3), activation='relu')(image_input)
x_img = layers.Flatten()(x_img)
x_img = layers.Dense(16, activation='relu')(x_img)

# Combine representations
merged = layers.concatenate([x_tab, x_img], name='feature_fusion')

# Output branch
outputs = layers.Dense(1, activation='sigmoid', name='prediction')(merged)

# Model instantiation
multimodal_model = keras.Model(inputs=[tabular_input, image_input], outputs=outputs)

# Compile the model
multimodal_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Inspect the topology
multimodal_model.summary()

# Train the model by passing a list of inputs
print("\n--- Starting Model Training ---")
history = multimodal_model.fit(
    [X_tabular, X_images], y,
    epochs=3,
    batch_size=16,
    verbose=1
)
```

### Trade-offs
Multi-input models allow the network to learn interactions between different data modalities, significantly improving performance over models trained on a single source.

The trade-off is data alignment and training complexity.
- **Data Alignment:** Every training sample must contain both inputs. If a patient has an X-ray but is missing their tabular records, the sample cannot be easily processed unless missing data is imputed.
- **Branch Balancing:** The branches might learn at different rates. If the image branch is complex and trains slowly, the optimizer might focus entirely on the tabular branch, leading to sub-optimal performance. Tuning these networks requires careful initialization and sometimes separate learning rates per branch.

### Real-World Applications (Rule of 4)

1. **Example 1: Real Estate Valuation (Table + Image)**
   - **Input/Scenario:** We predict house price using 10 tabular features and a $64\times64\times3$ satellite image of the property.
   - **Expected Output:**
     ```python
     tab_in = keras.layers.Input(shape=(10,))
     img_in = keras.layers.Input(shape=(64, 64, 3))
     
     x_tab = keras.layers.Dense(16, activation='relu')(tab_in)
     x_img = keras.layers.Flatten()(img_in)
     x_img = keras.layers.Dense(32, activation='relu')(x_img)
     
     merged = keras.layers.concatenate([x_tab, x_img])
     outputs = keras.layers.Dense(1)(merged)
     model = keras.Model(inputs=[tab_in, img_in], outputs=outputs)
     ```
     The model accepts a list of two inputs: `model.fit([X_tab, X_img], y)`.
2. **Example 2: Multimodal Sentiment Analysis (Text + Audio)**
   - **Input/Scenario:** A customer service model predicts sentiment using a 300-dimensional text embedding and a 40-dimensional audio feature vector.
   - **Expected Output:** The text features and audio features are concatenated to form a 340-dimensional fused vector. This vector is passed to a dense classifier, allowing the model to detect sarcasm (e.g., positive words spoken with negative tone).
3. **Example 3: E-commerce Recommendation (User Profile + Item Image)**
   - **Input/Scenario:** A recommendation system matches user tabular profiles (age, history) to item images to predict purchase probability.
   - **Expected Output:** The network learns a joint representation space. If the user profile indicates an interest in sports, and the item image depicts running shoes, the combined activations trigger a high purchase probability output.
4. **Example 4: Missing Input Handling via Zero Padding**
   - **Input/Scenario:** A multi-input model is deployed, but 10% of testing samples are missing image inputs.
   - **Expected Output:** The developer feeds a placeholder image of zeros (`np.zeros((64, 64, 3))`) for missing inputs. The network relies entirely on the tabular branch for these samples, maintaining system uptime.

> **Metacognitive Checkpoint:** If you concatenate a tensor of shape $(32, 64)$ and a tensor of shape $(32, 128)$ along the feature axis (axis 1), what is the shape of the resulting tensor? What would happen if the batch dimensions (axis 0) of the two tensors did not match?

---

## Topic 3: Multi-Output and Skip-Connection Architectures

### Rationale and Mechanics
In multi-task learning, we train a single model to perform multiple predictions simultaneously. For example, a self-driving car network must predict steering angle (regression) and detect traffic signs (classification).

Using the Functional API, we can branch the network at the output stage, creating **Multi-Output Models**. We define a shared feature extractor and split the architecture into separate output heads:
```python
shared = keras.layers.Dense(64, activation='relu')(inputs)
price_out = keras.layers.Dense(1, name='price')(shared)
class_out = keras.layers.Dense(3, activation='softmax', name='class')(shared)

model = keras.Model(inputs=inputs, outputs=[price_out, class_out])
```

During compilation, we specify a loss function for each output and assign weights to balance them:
```python
model.compile(
    optimizer='adam',
    loss={'price': 'mse', 'class': 'categorical_crossentropy'},
    loss_weights={'price': 1.0, 'class': 0.2}
)
```
The total loss optimized by the network is:
$$\mathcal{L}_{\text{total}} = w_{\text{price}} \mathcal{L}_{\text{price}} + w_{\text{class}} \mathcal{L}_{\text{class}}$$

The Functional API also allows us to build **Skip Connections** (or residual connections). In deep networks, gradients decay as they backpropagate through layers. A skip connection bypasses one or more layers, adding the input directly to the output:
$$\mathbf{y} = f(\mathbf{x}) + \mathbf{x}$$

We implement this in Keras using the `Add` layer:
```python
x = keras.layers.Dense(32, activation='relu')(inputs)
# Residual addition
x = keras.layers.add([x, inputs])
outputs = keras.layers.Activation('relu')(x)
```
During backpropagation, the gradient flows directly through the addition operation to the earlier layers without decay, enabling the training of networks with hundreds of layers.

### Python Code Implementation
Here is a Python script implementing a model that features a residual skip connection and branches into two output heads (one regression output and one multi-class output):

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Define input (8 features)
inputs = layers.Input(shape=(8,), name='network_input')

# Hidden layer 1
x = layers.Dense(8, activation='relu', name='dense_1')(inputs)

# Skip connection: add input vector directly to the output of dense_1
# Both must have the same dimension (8,)
shortcut = layers.add([x, inputs], name='residual_addition')
x = layers.Activation('relu', name='residual_activation')(shortcut)

# 2. Branch into two output heads
regression_output = layers.Dense(1, name='reg_output')(x)
classification_output = layers.Dense(3, activation='softmax', name='class_output')(x)

# 3. Create model
multi_task_model = keras.Model(inputs=inputs, outputs=[regression_output, classification_output])

# 4. Compile with target-specific loss functions and weights
multi_task_model.compile(
    optimizer='adam',
    loss={
        'reg_output': 'mse',
        'class_output': 'sparse_categorical_crossentropy'
    },
    loss_weights={
        'reg_output': 1.0,
        'class_output': 0.5
    }
)

multi_task_model.summary()

# 5. Dummy training run
X_dummy = np.random.randn(100, 8)
y_reg = np.random.randn(100, 1)
y_class = np.random.randint(0, 3, size=(100, 1))

print("\n--- Training Multi-Task Model ---")
multi_task_model.fit(
    X_dummy,
    {'reg_output': y_reg, 'class_output': y_class},
    epochs=3,
    batch_size=16,
    verbose=1
)
```

### Trade-offs
- **Multi-Output Trade-off:** Multi-task learning acts as a regularizer, forcing the network to learn generalized features in the shared layers. However, tuning the loss weights $w_i$ is difficult. If one weight is too large, the optimizer will focus on that task and ignore the others.
- **Skip Connection Trade-off:** Bypassing layers prevents vanishing gradients, but requires the added tensors to have identical shapes. If Layer output $f(\mathbf{x})$ has shape $(32, 64)$ and $\mathbf{x}$ has shape $(32, 128)$, you must project $\mathbf{x}$ to shape $(32, 64)$ using a $1\times1$ convolutional or dense layer before adding them, increasing model complexity.

### Real-World Applications (Rule of 4)

1. **Example 1: Multi-Task Demographics Predictor**
   - **Input/Scenario:** A face analysis network predicts both age (regression) and gender (binary classification) from a shared feature representation.
   - **Expected Output:**
     - `loss={'age': 'mae', 'gender': 'binary_crossentropy'}`
     - `loss_weights={'age': 0.01, 'gender': 1.0}` (scaled to balance output magnitudes).
     The model optimizes both heads simultaneously, saving compute compared to running two separate models.
2. **Example 2: ResNet Residual Block**
   - **Input/Scenario:** We build a deep ResNet-style block where input features are passed through two dense layers and then added back to the input.
   - **Expected Output:**
     ```python
     inputs = keras.layers.Input(shape=(64,))
     x = keras.layers.Dense(64, activation='relu')(inputs)
     x = keras.layers.Dense(64)(x)
     x = keras.layers.add([x, inputs])
     outputs = keras.layers.Activation('relu')(x)
     ```
     The gradient flows unimpeded through the identity path, ensuring the model trains successfully even at extreme depths.
3. **Example 3: Auto-Encoder with Skip Connections (U-Net style)**
   - **Input/Scenario:** An image segmentation network links encoder layers directly to decoder layers using skip connections to preserve spatial details.
   - **Expected Output:** The decoder combines high-level semantic features with low-level spatial coordinates (via skip path concatenation), producing precise, pixel-level boundaries.
4. **Example 4: Unbalanced Multi-Loss Failure**
   - **Input/Scenario:** A developer compiles a multi-output model with MSE loss (magnitude $\approx 10,000$) and binary cross-entropy loss (magnitude $\approx 0.5$) using equal weights ($1.0$).
   - **Expected Output:** The optimizer focuses entirely on reducing the MSE loss, and validation accuracy for the classification task degrades. The developer must scale the classification loss weight to $20,000$ to balance the gradients.

> **Metacognitive Checkpoint:** Why are skip connections mathematically effective at preventing vanishing gradients in very deep networks? Analyze the derivative of the residual equation $\mathbf{y} = f(\mathbf{x}) + \mathbf{x}$ with respect to $\mathbf{x}$.

---

## Summary & Next Steps

- **Functional API Builds Graphs:** Treating layers as functions allows us to define neural networks as Directed Acyclic Graphs (DAGs) using symbolic input and output tensors.
- **Multi-Input Fuses Data:** Using `Concatenate` layers allows us to combine features from different data sources (e.g., tables and images) into a single representation.
- **Multi-Output & Skip Connections Expand Topologies:** Multi-task models make multiple predictions from shared layers, balanced by loss weights. Skip connections bypass layers to assist gradient flow.

In the next lesson, we will conquer spatial data, transitioning to **Computer Vision** to learn why dense networks fail on images and how convolutional layers extract spatial feature hierarchies.
