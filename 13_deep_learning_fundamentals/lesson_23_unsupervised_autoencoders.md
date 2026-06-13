# Lesson 23: Unsupervised Learning with Autoencoders

## Introduction & The "Why"

In previous modules, we trained models using supervised learning: every input sample was paired with a label ($y$), and the model optimized parameters to minimize classification or regression error. However, in many industrial settings, labeled data is scarce or non-existent. Labeling datasets requires manual human review, which is expensive and slow.

To exploit the massive amounts of unlabeled data available, we use **Unsupervised Learning**. The most fundamental deep learning architecture for unsupervised learning is the **Autoencoder**.

An Autoencoder is a neural network trained to reconstruct its input at its output layer. If the network was unrestricted, it would simply learn the identity function—copying the input pixels or values directly to the output. To force the network to learn meaningful features, we pass the inputs through a low-dimensional bottleneck. This bottleneck forces the network to compress the data, capturing its core structure. This lesson covers the architecture of undercomplete autoencoders, compares them mathematically to Principal Component Analysis (PCA), and explains how to use reconstruction error for unsupervised anomaly and fraud detection.

---

## Topic 1: Autoencoder Architecture: The Encoder, Bottleneck, and Decoder

### Rationale and Mechanics
In classical data compression, algorithms (like JPEG for images or MP3 for audio) discard high-frequency details to store files in smaller sizes. An Autoencoder performs a similar task: it learns a lossy compression pipeline directly from data.

An Autoencoder consists of two sequential networks:
1.  **The Encoder ($f_{\text{enc}}$):** Maps the high-dimensional input vector $\mathbf{x} \in \mathbb{R}^D$ to a low-dimensional latent space vector $\mathbf{z} \in \mathbb{R}^d$, where $d \ll D$:
    $$\mathbf{z} = f_{\text{enc}}(\mathbf{x})$$
2.  **The Decoder ($f_{\text{dec}}$):** Takes the compressed vector $\mathbf{z}$ and attempts to reconstruct the original input vector, outputting $\hat{\mathbf{x}} \in \mathbb{R}^D$:
    $$\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{z})$$

The low-dimensional vector $\mathbf{z}$ is called the **Bottleneck** or **Latent Representation**.

```
       Input x (D-dim) ---> [ Encoder ] ---> Bottleneck z (d-dim) ---> [ Decoder ] ---> Reconstructed x_hat (D-dim)
                                                     ^
                                                     |
                                            Information Bottleneck
```

Under the hood, let's write the mathematical updates for a single-layer feedforward autoencoder. The encoder projects the input:
$$\mathbf{z} = g\left( \mathbf{W}_e \mathbf{x} + \mathbf{b}_e \right)$$
where $\mathbf{W}_e$ is a weight matrix of shape $d \times D$.

The decoder reconstructs the input:
$$\hat{\mathbf{x}} = g\left( \mathbf{W}_d \mathbf{z} + \mathbf{b}_d \right)$$
where $\mathbf{W}_d$ is a weight matrix of shape $D \times d$.

We train the model by minimizing the difference between the input $\mathbf{x}$ and the output $\hat{\mathbf{x}}$. The most common objective function is the **Reconstruction Mean Squared Error (MSE)**:
$$\mathcal{L}_{\text{reconstruction}} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$$

If the bottleneck dimension $d$ was equal to or larger than $D$, the model could copy the inputs directly without learning anything. By forcing $d \ll D$ (called an **Undercomplete Autoencoder**), we create an information bottleneck that forces the network to learn the most salient features (the coordinates of the underlying data manifold) to minimize reconstruction loss.

### Python Code Implementation
The following code constructs a complete Autoencoder in Keras/TensorFlow, trains it on synthetic 10-dimensional data, compresses the input to a 2D bottleneck, and extracts the encoder and decoder components to show their outputs.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Generate synthetic data (320 samples, 10 dimensions)
np.random.seed(42)
data = np.random.randn(320, 10)

# 2. Define the Autoencoder model using Keras Functional API
input_dim = 10
latent_dim = 2

# Encoder Network
inputs = keras.Input(shape=(input_dim,), name="input_layer")
x = layers.Dense(6, activation="relu", name="enc_hidden")(inputs)
latent = layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)

# Decoder Network
decoder_hidden = layers.Dense(6, activation="relu", name="dec_hidden")
outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(decoder_hidden(latent))

# Full Autoencoder Model
autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
autoencoder.compile(optimizer="adam", loss="mse")

# 3. Train the Autoencoder
print("Training Autoencoder...")
autoencoder.fit(data, data, epochs=5, batch_size=16, verbose=1)

# 4. Extract Encoder as a separate model
encoder = keras.Model(inputs=inputs, outputs=latent, name="encoder_only")

# 5. Extract Decoder as a separate model
decoder_input = keras.Input(shape=(latent_dim,))
dec_x = decoder_hidden(decoder_input)
dec_out = layers.Dense(input_dim, activation="linear")(dec_x)
decoder = keras.Model(inputs=decoder_input, outputs=dec_out, name="decoder_only")

# Run compressions and reconstructions
sample_input = data[0:1]
compressed_z = encoder.predict(sample_input)
reconstructed_x = decoder.predict(compressed_z)

print("\nOriginal Input Shape:", sample_input.shape)
print("Compressed Bottleneck Vector:", np.round(compressed_z, 4))
print("Reconstructed Vector Shape:", reconstructed_x.shape)
```

### Trade-offs
*   **Advantages:** Unsupervised feature extraction requires zero labels. The learned latent vector $\mathbf{z}$ acts as a compressed representation that can be used for downstream tasks like visualization, clustering, or feeding to classical classifiers.
*   **Disadvantages:** The compression is lossy. If the bottleneck is too small, the reconstructed outputs will be blurry or lose key details. The encoder and decoder must be balanced; if the decoder is too complex, it can memorize representations even with a small bottleneck, rendering the bottleneck features useless.

### Real-World Applications (Rule of 4)
1.  **Example 1: Dense Dimension Reduction**
    *   **Input/Scenario:** A tabular dataset contains 100 features. We design an undercomplete autoencoder with a bottleneck of size 10.
    *   **Expected Output:** The input is compressed from 100 dimensions to 10 dimensions, which can be extracted for clustering.
2.  **Example 2: Image Denoising**
    *   **Input/Scenario:** We train an autoencoder where the inputs are noisy images (e.g., scans with grain) and target targets are clean original images.
    *   **Expected Output:** The model learns to ignore the random grain noise (which cannot be easily compressed through the bottleneck) and reconstructs only the clean structural shapes.
3.  **Example 3: Tabular Imputation**
    *   **Input/Scenario:** A dataset contains missing values (NaN) zeroed out.
    *   **Expected Output:** The decoder outputs a complete reconstruction, automatically imputing the missing values based on the learned relationships.
4.  **Example 4: Neural Compression Formats**
    *   **Input/Scenario:** A developer implements an image-to-bottleneck autoencoder to compress images for bandwidth-limited storage.
    *   **Expected Output:** The compressed bottleneck files require $90\%$ less bandwidth, while the decoder reconstructs readable images at the client end.

> **Metacognitive Checkpoint:** Why is it critical that the bottleneck dimension $d$ is strictly smaller than the input dimension $D$ in an undercomplete autoencoder? What would happen if $d \ge D$?

---

## Topic 2: Autoencoders vs. PCA: Linear vs. Non-linear Projection

### Rationale and Mechanics
In classical machine learning, Principal Component Analysis (PCA) is the default algorithm for dimensionality reduction. It projects data onto orthogonal axes (principal components) that maximize variance.

There is a direct mathematical connection between PCA and Autoencoders. If we build a single-layer undercomplete autoencoder using only **linear activation functions** ($g(z) = z$), the subspace learned by the bottleneck is equivalent to the subspace spanned by the first $d$ principal components of PCA.

Under the hood:
*   **PCA:** Projects data using an orthogonal projection matrix $\mathbf{P}$. The transformation is strictly linear:
    $$\mathbf{z} = \mathbf{P}\mathbf{x}$$
    PCA is restricted to finding linear flat subspaces (hyperplanes) in the feature space.
*   **Non-linear Autoencoder:** By introducing non-linear activation functions (like ReLU, Tanh, or Sigmoid) and stacking multiple hidden layers, the autoencoder can learn a curved **Non-linear Manifold**.

```
       PCA (Linear): Projects onto a flat line      Autoencoder (Non-linear): Projects onto a curved manifold
       
              y                                           y
              |   /                                       |   .---.
              |  /   (Data points along line)             |  /     \  (Data points along curve)
       -------+/------- x                          -------+-/-------\- x
              /|                                          |/         \
             / |                                          /           \
```

For example, if data points lie along a curved, spiral shape (like the Swiss Roll dataset), PCA cannot compress it without losing the spiral structure because it is forced to project it onto a flat plane. A non-linear autoencoder can learn to "unroll" the spiral, mapping the coordinates along the curve into a 1D bottleneck coordinate.

### Python Code Implementation
The following code generates a non-linear S-curve dataset in 3D and compares the reconstruction error of standard linear PCA (from Scikit-Learn) with a non-linear Keras Autoencoder. This shows how non-linear connections preserve structure better.

```python
import numpy as np
from sklearn.datasets import make_s_curve
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Generate non-linear 3D S-curve data
X, _ = make_s_curve(n_samples=1000, noise=0.05, random_state=42)
# Standardize inputs
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 2. Linear Dimensionality Reduction: PCA (reduce 3D to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_reconstructed_pca = pca.inverse_transform(X_pca)
pca_mse = np.mean((X - X_reconstructed_pca) ** 2)

# 3. Non-linear Dimensionality Reduction: Autoencoder (reduce 3D to 2D)
inputs = keras.Input(shape=(3,))
x = layers.Dense(12, activation="tanh")(inputs)
latent = layers.Dense(2, activation="tanh")(x) # 2D Bottleneck
y = layers.Dense(12, activation="tanh")(latent)
outputs = layers.Dense(3, activation="linear")(y)

autoencoder = keras.Model(inputs=inputs, outputs=outputs)
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
autoencoder.fit(X, X, epochs=20, batch_size=32, verbose=0)

X_reconstructed_ae = autoencoder.predict(X)
ae_mse = np.mean((X - X_reconstructed_ae) ** 2)

print(f"Linear PCA Reconstruction MSE:          {pca_mse:.6f}")
print(f"Non-linear Autoencoder Reconstruction MSE: {ae_mse:.6f}")
print(f"Reconstruction Error Reduction: {(pca_mse - ae_mse) / pca_mse * 100:.2f}%")
```

### Trade-offs
*   **PCA Advantages:** Computationally fast, deterministic (always yields the exact same solution), has a global analytical solution (via Singular Value Decomposition), and has no hyperparameters to tune.
*   **Autoencoder Advantages:** Can learn complex, non-linear representations, retaining far more information than PCA at very small bottleneck sizes.
*   **Autoencoder Disadvantages:** Highly non-convex loss surface with multiple local minima. Training is computationally expensive, requires GPU resources, and is sensitive to hyperparameter selection.

### Real-World Applications (Rule of 4)
1.  **Example 1: Swiss Roll Compression**
    *   **Input/Scenario:** We compress a 3D dataset shaped like a spiral Swiss Roll into 2D.
    *   **Expected Output:** PCA projects the spiral onto a flat 2D plane, tangling the layers of the spiral together. A non-linear autoencoder learns the curved manifold, unrolling the spiral into a flat 2D sheet, preserving neighbor distances.
2.  **Example 2: Linear Equivalence Test**
    *   **Input/Scenario:** We train a 1-layer autoencoder with linear activations on a dataset.
    *   **Expected Output:** The weights of the bottleneck project the inputs onto the same subspace as the first $d$ principal components of PCA.
3.  **Example 3: High-Dimensional Image Compression**
    *   **Input/Scenario:** We compress $128\times128$ face images to a bottleneck of size 64.
    *   **Expected Output:** PCA reconstructions are highly blurry and ghost-like because facial shapes are non-linear. The non-linear autoencoder produces sharp reconstructions, capturing eyes and noses.
4.  **Example 4: Computation Speed Comparison**
    *   **Input/Scenario:** We compress a dataset with 1 million rows and 100 features.
    *   **Expected Output:** PCA computes the decomposition in seconds on a CPU, whereas training the autoencoder takes several minutes on a GPU.

> **Metacognitive Checkpoint:** Under what mathematical conditions is a single-layer autoencoder equivalent to Principal Component Analysis (PCA)? Explain the role of activation functions in this equivalence.

---

## Topic 3: Anomaly and Fraud Detection using Reconstruction Error

### Rationale and Mechanics
In credit card fraud detection, fraudulent transactions are extremely rare (e.g., $0.1\%$ of all transactions). If we attempt to train a supervised classifier, the model will struggle to learn the patterns of fraud due to the extreme class imbalance.

To solve this, we use an autoencoder for **Unsupervised Anomaly Detection**.

Under the hood:
1.  **Train on Normal Data Only:** We train the autoencoder exclusively on normal, non-fraudulent transactions. The network optimizes its weights to compress and reconstruct normal transaction patterns.
2.  **Compute Reconstruction Error:** During testing, we feed a new transaction $\mathbf{x}$ to the autoencoder and compute the reconstruction error (MSE):
    $$e = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$
3.  **Analyze Error Magnitude:**
    *   If the transaction is normal, it matches the patterns the network saw during training, resulting in a **low reconstruction error**.
    *   If the transaction is fraudulent, its features (e.g., an unusual purchase location combined with a high amount) differ from the normal pattern. Because the autoencoder never learned how to reconstruct this pattern, it fails, resulting in a **high reconstruction error**.
4.  **Apply Threshold ($\tau$):** We classify any sample with error exceeding a threshold $\tau$ as an anomaly:
    $$\text{Anomaly}(\mathbf{x}) = \begin{cases} 1 & \text{if } e > \tau \\ 0 & \text{if } e \le \tau \end{cases}$$

```
       Normal Transaction   ---> [ Autoencoder ] ---> High-Quality Reconstruction ---> Low Error (0.01)
       
       Fraudulent Transaction -> [ Autoencoder ] ---> Poor-Quality Reconstruction ---> High Error (4.50)  <-- Flagged
```

To set the threshold $\tau$, we evaluate the reconstruction errors on a validation set containing only normal transactions and choose a percentile (e.g., the 99th percentile, which flags the 1% most unusual normal transactions as potential anomalies).

### Python Code Implementation
The following code simulates an unsupervised credit card fraud detector using NumPy and Keras. We train the model on normal data only, then test it on both normal and anomalous data, demonstrating how reconstruction error flags outliers.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Simulate Dataset (Normal transactions = cluster around 0.0, Fraud = outliers)
np.random.seed(42)
normal_train = np.random.normal(loc=0.0, scale=0.5, size=(1000, 5))
normal_test = np.random.normal(loc=0.0, scale=0.5, size=(200, 5))
anomalies_test = np.random.normal(loc=4.0, scale=1.0, size=(15, 5)) # Far outliers

# 2. Build and train Autoencoder on normal data only
inputs = keras.Input(shape=(5,))
encoded = layers.Dense(3, activation="relu")(inputs)
decoded = layers.Dense(5, activation="linear")(encoded)

model = keras.Model(inputs=inputs, outputs=decoded)
model.compile(optimizer="adam", loss="mse")
model.fit(normal_train, normal_train, epochs=15, batch_size=32, verbose=0)

# 3. Calculate reconstruction errors (MSE) for train and test sets
train_reconstructed = model.predict(normal_train)
train_errors = np.mean((normal_train - train_reconstructed) ** 2, axis=1)

# Set anomaly threshold at 98th percentile of normal training errors
threshold = np.percentile(train_errors, 98)
print(f"Set anomaly threshold (98th percentile of train error): {threshold:.4f}")

# 4. Evaluate on test data
test_normal_recon = model.predict(normal_test)
test_normal_errors = np.mean((normal_test - test_normal_recon) ** 2, axis=1)

anomalies_recon = model.predict(anomalies_test)
anomalies_errors = np.mean((anomalies_test - anomalies_recon) ** 2, axis=1)

# Count flagged samples
flagged_normal = np.sum(test_normal_errors > threshold)
flagged_anomalies = np.sum(anomalies_errors > threshold)

print(f"\nTest normal transactions: {len(normal_test)}")
print(f" - Flagged as anomaly (False Positives): {flagged_normal} ({flagged_normal/len(normal_test)*100:.2f}%)")

print(f"\nActual Anomalous transactions: {len(anomalies_test)}")
print(f" - Flagged as anomaly (True Positives): {flagged_anomalies} ({flagged_anomalies/len(anomalies_test)*100:.2f}%)")
```

### Trade-offs
*   **Advantages:** Does not require labeled anomaly data. It can detect novel, unseen types of fraud because it only needs to know what "normal" looks like.
*   **Disadvantages:** Setting the threshold $\tau$ involves a trade-off:
    *   Setting $\tau$ too low increases **False Positives** (flagging legitimate users).
    *   Setting $\tau$ too high increases **False Negatives** (missing fraud).
    Additionally, if the training set contains unflagged anomalies, the autoencoder will learn to reconstruct them, reducing its anomaly detection sensitivity.

### Real-World Applications (Rule of 4)
1.  **Example 1: Credit Card Fraud (Imbalanced Data)**
    *   **Input/Scenario:** A bank has 1 million transactions; only 500 are labeled as fraud. We train an autoencoder on the 999,500 normal transactions.
    *   **Expected Output:** When a transaction with $x = [\text{Midnight}, \text{Foreign Country}, \text{10,000 USD}]$ is input, the model outputs a high reconstruction error ($e = 8.5$), flagging it for fraud.
2.  **Example 2: Industrial Machinery Health Monitoring**
    *   **Input/Scenario:** Sensors record turbine vibration. We train an autoencoder on vibration data recorded when the turbine is healthy.
    *   **Expected Output:** When a bearing wears down, it generates a unique vibration. The autoencoder outputs a high reconstruction error, triggering a maintenance alert.
3.  **Example 3: Threshold Selection via Validation Percentiles**
    *   **Input/Scenario:** A validation set of normal transactions yields reconstruction errors. The 99th percentile of these errors is $1.2$.
    *   **Expected Output:** We set the anomaly threshold to $\tau = 1.2$. During testing, any transaction with reconstruction error $e > 1.2$ is flagged.
4.  **Example 4: Network Intrusion Detection**
    *   **Input/Scenario:** A server monitors incoming network packets. We train an autoencoder on normal web traffic.
    *   **Expected Output:** A DDoS attack generates an unusual volume of synchronized packets. The autoencoder outputs a massive reconstruction error, alerting security.

> **Metacognitive Checkpoint:** Why is the reconstruction error of an autoencoder a valid metric for unsupervised anomaly detection? Explain how training only on normal data guarantees high errors for anomalous inputs.

---

## Summary & Next Steps

*   **Autoencoders Compress Data:** Undercomplete autoencoders pass inputs through a low-dimensional bottleneck, forcing the network to learn lossy compression representations.
*   **Manifolds are Non-Linear:** Stacking non-linear layers allows autoencoders to map data onto curved manifolds, retaining more information than linear PCA.
*   **High Reconstruction Errors Signal Anomalies:** Training a model only on normal patterns ensures that anomalous data points yield high reconstruction errors, enabling unsupervised fraud detection.

In the next lesson, we will transition to the course capstone project, starting **Lesson 24: Capstone Project Part 1 (Architecture & Training)** to guide students in selecting, designing, and regularizing a custom deep learning network.
