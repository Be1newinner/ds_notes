# Lesson 12: Data Augmentation & Robustness

## Introduction & The "Why"

In deep learning, the performance of convolutional neural networks is heavily dependent on the size and diversity of the training dataset. If a model is trained on a small dataset, it will overfit, learning to recognize training-specific details (such as a specific background color or angle) rather than the general shape of the object. 

Collecting and labeling thousands of new images is expensive and time-consuming. To solve this data scarcity problem, we use **Data Augmentation**. Data Augmentation is a regularization technique that artificially expands our training dataset by applying random, label-preserving transformations to the images. By rotating, zooming, shifting, and flipping images during training, we force the network to become invariant to these variations.

This lesson explores the mathematics of image transformations, compares online (dynamic) and offline (static) data augmentation, and demonstrates how to integrate Keras preprocessing layers directly into the model's computational graph to build robust classifiers.

---

## Topic 1: The Mathematics of Image Transformations: Rotation, Translation, Zoom, and Flips

### Rationale and Mechanics
In computer vision, image transformations are formulated as coordinate mappings. To transform an image, we map each pixel coordinate $\mathbf{x} = [x, y]^T$ in the original image to a new coordinate $\mathbf{x}' = [x', y']^T$ in the augmented image. These transformations are called **Affine Transformations**.

Under the hood, we represent affine transformations using matrix multiplication in homogeneous coordinates:
$$\begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} = \begin{pmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$
where the $2\times2$ submatrix $\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$ controls scaling, rotation, and shearing, while the vector $\mathbf{t} = [t_x, t_y]^T$ controls translation (shifting).

Let's analyze the mathematical matrices for the most common augmentations:
- **Rotation:** To rotate an image by an angle $\theta$ around the origin, we use:
  $$\mathbf{A}_{\text{rot}} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}, \quad \mathbf{t} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$
- **Scaling (Zoom):** To scale (zoom in/out) the image by factors $s_x$ and $s_y$ along the horizontal and vertical axes:
  $$\mathbf{A}_{\text{scale}} = \begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix}, \quad \mathbf{t} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$
  If $s_x = s_y > 1$, we scale up (zoom in); if $s_x = s_y < 1$, we scale down (zoom out).
- **Horizontal Reflection (Flip):** To reflect the image horizontally across its vertical axis:
  $$\mathbf{A}_{\text{flip}} = \begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}, \quad \mathbf{t} = \begin{pmatrix} W \\ 0 \end{pmatrix}$$
  where $W$ is the image width, translating the coordinates back into the positive quadrant.

Because the calculated coordinates $x'$ and $y'$ are usually real numbers (floats) rather than integers, they do not align perfectly with the target pixel grid. To determine the pixel intensity at the new coordinate, we use **Bilinear Interpolation**. We calculate a weighted average of the intensities of the four nearest integer coordinates surrounding the target point:
$$I(x', y') \approx (1-d_x)(1-d_y)I(x_0, y_0) + d_x(1-d_y)I(x_1, y_0) + (1-d_x)d_yI(x_0, y_1) + d_xd_yI(x_1, y_1)$$
where $d_x$ and $d_y$ are the fractional distances from the top-left integer pixel $(x_0, y_0)$ to the target point $(x', y')$.

### Python Code Implementation
Here is a Python script illustrating how to define and run a 2D coordinate rotation matrix from scratch, transforming coordinates and demonstrating how bilinear interpolation coefficients are derived:

```python
import numpy as np

def rotate_coordinate(coord, angle_degrees):
    # Convert degrees to radians
    theta = np.radians(angle_degrees)
    
    # Define rotation matrix R
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Matrix multiplication to find transformed coordinates
    return np.dot(R, coord)

# Initial coordinate in our image space
coord = np.array([2.0, 5.0])
rotated = rotate_coordinate(coord, angle_degrees=30.0)

print(f"Original Coordinate: {coord}")
print(f"Rotated (30 deg):     {np.round(rotated, 4)}")

# Calculate bilinear interpolation weights for the new coordinate
x_float, y_float = rotated[0], rotated[1]
x0, y0 = int(np.floor(x_float)), int(np.floor(y_float))
x1, y1 = x0 + 1, y0 + 1

dx = x_float - x0
dy = y_float - y0

print(f"\nNearest grid coordinates: ({x0}, {y0}) to ({x1}, {y1})")
print(f"Fractional distances (dx, dy): {dx:.4f}, {dy:.4f}")
print(f"Interpolation weights:")
print(f"  Top-Left weight:  {(1-dx)*(1-dy):.4f}")
print(f"  Top-Right weight: {dx*(1-dy):.4f}")
print(f"  Bottom-Left weight: {(1-dx)*dy:.4f}")
print(f"  Bottom-Right weight: {dx*dy:.4f}")
```

### Trade-offs
Affine transformations are highly effective at introducing variation.

The trade-off is computational cost. Applying matrix transformations and performing bilinear interpolation for every pixel in a batch of high-resolution images is computationally expensive. If performed inefficiently, data augmentation can become a bottleneck, leaving the GPU idle while waiting for the CPU to process the next batch of images. Additionally, transformations must be **label-preserving**. For example, flipping an image of a cat is label-preserving (it is still a cat). However, flipping the digit "6" horizontally can turn it into the digit "2" or a meaningless shape, degrading classification performance.

### Real-World Applications (Rule of 4)

1. **Example 1: Horizontal Flip Matrix Calculation**
   - **Input/Scenario:** We reflect a pixel at coordinate $[10, 20]^T$ in a $100\times100$ image horizontally.
   - **Expected Output:** The transformation is:
     $$\begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} = \begin{pmatrix} -1 & 0 & 100 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} 10 \\ 20 \\ 1 \end{pmatrix} = \begin{pmatrix} -10 + 100 \\ 20 \\ 1 \end{pmatrix} = \begin{pmatrix} 90 \\ 20 \\ 1 \end{pmatrix}$$
     The pixel has been successfully mapped to coordinate $[90, 20]^T$.
2. **Example 2: Bilinear Interpolation Calculation**
   - **Input/Scenario:** A coordinate maps to fractional location $x' = 2.4, y' = 5.2$. The neighboring pixel values are $I(2, 5) = 100, I(3, 5) = 120, I(2, 6) = 150, I(3, 6) = 180$.
   - **Expected Output:**
     - Fractional parts: $d_x = 0.4, d_y = 0.2$.
     - Interpolated intensity:
       $$I(2.4, 5.2) = 0.6(0.8)(100) + 0.4(0.8)(120) + 0.6(0.2)(150) + 0.4(0.2)(180)$$
       $$= 38.4 + 38.4 + 18.0 + 14.4 = 109.2$$
       The output intensity at the coordinate is $109.2$.
3. **Example 3: Medical Image Segmentation (Rotation Invariance)**
   - **Input/Scenario:** An MRI scanner captures tumors at various rotations. We apply random rotation augmentations ($0$ to $360^\circ$) to the training dataset.
   - **Expected Output:** The network learns to segment tumors based on shape and contrast, regardless of scanner alignment.
4. **Example 4: Street Sign Recognition (Zoom/Scale Invariance)**
   - **Input/Scenario:** A self-driving car model classifies stop signs. Signs appear at various sizes depending on the car's distance. We apply random zoom augmentations ($0.8$ to $1.2$).
   - **Expected Output:** The network learns features that generalize across scales, ensuring stop signs are recognized from a distance.

> **Metacognitive Checkpoint:** Why is horizontal flipping a valid data augmentation technique for classifying images of cars, but invalid for classifying handwritten digits (like MNIST)? Explain this in terms of label preservation.

---

## Topic 2: Online vs Offline Augmentation

### Rationale and Mechanics
There are two primary ways to implement data augmentation: **Offline Augmentation** and **Online Augmentation**.

Under the hood:
- **Offline Augmentation:** We apply transformations to the training images beforehand and save the augmented images to disk. For a dataset of size $N$, if we generate 5 augmentations per image, the dataset size increases to $5N$. During training, the optimizer reads these static files from disk.
- **Online Augmentation:** We do not save augmented images to disk. Instead, when loading a batch of size $B$ during training, we apply random transformations to the images on the fly in system memory. In each epoch, the model receives a slightly different set of random transformations, meaning it never sees the exact same image twice.

### Python Code Implementation
Here is a Python simulation comparing the unique training samples generated using static offline augmentation (limited set of pre-rendered options) versus online augmentation (infinite variations on the fly):

```python
import numpy as np

# Simulate a training dataset of 3 images
raw_images = [np.array([1.0]), np.array([2.0]), np.array([3.0])]

# 1. Offline Simulation: Generate 3 static variations per image beforehand
offline_database = []
for img in raw_images:
    offline_database.append(img)  # original
    offline_database.append(img + 0.1)  # variation 1
    offline_database.append(img + 0.2)  # variation 2

print(f"Offline Database Size: {len(offline_database)} samples")
print("Offline Samples:       ", [x[0] for x in offline_database])

# 2. Online Simulation: Apply random continuous variations on the fly during training
def online_augment(img):
    # Apply a continuous random shift
    noise = np.random.uniform(0.0, 0.5)
    return img + noise

print("\n--- Training Simulation (3 Epochs) ---")
for epoch in range(1, 4):
    print(f"Epoch {epoch}:")
    # Generate batch values dynamically
    epoch_batch = [round(online_augment(img)[0], 4) for img in raw_images]
    print(f"  Online Batch Values: {epoch_batch}")
```

### Trade-offs
- **Offline Augmentation Trade-off:** Extremely simple to implement and requires no CPU overhead during training. However, it requires significant disk space. For large datasets (e.g., ImageNet), saving multiple augmentations to disk is impractical. Additionally, the model is still exposed to a static set of images, limiting the regularization effect.
- **Online Augmentation Trade-off:** Requires zero additional disk space and provides infinite variety: the model is exposed to unique variations in every epoch. The trade-off is computational overhead. If the transformations are calculated on the CPU, they can bottleneck training. To prevent this, modern pipelines perform online augmentation using the GPU, or use asynchronous multi-threaded prefetching (`tf.data` API with `AUTOTUNE`) to prepare the next batch of augmented images while the GPU is processing the current batch.

### Real-World Applications (Rule of 4)

1. **Example 1: Disk Space Constraints (Offline)**
   - **Input/Scenario:** A dataset contains $100,000$ high-resolution images, requiring 50 GB of disk space. We want to apply 10 augmentations per image.
   - **Expected Output:** Offline augmentation would require 500 GB of disk space. Online augmentation uses 0 GB of additional space, keeping storage requirements at 50 GB.
2. **Example 2: Epoch Variety (Online)**
   - **Input/Scenario:** We train a model for 100 epochs on a dataset of 1,000 images using online augmentation.
   - **Expected Output:** In each epoch, the model receives a unique set of random transformations. Over 100 epochs, the model is exposed to 100,000 unique variations, reducing overfitting compared to static offline datasets.
3. **Example 3: CPU Bottleneck (Online)**
   - **Input/Scenario:** A developer implements online data augmentation using a slow CPU-bound library.
   - **Expected Output:** The GPU utilization drops from 95% to 15%, slowing down training. The developer must transition to GPU-accelerated pre-processing layers to restore performance.
4. **Example 4: Asynchronous Prefetching (tf.data)**
   - **Input/Scenario:** A developer configures a training pipeline using `tf.data` with prefetching.
   - **Expected Output:**
     ```python
     dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
     dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
     ```
     The CPU processes and augments Batch $T+1$ in parallel while the GPU is executing the forward and backward passes for Batch $T$, maintaining 100% GPU utilization.

> **Metacognitive Checkpoint:** Why does online data augmentation act as a stronger regularizer than offline data augmentation? Explain this in terms of the number of unique training samples the model sees over 100 epochs.

---

## Topic 3: Keras Preprocessing Layers: In-Graph Augmentation

### Rationale and Mechanics
In early versions of Keras and TensorFlow, data augmentation was performed outside the model's computational graph (e.g., using Python generators). This separated the augmentation logic from the model files, making deployment complex.

Modern Keras integrates data augmentation directly into the model's computational graph using **Keras Preprocessing Layers**.

Under the hood, we can define a Sequential model of augmentation layers:
```python
import tensorflow as tf
from tensorflow import keras

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2)
])
```

We can insert this stack at the beginning of our classification model:
```python
model = keras.Sequential([
    keras.layers.Input(shape=(256, 256, 3)),
    data_augmentation,
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # Downstream layers...
])
```

Because these layers are built into the Keras model graph:
- **During Training (`training=True`):** The layers generate random parameters (e.g., drawing a random angle between $-0.2\times2\pi$ and $+0.2\times2\pi$ rad) and apply the transformation to the batch.
- **During Testing/Inference (`training=False`):** The layers are automatically bypassed. They act as identity mappings, passing the clean input images directly to the convolutional layers. This ensures predictions are made on unaltered data.

### Python Code Implementation
Here is a Python script showing how to build a preprocessing and augmentation pipeline using Keras layers, and confirming how they dynamically alter data during training but remain inactive during inference:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define in-graph augmentation layers
augment_pipeline = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1)
])

# Create a dummy image of shape (1, 4, 4, 1) - single channel 4x4
dummy_image = np.array([[[[1], [2], [3], [4]],
                         [[5], [6], [7], [8]],
                         [[9], [10], [11], [12]],
                         [[13], [14], [15], [16]]]]).astype(float)

print("Original Image Matrix (spatial view):\n", dummy_image[0, :, :, 0])

# 1. Simulate training forward pass (applying transformations)
# We set training=True explicitly
training_outputs = [augment_pipeline(dummy_image, training=True)[0, :, :, 0].numpy() for _ in range(3)]

print("\n--- Training Passes (Dynamic Outputs) ---")
for i, out in enumerate(training_outputs):
    print(f"Pass {i+1} Output:\n", out)

# 2. Simulate evaluation/testing pass (bypass transformations)
# We set training=False explicitly
testing_output = augment_pipeline(dummy_image, training=False)[0, :, :, 0].numpy()

print("\n--- Testing Pass (Identity Output) ---")
print("Test Output:\n", testing_output)

# Check if identical to input
assert np.all(testing_output == dummy_image[0, :, :, 0]), "Transformation applied in testing!"
print("\nValidation Succeeded: Augmentations were successfully bypassed in testing.")
```

### Trade-offs
- **Advantages:** The entire preprocessing and augmentation logic is compiled into the model. When we export the model (`model.save('my_model')`), the augmentation layers are saved inside the file. When we load the model in a production environment (e.g., using TensorFlow Serving), it automatically handles input preprocessing, eliminating the need to write matching preprocessing code in the production client.
- **Disadvantages:** Including augmentation inside the graph can increase model loading times and GPU memory footprint. Additionally, debugging custom transformations inside a compiled TensorFlow graph is difficult compared to standard Python functions.

### Real-World Applications (Rule of 4)

1. **Example 1: Random Rotation Graph Execution**
   - **Input/Scenario:** We train a model with a `RandomRotation(0.1)` layer. A batch of 32 images is passed to the input.
   - **Expected Output:** Keras draws 32 random angles in the range $[-0.1 \times 2\pi, +0.1 \times 2\pi]$ rad, applies the rotations to the batch, and passes the augmented images to the Conv2D layer.
2. **Example 2: Inference Bypassing**
   - **Input/Scenario:** The model is compiled and we call `model.evaluate(X_val, y_val)` or `model.predict(new_image)`.
   - **Expected Output:** Keras sets `training=False`. The rotation, flip, and zoom layers are bypassed, passing the validation and test images to the classifier without alteration.
3. **Example 3: Exporting Self-Contained Models**
   - **Input/Scenario:** A developer trains a model with in-graph augmentation and exports it using `model.save()`.
   - **Expected Output:** The saved model directory contains the graph definition for the augmentation layers. When loaded in a mobile app using TensorFlow Lite, the input image is processed without needing external image manipulation libraries.
4. **Example 4: Feature Map Dimension Retention**
   - **Input/Scenario:** A model uses `RandomZoom(0.2)` on an input image of shape $(256, 256, 3)$.
   - **Expected Output:** Even though the zoom operation scales the image coordinates, Keras automatically crops or pads the output to maintain the target shape $(256, 256, 3)$, preventing dimension mismatch errors in downstream layers.

> **Metacognitive Checkpoint:** How does Keras ensure that data augmentation layers do not modify the validation or test images during evaluation? Explain the role of the `training` argument in the Keras layer execution logic.

---

## Summary & Next Steps

- **Transformations Use Affine Matrices:** Image rotations, scaling, and flips are calculated using affine matrices in homogeneous coordinates, using bilinear interpolation to map pixels back to integer grids.
- **Online Augmentation Saves Disk Space:** Online augmentation applies transformations dynamically in memory, providing infinite variety over training epochs without requiring disk storage.
- **Keras Layers Integrate Preprocessing:** Preprocessing layers compile augmentation directly into the model graph, automatically deactivating during inference to simplify deployment.

In the next lesson, we will explore **Transfer Learning & Pre-trained Backbones**, learning how to import massive architectures like ResNet50 and freeze convolutional weights to solve specific classification tasks.
