# Tutorial 17: GPU Performance & Benchmarking

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=YmDaqXMIoeY)

## 01 Executive Summary

This analysis benchmarks the performance difference between a **CPU** and a **GPU** when training a deep learning model for image classification. Using an **Artificial Neural Network (ANN)** and the **CIFAR-10 dataset**, the demonstration highlights how specialized hardware like NVIDIA GPUs can accelerate training speeds by approximately **15 times** compared to standard processors.

## 02 The "Why" (First Principles)

Deep learning involves massive **matrix multiplications** that are computationally expensive for a standard CPU, which is designed for general-purpose sequential tasks. GPUs are built for **massive parallelism**, allowing thousands of small operations to occur simultaneously. This tutorial addresses the common bottleneck of long training durations, showing that leveraging **CUDA-enabled hardware** is essential for scaling deep learning experiments.

## 03 Detailed Step-by-Step Notes

### 1. Environment Configuration and Verification

* Import core libraries including **TensorFlow**, **Keras**, **Matplotlib**, and **NumPy**.

* Use TensorFlow configuration commands to list physical devices and verify if a **GPU is detected**.

* Ensure **CUDA** and **cuDNN** libraries are correctly installed and version-compatible with the TensorFlow build.

### 2. Data Loading and Exploration

* Load the **CIFAR-10 dataset** directly through the Keras API.

* The dataset consists of **60,000 images** (50,000 training, 10,000 test) across **10 classes** (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

* Each image is a small **32x32 pixel array** with 3 color channels (RGB).

### 3. Data Preprocessing

* **Scaling**: Normalize pixel values by dividing the entire dataset by **255**. This converts the range from 0-255 to 0-1, which helps the neural network learn more effectively.

* **Categorical Encoding**: Apply **One-Hot Encoding** to the target labels using the Keras `to_categorical` utility. This transforms discrete integers (e.g., 6) into binary vectors (e.g., [0,0,0,0,0,0,1,0,0,0]).

### 4. Designing the Neural Network

* Instantiate a **Keras Sequential model**.

* Add a **Flatten layer** to convert the 32x32x3 input into a single vector of **3,072 elements**.

* Create two large hidden **Dense layers** (3,000 and 1,000 neurons) using the **ReLU activation function** to capture complex patterns.

* Define an output **Dense layer** with 10 neurons using the **Sigmoid activation function** for class probability prediction.

### 5. Model Compilation and Benchmarking

* Compile the model using the **Stochastic Gradient Descent (SGD)** optimizer and **categorical cross-entropy** loss.

* Utilize the Jupyter magic command `%%timeit` to measure the execution time of training one epoch.

* Compare the **wall-clock time** between training on the CPU (**/CPU:0**) and the GPU (**/GPU:0**).

## 04 Highlighted Examples

Visual Confirmation

**Dataset Visualization**: Plotting the first few images of the training set confirms that **label 6** corresponds to a frog and **label 9** corresponds to a truck.

Inference Logic

**Prediction Logic**: The `model.predict` function returns a list of probabilities; using NumPy's `argmax` function identifies the index with the highest probability, representing the final classification.

Real-world Metric

**Speed Comparison**: In a test of 10 epochs, a **CPU might take over 7 minutes** to complete the task, whereas a high-end GPU like the **NVIDIA Titan RTX finishes in just 30 seconds**.

## 05 Technical Execution
Checking for GPU

```python
# Check and list all physical graphics processing units (GPUs) available to TensorFlow
tf.config.experimental.list_physical_devices('GPU')
```
Normalization

```python
# Scale the training images by dividing pixel values by 255 to normalize the data to a [0, 1] range
X_train_scaled = X_train / 255
```
One-Hot Encoding

```python
# Convert integer training labels into binary category vectors (one-hot encoding)
y_train_categorical = tf.keras.utils.to_categorical(
   # Pass in the array of integer labels to be converted
   y_train,
   # Set total number of target classes to 10 for CIFAR-10 classification
   num_classes=10,
   # Define the data type of the output categorical arrays as float32
   dtype='float32'
)
```
Model Architecture

```python
# Define a Sequential deep neural network model for the CIFAR-10 dataset
model = keras.Sequential([
   # Flatten the 32x32x3 input images into a 1D vector of shape (3072,)
   keras.layers.Flatten(input_shape=(32,32,3)),
   # Add a dense hidden layer with 3000 neurons using ReLU activation to identify local structures
   keras.layers.Dense(3000, activation='relu'),
   # Add a dense hidden layer with 1000 neurons using ReLU activation to extract higher level features
   keras.layers.Dense(1000, activation='relu'),
   # Add a dense output layer with 10 classes using Sigmoid activation to estimate categorical outputs
   keras.layers.Dense(10, activation='sigmoid')
])
```
Benchmarking Block

```python
# Force execution of the enclosed model operations on the first detected GPU device
with tf.device('/GPU:0'):
   # Initialize and compile the model architecture by calling get_model()
   gpu_model = get_model()
   # Train the compiled model on scaled images and categorical labels for exactly 1 epoch
   gpu_model.fit(X_train_scaled, y_train_categorical, epochs=1)
```

### 💡 Beginner's Analogy: The Genius vs. The Schoolchildren

Why are GPUs so much faster at training deep learning models?

* **The CPU (Central Processing Unit)**:
  Imagine a **single genius mathematician** who can solve incredibly complex algebraic formulas. However, they can only write down and solve **one equation at a time** (sequential execution).
  If you give them 1,000,000 simple addition problems, they will do them very fast, but they still have to do them one-by-one.

* **The GPU (Graphics Processing Unit)**:
  Imagine a massive stadium filled with **10,000 elementary school students**. None of them can solve complex calculus, but every single one of them knows how to do simple addition and multiplication.
  If you give this stadium the 1,000,000 simple addition problems, they can divide the workload and solve thousands of them **at the exact same time** (parallel execution).

Since training neural networks is mostly billions of simple matrix multiplications (additions and multiplications), the GPU (the stadium of kids) finishes the training exponentially faster than the CPU (the single genius).

---

### 💡 Supplementary Notes

* **Mixed Precision Training**: Utilizing 16-bit floats (`float16` or `bfloat16`) instead of 32-bit floats (`float32`) during training doubles memory bandwidth, reduces storage, and leverages specialized hardware acceleration (like Tensor Cores) without loss in final accuracy.

## 06 Active Recall Checkpoint

1. 1

What is the specific **mathematical benefit** of scaling pixel values to a range of 0 to 1 before training?

1. 2

When would you choose **sparse_categorical_crossentropy** over **categorical_crossentropy** as your loss function?

1. 3

How does the **Flatten layer** transform a 32x32x3 image array into a format usable by a standard Dense layer?

Try to answer these without looking back at the notes to reinforce your learning!