# Tutorial 6.1: Core Keras Layers

> Mastering Architecture

## Executive Summary

In Keras, a model is essentially a graph of layers. Understanding which layer to use, how to configure it, and when to apply it is the difference between a model that learns and one that fails. This guide covers the essential layers required for 90% of beginner to intermediate deep learning tasks.

## 1. The Fundamentals
keras.layers.Dense

### Fully Connected Layer

The most common and basic layer. Every neuron in a Dense layer is connected to every neuron in the preceding layer. It learns global patterns across all its inputs.

```python
# Import the Dense layer class from Keras layers module to create fully connected layers
from keras.layers import Dense

# Instantiate a Dense layer with 64 units/neurons using the Rectified Linear Unit (ReLU) activation function for non-linearity
layer = Dense(units=64, activation='relu')
```

####  🌍 Real-Life Use Case

**Tabular Data & Final Classification:** Predicting house prices based on features (bedrooms, area, age) or outputting the final probabilities (using Softmax) in an image classification model.
keras.layers.Flatten

### Flattening Tensor

Takes a multi-dimensional tensor (like a 2D image) and unrolls it into a single, long 1D array. It does not have any parameters to learn; it just reshapes the data.

```python
# Import the Flatten layer class from Keras layers to reshape multi-dimensional tensors
from keras.layers import Flatten

# Instantiate a Flatten layer to unroll multi-dimensional arrays (e.g. (28, 28) images) into a single 1D vector of shape (784,)
layer = Flatten()
```

####  🌍 Real-Life Use Case

**Connecting CNNs to Dense Layers:** After a Convolutional Neural Network processes an image in 2D, the data must be Flattened before it can be fed into a standard Dense layer for final classification.

## 2. Computer Vision Layers
keras.layers.Conv2D

### 2D Convolution

Creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. It learns spatial hierarchies and localized patterns (like edges, textures).

```python
# Import the Conv2D convolutional layer class to extract spatial features from 2D arrays
from keras.layers import Conv2D

# Instantiate a Conv2D layer with 32 filters, a 3x3 kernel sliding window, and ReLU activation to detect features like edges
layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
```

####  🌍 Real-Life Use Case

**Image Recognition:** Analyzing X-ray scans for tumors, self-driving cars detecting pedestrians, or facial recognition systems on smartphones.
keras.layers.MaxPooling2D

### Downsampling

Reduces the spatial dimensions (height, width) of the input volume by taking the maximum value over an input window. This reduces computational cost and provides translational invariance.

```python
# Import the MaxPooling2D layer class to downsample 2D spatial feature maps
from keras.layers import MaxPooling2D

# Instantiate a MaxPooling2D layer with a 2x2 pooling window to take maximum values and reduce spatial dimensions by half
layer = MaxPooling2D(pool_size=(2, 2))
```

## 3. Sequence & Regularization
keras.layers.LSTM

### Long Short-Term Memory

A type of Recurrent Neural Network (RNN) layer capable of learning long-term dependencies in sequence data, solving the vanishing gradient problem of standard RNNs.

```python
# Import the LSTM layer class to process sequential data and learn temporal dependencies
from keras.layers import LSTM

# Instantiate an LSTM layer with 64 units, returning only the final step's output to feed succeeding layers
layer = LSTM(units=64, return_sequences=False)
```

####  🌍 Real-Life Use Case

**Time Series & NLP:** Stock market price forecasting, language translation (Google Translate), and speech recognition.
keras.layers.Dropout

### Regularization

Randomly sets a fraction of input units to 0 at each step during training time, which helps prevent overfitting by forcing the network to learn redundant representations.

```python
# Import the Dropout regularization layer class to prevent model overfitting
from keras.layers import Dropout

# Instantiate a Dropout layer to randomly zero out 30% of inputs at each step during training
layer = Dropout(rate=0.3)
```

####  🌍 Real-Life Use Case

**Overfitting Prevention:** Used almost universally in large models where the model starts memorizing training data rather than generalizing to unseen data.

### 💡 Beginner's Perspective: Shape Flow Visualization

When building neural networks, the hardest part for a beginner is tracking how the dimensions (shapes) of your data change as they pass from layer to layer. Here is a visual pipeline of a typical Image Classification network (e.g., MNIST):

```text
[Input Image]                 Shape: (Batch, 28, 28, 1)  <-- Height=28, Width=28, Channels=1 (Grayscale)
      │
      ▼
[Conv2D (32 filters, 3x3)]    Shape: (Batch, 26, 26, 32) <-- Height/Width shrink slightly, channel dimension becomes 32 features
      │
      ▼
[MaxPooling2D (2x2)]          Shape: (Batch, 13, 13, 32) <-- Height/Width divided by 2 (downsampled)
      │
      ▼
[Flatten]                     Shape: (Batch, 5408)       <-- 13 * 13 * 32 = 5408 unrolled features
      │
      ▼
[Dense (128 units)]           Shape: (Batch, 128)        <-- Fully-connected layer learning combinations
      │
      ▼
[Dropout (0.3)]               Shape: (Batch, 128)        <-- Randomly kills 30% of signals during training
      │
      ▼
[Dense (10 units, Softmax)]   Shape: (Batch, 10)         <-- Final output: 10 probabilities summing to 1.0
```

---

### 💡 Supplementary Notes

* **Receptive Field Expansion**: In Convolutional layers, the receptive field is the local area of input pixels that a neuron 'sees'. Downsampling (MaxPooling) and strided convolutions expand the receptive field of subsequent layers, allowing the network to capture global structures with fewer parameters.

## Active Recall Checkpoint
1

Data Transformation

If you have a Conv2D layer outputting a shape of (5, 5, 32), what is the shape of the data after passing it through a Flatten layer?
2

Combating Overfitting

Why is Dropout only active during the training phase, and automatically disabled during the evaluation/testing phase?