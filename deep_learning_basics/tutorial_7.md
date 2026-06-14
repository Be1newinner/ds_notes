# Tutorial 7: Handwritten Digits Classification

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=iqQgED9vV7k)

# Computer Vision Foundations: Watch Video Tutorial

> Deep Dive into

[Watch Video Tutorial](https://www.youtube.com/watch?v=iqQgED9vV7k)

## Executive Summary

MNIST is the "Drosophila" of machine learning. We don't just "train a model"—we perform **Matrix Flattening**, **Min-Max Scaling**, and **Categorical Probability Mapping**. This guide covers why these steps are non-negotiable for convergence.

## The Mathematical Prerequisites

#### 1. Scaling & Numerical Stability

Input pixels range from 0 (Black) to 255 (White). If we feed 255 into a network, the gradients during backpropagation will "explode." Scaling to **[0, 1]** ensures the weights stay in a manageable range.
X_scaled = (X - X_min) / (X_max - X_min)

#### 2. Softmax: The Competition Layer

The final layer uses Softmax to turn raw "logits" into probabilities. It forces the output to sum to exactly 1.0, creating a "winner-takes-all" dynamic among the 10 possible digits.
Softmax(z_i) = e^(z_i) / Σ e^(z_j)

## Implementation Architecture

### 1. Basic Feedforward Model (Baseline)

```python
# Create a Sequential model to build a network step-by-step, where layers are stacked in linear order
model = keras.Sequential([
   # Flatten the 2D input of shape (28, 28) into a 1D vector of shape (784,) so dense layers can process it
   keras.layers.Flatten(input_shape=(28, 28)),
   # Add a dense hidden layer with 128 neurons using ReLU activation to introduce non-linearity and learn complex patterns
   keras.layers.Dense(128, activation='relu'),
   # Add a dense output layer with 10 neurons (one per class digit 0-9) using Softmax activation to convert outputs into probabilities
   keras.layers.Dense(10, activation='softmax')
])

# Compile model using Adam optimizer for adaptive learning rates and sparse categorical crossentropy loss since labels are integers
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

### 2. Advanced CNN Pipeline (Production Grade)

```python
# Import numpy for array manipulation (such as adding dimensions)
import numpy as np
# Import keras to access deep learning modules, datasets, and utilities
import keras
# Import layers module to construct neural network layer classes
from keras import layers
# Import image module to load/preprocess image inputs if needed
from tensorflow.keras.preprocessing import image

# Define the number of target classes (10 categories: digits 0 through 9)
num_classes = 10
# Define the input shape: 28x28 pixels with 1 color channel (grayscale)
input_shape = (28, 28, 1)

# 1) Load dataset
# Load the pre-split MNIST training and testing arrays from keras datasets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2) Preprocess
# Cast training images to float32 and divide by 255.0 to scale pixel values to [0, 1] range for gradient stability
x_train = x_train.astype("float32") / 255.0
# Cast testing images to float32 and divide by 255.0 to scale pixel values to [0, 1] range for consistency
x_test = x_test.astype("float32") / 255.0

# Add a channel dimension at the end to transform shape from (samples, 28, 28) to (samples, 28, 28, 1) as required by Conv2D layers
x_train = np.expand_dims(x_train, -1)
# Apply the same channel expansion to test data to match input requirements
x_test = np.expand_dims(x_test, -1)

# One-hot encode integer labels (e.g. 5 -> [0,0,0,0,0,1,0,0,0,0]) to match the categorical crossentropy loss targets
y_train = keras.utils.to_categorical(y_train, num_classes)
# One-hot encode test labels using the same function for evaluation compatibility
y_test = keras.utils.to_categorical(y_test, num_classes)

# 3) Build CNN model
# Create a Sequential network structure where layers execute consecutively
model = keras.Sequential([
   # Explicitly declare the model's input tensor shape to initialize weights properly
   keras.Input(shape=input_shape),
   # Add convolutional layer with 32 filters of size 3x3 to extract low-level patterns like edges using ReLU activation
   layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
   # Add max pooling layer of size 2x2 to downsample feature maps and achieve translation invariance
   layers.MaxPooling2D(pool_size=(2, 2)),
   # Add convolutional layer with 64 filters of size 3x3 to capture more abstract, higher-level features
   layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
   # Add max pooling layer of size 2x2 to further reduce dimensions and parameters
   layers.MaxPooling2D(pool_size=(2, 2)),
   # Flatten the 2D feature maps into a 1D vector before feeding into dense classifier layers
   layers.Flatten(),
   # Apply 50% dropout during training to randomly deactivate neurons, preventing overfitting and encouraging generalization
   layers.Dropout(0.5),
   # Add a dense output layer with 10 units (num_classes) using Softmax to produce probability distributions over the 10 classes
   layers.Dense(num_classes, activation="softmax"),
])

# 4) Compile
# Configure the model's learning configuration by specifying the optimizer, loss, and evaluation metrics
model.compile(
   # Use the Adam optimizer, which automatically adjusts learning rates for faster convergence
   optimizer="adam",
   # Use categorical crossentropy loss since we one-hot encoded the training labels
   loss="categorical_crossentropy",
   # Track accuracy throughout training and validation to monitor classification performance
   metrics=["accuracy"]
)

# 5) Train
# Train the compiled model on the preprocessed training dataset
model.fit(
   # Provide the training image tensor
   x_train,
   # Provide the corresponding one-hot encoded targets
   y_train,
   # Process 128 images at a time before updating internal model weights
   batch_size=128,
   # Train the model for 10 full passes (epochs) over the dataset
   epochs=10,
   # Split 10% of training data to check validation loss and accuracy after each epoch
   validation_split=0.1
)

# 6) Evaluate
# Evaluate the final trained model on the unseen test set to measure generalization performance
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
# Print the final calculated test accuracy to the console
print("Test accuracy:", test_acc)

# 7) Save model
# Save the model's architecture, weights, and compilation details to a single Keras v3 file format for deployment
model.save("mnist_digit_model.keras")
```

### 💡 Beginner's Blueprint: Demystifying MNIST Preprocessing

When working with digits or images like MNIST, a few helper steps always trip up beginners:

1. **What is MNIST?**
   It is a collection of 70,000 grayscale images of handwritten digits (0-9). Each image is a $28 \times 28$ grid of pixels. In code, it starts as a 3D NumPy array of shape `(60000, 28, 28)`.

2. **Why scale / divide by 255.0?**
   Grayscale pixel values range from $0$ (pure black) to $255$ (pure white). Feeding large values like $255$ directly into neural networks triggers massive changes in weights, leading to **exploding gradients** (gradients getting too large) and causing the model's training to crash or learn extremely slowly. Scaling by dividing by $255.0$ squashes the values into a friendly $[0, 1]$ range.

3. **`categorical_crossentropy` vs. `sparse_categorical_crossentropy`**:
   Both compute the classification loss, but they expect different formats for target labels ($y$):
   - **`sparse_categorical_crossentropy`**: Expects $y$ to be raw integer labels (e.g., `y_train = [5, 0, 4]`).
   - **`categorical_crossentropy`**: Expects $y$ to be one-hot encoded binary vectors (e.g., `y_train = [[0,0,0,0,0,1,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0]]`).

   If you don't want to run `to_categorical()` to change your labels into vectors, simply use `sparse_categorical_crossentropy`.

### 1. Why use the `Dense` layer twice? (Hidden vs. Output)

The word "Dense" doesn't mean "hidden" or "output." **Dense just describes the _wiring_**. It means that every neuron in this layer is physically connected to every single piece of data coming from the layer before it.

We use `Dense` layers multiple times because they are doing different jobs based on where they are placed in the assembly line:

- **The Hidden Layer's Job:** This is an intermediate step. It sits between the input and the final decision. Its only job is to look at the raw pixels and extract useful patterns, like loops, lines, and curves. We call it "hidden" simply because it is not the input layer, and it is not the final output layer.
- **The Output Layer's Job:** This is the end of the line. It uses the exact same `Dense` wiring, but its job is strictly to make the final prediction. Because we are trying to predict digits from 0 to 9, this layer **must** have exactly 10 neurons.

### 2. The 128 Neurons: Can I use any number? Does more = better?

**Yes, you can use absolutely any number of neurons in a hidden layer!** You could use 10, 56, 300, or 10,000.

- **Why 128?** In computer science, we often pick powers of 2 (32, 64, 128, 256, 512) simply because computers and their memory systems (like GPUs) process data in these chunks much faster. It is a convention for efficiency, not a strict mathematical rule.
- **Does more neurons mean better output?** **No!** This is one of the most common misconceptions in data science.
  - If you have **too few** neurons (e.g., 2 neurons), the network doesn't have enough "brainpower" to understand the complexity of the handwriting. It will fail.
  - If you have **too many** neurons (e.g., 10,000 neurons), the network becomes _too_ smart for its own good. Instead of learning the general patterns of a "3", it simply memorizes the exact pixels of the training images. When you show it a new, slightly different "3" it has never seen before, it gets confused and fails. This is a very common trap called **Overfitting**.

You want the "Goldilocks" number of neurons—just enough to learn the patterns, but few enough that it is forced to generalize rather than memorize. For the MNIST dataset, 128 happens to be a very solid sweet spot!

### 3. When to use a hidden layer, and how many can I create?

**When to use them:**
You should use at least one hidden layer for almost any modern machine learning problem. If you _don't_ have a hidden layer (meaning you connect your input pixels directly to your 10 output neurons), your network can only draw straight mathematical lines to separate the numbers. Hidden layers (combined with the ReLU activation) are what allow the network to learn complex, non-linear shapes.

**How many can you create?**
Technically, you can stack as many hidden layers as you want!

If you add 5, 10, or 50 hidden layers, you are officially doing **Deep Learning** (the "deep" just refers to having many hidden layers).

However, adding more layers introduces the same trade-offs as adding more neurons:

1. **Overfitting:** Too many layers will just memorize the data.
2. **Diminishing Returns:** For a simple dataset like MNIST, 1 or 2 hidden layers will get you to 98%+ accuracy. Adding 10 more layers might only increase accuracy by 0.1% but will take 10 times longer to train.
3. **The Rule of Thumb:** Always start simple. Build a model with 1 hidden layer. If it doesn't perform well, try adding more neurons, or add a second hidden layer. Data science is highly experimental!

---

## How to Decide How Many Neurons to Add in a Hidden/Dense Layer?

The hard truth is: **There is no single mathematical formula that tells you the perfect number of neurons.** Unlike the input layer (which _must_ be 784 for MNIST) and the output layer (which _must_ be 10 for MNIST), the hidden layers are completely up to you. Choosing the number of neurons is an empirical process, which is a fancy way of saying **"trial and error guided by best practices."**

Here are the standard rules of thumb data scientists use to decide where to start:

### 1. The "Somewhere in Between" Rule

A very common starting point is to choose a number of hidden neurons that falls somewhere between the size of your input layer and the size of your output layer.

- Your input is **784**.
- Your output is **10**.
- Therefore, any number between 10 and 784 is a reasonable guess. This is why 128, 64, or 256 are very common choices for this specific dataset.

### 2. Stick to Powers of 2

As mentioned previously, keep your numbers to powers of 2 for computational efficiency. When you are guessing how many neurons to use, limit your choices to a "menu" like: **32, 64, 128, 256, or 512**.

### 3. The "Funnel" Architecture

If you decide to use _more than one_ hidden layer, the standard practice is to create a funnel (or pyramid) shape. You start with a larger number of neurons to capture broad patterns, and shrink the number as you get closer to the final decision.

> **Example of a Funnel:**
> Input (784) → Hidden 1 (**256**) → Hidden 2 (**128**) → Hidden 3 (**64**) → Output (10)

### 4. The Practical Step-by-Step Workflow

Since there is no magic formula, here is exactly how you should approach it in practice:

1. **Start Small:** Always begin with a smaller, simple network. Start with one hidden layer of **32 or 64** neurons.
2. **Check the Performance:** Train the model and look at its accuracy. Does it perform terribly on the data you trained it on? This means the network isn't smart enough (Underfitting).
3. **Scale Up:** If it is underfitting, double the neurons (e.g., go from 64 to 128) and try again.
4. **Watch for the Trap:** Keep scaling up until the model starts performing _worse_ on new, unseen data. That means it has stopped learning and started memorizing (Overfitting). When that happens, you step back to the previous number.

Eventually, you will write code (using tools called Hyperparameter Tuners) that automatically loops through 32, 64, 128, and 256, tests them all, and just hands you the best result!

---

### 💡 Supplementary Notes

- **Softmax & Cross-Entropy Stability**: In multiclass classifiers, computing softmax first and then categorical cross-entropy separately can lead to numerical instability (underflow/overflow). Modern frameworks solve this by combining them internally or offering loss functions that take raw logits.

---

#### Hyperparameter Tuners - that automatically loops through 32, 64, 128, and 256, tests them all, and just hands you the best result!

To do this in Keras, we use a dedicated library called **KerasTuner**. It acts as a project manager for your neural network. Instead of building one model, you tell KerasTuner how to build a _flexible_ model, give it a menu of options, and tell it to go find the best one.

Here is exactly what that code looks like for your MNIST model.

### 1. The Setup (Installing the Tool)

First, you would need to install the library if you haven't already. In a Jupyter Notebook or terminal, you would run:

```bash
pip install keras-tuner

```

### 2. The Code: Automating the Search

Here is the Python code to automate finding the perfect number of neurons.

```python
from tensorflow import keras
import keras_tuner as kt

# Step 1: Write a function that builds your model with a "variable" instead of a hardcoded number
def build_model(hp):
    model = keras.Sequential()

    # Prep step (always the same)
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # THE MAGIC HAPPENS HERE:
    # Instead of typing 128, we define a "Hyperparameter" (hp) with a menu of choices
    hp_neurons = hp.Choice('dense_neurons', values=[32, 64, 128, 256])

    # Add the hidden layer using our flexible hp_neurons variable
    model.add(keras.layers.Dense(units=hp_neurons, activation='relu'))

    # Output layer (always 10 for MNIST)
    model.add(keras.layers.Dense(10, activation='softmax'))

    # Compile the model (setting up the rules for how it learns)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Step 2: Set up the Tuner (The Project Manager)
tuner = kt.RandomSearch(
    build_model,                     # The function we just wrote
    objective='val_accuracy',        # The goal: find the highest validation accuracy
    max_trials=4,                    # How many different setups to try
    directory='mnist_tuning',        # Folder to save the results
    project_name='neuron_search'
)

# Step 3: Start the loop!
# (Assuming you already have your MNIST training and validation data loaded)
print("Starting the automated search...")
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

# Step 4: Ask the Tuner for the best model it found
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"The best number of neurons was: {best_hyperparameters.get('dense_neurons')}")

```

---

### What is actually happening here?

- **`hp.Choice`**: This is the most important part. You are telling the code, "Don't just run once. Run multiple times, and each time you run, pick a different number from this list: `[32, 64, 128, 256]`".

- **`kt.RandomSearch`**: This is the controller. It looks at your `build_model` function, sees the menu of choices, and methodically tests them.
- **`objective='val_accuracy'`**: This is how the Tuner knows what "best" means. It trains a model with 32 neurons, checks the accuracy. Then it trains a model with 64 neurons, checks the accuracy, and so on. At the end, it ranks them based on which one scored the highest on the unseen validation data.
- **`tuner.get_best_models`**: After the search is done, you don't have to rebuild anything. The Tuner hands you back the fully built, optimized model, ready to be used!
