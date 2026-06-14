# Tutorial 19: Classification Metrics & Evaluation

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=2osIZ-dSPGE)

## Executive Summary

In classification, **Accuracy is a trap**. If 99% of your emails are normal and 1% are spam, a model that predicts "Not Spam" for every single email is 99% accurate, but completely useless. This guide explains how to use the **Confusion Matrix**, **Precision**, **Recall**, and **F1-Score** to properly evaluate models, especially on imbalanced datasets.

---

## 1. The Foundation: The Confusion Matrix

A Confusion Matrix is a tabular summary of the model's predictions compared to the actual truths. To understand the terminology, think of a **medical test for a disease**:

| | Actual Positive (Sick) | Actual Negative (Healthy) |
| :--- | :--- | :--- |
| **Predicted Positive (Test Says Sick)** | **True Positive (TP)**<br>Model correctly identified a sick person. | **False Positive (FP)** (Type I Error)<br>Model said a healthy person is sick. |
| **Predicted Negative (Test Says Healthy)** | **False Negative (FN)** (Type II Error)<br>Model said a sick person is healthy. | **True Negative (TN)**<br>Model correctly identified a healthy person. |

---

## 2. Breaking Down the Metrics

### 1. Accuracy
* **Definition**: The percentage of correct predictions out of all predictions.
* **Formula**: $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **When to use**: Only when your classes are balanced (e.g., 50% cats and 50% dogs).

### 2. Precision (Quality of Positives)
* **Definition**: Out of all samples the model *predicted* as positive, how many were *actually* positive?
* **Formula**: $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Beginner Analogy**: A spam filter. You want to be absolutely sure that if an email is labeled as "Spam" (Positive), it is actually spam. If Precision is low, important work emails (False Positives) end up in the junk folder.

### 3. Recall / Sensitivity (Quantity of Positives Found)
* **Definition**: Out of all the *actual* positive samples in the dataset, how many did the model manage to *find/catch*?
* **Formula**: $$\text{Recall} = \frac{TP}{TP + FN}$$
* **Beginner Analogy**: Cancer detection. You want to catch *every single* cancer case. If the test misses a case (False Negative), a patient goes untreated. We tolerate a lower Precision (more false alarms/false positives) to ensure high Recall.

### 4. F1-Score (The Balance)
* **Definition**: The harmonic mean of Precision and Recall. It gives you a single score that balances both.
* **Formula**: $$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 3. Python Implementation

Here is how you evaluate your models using Scikit-Learn and TensorFlow/Keras:

### Scikit-Learn Implementation
```python
# Import evaluation metric classes confusion_matrix and classification_report from Scikit-Learn
from sklearn.metrics import confusion_matrix, classification_report

# Define an array of ground-truth target labels for evaluation
y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
# Define an array of corresponding model prediction outputs to compare against truths
y_pred = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0]

# Print header label for the Confusion Matrix
print("Confusion Matrix:")
# Generate and print the confusion matrix showing counts of TP, TN, FP, FN
print(confusion_matrix(y_true, y_pred))

# Print header label for the overall metrics table
print("\nClassification Report:")
# Generate and print the report summary containing precision, recall, f1-score, and support metrics
print(classification_report(y_true, y_pred))
```

### Keras Metric Compilation
```python
# Import the Keras deep learning framework to build and compile models
import keras
# Import the layers module from Keras to stack network structures
from keras import layers

# Define a Sequential network stack of linear layers
model = keras.Sequential([
    # Add a dense hidden layer with 16 neurons, expecting 10 input features, and using ReLU activation
    layers.Dense(16, activation='relu', input_shape=(10,)),
    # Add a single-neuron output layer with Sigmoid activation to estimate binary class probability
    layers.Dense(1, activation='sigmoid')
])

# Compile the model specifying its optimizer, loss function, and targeted metrics to track
model.compile(
    # Use the Adam optimizer for adaptive learning rate weight updates
    optimizer='adam',
    # Use binary crossentropy loss since our classification has exactly 2 classes (binary)
    loss='binary_crossentropy',
    # List of Keras metric classes to compute during training and validation
    metrics=[
        # Track binary prediction accuracy epoch-by-epoch
        keras.metrics.BinaryAccuracy(name='accuracy'),
        # Track model precision to assess positive prediction quality
        keras.metrics.Precision(name='precision'),
        # Track model recall to measure the percentage of actual positives successfully retrieved
        keras.metrics.Recall(name='recall')
    ]
)
```

---

### 💡 Supplementary Notes

* **Precision-Recall AUC**: For highly imbalanced datasets, the Receiver Operating Characteristic (ROC) curve can present an overly optimistic view of model performance. A **Precision-Recall (PR)** curve is more informative as it focuses directly on the minority class.

---

## 4. Active Recall Checkpoint

#### The Accuracy Trap
If you build a model to detect fraud (which happens in only 0.1% of transactions) and it simply predicts "No Fraud" for every transaction, what is its Accuracy? Why is it useless?

#### Precision vs. Recall Trade-off
If a security system triggers an alarm every time a leaf blows in the wind to ensure no intruder gets in, does it have **High Recall/Low Precision** or **High Precision/Low Recall**?

#### F1 Score Utility
Why can't we just use a simple arithmetic average (mean) of Precision and Recall instead of the Harmonic Mean (F1-Score)? (Hint: What happens to the F1 score if Precision is 1.0 but Recall is 0.0?)