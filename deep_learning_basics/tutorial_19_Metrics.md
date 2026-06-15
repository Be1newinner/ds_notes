# Tutorial 19: Classification Metrics & Evaluation

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=2osIZ-dSPGE)

## Executive Summary

In classification, **Accuracy is a trap**. If 99% of your emails are normal and 1% are spam, a model that predicts "Not Spam" for every single email is 99% accurate, but completely useless. This guide explains how to use the **Confusion Matrix**, **Precision**, **Recall**, and **F1-Score** to properly evaluate models, especially on imbalanced datasets.

---

## 1. The Foundation: The Confusion Matrix

A Confusion Matrix is the ultimate map of how your classification model is thinking. Instead of giving you a single summary number (like accuracy), it lays bare every success and every failure, categorizing them into four distinct buckets.

To understand the confusion matrix, it helps to look at two highly intuitive real-world scenarios:

1. **The Medical Diagnostics Test (Disease Screening)**
2. **The Fire Alarm System**

### The Core Terminology

Every prediction falls into one of these four categories:

- **True Positive (TP)**: You predicted positive, and it was positive (Correct).
- **True Negative (TN)**: You predicted negative, and it was negative (Correct).
- **False Positive (FP)** (Type I Error): You predicted positive, but it was negative (False Alarm).
- **False Negative (FN)** (Type II Error): You predicted negative, but it was positive (Missed Opportunity / Danger).

Here is how they align in a matrix:

|                                         | **Actual Positive (Truth: Yes)**                               | **Actual Negative (Truth: No)**                          |
| :-------------------------------------- | :------------------------------------------------------------- | :------------------------------------------------------- |
| **Predicted Positive (Model Says Yes)** | **True Positive (TP)**<br>✨ Correct hit!                      | **False Positive (FP)**<br>⚠️ Type I Error (False Alarm) |
| **Predicted Negative (Model Says No)**  | **False Negative (FN)**<br>🔥 Type II Error (Missed Detection) | **True Negative (TN)**<br>✨ Correct rejection!          |

---

### Scenario A: The COVID-19 Rapid Test (Medical Screening)

Imagine you take a rapid at-home test for a virus.

- **Actual Class**: Positive ($1$) = Sick with the virus, Negative ($0$) = Healthy.
- **Predicted Class**: Positive ($1$) = Test says "You are sick", Negative ($0$) = Test says "You are healthy".

Let's look at the consequences of each outcome:

1. **True Positive (TP)**: You actually have the virus, and the test says you have it.
   - _Consequence_: You isolate, get treatment, and recover safely.
2. **True Negative (TN)**: You are healthy, and the test says you are healthy.
   - _Consequence_: You go about your day with peace of mind.
3. **False Positive (FP) - Type I Error**: You are healthy, but the test displays a positive line.
   - _Consequence_: You panic, cancel your plans, isolate unnecessarily, and get a PCR test only to find out you were fine all along. _Cost: Temporary inconvenience and anxiety._
4. **False Negative (FN) - Type II Error**: You actually have the virus, but the test says you are healthy.
   - _Consequence_: You believe you are safe, go to a family gathering, and accidentally infect vulnerable relatives. _Cost: High danger/risk to lives._

> [!IMPORTANT]
> In medical testing, **False Negatives (Type II Errors) are far more dangerous** than False Positives. We want to minimize False Negatives at all costs, even if it means having a few more False Positives (false alarms).

---

### Scenario B: The Fire Alarm System

Imagine building a smart IoT fire detection sensor for a high-rise building.

- **Actual Class**: Positive ($1$) = There is a fire, Negative ($0$) = No fire (Normal).
- **Predicted Class**: Positive ($1$) = Sound the alarm, Negative ($0$) = Remain silent.

Let's look at the outcomes:

1. **True Positive (TP)**: There is a fire, and the alarm sounds.
   - _Consequence_: Everyone evacuates safely. The system saved lives.
2. **True Negative (TN)**: There is no fire, and the alarm remains silent.
   - _Consequence_: Everyone works or sleeps peacefully.
3. **False Positive (FP) - Type I Error**: There is no fire, but a dust particle triggers the sensor, sounding the alarm.
   - _Consequence_: The building evacuates in the middle of the night. Fire trucks arrive. Everyone is annoyed by the false alarm. _Cost: Annoyance, disruption, and potential "crying wolf" syndrome (people ignoring future alarms)._
4. **False Negative (FN) - Type II Error**: There is a roaring fire, but the sensor fails and the alarm remains silent.
   - _Consequence_: No one evacuates. Massive loss of property and lives. _Cost: Absolute catastrophe._

---

## 2. Breaking Down the Metrics

A single confusion matrix holds all the raw data. To summarize it for business decisions, we extract specific metrics.

### 1. Accuracy (The Default Trap)

- **What it answers**: _"Out of all predictions, what percentage did we get right?"_
- **Formula**:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **The Real-Life Trap**:
  Imagine you are building a Credit Card Fraud Detection model. In the real world, only **$0.1\%$** of all transactions are fraudulent ($1$ in $1000$).
  If you build a dumb model that simply outputs `No Fraud` for every single transaction:
  - **TP** = 0 (we caught no fraud)
  - **TN** = 999 (all normal transactions correctly predicted)
  - **FP** = 0
  - **FN** = 1 (the one fraud transaction we missed)

  $$\text{Accuracy} = \frac{0 + 999}{0 + 999 + 0 + 1} = 99.9\%$$

  A $99.9\%$ accuracy sounds amazing, but this model is **completely useless** because it lets $100\%$ of all fraud go through!

- **When to use it**: Use Accuracy **only** when your classes are balanced (e.g., classifying images as $50\%$ cats vs $50\%$ dogs).

---

### 2. Precision (Trustworthiness of Positives)

- **What it answers**: _"Of all the times the model predicted 'Positive', how often was it actually right?"_
- **Focus**: Minimizing **False Positives (FP)**.
- **Formula**:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Real-Life Hero Example: Email Spam Filter**
  - **Goal**: Detect spam emails and send them to the Spam folder.
  - If a normal email from your bank or boss is incorrectly labeled as "Spam" (**False Positive**), you could miss critical information, lose money, or get fired.
  - Therefore, you need your Spam Filter to have **extremely high Precision** (e.g., $99.9\%$). If it tells you an email is spam, it _must_ be spam. We would rather let a few spam emails slip into the inbox (allowing some False Negatives) than misclassify a crucial email (avoiding False Positives).
- **High Precision means**: "If I say it's positive, you can bet your life savings it is!"

---

### 3. Recall / Sensitivity (Coverage / Search Ability)

- **What it answers**: _"Of all the actual positives that existed, how many did the model manage to find?"_
- **Focus**: Minimizing **False Negatives (FN)**.
- **Formula**:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **Real-Life Hero Example: Cancer Detection in MRI Scans**
  - **Goal**: Detect tumors in medical scans.
  - If a patient has a malignant tumor but the model classifies it as benign (**False Negative**), the patient goes home untreated and the cancer spreads, leading to a fatal outcome.
  - In this scenario, we need **extremely high Recall** (e.g., $99.9\%$). We want to catch _every single_ tumor. If the model triggers a false alarm on a healthy patient (**False Positive**), further tests (like a biopsy) will quickly clear it up. A false alarm is a minor inconvenience; a missed tumor is fatal.
- **High Recall means**: "No positive case gets left behind!"

---

### 4. F1-Score (The Balanced Compromise)

- **What it answers**: _"How well does my model balance both Precision and Recall?"_
- **Formula**:
  $$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Why Harmonic Mean instead of Simple Average (Arithmetic Mean)?

Imagine we evaluated our model using a simple average of Precision and Recall: $\text{Average} = \frac{\text{Precision} + \text{Recall}}{2}$.

Let's test this with an extreme, useless model:

- A model that predicts **Positive** for every single sample.
- It achieves a **Recall of $1.0$** (it caught $100\%$ of all positive cases).
- But its **Precision is extremely low, say $0.01$** (only $1\%$ of its positive predictions were actual positives).

Let's compare the two scoring methods:

1. **Simple Arithmetic Average**:
   $$\text{Arithmetic Mean} = \frac{1.0 + 0.01}{2} = 0.505 \text{ (or } 50.5\% \text{)}$$
   _A score of $50.5\%$ makes the model look mediocre but somewhat functional, which is deceptive!_
2. **Harmonic Mean (F1-Score)**:
   $$\text{F1-Score} = 2 \cdot \frac{1.0 \cdot 0.01}{1.0 + 0.01} = 2 \cdot \frac{0.01}{1.01} \approx 0.0198 \text{ (or } 1.98\% \text{)}$$
   _The F1-score immediately flags this model as trash ($1.98\%$)._

> [!TIP]
> The **Harmonic Mean** acts like a severe penalty. If either Precision or Recall drops to near zero, the entire F1-Score collapses to near zero. It forces the model to perform well on _both_ metrics to get a high score.

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

- **Precision-Recall AUC**: For highly imbalanced datasets, the Receiver Operating Characteristic (ROC) curve can present an overly optimistic view of model performance. A **Precision-Recall (PR)** curve is more informative as it focuses directly on the minority class.

---

## 4. Active Recall Checkpoint

#### 1. The Accuracy Trap

If you build a model to detect credit card fraud (which happens in only $0.1\%$ of transactions) and it simply predicts "No Fraud" for every transaction:

- What is its Accuracy?
- Why is it useless?
- What metric would you track instead to ensure you catch fraud?

#### 2. Precision vs. Recall Trade-off

If a smart home security system triggers a loud alarm and alerts the police every time a leaf blows in the wind or a stray cat walks by, just to make sure no burglar ever goes undetected:

- Does this system have **High Recall & Low Precision** or **High Precision & Low Recall**?
- What is the real-world cost/consequence of this setting?

#### 3. F1-Score Utility

Explain why we cannot just use a simple arithmetic average (mean) of Precision and Recall instead of the Harmonic Mean (F1-Score). Walk through what happens to both scores if a model has a Precision of $1.0$ but a Recall of $0.0$.
