# Tutorial 18: Customer Churn Prediction

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=MSBY28IJ47U)

## Executive Summary
This video provides a comprehensive tutorial on predicting customer churn using an Artificial Neural Network built with Python, Pandas, and TensorFlow. The project covers the full machine learning lifecycle from data cleaning and visualization to model training and performance evaluation using metrics like precision and recall.

## The "Why" (First Principles)

Customer churn measures how many customers stop doing business with a company, which is a critical problem for industries like telecommunications and banking. By predicting which customers are likely to leave, businesses can take proactive actions to retain them, potentially saving significant revenue.

Deep learning models like **Artificial Neural Networks** are effective at identifying complex patterns in customer behavior data that correlate with churn, allowing for more nuanced predictions than traditional statistical methods.

## Detailed Step-by-Step Notes
1

**Dataset acquisition and initial exploration:** The Telco Customer Churn dataset is loaded using Pandas to examine features like tenure, monthly charges, and services used.
2

**Data cleaning:** Irrelevant columns like `CustomerID` are dropped. Strings in columns like `TotalCharges` are converted to numeric values, handling empty spaces by coercing errors and dropping null records.
3

**Feature engineering:** Categorical values with similar meanings, such as "No internet service", are consolidated into a standard "No" value. Yes and No strings are converted to binary 1 and 0 values.
4

**Categorical encoding:** Multi-category text data, such as contract types and payment methods, are converted into numerical columns via one-hot encoding using the `pd.get_dummies` function.
5

**Feature scaling:** Continuous variables such as tenure and monthly charges are scaled to a 0 to 1 range using `MinMaxScaler` to ensure training stability.
6

**Model architecture:** A sequential model is defined with an input layer matching features, hidden layers with `ReLU`, and a single-neuron output layer with `sigmoid` activation.
7

**Training and evaluation:** The model is compiled with the `Adam` optimizer and binary crossentropy loss, validated via confusion matrix and classification reports.

## Highlighted Examples

#### Telecommunications

A wireless carrier tracking when a user stops their service and moves to a competitor.

#### Banking Context

A financial institution monitoring when a customer closes their account.

#### Tenure Visualization

Analysis showing customers with tenure over 70 months are statistically much less likely to leave.

#### Monthly Charges

Trends showing customers with higher bills ($70–$110 range) have a higher frequency of churning.

## Technical Execution
pandas_preprocessing.py Python

```python
# Drop the customerID identifier column in-place since it doesn't contain useful patterns for predicting churn
df.drop(columns=['customerID'], axis='columns', inplace=True)
# Convert the TotalCharges column to numerical values, setting empty spaces or invalid strings to NaN (coerce errors)
pd.to_numeric(df.TotalCharges, errors='coerce')
# Select the tenure values specifically for customers who have not churned (Churn == 'No') for visualization
df1[df1.Churn == 'No'].tenure
# Apply one-hot encoding to specified multi-category text columns to convert them into binary column vectors
pd.get_dummies(data=df1, columns=['InternetService', 'Contract', 'PaymentMethod'])
```
keras_model.py TensorFlow

```python
# Instantiate a MinMaxScaler object to scale continuous numeric values to a uniform [0, 1] range
scaler = MinMaxScaler()
# Fit the scaler and transform the numerical columns in df2 in-place to ensure learning stability
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

# Define a Sequential model to build a feedforward network layer-by-layer
model = keras.Sequential([
   # Add a dense hidden layer with 20 neurons, expecting 26 input features, and using ReLU activation
   keras.layers.Dense(20, input_shape=(26,), activation='relu'),
   # Add a single-neuron output layer with Sigmoid activation to estimate binary churn probabilities
   keras.layers.Dense(1, activation='sigmoid')
])

# Compile the neural network with the Adam optimizer, binary crossentropy loss, and accuracy metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
Accuracy `Correct / Total`
Precision `TP / (TP + FP)`
Recall `TP / (TP + FN)`

### 💡 Beginner's Blueprint: The 80/20 Rule of Data Science

If you expect to spend all your time building cool deep learning architectures, you are in for a surprise. In real-world projects, **80% of your time is spent cleaning and preparing data**, while only 20% is spent writing the neural network code.

Why? Because neural networks are mathematical engines that follow the **"Garbage In, Garbage Out"** rule:
* If you leave empty spaces in numeric columns (like empty strings in `TotalCharges`), Keras will throw error messages.
* If you feed words directly to the model (like "Yes"/"No" or "InternetService"), the neural network cannot multiply them because it only understands numbers. This is why we use **One-Hot Encoding**.
* If you don't scale features, columns with massive numbers (like `MonthlyCharges` up to 120) will dominate columns with small numbers (like `tenure` under 10), confusing the optimizer.

Before you write `model.fit()`, always ensure your dataset is cleaned, scaled, and encoded!

---

### 💡 Supplementary Notes

* **Data Leakage in Feature Scaling**: Always fit scaler transforms (e.g., `StandardScaler.fit()`) solely on the **training set** and then transform the test set. Fitting on the entire dataset leads to data leakage, producing overly optimistic validation performance.

## Active Recall Checkpoint
01.

Why is it necessary to convert categorical text data into numerical values before feeding them into an artificial neural network?
02.

Explain how to handle numerical columns that are incorrectly formatted as objects or strings in a Pandas DataFrame.
03.

How does the choice of activation function for the output layer change when performing binary classification versus multi-class classification?

Try to answer these without looking back at the notes.