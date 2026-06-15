# Tutorial 18: Customer Churn Prediction

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=MSBY28IJ47U)

## Executive Summary

This video provides a comprehensive tutorial on predicting customer churn using an Artificial Neural Network built with Python, Pandas, and TensorFlow. The project covers the full machine learning lifecycle from data cleaning and visualization to model training and performance evaluation using metrics like precision and recall.

## The "Why" (First Principles)

Customer churn measures how many customers stop doing business with a company, which is a critical problem for industries like telecommunications and banking. By predicting which customers are likely to leave, businesses can take proactive actions to retain them, potentially saving significant revenue.

Deep learning models like **Artificial Neural Networks** are effective at identifying complex patterns in customer behavior data that correlate with churn, allowing for more nuanced predictions than traditional statistical methods.

## Detailed Step-by-Step Notes

**1. Dataset acquisition and initial exploration:** The Telco Customer Churn dataset is loaded using Pandas to examine features like tenure, monthly charges, and services used.

**2. Data cleaning:** Irrelevant columns like `CustomerID` are dropped. Strings in columns like `TotalCharges` are converted to numeric values, handling empty spaces by coercing errors and dropping null records.

**3. Feature engineering:** Categorical values with similar meanings, such as "No internet service", are consolidated into a standard "No" value. Yes and No strings are converted to binary 1 and 0 values.

**4. Categorical encoding:** Multi-category text data, such as contract types and payment methods, are converted into numerical columns via one-hot encoding using the `pd.get_dummies` function.

**5. Feature scaling:** Continuous variables such as tenure and monthly charges are scaled to a 0 to 1 range using `MinMaxScaler` to ensure training stability.

**6. Model architecture:** A sequential model is defined with an input layer matching features, hidden layers with `ReLU`, and a single-neuron output layer with `sigmoid` activation.

**7. Training and evaluation:** The model is compiled with the `Adam` optimizer and binary crossentropy loss, validated via confusion matrix and classification reports.

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

### Complete Churn Prediction Pipeline (`churn_prediction.py`)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras

print("Loading dataset...")
# Step 1: Load the Telco Customer Churn dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 2: Drop customerID column
df.drop(columns=['customerID'], inplace=True)

# Step 3: Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Step 4: Drop null records
df.dropna(subset=['TotalCharges'], inplace=True)

# Step 5: Clean categorical values with redundant meanings
df.replace('No internet service', 'No', inplace=True)
df.replace('No phone service', 'No', inplace=True)

# Step 6: Convert 'Yes' and 'No' text values to binary numeric values (1 and 0)
yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'PaperlessBilling', 'Churn']
for col in yes_no_cols:
    df[col] = df[col].replace({'Yes': 1, 'No': 0})

# Step 7: Map the 'gender' column to binary integers
df['gender'] = df['gender'].replace({'Female': 0, 'Male': 1})

# Step 8: Apply One-Hot Encoding to the remaining multi-category columns
df_encoded = pd.get_dummies(data=df, columns=['InternetService', 'Contract', 'PaymentMethod'], dtype=int)

# Step 9: Scale continuous numeric variables to a uniform [0, 1] range
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = MinMaxScaler()
df_encoded[cols_to_scale] = scaler.fit_transform(df_encoded[cols_to_scale])

# Step 10: Split features (X) and label target (y)
X = df_encoded.drop('Churn', axis='columns')
y = df_encoded['Churn']

# Step 11: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Step 12: Build the ANN Sequential model
model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(X_train.shape[1],), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Step 13: Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 14: Fit/Train the model (just run for 5 epochs to check compatibility and functionality)
print("Training model for 5 epochs...")
model.fit(X_train, y_train, epochs=5)

# Step 15: Run predictions
y_pred_probs = model.predict(X_test)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

# Step 16: Evaluate model performance
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Test completed successfully!")
```

Accuracy `Correct / Total`
Precision `TP / (TP + FP)`
Recall `TP / (TP + FN)`

### Customer Churn Predections

### 1. Bank Customer Churn Prediction

This dataset is perfect for practicing mixed data types and handling class imbalance.

- **Kaggle Link:** [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) _(Note: There are a few user-uploaded variations of this dataset on Kaggle, but this is a standard version that fits the exact feature set!)_

### 2. E-Commerce Customer Churn Prediction

This retail dataset is great for teaching students how to handle missing values and think critically about ordinal vs. categorical features.

- **Kaggle Link:** [E-Commerce Customer Churn Dataset](https://www.kaggle.com/datasets/vishardmehta/e-commerce-customer-churn-prediction-dataset)

### 3. KKBox's Churn Prediction Challenge

This is an advanced, capstone-level challenge from the WSDM Cup. It is fantastic for introducing relational databases, feature engineering, and dealing with temporal user logs.

- **Kaggle Link:** [WSDM - KKBox's Churn Prediction Challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge)

### 💡 Beginner's Blueprint: The 80/20 Rule of Data Science

If you expect to spend all your time building cool deep learning architectures, you are in for a surprise. In real-world projects, **80% of your time is spent cleaning and preparing data**, while only 20% is spent writing the neural network code.

Why? Because neural networks are mathematical engines that follow the **"Garbage In, Garbage Out"** rule:

- If you leave empty spaces in numeric columns (like empty strings in `TotalCharges`), Keras will throw error messages.
- If you feed words directly to the model (like "Yes"/"No" or "InternetService"), the neural network cannot multiply them because it only understands numbers. This is why we use **One-Hot Encoding**.
- If you don't scale features, columns with massive numbers (like `MonthlyCharges` up to 120) will dominate columns with small numbers (like `tenure` under 10), confusing the optimizer.

Before you write `model.fit()`, always ensure your dataset is cleaned, scaled, and encoded!

---

### 💡 Supplementary Notes

- **Data Leakage in Feature Scaling**: Always fit scaler transforms (e.g., `StandardScaler.fit()`) solely on the **training set** and then transform the test set. Fitting on the entire dataset leads to data leakage, producing overly optimistic validation performance.

## Active Recall Checkpoint

1.

Why is it necessary to convert categorical text data into numerical values before feeding them into an artificial neural network? 02.

Explain how to handle numerical columns that are incorrectly formatted as objects or strings in a Pandas DataFrame. 03.

How does the choice of activation function for the output layer change when performing binary classification versus multi-class classification?

Try to answer these without looking back at the notes.
