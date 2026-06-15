# Handling Imbalanced Datasets in Machine Learning

**A Complete Tutorial with Code, Real-Life Examples, and 2026 Best Practices**

Imbalanced datasets are one of the most common reasons a model looks great in testing but fails in production. When one class is rare, the model can learn to “play safe” by predicting the majority class most of the time and still get high accuracy. This tutorial teaches you why that happens, how to detect it, and how to fix it using modern techniques.

---

## 1. What Is an Imbalanced Dataset?

A dataset is imbalanced when the classes are not represented equally.

### Real-Life Examples

- **Fraud detection**: 99.9% genuine transactions, 0.1% fraud
- **Medical diagnosis**: far fewer positive disease cases than negative cases
- **Customer churn**: only a small percentage of customers actually leave
- **Manufacturing defect detection**: most products are fine, only a few are faulty
- **Rare event prediction**: equipment failure, cyberattacks, insurance claims

In all these cases, the minority class is usually the one you care about most. Missing rare events is often more costly than making a few extra false alarms.

### Synthetic Example

```python
import pandas as pd
import numpy as np

# Create a highly imbalanced dataset (99% class 0, 1% class 1)
n_samples = 10000
X = np.random.randn(n_samples, 5)
y = np.array([0] * 9900 + [1] * 100)

df = pd.DataFrame(X, columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
df['target'] = y

print(df['target'].value_counts(normalize=True))
```

Output:

```
target
0.0   0.99
1.0   0.01
```

---

## 2. Why Accuracy Fails on Imbalanced Data

Accuracy is not enough for imbalanced problems because a model can look good while being useless.

### The Classic Trap

- **Dataset**: 10,000 bank transactions, only 10 are fraud
- **Model**: predicts every transaction as “genuine”
- **Accuracy**: 9,990 / 10,000 = **99.9%**
- **Reality**: caught **zero fraud cases**

That model is useless in practice.

### Better Metrics to Use

- **Precision**: what fraction of predicted positives are real?
- **Recall (Sensitivity)**: what fraction of real positives did we catch?
- **F1 Score**: balance between precision and recall
- **ROC-AUC**: overall ranking quality
- **PR-AUC**: more informative than ROC-AUC for extreme imbalance
- **Confusion Matrix**: exact breakdown of errors

### When to Use Which Metric

| Scenario                                                 | Key Metric       | Why                      |
| -------------------------------------------------------- | ---------------- | ------------------------ |
| False alarms are expensive (e.g., medical follow-ups)    | Precision        | Avoid unnecessary costs  |
| Missing positives is dangerous (e.g., disease detection) | Recall           | Catch as many real cases |
| Balanced trade-off needed                                | F1 Score         | Combined performance     |
| Extreme imbalance (fraud, defects)                       | PR-AUC           | Better than ROC-AUC      |
| Full error breakdown                                     | Confusion Matrix | See exact mistakes       |

---

## 3. Real-World Intuition

### Medical Screening (High Recall Priority)

- Disease is rare but critical
- Missing a sick patient is very costly
- Model should predict “sick” more often
- Favor higher recall, even if it creates more false alarms
- Doctors use model as decision aid, not final authority

### Spam Filter (High Precision Priority)

- Too much false spam hides real emails
- False alarms disrupt communication
- Model should be conservative about marking spam
- Favor higher precision

### Fraud Detection (Balanced but Recall-Focused)

- Fraud is rare but expensive
- Missing fraud costs money
- False alarms cost investigation time
- Tune threshold to balance recall vs. precision based on business cost

---

## 4. Common Techniques to Handle Imbalance

There is no single universal solution. In practice, you combine several methods.

### 4.1 Under-Sampling

**What it does:** reduces majority class samples randomly.

```python
from sklearn.utils import resample

df_majority = df[df['target'] == 0]
df_minority = df[df['target'] == 1]

df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

df_balanced = pd.concat([df_majority_downsampled, df_minority])
```

**Pros:**

- Faster training
- More balanced training distribution

**Cons:**

- Throws away data
- May lose important patterns

**When to use:**

- Very large datasets
- Training speed matters
- Majority class is highly redundant

---

### 4.2 Over-Sampling (Simple Duplication)

**What it does:** duplicates minority class samples until classes are balanced.

```python
from sklearn.utils import resample

df_minority = df[df['target'] == 1]
df_majority = df[df['target'] == 0]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
```

**Pros:**

- Keeps all majority data
- More exposure to minority class

**Cons:**

- Duplicates cause overfitting
- Model may memorize repeated examples

**When to use:**

- Small to medium datasets
- You don't want to lose majority information

---

### 4.3 SMOTE (Synthetic Minority Over-sampling Technique)

**What SMOTE does:** creates _synthetic_ minority samples by interpolating between nearby minority points instead of copying.

```python
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**Why SMOTE is better:**

- Adds variation
- Reduces overfitting from duplication
- Often performs better than naive over-sampling

**Important correction for 2026:**

- **SMOTE must be applied only on the training set, after splitting**
- If you oversample before splitting, synthetic points derived from test data can leak into training, making evaluation unrealistically good

**SMOTE Variants:**

- **Borderline-SMOTE**: focuses on difficult boundary cases
- **ADASYN**: generates more samples in harder regions
- **SMOTE-ENN / SMOTE-Tomek**: combines oversampling with cleaning noisy points
- **KMeans-SMOTE**: groups data first, then generates synthetic samples strategically

These variants are often better than plain SMOTE when the minority class is noisy or scattered.

---

### 4.4 Class Weights (Algorithm-Level)

**What it does:** makes the model pay more penalty for minority-class mistakes.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)
```

Or with custom weights:

```python
class_weight = {0: 1.0, 1: 10.0}
```

**Pros:**

- Simple and effective
- No data manipulation needed
- Works with most models

**Cons:**

- Some models don't support it

**Where it works:**

- Logistic regression
- SVM
- Decision trees
- Random forest
- Neural networks

**2026 note:**
For tabular data, class weights are often the **first thing to try** before resampling tricks.

---

### 4.5 Focal Loss (Deep Learning)

**What focal loss does:** reduces the influence of easy majority examples and focuses learning on hard ones.

```python
import tensorflow as tf

def focal_loss(gamma=2., alpha=1.):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.log(y_pred)
        weight = alpha * (1 - y_pred) ** gamma
        loss = weight * cross_entropy
        return tf.reduce_mean(loss)
    return loss_fn

model.compile(optimizer='adam', loss=focal_loss())
model.fit(X_train, y_train, epochs=10)
```

**Where it's used:**

- Object detection
- Rare-event classification
- Some fraud and anomaly tasks

**2026 note:**
Focal loss is still relevant but usually not the first choice for tabular data. For tabular ML, class weights and boosting methods are more practical.

---

### 4.6 Ensemble Methods

**Balanced Bagging:**
Train several models on different balanced subsets.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier

models = []
kfold = StratifiedKFold(n_splits=5)

for train_idx, _ in kfold.split(X_train, y_train):
    X_sub, y_sub = X_train[train_idx], y_train[train_idx]
    smote = SMOTE(random_state=42)
    X_sub_res, y_sub_res = smote.fit_resample(X_sub, y_sub)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_sub_res, y_sub_res)
    models.append(model)

voting = VotingClassifier(estimators=models, voting='hard')
```

**Balanced Random Forest:**
Each tree sees all minority samples and a different sample of majority cases.

**Boosting-based methods (XGBoost, LightGBM, CatBoost):**
These are often the strongest for imbalanced tabular data.

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    random_state=42,
    scale_pos_weight=len(y_train) / y_train.sum()  # automatically sets class weight
)
model.fit(X_train, y_train)
```

**2026 note:**
Gradient-boosted trees remain one of the **strongest choices** for imbalanced tabular data.

---

## 5. Threshold Tuning (Critical and Often Overlooked)

A classifier predicts probabilities, not just classes. The default threshold is 0.5, but for imbalanced problems that's often not optimal.

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

y_proba = model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Find best threshold based on business cost
best_threshold = thresholds[np.argmax(precisions + recalls)]
```

**Example:**

- Fraud detection: 0.5 threshold may miss many fraud cases
- Lowering threshold to 0.2 or 0.1 improves recall (catches more fraud)
- Precision may drop (more false alarms), but that's acceptable if investigation cost is low

**When to use:**

- Medical screening: prefer lower threshold to catch more patients
- Spam filter: prefer higher threshold to avoid hiding important emails

---

## 6. Data Leakage Traps (Critical for 2026)

Never do these:

- Oversample before train-test split
- Scale using the full dataset before splitting
- Use test data to guide SMOTE or feature engineering
- Tune thresholds directly on the test set

Always do this:

- Split first
- Apply resampling only on the training fold
- Use validation or cross-validation for tuning
- Keep the test set untouched until the end

If you break this rule, your model will look stronger than it really is.

---

## 7. Complete Practical Workflow

Use this sequence for most projects:

1. **Understand business cost of errors**
2. **Check class distribution**
3. **Split data with stratification**
4. **Establish a baseline model**
5. **Evaluate with precision, recall, F1, PR-AUC, confusion matrix**
6. **Try class weights first**
7. **Try oversampling or SMOTE on training data only**
8. **Try balanced ensembles or boosting**
9. **Tune threshold based on real-world cost**
10. **Re-test on untouched data**

---

## 8. Which Method to Choose?

| Situation                                      | Best Starting Method                      |
| ---------------------------------------------- | ----------------------------------------- |
| Large dataset with redundant majority examples | Under-sampling                            |
| Small dataset, don't want to lose data         | SMOTE or over-sampling                    |
| Tabular classification                         | Class weights + XGBoost/LightGBM/CatBoost |
| Deep learning                                  | Class-weighted loss or focal loss         |
| Very noisy minority class                      | SMOTE variants or cleaning methods        |
| Extreme imbalance with rare events             | Threshold tuning + PR-AUC focus           |

---

## 9. Real-Life Case Studies

### Fraud Detection

- Fraud is extremely rare (0.1%)
- Model trained only for accuracy may predict "not fraud" for almost everything
- **Solution:** class weighting + boosting + threshold tuning
- Tune threshold to catch more fraud while keeping false alarms manageable

### Medical Diagnosis

- Missing disease is costly → high recall priority
- Doctors accept more false positives for follow-up testing
- Model is decision aid, not final authority

### Manufacturing Defect Detection

- Defective products are rare
- Manual inspection is expensive
- **Solution:** anomaly detection + class weighting + cost-sensitive thresholds

### Customer Churn

- Only small percentage leave
- High recall: catch more churners if campaigns are cheap
- High precision: focus on likely churners if campaigns are expensive

---

## 10. Updated Best Practices for June 2026

✅ Do not rely on accuracy as the main metric  
✅ Use stratified splits and leakage-safe pipelines  
✅ Prefer class weights or boosted tree models before complex sampling  
✅ Use SMOTE carefully and only on training data  
✅ Consider PR-AUC more often than ROC-AUC for extreme imbalance  
✅ Tune probability threshold instead of accepting 0.5 default  
✅ For tabular data, gradient boosting often beats deep learning  
✅ For very rare events, anomaly detection or semi-supervised approaches may be better  
✅ Monitor class drift in production—imbalance can change over time

---

## 11. Mental Model

Think of imbalance like a classroom where 99 students always answer the same way and 1 student has a different but important answer. If you only listen to the majority, you miss the valuable exception. Good imbalanced learning is about making sure the rare but important cases still influence the model enough.

---

## Final Takeaway

Imbalanced learning is not about making classes perfectly equal. It's about training a model that respects the true **cost of mistakes**. The best results come from combining:

- The right metric (precision, recall, F1, PR-AUC)
- A leakage-safe data pipeline
- A class-aware training strategy (class weights, boosting, SMOTE)
- Threshold tuning based on business cost
