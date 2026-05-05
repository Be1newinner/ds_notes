# Interview Questions: Imbalanced Classification

## Beginner Questions
1. **What does it mean if a dataset is "imbalanced"?**
   *Hint:* It means the target variable has a very uneven distribution of classes (e.g., 99% Class 0, 1% Class 1).
2. **Why shouldn't you use Accuracy to evaluate an imbalanced model?**
   *Hint:* Because a model can achieve very high accuracy simply by predicting the majority class every single time, while completely failing to identify the minority class.
3. **What is SMOTE?**
   *Hint:* Synthetic Minority Over-sampling Technique. It is an algorithm that creates synthetic, artificial rows of data for the minority class to balance the dataset.

## Conceptual Questions
4. **Explain the difference between Precision and Recall.**
   *Hint:* Precision asks: "Of all the times the model cried 'Wolf!', how many times was there actually a wolf?" Recall asks: "Out of all the actual wolves out there, how many did the model find?"
5. **How does the `class_weight='balanced'` parameter actually work under the hood?**
   *Hint:* It modifies the loss function of the algorithm. Normally, getting a Class 0 prediction wrong and a Class 1 prediction wrong incur the same penalty. `class_weight='balanced'` applies a massive mathematical penalty if the model gets the rare minority class wrong, forcing the model to care about it.
6. **If you have a strict budget and investigating false alarms costs you $10,000 each, do you want to optimize for Precision or Recall?**
   *Hint:* Precision. You want to make absolutely sure that when your model flags something, it is actually correct, to avoid wasting money on false alarms.

## Practical Questions
7. **Explain the "Golden Rule" of applying SMOTE.**
   *Hint:* You must NEVER apply SMOTE to the validation or test sets. You split your data *first*, apply SMOTE *only* to the training data, and then evaluate on the pristine, untouched test data.
8. **What is Data Leakage in the context of SMOTE and Cross-Validation?**
   *Hint:* If you apply SMOTE to the entire dataset *before* running K-Fold Cross Validation, the synthetic data generated using information from the test folds will leak into the training folds, resulting in massively inflated and fake evaluation scores.
9. **Instead of algorithmic weights or SMOTE, how can you solve an imbalance problem using just the `predict_proba()` method?**
   *Hint:* By default, models predict Class 1 if the probability is > 0.5. You can manually lower this decision threshold (e.g., to 0.1) so the model becomes much more aggressive in predicting the minority class.

## Output Interpretation
10. **Your model has a Precision of 0.99 and a Recall of 0.05. Is this a good model for detecting terminal cancer?**
    *Hint:* No, it is terrible. It means that when the model says you have cancer, it is almost certainly right (High Precision). However, it misses 95% of the actual cancer cases (Low Recall). For medical diagnoses, you generally want High Recall to ensure no sick patients slip through the cracks.
