# Interview Questions: Classification Metrics

## Beginner Questions
1. **If your classification model has 99% accuracy, does that mean it's ready for production?**
   - *Answer*: No. If the dataset is highly imbalanced (e.g., 99% of transactions are legitimate and 1% are fraud), a model that simply guesses "Legitimate" every single time will have 99% accuracy but will fail to catch any fraud. You must check Precision and Recall.
2. **What is the difference between a False Positive and a False Negative?**
   - *Answer*: A False Positive is a "False Alarm" (the model predicted the event would happen, but it didn't). A False Negative is a "Miss" (the model predicted the event would *not* happen, but it actually did).

## Conceptual Questions
3. **Explain Precision vs. Recall in simple terms.**
   - *Answer*: Precision is Quality: "When the model fires, how often is it right?" Recall is Quantity: "Out of all the targets that exist in reality, how many did the model manage to catch?"
4. **Why do we use the F1-Score instead of just taking the simple average of Precision and Recall?**
   - *Answer*: The F1-score uses the *harmonic mean*. The simple average is too forgiving. If a model has 100% Recall and 0% Precision, the simple average is 50%. The harmonic mean heavily penalizes extreme differences, dragging the F1-score down to near 0, which accurately reflects that the model is broken.
5. **What does the ROC curve represent, and what does the AUC number mean?**
   - *Answer*: The ROC curve plots the True Positive Rate against the False Positive Rate at every possible probability threshold. The AUC (Area Under Curve) is a single number summarizing that graph. An AUC of 1.0 means the model perfectly separates the classes. An AUC of 0.5 means the model is no better than flipping a coin.

## Practical / Scenario Questions
6. **You are building a spam filter. Do you tune the model for higher Precision or higher Recall?**
   - *Answer*: Higher Precision. In a spam filter, a False Positive means sending an important real email (like a job offer) to the spam folder, which is terrible. A False Negative means letting a spam email into the inbox, which is merely a mild annoyance.
7. **You are building an AI to detect defective parts in a car engine. Do you optimize for Precision or Recall?**
   - *Answer*: Higher Recall. A False Negative means letting a defective, dangerous engine part go into a car, which could cause a fatal crash. A False Positive means throwing away a perfectly good part, which just costs the company a little bit of money. Safety demands Recall.
