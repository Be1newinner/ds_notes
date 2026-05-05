# Interview Questions: KNN and Naive Bayes

## Beginner Questions
1. **Explain K-Nearest Neighbors in one sentence.**
   *Hint:* It classifies a new data point by looking at the majority class among its $K$ closest neighbors in the training data.
2. **Does KNN require a training step?**
   *Hint:* No, it is a "lazy learner." The "training" phase is essentially just memorizing the dataset. All computation happens at prediction time.
3. **What is the difference between Gaussian and Multinomial Naive Bayes?**
   *Hint:* Gaussian is for continuous numeric data (assumes a normal distribution). Multinomial is for discrete counts (like word frequencies in text).

## Conceptual Questions
4. **How do you choose the right value for $K$ in KNN? What happens if $K$ is too small or too large?**
   *Hint:* You choose $K$ using cross-validation. If $K$ is too small (e.g., $K=1$), the model captures noise and overfits. If $K$ is too large (e.g., $K=N$), the model underfits and simply predicts the majority class of the entire dataset.
5. **Why is feature scaling absolutely critical for KNN?**
   *Hint:* KNN relies on calculating physical distances (like Euclidean distance). If one feature is measured in millions (e.g., salary) and another in single digits (e.g., years of experience), the distance metric will completely ignore the smaller feature.
6. **Explain the "Naive" assumption in Naive Bayes.**
   *Hint:* It assumes that all features are conditionally independent of each other given the class label. This is rarely true in reality, but the algorithm still performs well.

## Practical Questions
7. **In a Natural Language Processing task (like spam classification), a word appears in the test set that was never seen in the training set. How does Naive Bayes handle this?**
   *Hint:* Without intervention, the probability of that word would be 0, causing the entire posterior probability to multiply to 0. We use Laplace Smoothing (the `alpha` parameter) to assign a tiny non-zero probability to unseen words.
8. **If you have a dataset with 10,000 features, would you prefer KNN or Naive Bayes? Why?**
   *Hint:* Naive Bayes. KNN suffers terribly from the "curse of dimensionality" because in high-dimensional space, the concept of "distance" breaks down (all points become roughly equidistant). Naive Bayes handles high-dimensional sparse data (like text) exceptionally well.

## Output Interpretation
9. **You run a Naive Bayes model and look at `predict_proba`. The probabilities are very extreme (e.g., 0.99999 and 0.00001). Can you trust these numbers as true confidence intervals?**
   *Hint:* Usually not. Naive Bayes is known for being a "bad estimator" of actual probabilities because its independence assumption pushes probabilities toward the extremes. You can trust the rank ordering (the class with 0.999 is definitely more likely than the one with 0.001), but not the absolute probability value.
