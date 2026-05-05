# Interview Questions: Support Vector Machines

## Beginner Questions
1. **Explain the goal of a Support Vector Machine in your own words.**
   *Hint:* The goal is to find a line (hyperplane) that separates different classes while maximizing the margin (the gap) between the line and the closest data points from each class.
2. **What are Support Vectors?**
   *Hint:* They are the data points closest to the hyperplane. They are the only points that dictate where the boundary is drawn.
3. **What is the "Kernel Trick"?**
   *Hint:* It's a mathematical technique that allows SVMs to classify non-linearly separable data by mapping it into a higher-dimensional space where it *is* linearly separable.

## Conceptual Questions
4. **Explain the role of the `C` parameter.**
   *Hint:* `C` is the regularization parameter. A high `C` creates a "hard margin" (penalizes misclassifications heavily, leading to a smaller margin and potential overfitting). A low `C` creates a "soft margin" (allows some misclassifications for the sake of a wider, more generalizable margin).
5. **How does the RBF kernel work intuitively?**
   *Hint:* Radial Basis Function measures the similarity between two points based on how close they are to each other, acting like a bell curve centered around each point. It essentially creates "peaks" around points of one class, allowing the SVM to draw circles/contours around them.
6. **Explain the role of the `gamma` parameter when using an RBF kernel.**
   *Hint:* `gamma` controls the "reach" of a single training example. High gamma = short reach (bumpy boundary, high variance/overfitting). Low gamma = far reach (smooth boundary, high bias/underfitting).

## Practical Questions
7. **If you have a dataset with 5 million rows, would you use `SVC(kernel='rbf')`?**
   *Hint:* No. Training time for non-linear SVMs scales terribly (between $O(n^2)$ and $O(n^3)$). It would take forever.
8. **Why must you scale your data before feeding it to an SVM?**
   *Hint:* SVMs maximize the physical distance between data points. If one feature is measured in thousands and another in decimals, the algorithm will completely ignore the decimal feature.
9. **How do you get probability estimates from an SVM?**
   *Hint:* You must pass `probability=True` when initializing the model. However, this slows down training significantly because it runs an internal 5-fold cross-validation (Platt scaling) to calibrate the probabilities.

## Output Interpretation
10. **You train an SVM with `C=1000` and `gamma=100`. Your training accuracy is 100%, but your validation accuracy is 50%. What happened?**
    *Hint:* You heavily overfit the data. `C=1000` forces a hard margin, and `gamma=100` makes the boundary hyper-sensitive to every single point. You should lower both values.
