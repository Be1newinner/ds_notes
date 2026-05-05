# Anomaly Detection Interview Questions

1. **Beginner**: What is the difference between an inlier and an outlier in scikit-learn's terminology?
2. **Conceptual**: How does Isolation Forest conceptually isolate an anomaly faster than a normal point?
3. **Practical**: You don't know the exact percentage of fraud in your dataset. How do you set the `contamination` parameter?
4. **Comparison**: Why might you use Isolation Forest instead of simply looking at Z-scores (standard deviations from the mean)? (Hint: Z-scores only look at one feature at a time; IsoForest looks at multi-dimensional relationships).
5. **Output**: If `model.predict()` returns an array containing `[1, 1, -1, 1]`, what does the `-1` represent?
