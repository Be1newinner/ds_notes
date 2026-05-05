# Most Used Algorithms

## Top Supervised Algorithms

| Algorithm                                       | Classification / Regression | Strengths & Benefits                                               | Real-World Use Cases                                    |
| ----------------------------------------------- | --------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------- |
| Linear Regression                               | Regression                  | Simple, interpretable, fast on linear problems                     | Sales forecasting, price prediction                     |
| Logistic Regression                             | Classification              | Probabilistic, interpretable, efficient for binary labels          | Credit scoring, medical diagnosis                       |
| Decision Trees                                  | Both                        | Easy to interpret, handles mixed data types, no scaling            | Customer segmentation, churn prediction                 |
| Random Forest                                   | Both                        | Robust, reduces overfitting, handles large feature sets            | Fraud detection, recommendation systems                 |
| Support Vector Machines (SVM)                   | Both                        | Effective in high-dimensional spaces, works well with clear margin | Text classification, image recognition                  |
| Naive Bayes                                     | Classification              | Fast, works well with small data, good for text data               | Spam filtering, sentiment analysis                      |
| K-Nearest Neighbors (KNN)                       | Both                        | Simple, no training phase, good for small datasets                 | Handwriting recognition, anomaly detection              |
| Gradient Boosting (XGBoost, LightGBM, CatBoost) | Both                        | High predictive power, handles missing data well                   | Customer lifetime value, credit risk                    |
| Neural Networks                                 | Both                        | Captures complex relationships, adaptable to many data types       | Image and speech recognition, complex pattern detection |

### Classification ALgorithms Mind Map:

```
Start:
│
├── Is the dataset small AND features are mostly independent?
│       ├── Yes: Use Naive Bayes
│       │       (Benefit: Fast, works well with small data and multi-class tasks)
│       │       (Loss: Assumes feature independence, struggles with correlated features)
│       └── No: Lookup Next
│
├── Is interpretability critical AND problem is binary classification?
│       ├── Yes: Use Logistic Regression
│       │       (Benefit: Simple, interpretable, efficient for binary tasks)
│       │       (Loss: Cannot model complex non-linear boundaries)
│       └── No: Lookup Next
│
├── Is data high-dimensional with clear class margins?
│       ├── Yes: Use Support Vector Machine (SVM)
│       │       (Benefit: Effective for high-dimensional data, flexible kernels)
│       │       (Loss: Poor scaling on large data, sensitive to noise)
│       └── No: Lookup Next
│
├── Does data contain mixed types OR is interpretability important?
│       ├── Yes: Use Decision Tree
│       │       (Benefit: Easy to interpret, handles mixed data, no scaling required)
│       │       (Loss: Prone to overfitting, unstable with small data changes)
│       └── No: Lookup Next
│
├── Is dataset large, complex, and risk of overfitting high?
│       ├── Yes: Use Random Forest
│       │       (Benefit: Robust ensemble, reduces overfitting, handles noisy data)
│       │       (Loss: Less interpretable, slower on very large datasets)
│       └── No: Lookup Next
│
├── Is highest accuracy needed AND computational resources are sufficient?
│       ├── Yes: Use Gradient Boosting (XGBoost, LightGBM, CatBoost)
│       │       (Benefit: High predictive power, handles missing data well)
│       │       (Loss: Complex tuning, longer training time)
│       └── No: Lookup Next
│
├── Is data complex with non-linear patterns AND large labeled dataset available?
│       ├── Yes: Use Neural Networks
│       │       (Benefit: Captures complex patterns, adaptable)
│       │       (Loss: Requires large data, less interpretable, computationally intensive)
│       └── No: Lookup Next
│
├── Is simple, instance-based method desired AND dataset is small?
│       ├── Yes: Use K-Nearest Neighbors (KNN)
│       │       (Benefit: Simple, no training phase, good for small data)
│       │       (Loss: Slow prediction on large data, sensitive to irrelevant features)
│       └── No: Lookup Next
│
└── Default: Use Random Forest (robust baseline for most classification problems)
```

#### Example Problems

1. Naive Bayes (Small, independent features):

   - Example: Email spam filtering, where words appear independently and the model quickly classifies emails as spam or not based on word frequencies.[1][2]

2. Logistic Regression (Interpretability and binary classification):

   - Example: Credit scoring for loan approvals, where the model predicts the binary outcome of default or non-default and decision makers need interpretable results.[9][10]

3. Support Vector Machine (High-dimensional data with clear margins):

   - Example: Text classification such as identifying topics in news articles using word vector spaces with many features but clear separation.[11][12]

4. Decision Tree (Mixed types and interpretability):

   - Example: Customer churn prediction in telecom, where numerical and categorical variables need to be handled and the model decision paths are easy to explain.[13][14]

5. Random Forest (Large complex datasets, overfitting risk):

   - Example: Fraud detection in banking transactional data, which is large, noisy, and complex and benefits from ensemble robustness.[15][16]

6. Gradient Boosting (Highest accuracy and compute available):

   - Example: Customer lifetime value prediction in marketing analytics, requiring highly accurate predictions with rich feature sets.[17][18]

7. Neural Networks (Complex non-linear patterns, large labeled data):

   - Example: Image recognition for medical diagnosis like classifying X-rays or MRIs to detect diseases.[19][11]

8. K-Nearest Neighbors (Simple, small dataset):
   - Example: Handwritten digit recognition in small labelled datasets where instance-based similarity works well.[12][13]

[1](https://www.pickl.ai/blog/naive-bayes-types-examples/)
[2](https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/)
[3](https://keylabs.ai/blog/naive-bayes-classifiers-types-and-use-cases/)
[4](https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python)
[5](https://scikit-learn.org/stable/modules/naive_bayes.html)
[6](https://ijarsct.co.in/Paper2028.pdf)
[7](https://www.datasciencewizards.ai/getting-started-with-machine-learning-algorithms-naive-bayes/)
[8](https://royalresearch.in/naive-bayes-classifier-fast-simple-and-surprisingly-powerful)
[9](https://www.tutorialspoint.com/pros-and-cons-of-different-classification-models)
[10](https://www.omdena.com/blog/machine-learning-classification-algorithms)
[11](https://www.coursera.org/in/articles/machine-learning-algorithms)
[12](https://builtin.com/data-science/supervised-machine-learning-classification)
[13](https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/)
[14](https://builtin.com/data-science/random-forest-algorithm)
[15](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/)
[16](https://shelf.io/blog/random-forests-in-machine-learning/)
[17](https://elitedatascience.com/machine-learning-algorithms)
[18](https://blog.quantinsti.com/machine-learning-classification/)
[19](https://www.datacamp.com/blog/classification-machine-learning)

### Regression Algorithms Mind Map

```
Start:
│
├── Is the dataset small AND relationship likely linear?
│       ├── Yes: Use Linear Regression
│       │       (Benefit: Simple, interpretable, fast training)
│       │       (Loss: Cannot model non-linear relationships)
│       └── No: Lookup Next
│
├── Is robustness to outliers and interpretability important?
│       ├── Yes: Use Ridge or Lasso Regression (Regularized Linear Models)
│       │       (Benefit: Handles multicollinearity, feature selection)
│       │       (Loss: Still linear, tuning required)
│       └── No: Lookup Next
│
├── Is data high-dimensional with complex but smooth trends?
│       ├── Yes: Use Support Vector Regression (SVR)
│       │       (Benefit: Good for high-dimensional, non-linear regression)
│       │       (Loss: Scaling issues on very large data, tuning complexity)
│       └── No: Lookup Next
│
├── Does data contain mixed types OR require interpretability?
│       ├── Yes: Use Decision Tree Regression
│       │       (Benefit: Handles non-linearity, interpretable, no scaling needed)
│       │       (Loss: Prone to overfitting, unstable)
│       └── No: Lookup Next
│
├── Is dataset large, complex, and prone to overfitting?
│       ├── Yes: Use Random Forest Regression
│       │       (Benefit: Robust, reduces overfitting, handles noisy data)
│       │       (Loss: Less interpretable, slower on huge data)
│       └── No: Lookup Next
│
├── Is highest accuracy needed and compute resources available?
│       ├── Yes: Use Gradient Boosting Regression (XGBoost, LightGBM, CatBoost)
│       │       (Benefit: State-of-the-art accuracy, handles missing data)
│       │       (Loss: Complex tuning, longer training time)
│       └── No: Lookup Next
│
├── Is non-linear and complex pattern detection required with lots of data?
│       ├── Yes: Use Neural Network Regression
│       │       (Benefit: Captures complex patterns, adaptable)
│       │       (Loss: Needs large data, less interpretable, compute heavy)
│       └── No: Lookup Next
│
├── Is a simple, instance-based method desired on a small dataset?
│       ├── Yes: Use K-Nearest Neighbors Regression (KNN)
│       │       (Benefit: Simple, no training needed, good for small data)
│       │       (Loss: Slow on large data, sensitive to irrelevant features)
│       └── No: Lookup Next
│
└── Default: Use Random Forest Regression (robust, general-purpose)
```

#### Example Problems

1. Linear Regression

   - Example: Predicting house prices based on square footage.
   - Benefit: Simple, interpretable model ideal for linear relationships; fast training and prediction.
   - Loss: More complex models (Random Forest, Neural Networks) unnecessarily add complexity and overfit with small, linear datasets.
   - Why: Data shows a linear pattern (price rises with size); small dataset suitable for simple models; interpretability is valuable for stakeholders wanting to understand price drivers.

2. Ridge or Lasso Regression

   - Example: Predicting blood pressure based on various correlated medical indicators.
   - Benefit: Handles multicollinearity between features; Lasso performs feature selection automatically.
   - Loss: Linear regression can't properly handle feature correlation; more complex models harder to interpret.
   - Why: Dataset has correlated features; interpretability and robustness to multicollinearity are key.

3. Support Vector Regression (SVR)

   - Example: Stock price prediction using many technical indicators.
   - Benefit: Effective in modeling complex, nonlinear relationships in high-dimensional spaces.
   - Loss: Linear models too simplistic; ensemble models require more compute, and simpler models may underfit.
   - Why: Data dimension is high with nonlinear trends; SVR balances expressiveness and generalization.

4. Decision Tree Regression

   - Example: Predicting loan default amounts where input data include both categorical and numeric features.
   - Benefit: Handles mixed data types, intuitive and interpretable decision rules, no need for feature scaling.
   - Loss: Linear and kernel methods need numeric data; neural nets lack interpretability and need more data.
   - Why: Dataset contains mixed feature types; interpretability needed to explain decisions.

5. Random Forest Regression

   - Example: Real estate price prediction on large, noisy, heterogeneous datasets.
   - Benefit: Ensemble technique reduces overfitting; robust and handles noisy data effectively.
   - Loss: Single decision trees overfit; linear models underfit; models like GBM can be slower to train.
   - Why: Large, complex dataset prone to overfit; RF offers a balance of accuracy and robustness.

6. Gradient Boosting Regression

   - Example: Precise customer lifetime value prediction in marketing with rich feature sets.
   - Benefit: High accuracy, handles missing data, adaptable to diverse data patterns.
   - Loss: Requires careful tuning; slower training compared to simpler models.
   - Why: Accuracy crucial; compute resources available; willing to manage increased complexity.

7. Neural Network Regression

   - Example: Forecasting energy consumption patterns from sensor networks with complex temporal dynamics.
   - Benefit: Captures highly nonlinear, complex patterns in large datasets.
   - Loss: Needs large data and compute; low interpretability.
   - Why: Complex data patterns; enough data and compute resources justify model complexity.

8. K-Nearest Neighbors Regression (KNN)
   - Example: Local housing price estimation based on sale prices of nearby houses.
   - Benefit: Simple, no training; good for small dataset with local similarity assumptions.
   - Loss: Inefficient on large datasets; sensitive to irrelevant/noisy features.
   - Why: Small, localized data; spatial similarity assumption holds; need for model simplicity.

This clear format links each regression algorithm to real-world use cases, highlights its key benefit, explains drawbacks of alternatives, and explicitly rationalizes the algorithm choice based on data and problem characteristics. It guides practical, informed algorithm selection decisions.[4][8][9]

[1](https://www.iieta.org/download/file/fid/85503)
[2](https://www.sciencedirect.com/science/article/pii/S2666546825001739)
[3](https://my.eng.utah.edu/~pingfant/Project%20Report.pdf)
[4](https://www.onlinemanipal.com/blogs/popular-regression-algorithms-in-machine-learning)
[5](https://pmc.ncbi.nlm.nih.gov/articles/PMC7472084/)
[6](https://f0nzie.github.io/machine_learning_compilation/comparison-of-six-linear-regression-algorithms.html)
[7](https://www.dataschool.io/comparing-supervised-learning-algorithms/)
[8](https://towardsdatascience.com/7-of-the-most-commonly-used-regression-algorithms-and-how-to-choose-the-right-one-fc3c8890f9e3/)
[9](https://www.geeksforgeeks.org/machine-learning/linear-regression-real-life-examples/)
