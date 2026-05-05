# Interview Questions: Classical NLP

## Beginner Level
1. How does a machine learning model like Logistic Regression handle text data?
2. What is the difference between Supervised NLP (like Sentiment Analysis) and Unsupervised NLP (like Topic Modeling)?

## Intermediate Level
3. Walk me through the exact pipeline of steps required to build a spam classifier from raw text to prediction.
4. Why is Naive Bayes such a popular baseline model for text classification tasks?
5. What is data leakage in NLP, and how does using `fit_transform` on the entire dataset before doing `train_test_split` cause it?

## Advanced / Practical Level
6. You built a sentiment classifier using TF-IDF and Logistic Regression. Your accuracy on the test set is 95%. However, when deployed, it classifies the phrase "Not bad at all, I actually loved it" as NEGATIVE. Why did the model likely make this mistake, and how could you fix the vectorizer to potentially solve it? 
   *(Answer: It likely focused on "Not" and "bad" independently. Fixing it involves changing `ngram_range=(1,2)` to capture the bigram "Not bad").*
7. How would you determine the optimal number of topics (`n_components`) when using LDA for topic modeling?
