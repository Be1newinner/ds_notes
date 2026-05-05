# Interview Questions: Text Preprocessing

## Beginner Level
1. What are stopwords, and why do we usually remove them in NLP?
2. Can you explain tokenization?

## Intermediate Level
3. What is the difference between stemming and lemmatization? When would you choose one over the other?
4. How would you handle punctuation in text data? Are there cases where you would keep it? (Hint: Emoticons like ":)" or exclamation marks in sentiment analysis).

## Advanced / Practical Level
5. You are building an NLP model to classify legal documents. Would you use stemming or lemmatization? Why?
6. If you notice your model is misclassifying sentences that contain negative sentiment (e.g., "The product was not good"), what preprocessing step might be causing the issue, and how would you fix it?

## Output Interpretation
7. If a lemmatizer turns "leaves" into "leaf", but you wanted it to mean "departing" (as in "he leaves the room"), what missing information does the lemmatizer need to get it right? (Answer: Part of Speech tag).
