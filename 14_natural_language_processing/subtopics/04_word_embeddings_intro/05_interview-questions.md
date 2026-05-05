# Interview Questions: Word Embeddings

## Beginner Level
1. Can you explain the basic idea behind Word Embeddings to someone without a technical background?
2. What is Cosine Similarity, and why is it used so often with word vectors?

## Intermediate Level
3. How does the Word2Vec algorithm actually learn the representations? What is the neural network trying to predict? *(Answer: It's a fake task! It tries to predict a word given its neighbors (CBOW) or predict neighbors given a word (Skip-Gram). The "hidden weights" learned become the embeddings).*
4. What does the famous equation `King - Man + Woman = Queen` actually mean in the context of vector mathematics?
5. How would you represent an entire sentence using word embeddings so that you can feed it into a standard classifier like Logistic Regression? *(Answer: The simplest way is to take the mean average of all the word vectors in the sentence).*

## Advanced / Practical Level
6. What is the "Out Of Vocabulary" (OOV) problem in Word2Vec, and how do subword models like FastText solve it?
7. Word2Vec produces "static" embeddings. The word "apple" has the same vector whether the sentence is "I ate an apple" or "I bought Apple stock". How do modern architectures like Transformers (BERT) address this limitation? *(Answer: They produce contextualized embeddings, where the vector for "apple" changes dynamically based on the surrounding words in the exact sentence).*
