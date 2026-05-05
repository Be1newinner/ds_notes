# Methods and Options: Word Embeddings

## `gensim.models.Word2Vec`
The most popular library for training your own embeddings from scratch.

### Syntax
```python
from gensim.models import Word2Vec
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1)
```

### Parameters
- `sentences`: A list of lists of tokens (e.g., `[["hello", "world"], ["i", "love", "nlp"]]`).
- `vector_size`: Dimensionality of the word vectors (typically 100-300).
- `window`: Maximum distance between the current and predicted word within a sentence.
- `min_count`: Ignores all words with total frequency lower than this.

### Common Methods
- `model.wv.most_similar("word")`: Returns a list of words most similar to the given word.
- `model.wv.similarity("word1", "word2")`: Computes the cosine similarity between two words.
- `model.wv["word"]`: Retrieves the actual numpy array (the vector) for the word.

## `spacy` (Pre-trained Models)
The easiest way to use embeddings trained by Google or Facebook on billions of words.

### Setup
You must download a medium (`md`) or large (`lg`) model. The small (`sm`) model does NOT include real word vectors to save space.
`python -m spacy download en_core_web_md`

### Syntax and Workflow
```python
import spacy
nlp = spacy.load("en_core_web_md")
doc = nlp("I love apples")

# Get vector for a specific word
apple_vector = doc[2].vector 

# Get similarity between two documents
doc1 = nlp("I like dogs")
doc2 = nlp("I prefer puppies")
print(doc1.similarity(doc2)) 
```

### Why SpaCy is great
It automatically averages the word vectors in a sentence to give you a single `doc.vector` representing the whole sentence's meaning.
