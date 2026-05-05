# Text Preprocessing Methods and Options

## NLTK (Natural Language Toolkit)
The standard library for basic NLP in Python.

### `nltk.tokenize.word_tokenize(text)`
- **Purpose:** Splits text into words and punctuation tokens.
- **Syntax:** `tokens = word_tokenize("Hello, world!")`
- **Return Type:** List of strings.
- **Workflow:** Usually the first step after lowercasing.

### `nltk.corpus.stopwords.words('english')`
- **Purpose:** Returns a list of common English stopwords.
- **Return Type:** List of strings.
- **Common Usage:** Used in a list comprehension to filter tokens: `[w for w in tokens if w not in stop_words]`

### `nltk.stem.PorterStemmer`
- **Purpose:** Aggressively chops off word endings.
- **Syntax:** `stemmer = PorterStemmer(); stemmer.stem("running")`
- **Return Type:** String.

### `nltk.stem.WordNetLemmatizer`
- **Purpose:** Linguistically maps words to their dictionary root. Requires POS tags for accuracy.
- **Syntax:** `lemmatizer = WordNetLemmatizer(); lemmatizer.lemmatize("better", pos="a")`
- **Return Type:** String.

## SpaCy
A modern, fast, production-ready NLP library.

### `spacy.load('en_core_web_sm')`
- **Purpose:** Loads a pre-trained language model.
- **Syntax:** `nlp = spacy.load("en_core_web_sm"); doc = nlp("This is a sentence.")`
- **Workflow:** SpaCy processes the entire document at once, automatically doing tokenization, POS tagging, and lemmatization.
- **Attributes on Tokens:**
  - `token.text`: The raw text.
  - `token.is_stop`: Boolean indicating if it's a stopword.
  - `token.is_punct`: Boolean indicating if it's punctuation.
  - `token.lemma_`: The lemmatized base form of the word.

## Common Mistakes & Best Practices
- **Best Practice:** When doing simple tasks like TF-IDF, NLTK stemming is often enough. For complex tasks needing grammatical context, use SpaCy lemmatization.
- **Mistake:** Forgetting to download NLTK data (`nltk.download('punkt')`) before running scripts.
