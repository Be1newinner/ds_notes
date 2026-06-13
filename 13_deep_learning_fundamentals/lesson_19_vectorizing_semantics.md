# Lesson 19: Vectorizing Semantics

## Introduction & The "Why"

In classical Natural Language Processing (NLP), computers treated words as arbitrary symbols. Methods like **Bag-of-Words (BoW)** or **One-Hot Encoding** represent text by matching exact strings. In this traditional view, the words "cat" and "kitten" are treated as completely independent, perpendicular dimensions in a giant grid. If a model learns that "cat" is a positive word in a review, it cannot automatically transfer that knowledge to the word "kitten" because their mathematical representations share zero overlap.

To bridge this gap, deep learning represents language using **Word Embeddings**. Instead of treating words as sparse, disconnected coordinates, we map them into a dense, continuous **Semantic Vector Space**. In this space, words with similar meanings are located near one another. The geometric distance and direction between word vectors represent real semantic relationships (like gender, tense, or location). 

This lesson covers how we build these semantic spaces, how raw text is broken down into numerical subword tokens, and the mechanics of the learnable Embedding layer that serves as the gateway to neural language processing.

---

## Topic 1: Moving Beyond Bag-of-Words: The Semantic Vector Space

### Rationale and Mechanics
In a one-hot representation, a vocabulary of size $V$ is mapped to a sparse vector space $\mathbb{R}^V$. If your vocabulary has $50,000$ words, each word is represented by a vector of length $50,000$ containing a single $1.0$ at its unique index and $49,999$ zeros.

If we calculate the similarity of any two distinct words represented by one-hot vectors $\mathbf{v}_i$ and $\mathbf{v}_j$ using the dot product, the result is always zero:
$$\mathbf{v}_i^T \mathbf{v}_j = 0 \quad (\text{for } i \neq j)$$

This tells the model that all words are completely unrelated. In contrast, deep learning represents words using dense continuous vectors $\mathbf{e} \in \mathbb{R}^d$, where the embedding dimension $d$ is much smaller than the vocabulary size $V$ (typically $d \in [50, 768]$).

In this dense space, semantic similarity is measured using **Cosine Similarity**, which computes the cosine of the angle $\theta$ between two vectors:
$$\text{CosSim}(\mathbf{e}_1, \mathbf{e}_2) = \cos(\theta) = \frac{\mathbf{e}_1^T \mathbf{e}_2}{\|\mathbf{e}_1\| \|\mathbf{e}_2\|}$$

*   **Cosine Similarity = $1.0$:** The vectors point in the exact same direction ($\theta = 0^\circ$), representing maximum similarity.
*   **Cosine Similarity = $0.0$:** The vectors are orthogonal ($\theta = 90^\circ$), indicating no semantic relationship.
*   **Cosine Similarity = $-1.0$:** The vectors point in opposite directions ($\theta = 180^\circ$), representing polar opposites or antonyms.

```
                         Semantic Vector Space Geometry
                         
                                       y
                                       ^      / e_king (0.8, 0.6)
                                       |    / 
                                       |  /   (Small Angle = High CosSim)
                                       |/  _  e_prince (0.75, 0.65)
                                       +---------> x
```

Because of this continuous geometry, the vectors exhibit **Linear Vector Arithmetic**. This means relationships are encoded as translation vectors. For example, the vector representing the difference between "man" and "woman" should be nearly identical to the vector representing the difference between "king" and "queen":
$$\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}} \approx \mathbf{e}_{\text{queen}}$$

### Python Code Implementation
The following code demonstrates how to calculate Cosine Similarity and perform Vector Analogy Arithmetic using NumPy. We set up a toy semantic vector space and search for the closest word in our vocabulary to a computed analogy vector.

```python
import numpy as np

# 1. Define a toy vocabulary and their dense embeddings (dim=3)
# Dimensions represent latent concepts: [Royal/Status, Masculinity, Youthfulness]
word_embeddings = {
    "man":    np.array([0.05,  0.90,  0.10]),
    "woman":  np.array([0.05, -0.90,  0.10]),
    "king":   np.array([0.95,  0.85,  0.05]),
    "queen":  np.array([0.95, -0.85,  0.05]),
    "boy":    np.array([0.01,  0.80,  0.85]),
    "girl":   np.array([0.01, -0.80,  0.85]),
    "apple":  np.array([-0.80, 0.01,  0.20])  # Unrelated fruit concept
}

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# 2. Test Cosine Similarity
similarity_cat_kitten = cosine_similarity(word_embeddings["king"], word_embeddings["queen"])
similarity_cat_apple = cosine_similarity(word_embeddings["king"], word_embeddings["apple"])

print(f"Cosine Similarity (king, queen): {similarity_cat_kitten:.4f}")
print(f"Cosine Similarity (king, apple): {similarity_cat_apple:.4f}")

# 3. Vector Arithmetic: king - man + woman = ?
target_vector = word_embeddings["king"] - word_embeddings["man"] + word_embeddings["woman"]
print(f"\nCalculated target vector (king - man + woman): {target_vector}")

# Find the nearest word in our vocabulary using Cosine Similarity
best_word = None
max_sim = -2.0

print("\nCandidate word similarities to target vector:")
for word, emb in word_embeddings.items():
    sim = cosine_similarity(target_vector, emb)
    print(f" - {word}: {sim:.4f}")
    if sim > max_sim:
        max_sim = sim
        best_word = word

print(f"\nClosest word to target vector is: '{best_word}' with similarity {max_sim:.4f}")
```

### Trade-offs
*   **Advantages:** Shrinks input feature dimensions from sparse $V$-dimensional arrays to dense $d$-dimensional representations. This enables generalization; models trained on "excellent review" can immediately generalize to "superb review" because their embedding vectors lie close together.
*   **Disadvantages:** Explanatory opacity. In a count-based model (like TF-IDF), coordinate $14$ might directly represent the exact frequency of "inflation." In a dense embedding, the coordinates represent abstract, latent mixture dimensions that are difficult to interpret without projection algorithms like t-SNE or PCA.

### Real-World Applications (Rule of 4)
1.  **Example 1: Evaluating Synonym Proximity**
    *   **Input/Scenario:** A search query checks the similarity between "buy" ($\mathbf{e}_1 = [0.2, 0.8]$) and "purchase" ($\mathbf{e}_2 = [0.22, 0.78]$).
    *   **Expected Output:** Cosine similarity yields $\approx 0.999$, allowing a shopping system to return matching products regardless of the specific search term used.
2.  **Example 2: Arithmetic Analogy Mapping**
    *   **Input/Scenario:** A vector search engine processes $\mathbf{e}_{\text{madrid}} - \mathbf{e}_{\text{spain}} + \mathbf{e}_{\text{france}}$ to identify capital-country relationships.
    *   **Expected Output:** The resulting vector points closest to $\mathbf{e}_{\text{paris}}$ in the database.
3.  **Example 3: Cross-Lingual Vector Alignments**
    *   **Input/Scenario:** A translation model projects English words ($\mathbf{e}_{\text{cat}}$) and Spanish words ($\mathbf{e}_{\text{gato}}$) into a shared multilingual vector space.
    *   **Expected Output:** The distance between $\mathbf{e}_{\text{cat}}$ and $\mathbf{e}_{\text{gato}}$ is minimized, letting the network translate text by mapping words to nearest neighbors in the shared space.
4.  **Example 4: Preventing Cold-Start Failure in Recommenders**
    *   **Input/Scenario:** A user likes video games related to "cyberpunk." A new game comes out containing tags that map close to "cyberpunk" in a dense embedding space.
    *   **Expected Output:** The system recommends the new game immediately, bypassing the need to wait for historical user reviews.

> **Metacognitive Checkpoint:** If two word vectors have a cosine similarity of $0.0$, what does this mean geometrically, and how does it affect the model's interpretation of their semantic relationship?

---

## Topic 2: Tokenization & Vocabulary Mapping

### Rationale and Mechanics
Computers cannot directly read text strings; they require numbers. Converting raw text into numbers involves **Tokenization** (splitting strings into smaller units like words or parts of words) and **Vocabulary Mapping** (assigning a unique integer index to each token).

Early NLP systems split text strictly at space characters (word-level tokenization). However, if your vocabulary size is restricted to 30,000 words to save memory, any word outside this set (like medical terms, typos, or rare names) gets converted to a generic `[UNK]` (Unknown) token. This is the **Out-of-Vocabulary (OOV)** problem, and it destroys information.

Modern deep learning uses **Subword Tokenization** algorithms, such as **Byte-Pair Encoding (BPE)**. Instead of choosing between character-by-character tokenization (which creates long sequences) or word-by-word tokenization (which creates giant vocabularies), subword tokenization splits rare words into common structural blocks.

Under the hood, BPE trains a vocabulary using these steps:
1.  Initialize the vocabulary with all basic characters (e.g., 'a', 'b', 'c').
2.  Represent all training words as sequences of characters.
3.  Count the frequency of adjacent symbol pairs across the dataset.
4.  Merge the most frequent pair to create a new token (e.g., merging 't' and 'h' to form 'th').
5.  Repeat merges until the vocabulary reaches a predefined size limit $V$.

```
       Raw Text: "The biodegradable cup."
       
       Word-level Tokenizer:      ["The", "biodegradable", "cup"] ---> OOV error for "biodegradable"
       
       Subword Tokenizer (BPE):   ["The", "bio", "##degrad", "##able", "cup"] ---> No OOV errors
```

If the trained BPE model encounters an unseen word like "biodegradable," it decomposes it into known subwords: `["bio", "##degrad", "##able"]` (where `##` shows that the subword attaches to the previous token). This ensures that every possible word can be tokenized without generating `[UNK]` tokens.

### Python Code Implementation
Below is a complete, self-contained implementation of a simple Byte-Pair Encoding (BPE) tokenizer. It trains a vocabulary of subwords on a short training text and then tokenizes an unseen word.

```python
import re
from collections import defaultdict

# 1. Training corpus and initial word counts
corpus = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3
}

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# 2. Train BPE by finding and merging the most frequent adjacent pairs
num_merges = 10
vocab = corpus.copy()
learned_merges = []

print("Starting BPE Training...")
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    learned_merges.append(best)
    print(f"Merge {i+1:2d}: {best} (Frequency: {pairs[best]})")

# Print the final BPE vocabulary tokens
tokens = set()
for word in vocab.keys():
    for tok in word.split():
        tokens.add(tok)
print("\nFinal Subword Vocabulary:", sorted(list(tokens)))

# 3. Tokenize an unseen word: "newer"
def tokenize(word, merges):
    # Initialize word as a list of characters, plus the end-of-word marker
    symbols = list(word) + ["</w>"]
    
    # Apply the learned merges in the order they were learned
    for pair in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                # Merge the pair
                symbols[i] = pair[0] + pair[1]
                del symbols[i+1]
            else:
                i += 1
    return symbols

unseen_word = "newer"
tokenized_output = tokenize(unseen_word, learned_merges)
print(f"\nTokenizing unseen word '{unseen_word}':")
print("Tokens:", tokenized_output)
```

### Trade-offs
*   **Advantages:** Eliminates Out-of-Vocabulary errors entirely since any word can be broken down to character levels if necessary. Keeps the global vocabulary size small, reducing memory.
*   **Disadvantages:** Increases the sequence length $T$ because words get split into multiple subwords. Downstream layers (like Transformers or RNNs) must process longer sequences, raising the computational cost.

### Real-World Applications (Rule of 4)
1.  **Example 1: Resolving Typos**
    *   **Input/Scenario:** A sentiment classifier encounters the word "greattttt" in a review.
    *   **Expected Output:** Rather than dropping it, the subword tokenizer outputs `["great", "##t", "##t", "##t", "##t"]`, enabling the downstream model to read the sentiment correctly.
2.  **Example 2: Processing Morphologically Rich Languages**
    *   **Input/Scenario:** A German translation model processes compound nouns like "Donaudampfschifffahrt".
    *   **Expected Output:** The subword tokenizer breaks it into standard subunits, avoiding vocabulary bloat.
3.  **Example 3: Handling Brand Names**
    *   **Input/Scenario:** A news classifier parses new brand names like "TikTok" or "ChatGPT".
    *   **Expected Output:** BPE breaks these into known root pieces, permitting context analysis without manual dictionary updates.
4.  **Example 4: Tokenizer Alignment in Fine-Tuning**
    *   **Input/Scenario:** A developer uses a model trained with GPT-4's tokenizer but tests it using a BERT tokenizer.
    *   **Expected Output:** The system fails or outputs gibberish because the integer ID assignments do not align.

> **Metacognitive Checkpoint:** Why does subword tokenization offer a better balance for neural network inputs compared to character-only or word-only approaches?

---

## Topic 3: The Embedding Layer: The Learnable Dictionary

### Rationale and Mechanics
After converting text into a sequence of integer IDs, the **Embedding Layer** maps these integers into dense vectors. You can think of this layer as a lookup table or a learnable dictionary.

This dictionary is represented by a weight matrix $\mathbf{E}$:
$$\mathbf{E} \in \mathbb{R}^{V \times d}$$
where $V$ is the vocabulary size, and $d$ is the embedding dimension.

Under the hood, if we represent an input token index $i$ as a one-hot vector $\mathbf{v}_i \in \mathbb{R}^V$, the dense embedding $\mathbf{e}_i$ is equivalent to multiplying the one-hot vector by the embedding matrix:
$$\mathbf{e}_i = \mathbf{v}_i^T \mathbf{E}$$

```
        One-Hot Input [0, 0, 1, 0, 0] (V=5)
              |
              v
        [ Embedding Matrix E ] (5 x 3)
        [  e11   e12   e13  ] - Row 0
        [  e21   e22   e23  ] - Row 1
        [  e31   e32   e33  ] - Row 2  <== Selected row (index 2)
        [  e41   e42   e43  ] - Row 3
        [  e51   e52   e53  ] - Row 4
              |
              v
        Dense Output: [ e31, e32, e33 ] (shape: 3,)
```

Multiplying large matrices by sparse one-hot vectors is computationally wasteful because we multiply mostly by zeros. To optimize performance, the Embedding layer performs a **direct index lookup** instead:
$$\mathbf{e}_i = \mathbf{E}[i, :]$$

The values in $\mathbf{E}$ are parameters updated during training. When calculating the loss gradient, the errors are backpropagated to update the specific rows of the tokens that were used in the forward pass:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{E}[i, :]} = \frac{\partial \mathcal{L}}{\partial \mathbf{e}_i}$$
All other rows in $\mathbf{E}$ remain unchanged. Over time, gradient descent adjusts these coordinates, automatically grouping words with similar contextual meanings.

### Python Code Implementation
The following code implements a custom `NumpyEmbedding` layer from scratch using NumPy. It demonstrates:
1.  Looking up vectors for input sequences.
2.  Verifying that lookup matches one-hot matrix multiplication.
3.  Updating the embedding weights using a simulated gradient update.

```python
import numpy as np

class NumpyEmbedding:
    def __init__(self, vocab_size, embed_dim):
        # Initialize the embedding weights randomly
        self.weights = np.random.randn(vocab_size, embed_dim) * 0.1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
    def forward_lookup(self, token_ids):
        # Save indices for the backward pass
        self.last_inputs = np.array(token_ids)
        # Perform highly efficient row index lookup
        return self.weights[self.last_inputs]
        
    def forward_matmul(self, one_hot_vectors):
        # Mathematically equivalent, but slower lookup
        return np.dot(one_hot_vectors, self.weights)
        
    def backward(self, dL_dout, lr=0.01):
        # Accumulate gradients for each index present in the input sequence
        # We only update the rows that were actually accessed
        for i, token_id in enumerate(self.last_inputs):
            self.weights[token_id] -= lr * dL_dout[i]

# --- Demonstration ---
np.random.seed(42)
V, D = 5, 3  # Vocab size = 5, Embedding dim = 3
embed_layer = NumpyEmbedding(vocab_size=V, embed_dim=D)

# 1. Forward Pass via Index Lookup
input_ids = [0, 2, 4]
embeddings_lookup = embed_layer.forward_lookup(input_ids)
print("Embedding matrix E:\n", np.round(embed_layer.weights, 4))
print("\nLooked up vectors for token IDs [0, 2, 4]:\n", np.round(embeddings_lookup, 4))

# 2. Verify equivalence to One-Hot Matrix Multiplication
one_hot = np.zeros((len(input_ids), V))
for row, idx in enumerate(input_ids):
    one_hot[row, idx] = 1.0
    
embeddings_matmul = embed_layer.forward_matmul(one_hot)
print("\nAre lookup and matrix multiplication identical?", np.allclose(embeddings_lookup, embeddings_matmul))

# 3. Simulate Backward Pass / Gradient Update
# Suppose we received gradients from downstream layers for our outputs
dL_dout = np.array([
    [0.1, -0.2, 0.05],  # Gradient for index 0
    [-0.1, 0.3, 0.1],   # Gradient for index 2
    [0.0,  0.0, -0.1]   # Gradient for index 4
])

embed_layer.backward(dL_dout, lr=0.1)
print("\nWeights after updating rows for active indexes [0, 2, 4]:\n", np.round(embed_layer.weights, 4))
```

### Trade-offs
*   **Advantages:** Extremely fast matrix lookup execution. Allows task-specific adaptation since weights are updated via standard backpropagation.
*   **Disadvantages:** Huge parameter footprint. For a large vocabulary of $100,000$ and embedding dimension $1024$, the table contains $1.024 \times 10^8$ float values ($\approx 400$ MB just for input mappings). In small datasets, this easily leads to overfitting. To prevent this, developers can initialize the matrix using frozen, pre-trained weights (e.g., GloVe or Word2Vec).

### Real-World Applications (Rule of 4)
1.  **Example 1: Parameter Count Calculation**
    *   **Input/Scenario:** A neural model has `input_dim=50000`, `output_dim=300`.
    *   **Expected Output:** The layer allocates exactly $15,000,000$ parameters, requiring about $60$ MB of memory storage space.
2.  **Example 2: Word Analogies via Tuning**
    *   **Input/Scenario:** A movie review sentiment model trains on positive and negative labels containing the word "stellar."
    *   **Expected Output:** Backpropagation updates the specific vector coordinates for "stellar," moving it closer to the coordinate for "excellent."
3.  **Example 3: Freezing Pre-trained Weights**
    *   **Input/Scenario:** A developer uses GloVe vectors to classify rare documents where training samples are scarce.
    *   **Expected Output:** Setting `trainable=False` freezes the lookup matrix, preventing overfitting on the training set.
4.  **Example 4: Handling Index Out of Bounds**
    *   **Input/Scenario:** The model processes a sequence containing index $45,000$, but the embedding layer is configured with a vocabulary limit of $30,000$.
    *   **Expected Output:** An index out-of-bounds runtime exception occurs.

> **Metacognitive Checkpoint:** Why is it computationally better to use index lookups rather than one-hot matrix multiplications in the forward pass of an embedding layer?

---

## Summary & Next Steps

*   **Embedding Vectors Represent Meaning:** Continuous vector spaces map words to low-dimensional coordinates. We measure their closeness using cosine similarity, shifting past the limits of orthogonal representations.
*   **Subword Tokenization Solves OOV:** Methods like BPE split words into subword fragments, avoiding unknown tokens while controlling vocabulary size.
*   **Embedding Layers are Efficient Lookups:** Instead of performing expensive sparse matrix multiplications, the Embedding layer serves as a direct index lookup table, updating only the accessed indices during training.

In the next lesson, we will look at **The Sequence-to-Sequence Bottleneck** to see why processing sequences step-by-step limits performance and how researchers designed methods to address it.
