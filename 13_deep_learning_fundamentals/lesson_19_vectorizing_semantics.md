# Lesson 19: Vectorizing Semantics

## Introduction & The "Why"

In classical natural language processing, text is represented using sparse count-based representations like **Bag-of-Words (BoW)** or **TF-IDF**. While these methods are effective for simple document classification, they fail to capture the semantic meaning of language. 

In a Bag-of-Words representation, each word is treated as an independent dimension in a high-dimensional sparse vector. Under this design, the words "cat" and "kitten" are represented as orthogonal features. The mathematical dot product between their vectors is exactly zero, meaning the model treats them as completely unrelated. Furthermore, BoW discards word order, treating the sentences "not bad, good" and "not good, bad" identically.

To solve this semantic bottleneck, deep learning represents language using **Word Embeddings**. Instead of high-dimensional, sparse, orthogonal vectors, we map words to low-dimensional, dense vectors in a continuous **Semantic Vector Space**. In this space, words with similar meanings are positioned close to each other, and the geometric relationships between vectors represent semantic relationships (e.g., gender, tense, or capital cities). This lesson covers the geometry of semantic spaces, details subword tokenization, and explains the mechanics of the learnable Embedding layer.

---

## Topic 1: Moving Beyond Bag-of-Words: The Semantic Vector Space

### Rationale and Mechanics
In a one-hot representation, a vocabulary of size $V$ is mapped to a vector space $\mathbb{R}^V$. For a vocabulary of $50,000$ words, each word is a vector of size $50,000$ containing a single $1.0$ and $49,999$ zeros.

Under the hood, if we compute the dot product between any two distinct one-hot word vectors $\mathbf{v}_i$ and $\mathbf{v}_j$:
$$\mathbf{v}_i^T \mathbf{v}_j = 0 \quad (\text{for } i \neq j)$$

This orthogonality assumes that all words are completely independent. It is impossible for a model trained on one-hot vectors to generalize: if it learns a relationship for the word "cat," it cannot apply that knowledge to the word "kitten" because the features do not overlap.

Deep learning maps words to a dense continuous vector space:
$$f: w \to \mathbf{e} \in \mathbb{R}^d$$
where $d \ll V$ (typically $d \in [100, 768]$).

In this dense space, semantic similarity is measured using **Cosine Similarity**, which calculates the cosine of the angle between two word vectors:
$$\text{CosSim}(\mathbf{e}_1, \mathbf{e}_2) = \cos(\theta) = \frac{\mathbf{e}_1^T \mathbf{e}_2}{\|\mathbf{e}_1\| \|\mathbf{e}_2\|}$$
- **Cosine Similarity = 1:** The vectors point in the same direction ($\theta = 0^\circ$), representing high semantic similarity.
- **Cosine Similarity = 0:** The vectors are orthogonal, representing no semantic relationship.
- **Cosine Similarity = -1:** The vectors point in opposite directions, representing antonyms or opposing concepts.

```
                         Semantic Vector Space Geometry
                         
                                      y
                                      ^      / e_king
                                      |    / 
                                      |  /   (Small Angle = High CosSim)
                                      |/  _  e_queen
                                      +---------> x
```

Furthermore, this continuous space exhibits **Linear Vector Arithmetic**. The difference vectors between concepts represent semantic transformations:
$$\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}} \approx \mathbf{e}_{\text{queen}}$$
$$\mathbf{e}_{\text{paris}} - \mathbf{e}_{\text{france}} + \mathbf{e}_{\text{italy}} \approx \mathbf{e}_{\text{rome}}$$

### Trade-offs
- **Advantages:** Captures complex semantic, syntactic, and contextual relationships. It reduces input dimensionality, allowing networks to train on dense features and generalize to unseen synonyms.
- **Disadvantages:** Dense embeddings are abstract and lack direct interpretability. In a TF-IDF representation, coordinate 5 directly represents the frequency of the word "tax." In a dense embedding, coordinate 5 is a floating-point value that represents a mixture of latent semantic concepts, making it impossible to analyze without projection techniques like t-SNE or PCA.

### Real-World Applications (Rule of 4)

1. **Example 1: One-Hot Orthogonality Calculation**
   - **Input/Scenario:** We represent "cat" as $\mathbf{v}_1 = [1, 0, 0]^T$ and "kitten" as $\mathbf{v}_2 = [0, 1, 0]^T$.
   - **Expected Output:** The dot product is $\mathbf{v}_1^T \mathbf{v}_2 = 1(0) + 0(1) + 0(0) = 0.0$. Geometrically, the words are orthogonal, representing no semantic similarity.
2. **Example 2: Dense Cosine Similarity Calculation**
   - **Input/Scenario:** We project words to a 2D space. The embedding for "cat" is $\mathbf{e}_1 = [1.0, 1.0]^T$ and "kitten" is $\mathbf{e}_2 = [0.9, 1.1]^T$.
   - **Expected Output:**
     - Dot product: $\mathbf{e}_1^T \mathbf{e}_2 = 1.0(0.9) + 1.0(1.1) = 2.0$
     - Norms: $\|\mathbf{e}_1\| = \sqrt{1^2+1^2} \approx 1.414$, $\|\mathbf{e}_2\| = \sqrt{0.9^2+1.1^2} \approx 1.421$
     - Cosine Similarity: $\frac{2.0}{1.414 \cdot 1.421} \approx 0.996$. The value is close to $1.0$, indicating high semantic similarity.
3. **Example 3: Analogy Navigation**
   - **Input/Scenario:** A search engine processes query embeddings to find similar concepts. It calculates $\mathbf{e}_{\text{berlin}} - \mathbf{e}_{\text{germany}} + \mathbf{e}_{\text{japan}}$.
   - **Expected Output:** The resulting vector points closest to $\mathbf{e}_{\text{tokyo}}$ in the embedding database, allowing the search engine to resolve country-capital relationships.
4. **Example 4: Sentiment Generalization**
   - **Input/Scenario:** A sentiment classifier is trained on sentences containing "excellent" (e.g., "The food was excellent"). During testing, it encounters the sentence "The food was superb."
   - **Expected Output:** Because $\mathbf{e}_{\text{excellent}}$ and $\mathbf{e}_{\text{superb}}$ are close in the semantic space, the model correctly classifies the test sentence as positive, despite never seeing the word "superb" during training.

> **Metacognitive Checkpoint:** Why are one-hot encoded vectors mathematically incapable of representing semantic similarity? Explain how continuous vector spaces resolve this limitation using the cosine similarity formula.

---

## Topic 2: Tokenization & Vocabulary Mapping

### Rationale and Mechanics
Neural networks cannot process raw text strings. We must convert text into a sequence of numbers. This conversion process consists of two steps: **Tokenization** (splitting text into discrete units called tokens) and **Vocabulary Mapping** (mapping each token to a unique integer ID).

Early deep learning models used word-level tokenization. However, if the vocabulary size is fixed to 30,000 words, any word not in this list (e.g., names, technical terms, or typos) is mapped to a special `[UNK]` (Unknown) token. This is the **Out-of-Vocabulary (OOV)** problem.

To solve this, modern language models use **Subword Tokenization** algorithms, such as Byte-Pair Encoding (BPE) or WordPiece.

Under the hood, subword tokenizers build a vocabulary of characters and frequent character combinations:
1. Initialize the vocabulary with all single characters (e.g., 'a', 'b', 'c').
2. Count co-occurrences of character sequences in a large training corpus.
3. Iteratively merge the most frequent pairs (e.g., 't' and 'h' $\to$ 'th', then 'th' and 'e' $\to$ 'the').
4. Stop when the vocabulary reaches a target size $V$ (typically 30,000 to 50,000 tokens).

```
       Raw Text: "The biodegradable cup."
       
       Word-level Tokenizer:      ["The", "biodegradable", "cup"] ---> OOV error for "biodegradable"
       
       Subword Tokenizer (BPE):   ["The", "bio", "##degrad", "##able", "cup"] ---> No OOV errors
```

If the tokenizer encounters an unseen word like "biodegradable," it splits it into subwords that exist in the vocabulary: `["bio", "##degrad", "##able"]`. The prefix `##` indicates that the subword is connected to a preceding token.

### Trade-offs
- **Advantages:** Subword tokenization completely eliminates OOV errors. Any word can be decomposed into subwords or characters. It keeps the vocabulary size manageable, preventing parameter explosion in the embedding layer.
- **Disadvantages:** Subword tokenization increases the sequence length $T$ because a single word is split into multiple tokens. This increases the memory usage of downstream layers (like attention modules) that scale quadratically with sequence length.

### Real-World Applications (Rule of 4)

1. **Example 1: WordPiece Tokenization Mapping**
   - **Input/Scenario:** We tokenize the sentence "Preprocessing text is fun." using a BERT-style subword tokenizer.
   - **Expected Output:** The tokenizer outputs the list of tokens: `['pre', '##process', '##ing', 'text', 'is', 'fun', '.']`. Each token is mapped to its vocabulary integer: `[3021, 19283, 1102, 3091, 1110, 2510, 119]`.
2. **Example 2: Typos and Unknown Words**
   - **Input/Scenario:** A user enters a typo: "greeeat".
   - **Expected Output:** A word-level tokenizer maps this to `[UNK]`. A subword tokenizer splits it into `['gree', '##e', '##at']`, preserving the root meaning.
3. **Example 3: Medical Coding Extraction**
   - **Input/Scenario:** A medical records processor encounters a rare chemical term: "acetaminophen".
   - **Expected Output:** The subword tokenizer decomposes it into standard prefix and suffix tokens, allowing the model to analyze the term's context.
4. **Example 4: Tokenizer-Model Alignment**
   - **Input/Scenario:** A developer uses a pre-trained GPT-2 model but tokenizes their dataset using a BERT tokenizer.
   - **Expected Output:** The model outputs gibberish. The integer IDs mapped by the BERT tokenizer do not align with the semantic weight coordinates learned by the GPT-2 model, highlighting that tokenizers and models must match.

> **Metacognitive Checkpoint:** Why has subword tokenization replaced word-level tokenization in modern Large Language Models? Explain how BPE prevents the OOV problem.

---

## Topic 3: The Embedding Layer: The Learnable Dictionary

### Rationale and Mechanics
Once text is converted into a sequence of integer IDs, we must project these integers into dense vectors. We do this using an **Embedding Layer**.

An Embedding layer is a learnable lookup table represented as a matrix $\mathbf{E}$:
$$\mathbf{E} \in \mathbb{R}^{V \times d}$$
where $V$ is the vocabulary size and $d$ is the embedding dimension.

Under the hood, let a token ID be represented as a sparse one-hot vector $\mathbf{v}_i \in \mathbb{R}^V$. Projecting this token into the dense space is mathematically equivalent to multiplying the one-hot vector by the embedding matrix:
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

In practice, multiplying a $1\times V$ vector by a $V \times d$ matrix is highly inefficient because it involves multiplying by thousands of zeros. Instead, the Embedding layer performs a **direct index lookup**:
$$\mathbf{e}_i = \mathbf{E}[i, :]$$

The weights of the embedding matrix $\mathbf{E}$ are parameters updated during training. When we calculate loss gradients, the gradient with respect to the embedding matrix is backpropagated to update the specific row coordinates of the active tokens:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{E}[i, :]} = \frac{\partial \mathcal{L}}{\partial \mathbf{e}_i}$$

As the model trains (e.g., on a classification task), the optimizer adjusts the row coordinates in $\mathbf{E}$. Words that appear in similar contexts are pushed closer together, automatically developing the semantic vector space geometry.

### Trade-offs
- **Advantages:** Highly efficient and optimized in hardware. It allows the model to learn custom, task-specific semantic mappings directly from training data.
- **Disadvantages:** The Embedding layer introduces a large number of parameters. For a vocabulary size $V = 50,000$ and dimension $d = 300$, the embedding matrix has:
$$\text{Params} = 50,000 \times 300 = 15,000,000 \text{ parameters}$$
This matrix dominates the size of shallow networks. To reduce parameter counts, we can initialize the embedding layer using pre-trained weights (such as GloVe or Word2Vec) and freeze them, preventing updates.

### Real-World Applications (Rule of 4)

1. **Example 1: Embedding Parameter Count Calculation**
   - **Input/Scenario:** A developer instantiates a Keras Embedding layer with vocabulary $V = 30,000$ and embedding dimension $d = 128$.
   - **Expected Output:**
     ```python
     keras.layers.Embedding(input_dim=30000, output_dim=128)
     ```
     This layer initializes $3,840,000$ trainable weight parameters.
2. **Example 2: Direct Row Lookup**
   - **Input/Scenario:** A token sequence is `[0, 2]`. The embedding matrix is $\mathbf{E} = \begin{pmatrix} 0.5 & 0.1 \\ -0.2 & 0.8 \\ 0.9 & -0.4 \end{pmatrix}$.
   - **Expected Output:** The layer outputs the tensor:
     $$\mathbf{A} = \begin{pmatrix} \mathbf{E}[0, :] \\ \mathbf{E}[2, :] \end{pmatrix} = \begin{pmatrix} 0.5 & 0.1 \\ 0.9 & -0.4 \end{pmatrix}$$
     The output shape is $(2, 2)$, corresponding to the sequence length and embedding dimension.
3. **Example 3: Pre-trained Weight Initialization**
   - **Input/Scenario:** A developer uses pre-trained GloVe weights to initialize their Keras Embedding layer.
   - **Expected Output:**
     ```python
     embedding_layer = keras.layers.Embedding(
         input_dim=vocab_size,
         output_dim=300,
         embeddings_initializer=keras.initializers.Constant(glove_matrix),
         trainable=False  # Freeze weights to prevent overfitting
     )
     ```
     The pre-trained semantic relations are preserved, and training time is reduced.
4. **Example 4: Out-of-Bounds Error**
   - **Input/Scenario:** An input token has ID 35,000, but the embedding layer was instantiated with `input_dim=30000`.
   - **Expected Output:** The runtime raises a `tf.errors.InvalidArgumentError` or index out-of-bounds exception because the lookup index exceeds the row count of the embedding matrix.

> **Metacognitive Checkpoint:** Why is the lookup operation of an Embedding layer mathematically equivalent to a matrix multiplication with a one-hot vector? Write the equations to prove this equivalence.

---

## Summary & Next Steps

- **Word Embeddings Map Meaning:** Dense continuous vector spaces allow semantic relationships to be captured using cosine similarity, moving beyond sparse, orthogonal Bag-of-Words vectors.
- **Subword Tokenizers Prevent OOV:** Algorithms like BPE decompose words into characters and frequent subwords, keeping vocabulary sizes small while resolving unseen words.
- **Embedding Layers are Lookup Tables:** The Embedding layer maps integer IDs to dense row vectors, acting as a learnable dictionary updated during backpropagation.

In the next lesson, we will explore **The Sequence-to-Sequence Bottleneck**, analyzing why Recurrent Neural Networks are too slow for modern NLP and learning the limitations of processing text one word at a time.
