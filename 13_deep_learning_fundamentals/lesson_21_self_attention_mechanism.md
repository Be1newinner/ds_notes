# Lesson 21: The Self-Attention Mechanism

## Introduction & The "Why"

In Lesson 20, we analyzed the limitations of traditional Sequence-to-Sequence (Seq2Seq) architectures. We proved that LSTMs and GRUs suffer from two bottlenecks: the information bottleneck (which compresses entire sentences into a single context vector, losing early details) and the sequential bottleneck (which processes tokens step-by-step, preventing parallelization on GPUs).

To solve both bottlenecks, Vaswani et al. published the landmark paper *"Attention Is All You Need"* in 2017, introducing the **Transformer** architecture. The core engine of the Transformer is the **Self-Attention Mechanism**.

Instead of compressing a sequence step-by-step, Self-Attention allows every word in a sequence to connect directly to every other word. By calculating mathematical similarity scores between all pairs of tokens, the model can determine which words are most relevant to its current context, resolving pronouns and syntactic structures in parallel. This lesson covers the intuition of self-attention, explains the queries, keys, and values projection mechanism, details Scaled Dot-Product Attention, and explains how Multi-Head Attention extracts diverse relationships.

---

## Topic 1: The Intuition of Self-Attention: Bidirectional Context Matching

### Rationale and Mechanics
In classical natural language processing, words are represented by static embeddings (e.g., Word2Vec). In a static embedding, the word "bank" has the same vector representation whether it appears in the sentence "I deposited money in the bank" or "The river bank was muddy." This is a major limitation because language is highly contextual.

**Self-Attention** resolves this by dynamically updating word embeddings based on their context. It allows each word in a sequence to "attend" (look at) all other words in the sequence to refine its own meaning.

To understand this intuition, let's analyze a famous pronoun resolution example:
- **Sentence A:** "The animal didn't cross the street because **it** was too tired."
- **Sentence B:** "The animal didn't cross the street because **it** was too wide."

In Sentence A, a human immediately understands that the pronoun "it" refers to the "animal." In Sentence B, "it" refers to the "street." 

```
       Sentence A:
       The animal didn't cross the street because it was too tired.
           ^                                      | (High attention weight)
           \--------------------------------------/
           
       Sentence B:
       The animal didn't cross the street because it was too wide.
                                    ^             | (High attention weight)
                                    \-------------/
```

Self-Attention calculates a soft alignment score between every pair of words.
- In Sentence A, the attention mechanism associates the adjective "tired" with "it", which matches the features of "animal" (animals get tired, streets do not). The model assigns a high attention weight between "it" and "animal."
- In Sentence B, the adjective "wide" matches the features of "street." The model assigns a high attention weight between "it" and "street."

Through this process, the model dynamically updates the embedding of "it" to blend in features from the noun it references.

### Trade-offs
Self-Attention solves the information bottleneck because the maximum distance between any two tokens in the sequence is exactly $1$ operation: any token can read from any other token directly. It also solves the sequential bottleneck because attention scores for all tokens are calculated in parallel.

The trade-off is computational complexity. Because we must calculate similarity scores between every pair of tokens, the computational and memory complexity scales **quadratically** with sequence length:
$$\text{Complexity} = O(T^2)$$
where $T$ is the sequence length. For very long documents (e.g., $T > 8,000$ tokens), calculating and storing the $T\times T$ attention matrix can exceed GPU memory, requiring specialized sparse attention approximations.

### Real-World Applications (Rule of 4)

1. **Example 1: Pronoun Resolution Alignment**
   - **Input/Scenario:** A self-attention layer processes Sentence A: "The animal didn't cross the street because it was too tired."
   - **Expected Output:** The attention weight matrix shows a high probability connection ($0.85$) between the query token "it" and the key token "animal," updating "it" with animal-related features.
2. **Example 2: Polysemy Resolution**
   - **Input/Scenario:** The model processes the word "bank" in "The river bank was muddy."
   - **Expected Output:** The self-attention mechanism identifies strong connections to "river" and "muddy," shifting the vector representation of "bank" toward the geographical meaning rather than the financial one.
3. **Example 3: Parallel Sentence Encoding**
   - **Input/Scenario:** An attention layer processes a 50-word sentence on a GPU with 50 parallel cores.
   - **Expected Output:** The GPU calculates the attention scores for all 50 words simultaneously, completing the forward pass in a single step instead of the 50 sequential steps required by an LSTM.
4. **Example 4: Quadratic Memory Explosion**
   - **Input/Scenario:** A developer attempts to train a standard Transformer on a document of length $T = 10,000$ tokens.
   - **Expected Output:** The self-attention matrix requires storing $10,000 \times 10,000 = 100,000,000$ attention values per head. The GPU runs out of memory (OOM), forcing the developer to shorten the input sequences.

> **Metacognitive Checkpoint:** Why does the self-attention mechanism scale quadratically ($O(T^2)$) with sequence length? Explain by describing the grid of similarity calculations required for a sequence of length $T$.

---

## Topic 2: Queries, Keys, and Values: The Information Retrieval Analogy

### Rationale and Mechanics
To calculate self-attention mathematically, we use an analogy from information retrieval databases: **Queries**, **Keys**, and **Values**.
- **Query (Q):** What a word is looking for (e.g., "I am a verb, looking for my subject").
- **Key (K):** What a word offers (e.g., "I am a noun, I can act as a subject").
- **Value (V):** The actual content of the word (e.g., "I am the word 'animal'").

Under the hood, let the input sequence of word embeddings be represented as a matrix $\mathbf{X} \in \mathbb{R}^{T \times d}$, where $T$ is the sequence length and $d$ is the embedding dimension. We project $\mathbf{X}$ into three matrices using three learnable weight matrices $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d \times d_k}$ and $\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$:
$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$

To calculate the attention outputs, we use **Scaled Dot-Product Attention**:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

Let's dissect this equation step-by-step:
1. **Dot Product ($\mathbf{Q}\mathbf{K}^T$):** Computes the raw similarity scores between every query and every key. The result is a matrix of shape $(T, T)$.
2. **Scaling Factor ($\sqrt{d_k}$):** We divide the dot product by the square root of the key dimension $d_k$. If $d_k$ is large, the dot products can grow very large in magnitude. This pushes the Softmax function into its flat, saturated regions where the derivative is close to zero, causing vanishing gradients during training. Scaling prevents this saturation.
3. **Softmax:** Normalizes the scaled similarity scores along each row, converting them into probabilities that sum to $1.0$. These are the **Attention Weights**.
4. **Weighted Sum ($\mathbf{A} \cdot \mathbf{V}$):** We multiply the attention weight matrix by the Value matrix $\mathbf{V}$. This generates a weighted combination of the values, outputting a tensor of shape $(T, d_v)$ where each word representation is enriched with context from the words it attended to.

```
       Q Matrix (T x dk) -----\
                               x (Dot Product) ---> (T x T) ---> [ / sqrt(dk) ] ---> [ Softmax ] ---> Weight Matrix (T x T)
       K Matrix (T x dk) -----/                                                                             |
                                                                                                             x (Matrix Multiply)
                                                                                                            |
       V Matrix (T x dv) ----------------------------------------------------------------------------------/
                                                                                                            |
                                                                                                            v
                                                                                                  Output Matrix (T x dv)
```

### Trade-offs
The Query-Key-Value projection is a fully differentiable operation, allowing the network to learn how to match features end-to-end using gradient descent.

The trade-off is the addition of parameter matrices $\mathbf{W}_Q$, $\mathbf{W}_K$, and $\mathbf{W}_V$, which increases the parameter count of each layer. However, because these projections are simple matrix multiplications, they are highly optimized on GPU tensor cores.

### Real-World Applications (Rule of 4)

1. **Example 1: Dot Product Scaling Importance**
   - **Input/Scenario:** We use a key dimension $d_k = 64$ ($\sqrt{d_k} = 8$). Two vectors yield a raw dot product of $q \cdot k = 24.0$.
   - **Expected Output:** Without scaling, $e^{24}$ is extremely large, saturating the Softmax and zeroing out gradients. With scaling, the input to the softmax is $\frac{24}{8} = 3.0$, keeping the Softmax active and preserving gradients.
2. **Example 2: Information Retrieval Search Analogy**
   - **Input/Scenario:** A user searches for "deep learning" on YouTube.
   - **Expected Output:** The search term is the **Query**. The database titles of videos are the **Keys**. The actual video streams are the **Values**. The search engine matches the Query to the Keys to retrieve the best Values.
3. **Example 3: Softmax Weight Extraction**
   - **Input/Scenario:** A sequence of length 3 yields scaled dot products $\mathbf{Q}\mathbf{K}^T / \sqrt{d_k} = \begin{pmatrix} 2.0 & 0.0 & 0.0 \end{pmatrix}$ for the first word.
   - **Expected Output:** Applying Softmax yields attention weights:
     $$\text{Softmax}([2.0, 0.0, 0.0]) = \left[ \frac{e^2}{e^2+1+1}, \frac{1}{e^2+1+1}, \frac{1}{e^2+1+1} \right] \approx [0.79, 0.11, 0.11]$$
     The output value for Word 1 will contain 79% of its own value and 11% of the values of Words 2 and 3.
4. **Example 4: Dimension Matching**
   - **Input/Scenario:** Input matrix $\mathbf{X}$ has shape $(32, 10, 512)$ (Batch, Timesteps, Embedding Dim). The projection matrices have shape $(512, 64)$.
   - **Expected Output:** The Query and Key matrices have shape $(32, 10, 64)$. The product $\mathbf{Q}\mathbf{K}^T$ has shape $(32, 10, 10)$, representing the attention grid for each batch sample.

> **Metacognitive Checkpoint:** Why do we divide the dot product $\mathbf{Q}\mathbf{K}^T$ by $\sqrt{d_k}$ in Scaled Dot-Product Attention? Explain the relationship between high-dimensional vector dot products and Softmax gradient saturation.

---

## Topic 3: Multi-Head Attention: Extracting Diverse Feature Relationships

### Rationale and Mechanics
If we calculate attention using a single set of Query, Key, and Value matrices, the model can only focus on a single type of relationship at a time. For example, if the model attends heavily to the pronoun relationship ("it" $\to$ "animal"), it might miss the syntactic verb-object relationship.

To solve this, Transformers use **Multi-Head Attention (MHA)**. MHA splits the Queries, Keys, and Values into $h$ smaller subspaces, performs attention on each subspace in parallel, and concatenates the results.

Under the hood:
1. We project the input embeddings $\mathbf{X}$ into $h$ sets of Queries, Keys, and Values using $h$ independent sets of projection matrices:
   $$\mathbf{Q}_i = \mathbf{X} \mathbf{W}_Q^i, \quad \mathbf{K}_i = \mathbf{X} \mathbf{W}_K^i, \quad \mathbf{V}_i = \mathbf{X} \mathbf{W}_V^i$$
   where $i \in \{1, \dots, h\}$.
2. We calculate Scaled Dot-Product Attention for each head independently:
   $$\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i)$$
3. We concatenate the outputs of all $h$ heads:
   $$\text{Concat}(\text{head}_1, \dots, \text{head}_h)$$
4. We project the concatenated vector back to the original dimension using a final output projection matrix $\mathbf{W}^O$:
   $$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O$$

```
                                  [ Input X ]
                                  /    |    \ (Project into h heads)
                            Head 1   Head 2  Head h
                              |        |       |
                            [Attn]   [Attn]  [Attn]
                              \        |       /
                             [ Concatenate Heads ]
                                       |
                                 [ Project W^O ]
                                       |
                                    Output
```

Each head initializes with different random weights, allowing them to focus on different aspects of language:
- **Head 1:** Focuses on immediate neighbor words (syntax/grammar).
- **Head 2:** Focuses on long-range pronoun references.
- **Head 3:** Focuses on subject-verb agreements.
- **Head 4:** Focuses on semantic synonym relationships.

By combining these heads, the model builds a rich, multi-dimensional representation of context.

### Trade-offs
- **Advantages:** Prevents the network from averaging out all semantic relationships. It allows the model to extract diverse feature structures simultaneously.
- **Disadvantages:** Implementation is complex. To run MHA efficiently on a GPU, we do not loop through heads. Instead, we use high-dimensional matrix transpositions to calculate all heads in a single GPU operation, which requires precise tracking of tensor shapes.

### Real-World Applications (Rule of 4)

1. **Example 1: Dimension Splitting Calculation**
   - **Input/Scenario:** A Transformer has embedding dimension $d=512$ and uses $h=8$ attention heads.
   - **Expected Output:** The key dimension for each head is $d_k = d / h = 512 / 8 = 64$. The projection matrices for each head have shape $(512, 64)$. Concatenating the 8 heads yields a vector of shape $8 \times 64 = 512$, matching the input dimension.
2. **Example 2: Visualizing Head Roles**
   - **Input/Scenario:** A translation model processes the sentence "The contract was signed by the company."
   - **Expected Output:** Head 1 highlights the connection between "signed" and "contract" (action-object). Head 2 highlights the connection between "signed" and "company" (action-actor). This diverse focus enables accurate translation.
3. **Example 3: Keras MultiHeadAttention Layer**
   - **Input/Scenario:** A developer instantiates a MultiHeadAttention layer in Keras.
   - **Expected Output:**
     ```python
     mha = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
     output = mha(query=x, value=x, key=x)
     ```
     The layer projects the inputs, calculates attention in parallel, and merges the outputs.
4. **Example 4: Performance Bottleneck Avoidance**
   - **Input/Scenario:** A developer compares looping through 8 heads sequentially in Python versus using tensor transposition.
   - **Expected Output:** The tensor transposition method runs $8\times$ faster because it executes the projections for all 8 heads as a single large GPU matrix multiplication.

> **Metacognitive Checkpoint:** Why is Multi-Head Attention superior to Single-Head Attention? Explain how splitting the embedding dimension into multiple heads prevents different semantic relationships from washing each other out.

---

## Summary & Next Steps

- **Self-Attention Computes Context:** Self-attention allows every word in a sequence to connect directly to every other word, calculating dynamic, context-aware embeddings in parallel.
- **Q, K, V Project Meaning:** Projecting inputs into Queries, Keys, and Values allows similarity to be calculated using Scaled Dot-Product Attention, protecting gradients from Softmax saturation.
- **Multi-Head Attention Extracts Variety:** Splitting projections into multiple heads allows the model to track different linguistic relationships (grammar, pronouns, subjects) simultaneously.

In the next lesson, we will explore **Generative AI Foundations**, learning how self-attention scales to build Large Language Models, and exploring causal language modeling and generation parameters.
