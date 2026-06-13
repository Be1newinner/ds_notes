# Lesson 21: The Self-Attention Mechanism

## Introduction & The "Why"

In Lesson 20, we analyzed the limitations of traditional Sequence-to-Sequence (Seq2Seq) architectures. We proved that LSTMs and GRUs suffer from two bottlenecks: the information bottleneck (which compresses entire sentences into a single context vector, losing early details) and the sequential bottleneck (which processes tokens step-by-step, preventing parallelization on GPUs).

To solve both bottlenecks, Vaswani et al. published the landmark paper *"Attention Is All You Need"* in 2017, introducing the **Transformer** architecture. The core engine of the Transformer is the **Self-Attention Mechanism**.

Instead of compressing a sequence step-by-step, Self-Attention allows every word in a sequence to connect directly to every other word. By calculating mathematical similarity scores between all pairs of tokens, the model can determine which words are most relevant to its current context, resolving pronouns and syntactic structures in parallel. This lesson covers the intuition of self-attention, explains the queries, keys, and values projection mechanism, details Scaled Dot-Product Attention, explains how Multi-Head Attention extracts diverse relationships, and explores Positional Encoding.

---

## Topic 1: The Intuition of Self-Attention: Bidirectional Context Matching

### Rationale and Mechanics
In classical natural language processing, words are represented by static embeddings (e.g., Word2Vec). In a static embedding, the word "bank" has the same vector representation whether it appears in the sentence "I deposited money in the bank" or "The river bank was muddy." This is a major limitation because language is highly contextual.

**Self-Attention** resolves this by dynamically updating word embeddings based on their context. It allows each word in a sequence to "attend" (look at) all other words in the sequence to refine its own meaning.

To understand this intuition, let's analyze a famous pronoun resolution example:
*   **Sentence A:** "The animal didn't cross the street because **it** was too tired."
*   **Sentence B:** "The animal didn't cross the street because **it** was too wide."

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
*   In Sentence A, the attention mechanism associates the adjective "tired" with "it", which matches the features of "animal" (animals get tired, streets do not). The model assigns a high attention weight between "it" and "animal."
*   In Sentence B, the adjective "wide" matches the features of "street." The model assigns a high attention weight between "it" and "street."

Through this process, the model dynamically updates the embedding of "it" to blend in features from the noun it references.

### Python Code Implementation
The following code demonstrates the intuition of context matching. We define simple word embeddings, construct an attention weight distribution representing how "it" connects to "animal" or "street" based on context, and show how the updated embedding of "it" shifts in vector space.

```python
import numpy as np

# 1. Define toy embeddings representing semantic features: [Is_Animal, Is_Structure, Is_Tired/Wide]
embeddings = {
    "animal":  np.array([0.9, 0.1, 0.0]),
    "street":  np.array([0.1, 0.9, 0.0]),
    "it":      np.array([0.0, 0.0, 0.0]), # Unassigned pronoun vector
    "tired":   np.array([0.8, 0.0, 0.8]),
    "wide":    np.array([0.0, 0.8, 0.8])
}

# 2. Simulate context-driven attention weights for "it"
# In Sentence A ("it was too tired"), "it" attends heavily to "animal"
attn_weights_A = {"animal": 0.80, "street": 0.15, "tired": 0.05}

# In Sentence B ("it was too wide"), "it" attends heavily to "street"
attn_weights_B = {"animal": 0.10, "street": 0.85, "wide": 0.05}

# 3. Update the representation of "it" by computing the weighted sum of its context
updated_it_A = sum(weight * embeddings[word] for word, weight in attn_weights_A.items())
updated_it_B = sum(weight * embeddings[word] for word, weight in attn_weights_B.items())

print("--- Sentence A ('it was too tired') ---")
print("Updated vector for 'it':", np.round(updated_it_A, 4))
print("Similarity to 'animal':", np.dot(updated_it_A, embeddings["animal"]):.4f)
print("Similarity to 'street':", np.dot(updated_it_A, embeddings["street"]):.4f)

print("\n--- Sentence B ('it was too wide') ---")
print("Updated vector for 'it':", np.round(updated_it_B, 4))
print("Similarity to 'animal':", np.dot(updated_it_B, embeddings["animal"]):.4f)
print("Similarity to 'street':", np.dot(updated_it_B, embeddings["street"]):.4f)
```

### Trade-offs
*   **Advantages:** Solves the information bottleneck because the maximum path length between any two tokens in the sequence is exactly $1$ operation: any token can directly read from any other token. It also solves the sequential bottleneck because attention scores for all tokens are calculated in parallel.
*   **Disadvantages:** Computational complexity. Because we must calculate similarity scores between every pair of tokens, the computational and memory complexity scales **quadratically** with sequence length:
    $$\text{Complexity} = O(T^2)$$
    where $T$ is the sequence length. For very long documents (e.g., $T > 8,000$ tokens), calculating and storing the $T\times T$ attention matrix can exceed GPU memory.

### Real-World Applications (Rule of 4)
1.  **Example 1: Pronoun Resolution Alignment**
    *   **Input/Scenario:** A self-attention layer processes "The animal didn't cross the street because it was too tired."
    *   **Expected Output:** The attention weight matrix shows a high connection ($0.85$) between "it" and "animal."
2.  **Example 2: Polysemy Resolution**
    *   **Input/Scenario:** The model processes the word "bank" in "The river bank was muddy."
    *   **Expected Output:** The self-attention mechanism identifies strong connections to "river" and "muddy," shifting the vector representation of "bank" toward its geographical meaning.
3.  **Example 3: Parallel Sentence Encoding**
    *   **Input/Scenario:** An attention layer processes a 50-word sentence on a GPU with 50 parallel cores.
    *   **Expected Output:** The GPU calculates attention scores for all 50 words simultaneously, completing the forward pass in a single step instead of the 50 sequential steps required by an RNN.
4.  **Example 4: Quadratic Memory Explosion**
    *   **Input/Scenario:** A developer trains a standard Transformer on a document of length $T = 10,000$ tokens.
    *   **Expected Output:** The attention matrix requires storing $10,000 \times 10,000 = 100,000,000$ values per head, causing a GPU out-of-memory error.

> **Metacognitive Checkpoint:** Why does the self-attention mechanism scale quadratically ($O(T^2)$) with sequence length? Explain by describing the grid of similarity calculations required for a sequence of length $T$.

---

## Topic 2: Queries, Keys, and Values: The Information Retrieval Analogy

### Rationale and Mechanics
To calculate self-attention mathematically, we use an analogy from information retrieval databases: **Queries**, **Keys**, and **Values**.
*   **Query (Q):** What a word is looking for (e.g., "I am a verb, looking for my subject").
*   **Key (K):** What a word offers (e.g., "I am a noun, I can act as a subject").
*   **Value (V):** The actual semantic content of the word.

Under the hood, let the input sequence of word embeddings be represented as a matrix $\mathbf{X} \in \mathbb{R}^{T \times d}$, where $T$ is the sequence length and $d$ is the embedding dimension. We project $\mathbf{X}$ into three matrices using three learnable weight matrices $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d \times d_k}$ and $\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$:
$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$

To calculate the attention outputs, we use **Scaled Dot-Product Attention**:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

Let's dissect this equation step-by-step:
1.  **Dot Product ($\mathbf{Q}\mathbf{K}^T$):** Computes the raw similarity scores between every query and every key. The result is a matrix of shape $(T, T)$.
2.  **Scaling Factor ($\sqrt{d_k}$):** We divide the dot product by the square root of the key dimension $d_k$. If $d_k$ is large, the dot products can grow very large in magnitude. This pushes the Softmax function into its flat, saturated regions where the derivative is close to zero, causing vanishing gradients during training. Scaling prevents this saturation.
3.  **Softmax:** Normalizes the scaled similarity scores along each row, converting them into probabilities that sum to $1.0$. These are the **Attention Weights**.
4.  **Weighted Sum ($\mathbf{A} \cdot \mathbf{V}$):** We multiply the attention weight matrix by the Value matrix $\mathbf{V}$. This generates a weighted combination of the values, outputting a tensor of shape $(T, d_v)$ where each word representation is enriched with context from the words it attended to.

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

### Python Code Implementation
The following code implements Scaled Dot-Product Attention from scratch in NumPy, demonstrating the tensor shape transitions and verifying how the scaling factor protects gradients from vanishing.

```python
import numpy as np

def softmax(x, axis=-1):
    # Stabilized softmax to avoid overflow
    exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exps / np.sum(exps, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    
    # 1. Compute dot product similarity scores: shape (T, T)
    scores = np.dot(Q, K.T)
    print("Raw attention scores:\n", np.round(scores, 4))
    
    # 2. Divide by scaling factor
    scaled_scores = scores / np.sqrt(d_k)
    print("\nScaled attention scores:\n", np.round(scaled_scores, 4))
    
    # 3. Apply Softmax to get attention weights
    attn_weights = softmax(scaled_scores, axis=-1)
    print("\nAttention Weights (Softmax output):\n", np.round(attn_weights, 4))
    
    # 4. Multiply by Value matrix
    output = np.dot(attn_weights, V)
    return output, attn_weights

# Define toy Query, Key, Value matrices for a 3-word sequence
# Sequence length T=3, dimension dk=dv=4
np.random.seed(42)
Q = np.random.randn(3, 4) * 2.0  # Simulating large outputs
K = np.random.randn(3, 4) * 2.0
V = np.random.randn(3, 4)

output, weights = scaled_dot_product_attention(Q, K, V)
print("\nFinal Attention Output Shape:", output.shape)
print("Final Attention Output:\n", np.round(output, 4))
```

### Trade-offs
*   **Advantages:** The Query-Key-Value projection is fully differentiable, allowing the network to learn how to match features end-to-end using gradient descent.
*   **Disadvantages:** Introduces parameter matrices $\mathbf{W}_Q$, $\mathbf{W}_K$, and $\mathbf{W}_V$, increasing parameters. However, since these projections are simple matrix multiplications, they are highly optimized on GPU tensor cores.

### Real-World Applications (Rule of 4)
1.  **Example 1: Dot Product Scaling Importance**
    *   **Input/Scenario:** We use a key dimension $d_k = 64$ ($\sqrt{d_k} = 8$). Two vectors yield a raw dot product of $q \cdot k = 24.0$.
    *   **Expected Output:** Without scaling, $e^{24}$ is extremely large, saturating the Softmax and zeroing out gradients. With scaling, the input to the softmax is $\frac{24}{8} = 3.0$, keeping the Softmax active and preserving gradients.
2.  **Example 2: Information Retrieval Search Analogy**
    *   **Input/Scenario:** A user searches for "deep learning" on YouTube.
    *   **Expected Output:** The search term is the **Query**. The database titles of videos are the **Keys**. The actual video streams are the **Values**. The search engine matches the Query to the Keys to retrieve the best Values.
3.  **Example 3: Softmax Weight Extraction**
    *   **Input/Scenario:** A sequence of length 3 yields scaled dot products $\mathbf{Q}\mathbf{K}^T / \sqrt{d_k} = \begin{pmatrix} 2.0 & 0.0 & 0.0 \end{pmatrix}$ for the first word.
    *   **Expected Output:** Softmax normalization outputs $\approx [0.79, 0.11, 0.11]$, focusing attention on Word 1.
4.  **Example 4: Tensor Dimension Matching**
    *   **Input/Scenario:** Input matrix $\mathbf{X}$ has shape $(32, 10, 512)$ (Batch, Timesteps, Embedding Dim). The projection matrices have shape $(512, 64)$.
    *   **Expected Output:** The Query and Key matrices have shape $(32, 10, 64)$. The product $\mathbf{Q}\mathbf{K}^T$ has shape $(32, 10, 10)$, representing the attention grid for each batch sample.

> **Metacognitive Checkpoint:** Why do we divide the dot product $\mathbf{Q}\mathbf{K}^T$ by $\sqrt{d_k}$ in Scaled Dot-Product Attention? Explain the relationship between high-dimensional vector dot products and Softmax gradient saturation.

---

## Topic 3: Multi-Head Attention: Extracting Diverse Feature Relationships

### Rationale and Mechanics
If we calculate attention using a single set of Query, Key, and Value matrices, the model can only focus on a single type of relationship at a time. For example, if the model attends heavily to the pronoun relationship ("it" $\to$ "animal"), it might miss the syntactic verb-object relationship.

To solve this, Transformers use **Multi-Head Attention (MHA)**. MHA splits the Queries, Keys, and Values into $h$ smaller subspaces, performs attention on each subspace in parallel, and concatenates the results.

Under the hood:
1.  We project the input embeddings $\mathbf{X}$ into $h$ sets of Queries, Keys, and Values using $h$ independent sets of projection matrices:
    $$\mathbf{Q}_i = \mathbf{X} \mathbf{W}_Q^i, \quad \mathbf{K}_i = \mathbf{X} \mathbf{W}_K^i, \quad \mathbf{V}_i = \mathbf{X} \mathbf{W}_V^i$$
    where $i \in \{1, \dots, h\}$.
2.  We calculate Scaled Dot-Product Attention for each head independently:
    $$\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i)$$
3.  We concatenate the outputs of all $h$ heads:
    $$\text{Concat}(\text{head}_1, \dots, \text{head}_h)$$
4.  We project the concatenated vector back to the original dimension using a final output projection matrix $\mathbf{W}^O$:
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

Each head initializes with different random weights, allowing them to focus on different aspects of language (e.g. grammar structure, long-range pronoun references, or semantic synonyms).

### Python Code Implementation
The following code implements Multi-Head Attention from scratch using NumPy. It projects a batch of sequences into multi-head queries, keys, and values, performs parallel scaled dot-product attention using tensor reshapes, and projects the concatenated output back to the original space.

```python
import numpy as np

def softmax(x, axis=-1):
    exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exps / np.sum(exps, axis=axis, keepdims=True)

class NumpyMultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize learnable projection matrices
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1
        
    def forward(self, X):
        batch_size, seq_len, d_model = X.shape
        
        # 1. Linear projections
        Q = np.dot(X, self.W_Q)  # Shape: (B, T, d_model)
        K = np.dot(X, self.W_K)
        V = np.dot(X, self.W_V)
        
        # 2. Reshape and transpose to split into heads: (B, num_heads, T, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 3. Scaled dot-product attention over all heads in parallel
        # Scores shape: (B, num_heads, T, T)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attn_weights = softmax(scores, axis=-1)
        
        # Output shape: (B, num_heads, T, d_k)
        head_outputs = np.matmul(attn_weights, V)
        
        # 4. Concatenate heads back: (B, T, d_model)
        concat_outputs = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # 5. Output projection
        output = np.dot(concat_outputs, self.W_O)
        return output, attn_weights

# Test the implementation
d_model, num_heads = 8, 2  # Small dimensions for demonstration
mha = NumpyMultiHeadAttention(d_model=d_model, num_heads=num_heads)

# Input X shape: (Batch=1, SeqLen=3, d_model=8)
X = np.random.randn(1, 3, 8)
output, weights = mha.forward(X)

print("Input Shape:", X.shape)
print("Output Shape (should match input shape):", output.shape)
print("Attention Weights shape (B, heads, T, T):", weights.shape)
```

### Trade-offs
*   **Advantages:** Prevents the network from averaging out all semantic relationships. It allows the model to extract diverse feature structures simultaneously.
*   **Disadvantages:** Harder to implement. Splitting tensors and transposing them requires precise tracking of shapes to compute attention efficiently on a GPU.

### Real-World Applications (Rule of 4)
1.  **Example 1: Dimension Splitting Calculation**
    *   **Input/Scenario:** A Transformer has embedding dimension $d=512$ and uses $h=8$ attention heads.
    *   **Expected Output:** The key dimension for each head is $d_k = d / h = 512 / 8 = 64$. Concatenating the 8 heads yields a vector of shape $8 \times 64 = 512$.
2.  **Example 2: Visualizing Head Roles**
    *   **Input/Scenario:** A translation model processes "The contract was signed by the company."
    *   **Expected Output:** Head 1 connects "signed" and "contract" (action-object). Head 2 connects "signed" and "company" (action-actor).
3.  **Example 3: Keras MultiHeadAttention Layer**
    *   **Input/Scenario:** A developer instantiates a MultiHeadAttention layer in Keras.
    *   **Expected Output:**
        ```python
        mha = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
        output = mha(query=x, value=x, key=x)
        ```
4.  **Example 4: Performance Bottleneck Avoidance**
    *   **Input/Scenario:** A developer compares looping through 8 heads sequentially versus using tensor transposition.
    *   **Expected Output:** The tensor transposition method runs $8\times$ faster on GPU because it executes projections for all heads as a single matrix multiplication.

---

## Topic 4: Positional Encoding: Injecting Sequence Order without Recurrence

### Rationale and Mechanics
Because the self-attention formula calculates dot products between all pairs of tokens simultaneously, it does not have any built-in concept of word order. The attention calculation for "The cat ate the fish" is identical to "The fish ate the cat."

In recurrent networks (RNNs) and convolutions, sequence order is naturally preserved by step-by-step updates or sliding kernels. To make the non-recurrent Transformer aware of order, we must add **Positional Encodings** to our input embeddings before feeding them to the attention layers.

Under the hood, we create a positional encoding matrix $\mathbf{PE}$ of the same shape as the input embeddings ($T \times d$). We compute each element of this matrix using sinusoidal functions of different frequencies:
$$\mathbf{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$\mathbf{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
where $pos$ is the token position index in the sequence, and $i$ represents the dimension index ($i \in [0, d/2-1]$).

```
         Word Embeddings X (T x d)  -----+
                                         |
                                         +---> [ Element-wise Sum ] ---> Transformer Input
                                         |
         Positional Encoding PE (T x d) -+
```

Adding these values directly to the input embeddings ensures that the resulting vectors contain both the semantic meaning of the word and its position in the sentence.

### Python Code Implementation
The following code calculates and visualizes sinusoidal Positional Encodings in NumPy, showing how positions are mapped to sine-cosine patterns.

```python
import numpy as np

def get_positional_encoding(seq_len, d_model):
    # Initialize PE matrix
    pe = np.zeros((seq_len, d_model))
    
    # Calculate pos vector shape: (seq_len, 1) and term index i shape: (d_model // 2,)
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Compute sinusoidal patterns
    pe[:, 0::2] = np.sin(position * div_term)  # Apply sine to even indices
    pe[:, 1::2] = np.cos(position * div_term)  # Apply cosine to odd indices
    
    return pe

# Test Positional Encoding construction
seq_len = 5
d_model = 6
pe_matrix = get_positional_encoding(seq_len, d_model)

print(f"Positional Encoding Matrix (T={seq_len}, d={d_model}):\n")
print(np.round(pe_matrix, 4))
```

### Trade-offs
*   **Advantages:** Preserves word order without using recurrent steps, allowing the Transformer to retain full parallelizability.
*   **Disadvantages:** Sinusoidal positional encodings are fixed and cannot adapt to specific training datasets. In modern systems, developers often use **Learnable Absolute Positional Embeddings** (which act like a second Embedding layer for position indices) or **Relative Positional Encodings** (like RoPE - Rotary Position Embeddings), which are more complex to implement but generalize better to longer sequences.

### Real-World Applications (Rule of 4)
1.  **Example 1: Order Sensitivity**
    *   **Input/Scenario:** The model processes "The dog chased the cat" versus "The cat chased the dog".
    *   **Expected Output:** Thanks to positional encoding, the token vectors for "dog" at index 1 and index 5 have different coordinates, allowing the model to identify the actor and the recipient.
2.  **Example 2: Extrapolating Sequence Lengths**
    *   **Input/Scenario:** A model trained on sequences of length 512 is fed a sequence of length 1024.
    *   **Expected Output:** Sinusoidal encodings naturally generalize to longer sequences because sine/cosine functions are defined for all values, unlike learnable positional encodings which fail on unseen index ranges.
3.  **Example 3: Rotary Positional Embeddings (RoPE)**
    *   **Input/Scenario:** A modern LLM (like Llama) uses Rotary Position Embeddings to process text.
    *   **Expected Output:** The model encodes positions by rotating Query and Key vectors in 2D planes, which preserves the relative distance between words better than additive sinusoidal constants.
4.  **Example 4: Adding Embeddings**
    *   **Input/Scenario:** Input embeddings $\mathbf{X}$ has shape $(1, 10, 512)$ and Positional Encoding $\mathbf{PE}$ has shape $(10, 512)$.
    *   **Expected Output:** Summing them `X + PE` uses NumPy broadcasting to yield a shape of $(1, 10, 512)$, successfully embedding positional information.

> **Metacognitive Checkpoint:** Why are sinusoidal functions chosen for positional encoding instead of simply appending the raw integers `[0, 1, 2, ...]` to the embeddings? Explain in terms of scale stability.

---

## Summary & Next Steps

*   **Self-Attention Computes Context:** Self-attention allows every word in a sequence to connect directly to every other word, calculating dynamic, context-aware embeddings in parallel.
*   **Q, K, V Project Meaning:** Projecting inputs into Queries, Keys, and Values allows similarity to be calculated using Scaled Dot-Product Attention, protecting gradients from Softmax saturation.
*   **Multi-Head Attention Extracts Variety:** Splitting projections into multiple heads allows the model to track different linguistic relationships (grammar, pronouns, subjects) simultaneously.
*   **Positional Encodings Restore Order:** Because attention does not process words sequentially, we add positional sine-cosine waves to restore order sensitivity.

In the next lesson, we will explore **Generative AI Foundations**, learning how self-attention scales to build Large Language Models, and exploring causal language modeling and generation parameters.
