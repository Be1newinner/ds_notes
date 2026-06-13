# Lesson 22: Generative AI Foundations

## Introduction & The "Why"

In Lesson 21, we explored the Self-Attention mechanism and saw how it allows models to capture bidirectional context across a sequence in parallel. This mechanism is the foundation of the Transformer architecture, which has replaced recurrent networks in natural language processing.

But how do we use this architecture to generate text, write code, or act as a digital assistant? This capability is unlocked by **Large Language Models (LLMs)**, which are decoder-only Transformers trained at scale.

A pre-trained LLM is essentially a highly advanced autocomplete engine. It is trained on the simple task of **Causal Language Modeling**—predicting the next word in a sequence given the words that came before it. By scaling this simple task to billions of parameters and training on terabytes of web text, the model develops emergent reasoning capabilities. This lesson covers the mechanics of next-token prediction, explains how causal masking enables parallel training, details the parameters that control text generation, and explores SFT and RLHF alignment.

---

## Topic 1: Causal Language Modeling: Predicting the Next Token

### Rationale and Mechanics
In classical natural language processing, language was modeled using Markov chains or N-gram models. An N-gram model calculates the probability of a word based on the frequencies of the preceding $N-1$ words. While fast, N-grams cannot capture long-range dependencies: a 3-gram model forgets the beginning of a sentence after only 3 words.

A **Causal Language Model** (such as GPT-3 or GPT-4) uses the Transformer decoder architecture to model language. The goal is to learn the joint probability distribution of a sequence of tokens $w_1, \dots, w_T$. Using the chain rule of probability, we decompose this joint probability into a product of conditional probabilities:
$$P(w_1, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

The model is trained to maximize this conditional probability across a massive text corpus using the Cross-Entropy loss at each timestep:
$$\mathcal{L} = -\sum_{t=1}^T \log P(w_t \mid w_1, \dots, w_{t-1})$$

During training, we want to calculate the loss for all timesteps in parallel to utilize the GPU. However, standard self-attention allows tokens to look both backward and forward. If token $t$ is allowed to attend to token $t+1$, it can see the correct answer, preventing the network from learning.

To prevent this "cheating," we apply a **Causal Mask** to the attention scores.

Under the hood, we add a mask matrix $\mathbf{M}$ to the scaled dot-product attention before applying the Softmax function:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V}$$

The mask matrix $\mathbf{M}$ is a lower-triangular matrix filled with $0.0$ and negative infinity ($-\infty$):
$$M(i, j) = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

```
       Causal Mask Matrix M:
       [   0  -inf  -inf  -inf ] - Token 1 (Can only attend to Token 1)
       [   0     0  -inf  -inf ] - Token 2 (Can attend to Token 1, 2)
       [   0     0     0  -inf ] - Token 3 (Can attend to Token 1, 2, 3)
       [   0     0     0     0 ] - Token 4 (Can attend to all tokens)
```

Since the Softmax function exponentiates the scaled scores, the terms containing $-\infty$ become zero ($e^{-\infty} = 0$). This mathematically blocks any attention weights from flowing from future tokens, preserving the causal property during parallel training.

### Python Code Implementation
The following code implements causal masking within scaled dot-product attention in NumPy, demonstrating how future tokens are zeroed out after softmax.

```python
import numpy as np

def masked_softmax(x, mask=None):
    # If a mask is provided, add it before exponentiating
    if mask is not None:
        x = x + mask
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

# Define sequence parameters (T=4, d_k=2)
T = 4
d_k = 2
np.random.seed(42)

Q = np.random.randn(T, d_k)
K = np.random.randn(T, d_k)

# 1. Compute raw similarity scores
scores = np.dot(Q, K.T) / np.sqrt(d_k)
print("Raw similarity scores (unmasked):\n", np.round(scores, 4))

# 2. Build the lower triangular causal mask matrix M
# Filled with 0 on and below diagonal, and -infinity above diagonal
mask = np.zeros((T, T))
mask[np.triu_indices(T, k=1)] = -np.inf
print("\nCausal Mask Matrix M:\n", mask)

# 3. Apply masked softmax
attn_weights_masked = masked_softmax(scores, mask=mask)
print("\nAttention Weights (Masked Softmax output):\n", np.round(attn_weights_masked, 4))

# Verification: Row 0 must only attend to index 0, Row 1 only to indices 0 & 1, etc.
```

### Trade-offs
*   **Advantages:** Causal masking allows the model to calculate gradients and losses for all tokens in a sequence simultaneously during training, enabling massive parallelization.
*   **Disadvantages:** During inference (generation), we cannot parallelize. The model must generate tokens **auto-regressively**: it must predict token $t$, append it to the input sequence, and pass the updated sequence back to the model to predict token $t+1$. This step-by-step loop is slow and computationally expensive.

### Real-World Applications (Rule of 4)
1.  **Example 1: Causal Masking Calculation**
    *   **Input/Scenario:** Raw attention score at row 0, column 1 is 5.0. We apply a causal mask value of $-\infty$.
    *   **Expected Output:** The masked score is $-\infty$. Softmax converts this to exactly 0.0 attention weight, preventing token 0 from looking ahead at token 1.
2.  **Example 2: Auto-regressive Generation Loop**
    *   **Input/Scenario:** A model starts generating text with the prompt "Artificial intelligence is".
    *   **Expected Output:**
        *   Step 1: Input "Artificial intelligence is" $\to$ Predicts "changing"
        *   Step 2: Input "Artificial intelligence is changing" $\to$ Predicts "the"
        *   Step 3: Input "Artificial intelligence is changing the" $\to$ Predicts "world"
3.  **Example 3: Autocomplete Engines**
    *   **Input/Scenario:** A smartphone keyboard predicts the next word as a user types.
    *   **Expected Output:** The keyboard runs a small causal language model, outputting the most likely next tokens based on the current sentence context.
4.  **Example 4: KV Caching**
    *   **Input/Scenario:** A developer notices that auto-regressive generation slows down as the generated text becomes longer.
    *   **Expected Output:** Enabling Key-Value (KV) Caching stores past keys and values instead of recalculating them at every step, speeding up inference.

> **Metacognitive Checkpoint:** Why is causal masking necessary during the training of decoder-only Transformers? What would happen to the model's ability to generate text auto-regressively if we did not use a causal mask during training?

---

## Topic 2: Decoding Parameters: Temperature, Top-K, and Top-P

### Rationale and Mechanics
When a causal language model predicts the next token, it outputs a vector of raw logits $\mathbf{z}$ over the entire vocabulary size $V$. Passing these logits through Softmax yields a probability distribution $\mathbf{p}$:
$$p_i = \text{Softmax}(z_i)$$

If we always choose the token with the highest probability, a technique called **Greedy Search**, the model will generate highly repetitive, dry, and predictable text. It can also get stuck in infinite loops.

To generate natural, human-like text, we sample from the probability distribution, controlling the randomness using three decoding parameters: **Temperature**, **Top-K**, and **Top-P (Nucleus Sampling)**.

Under the hood:
1.  **Temperature ($T > 0$):** Scales the logits before applying the Softmax function:
    $$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
    *   **Low Temperature ($T \to 0$):** Exaggerates differences between logits. The highest probability token dominates, approaching greedy search. Used for factual tasks.
    *   **High Temperature ($T > 1.0$):** Flattens the distribution, making all tokens more equally likely. This increases creativity but also increases grammatical errors.
2.  **Top-K:** Restricts the sampling pool to the $K$ most likely tokens. The remaining tokens are discarded, and the probabilities of the $K$ tokens are re-normalized to sum to 1.0. This prevents the model from sampling highly incorrect words.
3.  **Top-P (Nucleus Sampling):** Restricts the sampling pool dynamically. It selects the smallest set of tokens whose cumulative probability exceeds a threshold $p$ (typically $0.9$).

```
       Logit Vector z ---> [ Scale by 1/T ] ---> [ Top-K Filter ] ---> [ Top-P Filter ] ---> Softmax & Sample
```

### Python Code Implementation
The following code implements a complete sampling function in Python that supports Temperature scaling, Top-K filtering, and Top-P filtering on logits.

```python
import numpy as np

def sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.0):
    # 1. Apply Temperature Scaling
    if temperature > 0:
        logits = logits / temperature
    
    # 2. Apply Top-K Filtering
    if top_k > 0:
        # Find threshold value of top_k logit
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = -np.inf
        
    # 3. Apply Top-P (Nucleus) Filtering
    if top_p > 0.0 and top_p < 1.0:
        # Sort logits and their original indices in descending order
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        
        # Calculate Softmax probabilities of sorted logits
        sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
        sorted_probs /= np.sum(sorted_probs)
        
        # Compute cumulative probabilities
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Identify indices to exclude (cumulative probability exceeds top_p)
        # We want to keep the first token that pushes us past threshold, so shift indices
        exclude_sorted_indices = cumulative_probs > top_p
        exclude_sorted_indices[1:] = exclude_sorted_indices[:-1].copy()
        exclude_sorted_indices[0] = False
        
        # Mask out excluded logits
        indices_to_remove = sorted_indices[exclude_sorted_indices]
        logits[indices_to_remove] = -np.inf

    # 4. Final Softmax over filtered logits
    probs = np.exp(logits - np.max(logits))
    probs /= np.sum(probs)
    
    # 5. Sample token index from distribution
    vocab_indices = np.arange(len(logits))
    sampled_idx = np.random.choice(vocab_indices, p=probs)
    return sampled_idx, probs

# Toy setup: vocabulary of 6 words
vocab = ["the", "cat", "sat", "apple", "banana", "spaceship"]
# raw outputs from language model
raw_logits = np.array([4.0, 3.5, 3.0, 1.0, 0.5, -2.0])

np.random.seed(42)

# Test low temperature (more focused) vs high temperature (more creative)
_, probs_low = sample_next_token(raw_logits.copy(), temperature=0.2)
_, probs_high = sample_next_token(raw_logits.copy(), temperature=1.5)

print("Vocabulary:", vocab)
print("\nProbabilities (T=0.2):", np.round(probs_low, 4))
print("Probabilities (T=1.5):", np.round(probs_high, 4))

# Test Top-P (Nucleus Sampling) with top_p = 0.90
_, probs_topp = sample_next_token(raw_logits.copy(), temperature=1.0, top_p=0.90)
print("\nProbabilities (Top-P = 0.90):", np.round(probs_topp, 4))
# Note that words like 'spaceship' are filtered out (probabilities set to 0)
```

### Trade-offs
*   **Advantages:** Tuning decoding parameters allows you to balance precision and creativity.
*   **Disadvantages:** High temperatures increase the rate of **Hallucinations**—where the model confidently generates false facts because it was forced to sample a lower-probability token.

### Real-World Applications (Rule of 4)
1.  **Example 1: Temperature Scaling Calculation**
    *   **Input/Scenario:** The top two logits for a prediction are $z_1 = 4.0$ ("yes") and $z_2 = 2.0$ ("no").
    *   **Expected Output:** At $T=0.5$ (Low), Softmax yields $P(\text{"yes"}) \approx 0.982$. At $T=2.0$ (High), Softmax yields $P(\text{"yes"}) \approx 0.731$, increasing the chance of choosing "no".
2.  **Example 2: Top-K Filtering**
    *   **Input/Scenario:** We set `top_k=3`. The top three tokens have probabilities $[0.6, 0.2, 0.1]$, and the fourth has $0.05$.
    *   **Expected Output:** The fourth token and all subsequent words are discarded, restricting sampling to the top 3 options.
3.  **Example 3: Top-P Dynamic Sizing**
    *   **Input/Scenario:** We use `top_p=0.9`. The model predicts the next word after "The capital of France is". The token "Paris" has probability $0.95$.
    *   **Expected Output:** Since $0.95 > 0.9$, the sampling pool contains only "Paris," preventing incorrect capitals from being chosen.
4.  **Example 4: Repetition Loop Escape**
    *   **Input/Scenario:** A model using greedy search enters a loop: "I went to the store and the store and the store...".
    *   **Expected Output:** The developer sets $T = 0.7$ and `top_p=0.9`. The stochastic sampling breaks the loop by choosing an alternative word.

> **Metacognitive Checkpoint:** Why does setting the temperature parameter $T > 1.0$ increase the "creativity" of an LLM's output? Describe the effect of high temperature on the entropy of the output probability distribution.

---

## Topic 3: Fine-Tuning and Alignment: SFT & RLHF

### Rationale and Mechanics
Pre-training on raw web text only teaches a model to perform next-token prediction. If you prompt a pre-trained model with "Write a python function to sort a list," it might reply with "Write a java function to reverse a string," because it is autocompleting a list of coding prompts it saw on the web. It does not know it is supposed to act as an assistant.

To transform a raw base model into a helpful assistant, we must perform **Alignment**. This consists of two main stages: **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning from Human Feedback (RLHF)**.

Under the hood:
1.  **Supervised Fine-Tuning (SFT):** The model is trained on a curated dataset of high-quality instruction-answer pairs (e.g., Query: "Explain gravity," Answer: "Gravity is a force..."). We apply standard cross-entropy loss only on the answer tokens, teaching the model to follow instructions.
2.  **Reinforcement Learning from Human Feedback (RLHF):** SFT models can still generate toxic or incorrect answers. RLHF aligns the model with human preferences:
    *   **Train a Reward Model ($R_\phi$):** Humans rank multiple model outputs for a single prompt. A reward model is trained to output a high score for helpful answers and a low score for toxic/unhelpful answers.
    *   **Optimize the Policy ($\pi_\theta$):** We update the language model weights using Reinforcement Learning (typically PPO - Proximal Policy Optimization). The model generates answers, receives a score from the reward model, and updates its weights to maximize the score.

```
       Base Model ---> [ SFT: Instruction dataset ] ---> aligned SFT Model
                             |
                             v
       [ RLHF: Reward Model Optimization ] ---> Final Aligned Assistant Model (Chatbot)
```

To prevent the RLHF model from drifting too far from the base model and generating gibberish, we add a **Kullback-Leibler (KL) Divergence** penalty to the objective function:
$$\text{Objective}(\theta) = \mathbb{E}_{(x, y)} [R_\phi(x, y)] - \beta D_{\text{KL}}\left( \pi_\theta(y|x) \parallel \pi_{\text{SFT}}(y|x) \right)$$
where $\pi_{\text{SFT}}$ is the SFT model weights and $\beta$ is a scaling hyperparameter.

### Python Code Implementation
The following code simulates Supervised Fine-Tuning loss calculation. It shows how prompt tokens are masked out (loss weights set to 0.0) so that gradient updates are computed only on the response tokens.

```python
import numpy as np

def compute_sft_loss(logits, targets, loss_mask):
    # Logits shape: (Sequence_Length, Vocab_Size)
    # Targets shape: (Sequence_Length,) - integer token IDs
    # Loss_mask shape: (Sequence_Length,) - 1.0 for response, 0.0 for prompt
    
    seq_len, vocab_size = logits.shape
    
    # 1. Stable Softmax over logits to get probabilities
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    
    # 2. Extract probability of target tokens
    target_probs = probs[np.arange(seq_len), targets]
    
    # 3. Calculate Cross-Entropy Loss: -log(p)
    # Add a small epsilon to prevent log(0)
    ce_loss = -np.log(target_probs + 1e-15)
    
    # 4. Apply Loss Mask (Ignore prompt tokens)
    masked_loss = ce_loss * loss_mask
    
    # Average loss over response tokens only
    final_loss = np.sum(masked_loss) / np.sum(loss_mask)
    return final_loss, ce_loss, masked_loss

# Simulate a sequence: prompt has 3 tokens, response has 2 tokens (T=5, V=4)
logits = np.array([
    [2.0, 0.5, 0.1, -1.0], # Prompt token 1
    [1.0, 3.0, 0.0, 0.2],  # Prompt token 2
    [0.5, 0.5, 2.0, 0.1],  # Prompt token 3
    [0.1, 4.0, 0.5, 0.2],  # Response token 1 (target is index 1)
    [3.0, 1.0, 0.5, 0.1]   # Response token 2 (target is index 0)
])
targets = np.array([0, 1, 2, 1, 0])
# Mask out prompt: first 3 tokens are 0.0, next 2 tokens are 1.0
loss_mask = np.array([0.0, 0.0, 0.0, 1.0, 1.0])

final_loss, raw_ce, masked_ce = compute_sft_loss(logits, targets, loss_mask)

print("Raw Cross-Entropy per token:\n", np.round(raw_ce, 4))
print("\nMasked Cross-Entropy (Prompt zeroed out):\n", np.round(masked_ce, 4))
print(f"\nFinal SFT Loss (Average over response tokens): {final_loss:.4f}")
```

### Trade-offs
*   **Advantages:** Alignment transforms next-token predictors into helpful chatbots, reducing toxic outputs, improving safety, and enabling conversational interfaces.
*   **Disadvantages:** Aligned models often exhibit a drop in raw capabilities (such as mathematical reasoning or coding accuracy) compared to the base model. This is known as the **Alignment Tax**. Additionally, RLHF is complex, expensive, and sensitive to bias in the reward model.

### Real-World Applications (Rule of 4)
1.  **Example 1: SFT Instruction Masking**
    *   **Input/Scenario:** An SFT sample is: `"[PROMPT] How are you? [RESPONSE] I am doing well."`
    *   **Expected Output:** During backpropagation, the loss function masks out the prompt tokens. The model only calculates gradients for the response tokens `"I", "am", "doing", "well"`.
2.  **Example 2: Reward Hacking Prevention**
    *   **Input/Scenario:** During RLHF, the model discovers it can maximize reward by outputting a specific sequence of punctuation marks that confuses the reward model.
    *   **Expected Output:** The KL divergence penalty detects that the output distribution has drifted away from the base SFT distribution, stabilizing training.
3.  **Example 3: Safe Chatbot Deployments**
    *   **Input/Scenario:** A user prompts a customer support chatbot: "How do I hack a bank?"
    *   **Expected Output:** Because of RLHF safety alignment, the model refuses the request.
4.  **Example 4: Instruction Following**
    *   **Input/Scenario:** We prompt a base model versus an SFT-aligned model with: "Translate this to French: Apple".
    *   **Expected Output:** The base model outputs: "Translate this to Spanish: Manzana...". The SFT model outputs: "Pomme".

> **Metacognitive Checkpoint:** What is the "alignment tax" in Large Language Models? Explain why optimizing a model to be helpful and safe can sometimes degrade its performance on complex reasoning tasks.

---

## Summary & Next Steps

*   **Causal Modeling Autocompletes Text:** Decoder-only Transformers are trained to predict the next token in a sequence, using a causal mask to prevent tokens from looking ahead.
*   **Decoding Parameters Control Randomness:** Temperature scales logits to adjust predictability, while Top-K and Top-P restrict the sampling pool to prevent gibberish.
*   **Alignment Enables Interaction:** Supervised Fine-Tuning teaches the model to follow instructions, and RLHF aligns the outputs with human preferences using reward models.

In the next module, we will explore **Module 6: Unsupervised Architectures & Capstone**, starting with **Lesson 23: Unsupervised Learning with Autoencoders** to learn how to compress data into latent spaces.
