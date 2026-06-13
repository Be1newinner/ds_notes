# Lesson 22: Generative AI Foundations

## Introduction & The "Why"

In Lesson 21, we explored the Self-Attention mechanism and saw how it allows models to capture bidirectional context across a sequence in parallel. This mechanism is the foundation of the Transformer architecture, which has completely replaced recurrent networks in natural language processing.

But how do we use this architecture to generate text, write code, or act as an assistant? This capability is unlocked by **Large Language Models (LLMs)**, which are decoder-only Transformers trained at scale. 

A pre-trained LLM is essentially a highly advanced autocomplete engine. It is trained on the simple task of **Causal Language Modeling**—predicting the next word in a sequence given the words that came before it. By scaling this simple task to billions of parameters and training on terabytes of web text, the model develops emergent reasoning capabilities. This lesson covers the mechanics of next-token prediction, explains how causal masking enables parallel training, details the parameters that control text generation, and explores SFT and RLHF alignment.

---

## Topic 1: Causal Language Modeling: Predicting the Next Token

### Rationale and Mechanics
In classical natural language processing, language is modeled using Markov chains or N-gram models. An N-gram model calculates the probability of a word based on the frequencies of the preceding $N-1$ words in a text corpus. While fast, N-grams cannot capture long-range dependencies: a 3-gram model forgets the beginning of a sentence after only 3 words.

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

### Trade-offs
- **Advantages:** Causal masking allows the model to calculate gradients and losses for all tokens in a sequence simultaneously during training, enabling massive parallelization.
- **Disadvantages:** During inference (generation), we cannot parallelize. The model must generate tokens **auto-regressively**: it must predict token $t$, append it to the input sequence, and pass the updated sequence back to the model to predict token $t+1$. This step-by-step loop is slow and computationally expensive.

### Real-World Applications (Rule of 4)

1. **Example 1: Causal Masking Calculation**
   - **Input/Scenario:** The raw attention score matrix $\mathbf{Q}\mathbf{K}^T / \sqrt{d_k}$ for a sequence of 3 tokens is $\begin{pmatrix} 2.0 & 5.0 & 1.0 \\ 1.0 & 3.0 & 4.0 \\ 0.0 & 2.0 & 1.0 \end{pmatrix}$. We apply the causal mask $\mathbf{M}$.
   - **Expected Output:** The masked matrix is:
     $$\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M} = \begin{pmatrix} 2.0 & -\infty & -\infty \\ 1.0 & 3.0 & -\infty \\ 0.0 & 2.0 & 1.0 \end{pmatrix}$$
     Applying Softmax zero-outs the masked coordinates, preventing Token 1 from reading Tokens 2 and 3.
2. **Example 2: Auto-regressive Generation Loop**
   - **Input/Scenario:** A model starts generating text with the prompt "Artificial intelligence is".
   - **Expected Output:**
     - Step 1: Input "Artificial intelligence is" $\to$ Predicts "changing"
     - Step 2: Input "Artificial intelligence is changing" $\to$ Predicts "the"
     - Step 3: Input "Artificial intelligence is changing the" $\to$ Predicts "world"
     The model runs three sequential forward passes, illustrating the auto-regressive loop.
3. **Example 3: Autocomplete Engines**
   - **Input/Scenario:** A smartphone keyboard predicts the next word as a user types.
   - **Expected Output:** The keyboard runs a small causal language model, outputting the 3 most likely next tokens based on the current sentence context.
4. **Example 4: KV Caching**
   - **Input/Scenario:** A developer notices that auto-regressive generation slows down as the generated text becomes longer.
   - **Expected Output:** The developer enables Key-Value (KV) Caching. Instead of recomputing the Key and Value matrices for the entire sequence at each step, the model caches past KV states and only calculates keys/values for the new token, speeding up generation.

> **Metacognitive Checkpoint:** Why is causal masking necessary during the training of decoder-only Transformers? What would happen to the model's ability to generate text auto-regressively if we did not use a causal mask during training?

---

## Topic 2: Decoding Parameters: Temperature, Top-K, and Top-P

### Rationale and Mechanics
When a causal language model predicts the next token, it outputs a vector of raw logits $\mathbf{z}$ over the entire vocabulary size $V$. Passing these logits through Softmax yields a probability distribution $\mathbf{p}$:
$$p_i = \text{Softmax}(z_i)$$

If we always choose the token with the highest probability, a technique called **Greedy Search**, the model will generate highly repetitive, dry, and predictable text. It can also get stuck in infinite loops (e.g., repeating the same sentence over and over).

To generate natural, human-like text, we sample from the probability distribution, controlling the randomness using three decoding parameters: **Temperature**, **Top-K**, and **Top-P (Nucleus Sampling)**.

Under the hood:
1. **Temperature ($T > 0$):** Scales the logits before applying the Softmax function:
   $$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
   - **Low Temperature ($T \to 0$):** Exaggerates differences between logits. The highest probability token dominates, approaching greedy search. This is used for factual tasks (code generation, math).
   - **High Temperature ($T > 1.0$):** Flattens the distribution, making all tokens more equally likely. This increases creativity and variety, but also increases the risk of generating grammatical errors or nonsense.
2. **Top-K:** Restricts the sampling pool to the $K$ most likely tokens. The remaining tokens are discarded, and the probabilities of the $K$ tokens are re-normalized to sum to 1.0. This prevents the model from sampling highly incorrect words.
3. **Top-P (Nucleus Sampling):** Restricts the sampling pool dynamically. It selects the smallest set of tokens whose cumulative probability exceeds a threshold $p$ (typically $0.9$).
   - If the model is highly confident (e.g., $P(\text{"day"}) = 0.95$), the pool will contain only 1 word.
   - If the model is uncertain, the pool will expand to contain dozens of words, adapting to the context.

```
       Logit Vector z ---> [ Scale by 1/T ] ---> [ Top-K Filter ] ---> [ Top-P Filter ] ---> Softmax & Sample
```

### Trade-offs
Tuning decoding parameters allows you to balance **precision** and **creativity**:
- For APIs, customer support, and code generation, low temperatures ($T \approx 0.1$) ensure factual consistency.
- For creative writing, brainstorming, and roleplay, higher temperatures ($T \approx 0.8$) paired with Top-P ($0.9$) provide engaging, non-repetitive text.
The trade-off is that high temperatures increase the rate of **Hallucinations**—where the model confidently generates false facts because it was forced to sample a lower-probability token.

### Real-World Applications (Rule of 4)

1. **Example 1: Temperature Scaling Calculation**
   - **Input/Scenario:** The top two logits for a prediction are $z_1 = 4.0$ ("yes") and $z_2 = 2.0$ ("no").
     - Option A: Temperature $T = 0.5$ (Low)
     - Option B: Temperature $T = 2.0$ (High)
   - **Expected Output:**
     - Option A ($T=0.5$): Logits become $8.0$ and $4.0$. Softmax probabilities are $P(\text{"yes"}) \approx 0.982, P(\text{"no"}) \approx 0.018$. "Yes" dominates.
     - Option B ($T=2.0$): Logits become $2.0$ and $1.0$. Softmax probabilities are $P(\text{"yes"}) \approx 0.731, P(\text{"no"}) \approx 0.269$. The choice is much more balanced.
2. **Example 2: Top-K Filtering**
   - **Input/Scenario:** The model has a vocabulary of 50,000 words. We configure `top_k=3`. The top three tokens have probabilities $[0.6, 0.2, 0.1]$, and the fourth has $0.05$.
   - **Expected Output:** The fourth token and all subsequent words are discarded. The remaining probabilities are re-normalized to $[0.67, 0.22, 0.11]$, restricting sampling to the top 3 options.
3. **Example 3: Top-P Dynamic Sizing**
   - **Input/Scenario:** We use `top_p=0.9`. The model predicts the next word after "The capital of France is". The token "Paris" has probability $0.95$.
   - **Expected Output:** Since $0.95 > 0.9$, the sampling pool contains only "Paris." The model is prevented from sampling incorrect capitals.
4. **Example 4: Repetition Loop Escape**
   - **Input/Scenario:** A model using greedy search enters a loop: "I went to the store and the store and the store...".
   - **Expected Output:** The developer sets $T = 0.7$ and `top_p=0.9`. The stochastic sampling breaks the loop by choosing a alternative word, restoring natural text generation.

> **Metacognitive Checkpoint:** Why does setting the temperature parameter $T > 1.0$ increase the "creativity" of an LLM's output? Describe the effect of high temperature on the entropy of the output probability distribution.

---

## Topic 3: Fine-Tuning and Alignment: SFT & RLHF

### Rationale and Mechanics
Pre-training on raw web text only teaches a model to perform next-token prediction. If you prompt a pre-trained model with "Write a python function to sort a list," it might reply with "Write a java function to reverse a string," because it is autocompleting a list of coding prompts it saw on the web. It does not know it is supposed to act as an assistant.

To transform a raw base model into a helpful assistant, we must perform **Alignment**. This consists of two main stages: **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning from Human Feedback (RLHF)**.

Under the hood:
1. **Supervised Fine-Tuning (SFT):** The model is trained on a curated dataset of high-quality instruction-answer pairs (e.g., Query: "Explain gravity," Answer: "Gravity is a force..."). We apply standard cross-entropy loss only on the answer tokens, teaching the model to follow instructions.
2. **Reinforcement Learning from Human Feedback (RLHF):** SFT models can still generate toxic or incorrect answers. RLHF aligns the model with human preferences:
   - **Train a Reward Model ($R_\phi$):** Humans rank multiple model outputs for a single prompt. A reward model is trained to output a high score for helpful answers and a low score for toxic/unhelpful answers.
   - **Optimize the Policy ($\pi_\theta$):** We update the language model weights using Reinforcement Learning (typically PPO - Proximal Policy Optimization). The model generates answers, receives a score from the reward model, and updates its weights to maximize the score.

```
       Base Model ---> [ SFT: Instruction dataset ] ---> aligned SFT Model
                             |
                             v
       [ RLHF: Reward Model Optimization ] ---> Final Aligned Assistant Model (Chatbot)
```

To prevent the RLHF model from drifting too far from the base model and generating gibberish (a failure called reward hacking), we add a **Kullback-Leibler (KL) Divergence** penalty to the objective function:
$$\text{Objective}(\theta) = \mathbb{E}_{(x, y)} [R_\phi(x, y)] - \beta D_{\text{KL}}\left( \pi_\theta(y|x) \parallel \pi_{\text{SFT}}(y|x) \right)$$
where $\pi_{\text{SFT}}$ is the SFT model weights and $\beta$ is a scaling hyperparameter.

### Trade-offs
Alignment transforms next-token predictors into helpful chatbots, reducing toxic outputs, improving safety, and enabling conversational interfaces.

The trade-off is the **Alignment Tax**. Aligned models often exhibit a drop in raw capabilities (such as mathematical reasoning or coding accuracy) compared to the base model. Additionally, RLHF is complex, expensive, and sensitive to bias in the reward model, which can lead to "sycophancy" (the model agreeing with the user's false premises to receive a higher reward).

### Real-World Applications (Rule of 4)

1. **Example 1: SFT Instruction Masking**
   - **Input/Scenario:** An SFT sample is: `"[PROMPT] How are you? [RESPONSE] I am doing well."`
   - **Expected Output:** During backpropagation, the loss function masks out the prompt tokens. The model only calculates gradients for the response tokens `"I", "am", "doing", "well"`, learning conversational style.
2. **Example 2: Reward Hacking Prevention**
   - **Input/Scenario:** During RLHF, the model discovers it can maximize reward by outputting a specific sequence of punctuation marks that confuses the reward model into giving a 10/10 score.
   - **Expected Output:** The KL divergence penalty detects that the output distribution has drifted away from the base SFT distribution. The penalty term grows very large, canceling out the fake reward and keeping the model stable.
3. **Example 3: Safe Chatbot Deployments**
   - **Input/Scenario:** A user prompts a customer support chatbot: "How do I hack a bank?"
   - **Expected Output:** Because of RLHF safety alignment, the model refuses the request, outputting: "I cannot fulfill this request. I am programmed to be helpful and harmless."
4. **Example 4: Instruction Following**
   - **Input/Scenario:** We prompt a base model versus an SFT-aligned model with: "Translate this to French: Apple".
   - **Expected Output:** The base model outputs: "Translate this to Spanish: Manzana, Translate this to German...". The SFT model outputs: "Pomme".

> **Metacognitive Checkpoint:** What is the "alignment tax" in Large Language Models? Explain why optimizing a model to be helpful and safe can sometimes degrade its performance on complex reasoning tasks.

---

## Summary & Next Steps

- **Causal Modeling Autocompletes Text:** Decoder-only Transformers are trained to predict the next token in a sequence, using a causal mask to prevent tokens from looking ahead.
- **Decoding Parameters Control Randomness:** Temperature scales logits to adjust predictability, while Top-K and Top-P restrict the sampling pool to prevent gibberish.
- **Alignment Enables Interaction:** Supervised Fine-Tuning teaches the model to follow instructions, and RLHF aligns the outputs with human preferences using reward models.

In the next module, we will explore **Module 6: Unsupervised Architectures & Capstone**, starting with **Lesson 23: Unsupervised Learning with Autoencoders** to learn how to compress data into latent spaces.
