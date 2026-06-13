## Deep Learning Mechanics for Data Scientists

### Module 1: The First Principles of Neural Networks

_Goal: Bridge classical ML to deep learning math and build intuition for how networks actually learn._

- **Lesson 1: The Perceptron & The XOR Problem:** Mapping classical logistic regression to a single neuron, and proving why a single neuron fails at non-linear data.
- **Lesson 2: The Multi-Layer Perceptron (MLP):** Stacking layers. Understanding the exact mathematical role of weights and biases in hidden layers.
- **Lesson 3: Activation Dynamics:** Why linear networks collapse. A deep dive into Sigmoid, Tanh, and the shift to ReLU and its variants (Leaky ReLU, GeLU) to solve the "dead neuron" problem.
- **Lesson 4: Loss Landscapes & Objective Functions:** Connecting Mean Squared Error and Categorical Cross-Entropy to neural network optimization. How geometry dictates model convergence.
- **Lesson 5: Backpropagation & Optimizers:** Demystifying the chain rule. We explore how Stochastic Gradient Descent (SGD) works and why the industry relies on adaptive optimizers like Adam and RMSprop.

### Module 2: Applied Network Engineering (Keras Basics)

_Goal: Translate theory into code while mastering the art of preventing neural network overfitting._

- **Lesson 6: The Keras Sequential API:** Building the first MLP. Translating Scikit-Learn data prep into tensor shapes and executing `model.fit()`.
- **Lesson 7: The Overparameterization Problem (Regularization):** Neural networks will memorize data if allowed. Teaching Dropout layers, L1/L2 weight decay, and Early Stopping.
- **Lesson 8: Internal Covariate Shift & Normalization:** Why deep networks stall during training. Implementing and understanding Batch Normalization and Layer Normalization.
- **Lesson 9: The Keras Functional API:** Moving beyond straight lines. How to build networks with multiple inputs (e.g., predicting house price using both tabular data and an image) or multiple outputs.

### Module 3: Conquering Spatial Data (Computer Vision)

_Goal: Understand automated spatial feature extraction and the shift to transfer learning._

- **Lesson 10: The Spatial Hierarchy & Convolutions:** Why MLPs fail on images. Understanding spatial filters, stride, and padding in 2D Convolutions.
- **Lesson 11: Pooling & Translational Invariance:** How MaxPooling reduces dimensionality while ensuring the network recognizes a cat regardless of where it sits in the frame.
- **Lesson 12: Data Augmentation & Robustness:** Using Keras preprocessing layers to artificially expand datasets (rotations, zooms, flips) to force the model to learn generalized features.
- **Lesson 13: Transfer Learning & Pre-trained Backbones:** Why train from scratch? Importing ResNet50/EfficientNet, freezing convolutional bases, and training custom dense heads for specific business tasks.
- **Lesson 14: Explainable AI for Vision (Grad-CAM):** Opening the black box. Generating heatmaps to prove _why_ the CNN made its prediction to business stakeholders.

### Module 4: Sequential Data (Time-Series & RNNs)

_Goal: Mastering chronological data, where past context dictates future predictions._

- **Lesson 15: Time-Series Preparation & Vanilla RNNs:** Structuring 3D tensors for sequential data (Samples, Timesteps, Features). The mathematical structure of a Recurrent Neural Network.
- **Lesson 16: Backpropagation Through Time (BPTT):** The systemic flaw of Vanilla RNNs. Understanding the "Vanishing Gradient" problem that causes models to "forget" early sequence data.
- **Lesson 17: Long Short-Term Memory (LSTM) Networks:** How LSTMs solve the vanishing gradient. A granular look at the Forget, Input, and Output memory gates.
- **Lesson 18: Gated Recurrent Units (GRUs) & 1D Convolutions:** A lightweight alternative to LSTMs. We also introduce Conv1D as a highly efficient alternative for fast sequence processing.

### Module 5: Text, Semantics, and Transformers

_Goal: Moving from word frequencies to continuous vector spaces and attention mechanisms._

- **Lesson 19: Vectorizing Semantics:** Moving beyond classical ML's Bag-of-Words. Teaching Tokenization and the geometry of dense Embedding layers.
- **Lesson 20: The Sequence-to-Sequence Bottleneck:** Why LSTMs are too slow for modern NLP. The limitations of processing text one word at a time.
- **Lesson 21: The Self-Attention Mechanism:** The core of the Transformer. How mathematical Queries, Keys, and Values allow a network to look at an entire sentence simultaneously.
- **Lesson 22: Generative AI Foundations:** How Large Language Models work. Understanding causal language modeling (predicting the next token) and generation parameters (Temperature, Top-K).

### Module 6: Unsupervised Architectures & Capstone

_Goal: Applying neural networks to unlabeled data and verifying holistic course comprehension._

- **Lesson 23: Unsupervised Learning with Autoencoders:** Compressing data into a latent space. Comparing neural autoencoders to classical PCA for complex anomaly and fraud detection.
- **Lesson 24: Capstone Project Part 1 (Architecture & Training):** Students select a complex, messy dataset. They must design a custom architecture (CNN, LSTM, or Functional ensemble), apply strict regularization, and train it.
- **Lesson 25: Capstone Project Part 2 (Evaluation & Defense):** Students defend their architectural choices, visualize their loss trajectories, explain their models using XAI, and compare their neural network's performance against a classical ML baseline (like XGBoost).
