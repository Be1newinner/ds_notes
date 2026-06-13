# Lesson 13: Transfer Learning & Pre-trained Backbones

## Introduction & The "Why"

In previous lessons, we designed convolutional neural networks and trained them from scratch (using random weight initializations). While this works well for datasets with millions of samples (like ImageNet), it is highly impractical for most real-world business applications. In practice, we rarely have access to millions of labeled images. If we attempt to train a deep network from scratch on a small dataset (e.g., 500 images), the model will overfit immediately, regardless of the regularization techniques used.

To solve this, deep learning relies on **Transfer Learning**. Transfer Learning is a design paradigm where a model trained on a massive source task (where data is abundant) is reused as the starting point for a model on a target task (where data is scarce). 

Under the hood, a deep network trained on ImageNet learns a rich hierarchy of visual features. The early layers learn to detect simple edges and gradients; the middle layers learn shapes and textures; and the deep layers combine these into complex object parts. By importing these pre-trained feature extractors (backbones) and appending custom classification heads, we can train highly accurate models in minutes using very little labeled data. This lesson covers the theory of inductive bias in transfer learning, demonstrates how to freeze and train pre-trained backbones, and explains the mechanics of fine-tuning.

---

## Topic 1: The Rationale of Transfer Learning: Inductive Bias & Feature Reuse

### Rationale and Mechanics
In classical machine learning, we assume that the training and testing data are drawn from the same domain and feature space. If the distribution changes, we must train a new model from scratch. Transfer learning breaks this assumption: it allows us to transfer knowledge across different domains and tasks.

Under the hood, let's define this mathematically. We define a **Domain** $\mathcal{D}$ as consisting of a feature space $\mathcal{X}$ and a marginal probability distribution $P(X)$:
$$\mathcal{D} = \{\mathcal{X}, P(X)\}$$

Given a specific domain $\mathcal{D}$, a **Task** $\mathcal{T}$ consists of a label space $\mathcal{Y}$ and a predictive function $f(\cdot)$ (which is learned from the training data pairs $\{x_i, y_i\}$):
$$\mathcal{T} = \{\mathcal{Y}, f(\cdot)\} = \{\mathcal{Y}, P(Y|X)\}$$

In transfer learning, we have a **Source Domain** $\mathcal{D}_s$ and a **Source Task** $\mathcal{T}_s$ (e.g., classifying 1.4 million images into 1,000 classes on ImageNet), and a **Target Domain** $\mathcal{D}_t$ and a **Target Task** $\mathcal{T}_t$ (e.g., classifying 500 medical images into "benign" or "malignant"). The goal is to learn the target predictive function $f_t(\cdot)$ using the knowledge already learned in $\mathcal{D}_s$ and $\mathcal{T}_s$, where:
$$\mathcal{D}_s \neq \mathcal{D}_t \quad \text{or} \quad \mathcal{T}_s \neq \mathcal{T}_t$$

```
       Source Task (ImageNet)                      Target Task (Medical Scan Classifier)
       [ 1.4M general images ]                     [ 500 medical scans ]
                 |                                           |
                 v                                           v
       Train Deep ResNet50                         Import Pre-trained ResNet50
       (Learns edge, shape filters)                (Keep filters, replace output head)
                 |                                           |
                 \------------------------------------------>/ (Transfer weights)
```

Reusing the pre-trained weights provides a powerful **Inductive Bias**. Instead of initializing the weights randomly (which starts optimization at a random point in the loss landscape), we start the optimization process close to a high-quality local minimum. The network does not need to learn how to detect edges, contrast, or textures from scratch—it already knows how to extract these features, and only needs to learn how to combine them for the new task.

### Trade-offs
- **Advantages:** Reduces the required size of the target dataset by $99\%$. It saves massive amounts of compute time, electricity, and cloud training costs.
- **Disadvantages:** If the source domain is completely unrelated to the target domain, **Negative Transfer** can occur. For example, using a backbone pre-trained on natural ImageNet photos (dogs, cars, cups) to classify monochrome astronomical spectrograms or synthetic aperture radar (SAR) images can yield worse performance than training a smaller model from scratch.

### Real-World Applications (Rule of 4)

1. **Example 1: Medical Diagnosis with Limited Data**
   - **Input/Scenario:** A hospital has 200 labeled scans of a rare lung disease. Training a CNN from scratch yields 55% accuracy due to severe overfitting.
   - **Expected Output:** By importing a ResNet50 backbone pre-trained on ImageNet and training a custom classification head, the model achieves 92% accuracy, reusing the general feature filters to detect lung textures.
2. **Example 2: Visual Industrial Inspection**
   - **Input/Scenario:** A manufacturing line detects micro-cracks in metal casings. The company has only 1,000 images of defects.
   - **Expected Output:** The developer uses a pre-trained MobileNet backbone. The model learns to detect cracks within a few minutes of training, leveraging edge-detection features learned from ImageNet.
3. **Example 3: E-commerce Product Recommendation**
   - **Input/Scenario:** An e-commerce platform classifies user-uploaded clothing photos into 50 categories.
   - **Expected Output:** Reusing a pre-trained EfficientNet backbone allows the model to extract robust fashion features (patterns, colors, collar shapes) without needing millions of labeled fashion photos.
4. **Example 4: Negative Transfer (Satellite Radar)**
   - **Input/Scenario:** A developer uses an ImageNet pre-trained ResNet50 to classify raw, complex-valued Synthetic Aperture Radar (SAR) satellite data.
   - **Expected Output:** The model performs poorly because ImageNet features (focusing on natural lighting, perspective, and color gradients) do not translate to SAR radar backscatter signatures.

> **Metacognitive Checkpoint:** Why does using pre-trained weights act as a form of inductive bias during training? Explain how starting optimization near a pre-trained minimum differs from random initialization in a non-convex loss landscape.

---

## Topic 2: Freezing the Backbone: Feature Extraction Pipeline

### Rationale and Mechanics
When applying transfer learning, we split the architecture into two components: the **Convolutional Base** (backbone) and the **Classification Head**. 

The custom head is initialized with random weights, while the backbone contains pre-trained weights. If we train the entire model immediately, the large gradients generated by the random head during early epochs will backpropagate through the backbone. This will destroy the pre-trained weights, a failure known as **Feature Disruption**.

To prevent this, we **Freeze the Backbone**. We set the trainable status of the backbone's layers to `False`.

Under the hood, we implement this in Keras by setting the `trainable` attribute:
```python
from tensorflow import keras

# Import pre-trained backbone
backbone = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,  # Exclude the 1000-class ImageNet classification head
    input_shape=(224, 224, 3)
)

# Freeze the convolutional base
backbone.trainable = False
```

We then append our custom classification head using the Functional or Sequential API:
```python
model = keras.Sequential([
    backbone,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])
```

During model compilation and fitting, the optimizer calculates the parameter count:
- **Trainable Parameters:** Only the weights and biases of the custom Dense layers are updated.
- **Non-Trainable Parameters:** The weights of the ResNet50 backbone. Their gradients are not calculated, and their values remain frozen.

```
       Input ---> [ Pre-trained ResNet50 Base ] ---> [ Global Avg Pool ] ---> [ Custom Dense Head ]
                           |                                                        |
                     Frozen Weights                                          Trainable Weights
                     (No Updates)                                             (Optimized)
```

The frozen backbone acts as a static feature extraction pipeline, converting raw pixels into a high-level feature vector that the custom head learns to classify.

### Trade-offs
Freezing the backbone is safe, fast, and computationally cheap. Since we do not calculate gradients for the millions of parameters in the backbone, training is extremely fast and requires much less GPU memory.

The trade-off is that the model's feature extraction is restricted to what the backbone learned on the source dataset. If your target images have a different style (e.g., monochrome medical scans vs. color ImageNet photos), the frozen backbone's features might be sub-optimal. To resolve this, we must use fine-tuning after the head has converged.

### Real-World Applications (Rule of 4)

1. **Example 1: Parameter Count Verification**
   - **Input/Scenario:** We append a custom classifier to a frozen MobileNetV2 backbone ($2.2$ million parameters). The custom head has $100,000$ parameters.
   - **Expected Output:** Calling `model.summary()` shows:
     - Total params: $2,300,000$
     - Trainable params: $100,000$
     - Non-trainable params: $2,200,000$
     The optimizer only updates the $100,000$ weights in the head.
2. **Example 2: Fast Prototyping**
   - **Input/Scenario:** A developer wants to quickly train a classifier on a CPU.
   - **Expected Output:** By freezing the backbone, the model converges in 3 epochs. Since no gradients are calculated for the backbone, training takes only a few minutes on a standard CPU.
3. **Example 3: Protecting Pre-trained Weights**
   - **Input/Scenario:** We train the custom head with a standard learning rate ($\eta = 10^{-3}$).
   - **Expected Output:** The random weights in the head adjust quickly to minimize loss. Because the backbone is frozen, the pre-trained ImageNet filters are protected from being overwritten during the early, high-loss training phase.
4. **Example 4: Memory Optimization**
   - **Input/Scenario:** A model is trained on a GPU with limited memory.
   - **Expected Output:** Freezing the backbone eliminates the need to cache activation gradients for those layers, reducing GPU memory usage and allowing larger batch sizes.

> **Metacognitive Checkpoint:** Why do randomly initialized head weights generate large, destructive gradients during the first few epochs of training? Explain how freezing the backbone protects the pre-trained weights from these gradients.

---

## Topic 3: Fine-Tuning: Unfreezing and Adapting Deeper Layers

### Rationale and Mechanics
While a frozen backbone extracts excellent general features, we can improve performance by adapting the backbone's features to our specific target domain. This process is called **Fine-Tuning**.

We do not fine-tune immediately. First, we train the custom head until it converges (using a frozen backbone). Once the head is trained, we unfreeze a portion of the backbone and continue training the entire model.

Under the hood, we implement this by unfreezing the deeper layers of the backbone while keeping the early layers frozen:
```python
# Unfreeze the backbone
backbone.trainable = True

# Freeze all layers except the last few (e.g., from layer 140 onwards)
for layer in backbone.layers[:140]:
    layer.trainable = False
```

We freeze the early layers because they extract low-level, general features (edges, corners) that are useful for any computer vision task. We unfreeze the deeper layers because they extract high-level semantic shapes that need to be adjusted to match our specific target classes.

Crucially, when fine-tuning, we must recompile the model using an **extremely low learning rate**:
$$\eta_{\text{fine-tune}} \approx 10^{-5}$$

```
       [ Input ] ---> [ Early Layers: Frozen ] ---> [ Deeper Layers: Unfrozen ] ---> [ Custom Head: Trainable ]
                                |                                 |                                |
                        lr = 0 (No updates)                 lr = 1e-5 (Fine-tuned)           lr = 1e-5 (Fine-tuned)
```

This tiny learning rate ensures that we make small, incremental adjustments to the pre-trained weights, adapting them to the new data without destroying the learned feature structures.

### Trade-offs
Fine-tuning unlocks the full representation capacity of transfer learning, often boosting target classification accuracy by $3\%$ to $10\%$.

The trade-off is the risk of **Catastrophic Forgetting** (or feature disruption). If the learning rate is too high, the optimizer will overwrite the pre-trained weights with random adjustments, causing the model to overfit to the small target dataset and reducing accuracy. Fine-tuning also increases training time and memory usage because we must calculate gradients for the unfrozen backbone layers.

### Real-World Applications (Rule of 4)

1. **Example 1: Target Adaptation**
   - **Input/Scenario:** A model is trained to classify types of cars. The pre-trained ImageNet backbone detects "wheels" and "windows" generally. We unfreeze the last 10 layers of the backbone and fine-tune with $\eta = 10^{-5}$.
   - **Expected Output:** The unfrozen layers adjust their weights to detect specific grill and headlight shapes, improving classification accuracy for subtle car model differences.
2. **Example 2: Catastrophic Forgetting Avoidance**
   - **Input/Scenario:** A developer fine-tunes a ResNet50 backbone using a standard learning rate $\eta = 10^{-3}$ on a small dataset of 500 images.
   - **Expected Output:** The model's validation loss spikes after 2 epochs, and accuracy drops to 50% as the pre-trained feature extractors are overwritten, illustrating catastrophic forgetting. The developer must re-run with $\eta = 10^{-5}$.
3. **Example 3: Selective Layer Unfreezing**
   - **Input/Scenario:** A developer inspects a backbone with 150 layers.
   - **Expected Output:** By unfreezing only layers 130 to 150, the developer restricts training to the high-level semantic filters while keeping the general edge detector filters (layers 1 to 129) frozen.
4. **Example 4: Recompilation Requirement**
   - **Input/Scenario:** A developer unfreezes a backbone's layers in Python but forgets to call `model.compile()`.
   - **Expected Output:** The training updates are not updated. The model continues to train using the old compiled graph (keeping the backbone frozen). The developer must recompile the model to apply the unfreezing changes.

> **Metacognitive Checkpoint:** Why do we freeze the early layers of a backbone during fine-tuning while unfreezing the deeper layers? Connect this to the concept of spatial feature hierarchies in convolutional networks.

---

## Summary & Next Steps

- **Transfer Learning Reuses Features:** Reusing backbones pre-trained on large datasets (like ImageNet) provides a strong inductive bias, allowing us to train accurate models on small target datasets.
- **Freezing Protects Pre-trained Weights:** Freezing the backbone ensures that the pre-trained weights are not overwritten by the large gradients generated by the randomly initialized classification head.
- **Fine-Tuning Adapts Features:** Unfreezing the deep layers of the backbone and training with a tiny learning rate ($\eta \approx 10^{-5}$) adapts the feature extraction to the target domain, boosting accuracy.

In the next lesson, we will open the black box of computer vision, exploring **Explainable AI for Vision (Grad-CAM)** to generate spatial heatmaps that show exactly which parts of an image dictated the model's predictions.
