# Lesson 14: Explainable AI for Vision (Grad-CAM)

## Introduction & The "Why"

In previous lessons, we designed, regularized, normalized, and trained convolutional neural networks to classify images. However, deep neural networks are often criticized as being **black boxes**. A model might achieve $98\%$ accuracy on a validation dataset, but we do not know *how* it arrived at its decisions. It contains millions of weights interacting non-linearly, making it impossible to trace the decision path manually.

In regulated domains like healthcare, finance, or autonomous driving, high accuracy is not enough. A radiologist will not trust an AI that flags a chest X-ray as cancerous unless they can see exactly which regions of the image triggered the flag. Furthermore, models can easily overfit to **spurious correlations**—for example, a skin cancer model might learn to look for surgical markers or ruler lines on the skin rather than the lesion itself.

To open the black box and build trust with business stakeholders, we use **Explainable AI (XAI)**. This lesson covers the visual interpretability problem, explains the mathematics of Gradient-Weighted Class Activation Mapping (Grad-CAM), and demonstrates how to use spatial heatmaps to debug models and detect spurious correlations.

---

## Topic 1: Explainable AI (XAI) & The Black Box Problem

### Rationale and Mechanics
In classical machine learning, model interpretability is a built-in feature. For example, Decision Trees provide clear path rules, Random Forests compute permutation feature importances, and Logistic Regression outputs coefficients that represent the contribution of each input variable.

In deep neural networks, this direct interpretability is lost. A convolutional neural network processes pixels through layers of convolutions, activations, and pooling. By the time the image reaches the final layer, the spatial pixel coordinates are blended into abstract feature vectors.

```
       Image ---> [ Conv + Pool Layers ] ---> [ Dense Layer ] ---> Prediction
                     |                            |
               Spatial Pixels                Black Box
             (Direct Meaning)            (Abstract Vectors)
```

Explainable AI (XAI) for computer vision attempts to solve this by generating **visual attribution maps**. These are heatmaps overlaid on the input image, where color intensity represents the importance of each pixel to the model's final prediction.

### Trade-offs
The primary trade-off in XAI is the balance between **interpretability** and **accuracy**:
- Simple models (like linear classifiers) are highly interpretable but have low representational capacity.
- Deep models (like CNNs or Transformers) are highly accurate but difficult to interpret.

Visual attribution methods seek to provide post-hoc interpretability without changing the model architecture or reducing its accuracy. However, these methods are approximations: they show *what* regions the model looked at, but they do not explain the logic or decision rules the network used to combine those features.

### Real-World Applications (Rule of 4)

1. **Example 1: Medical Diagnosis Validation**
   - **Input/Scenario:** A CNN classifies X-rays for pneumonia. It predicts a scan has pneumonia with 95% confidence.
   - **Expected Output:** The XAI attribution map highlights the lung cavities (where fluid deposits appear), proving to the radiologist that the model is looking at the correct anatomical regions.
2. **Example 2: Autonomous Driving Safety Audit**
   - **Input/Scenario:** An autonomous vehicle model detects a stop sign.
   - **Expected Output:** The attribution heatmap highlights the hexagonal red shape of the sign and the text "STOP," proving to safety regulators that the model is detecting the sign's features rather than surrounding buildings.
3. **Example 3: Financial Document Classification**
   - **Input/Scenario:** A document classifier sorts invoices based on layout.
   - **Expected Output:** The heatmap highlights the location of the invoice total and vendor logo, verifying that the network is extracting key financial features.
4. **Example 4: Debugging Spurious Correlations**
   - **Input/Scenario:** A model trained to classify fish species achieves 99% accuracy on validation tests.
   - **Expected Output:** The XAI map highlights the bottom-right corner of the images (where the watermark of the camera source is located). This reveals that the model is cheating by reading watermarks rather than looking at the fish.

> **Metacognitive Checkpoint:** Why are deep neural networks considered "black boxes" compared to classical algorithms like Decision Trees? Explain how non-linear activations and fully connected layers obscure the relationship between inputs and outputs.

---

## Topic 2: The Math of Grad-CAM (Gradient-Weighted Class Activation Mapping)

### Rationale and Mechanics
Gradient-Weighted Class Activation Mapping (Grad-CAM), introduced by Ramprasaath R. Selvaraju et al. in 2017, is a popular visual interpretability method. It uses the gradients of a target class score flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image.

We target the **final convolutional layer** because it represents the highest-level spatial features before the spatial dimensions are collapsed by flattening or pooling.

Under the hood, the Grad-CAM algorithm calculates the heatmap using the following steps:
1. **Target Class Score ($y^c$):** We run a forward pass and extract the raw, unnormalized score (logit) $y^c$ for class $c$. We use the logit rather than the softmax probability to prevent gradient vanishing caused by the softmax denominator.
2. **Feature Map Activations ($A^k$):** We extract the activation maps of the final convolutional layer. Let $A^k$ represent the $k$-th feature channel map (shape: $H \times W$).
3. **Compute Gradients:** We calculate the gradient of the class score $y^c$ with respect to the feature map activations $A^k(i, j)$ at coordinate $(i, j)$:
   $$\frac{\partial y^c}{\partial A^k(i, j)}$$
4. **Calculate Channel Weights ($\alpha_k^c$):** We perform Global Average Pooling on the gradients to calculate the importance weight $\alpha_k^c$ for each channel $k$:
   $$\alpha_k^c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A^k(i, j)}$$
5. **Weighted Combination & ReLU:** We perform a weighted sum of the feature maps $A^k$ using the weights $\alpha_k^c$, and apply a Rectified Linear Unit (ReLU) to the result:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

```
        Feature Maps A^k                   Channel Weights alpha_k^c
        [  H x W Map 1  ] ---------------> Multiplied by alpha_1
        [  H x W Map 2  ] ---------------> Multiplied by alpha_2   ===> Sum maps ===> Apply ReLU ===> Heatmap
        [      ...      ]
        [  H x W Map K  ] ---------------> Multiplied by alpha_K
```

We apply the ReLU function because we are only interested in features that *positively* contribute to the target class $c$. Pixels that decrease the class score (negative values) are filtered out, leaving only the regions that support the prediction.

### Trade-offs
Grad-CAM can be applied to any CNN architecture without retraining or changing the network topology, and it is computationally cheap because it only requires one forward and one backward pass.

The trade-off is spatial resolution. Because the final convolutional layer has small spatial dimensions (e.g., $7\times7$ in ResNet50), the output Grad-CAM heatmap is coarse. While it tells you the general region of interest, it does not provide fine-grained, pixel-level boundaries. To get sharper attribution maps, we must combine Grad-CAM with pixel-space visualization methods (such as Guided Backpropagation) to generate **Guided Grad-CAM**.

### Real-World Applications (Rule of 4)

1. **Example 1: Gradient Weighting Calculation**
   - **Input/Scenario:** A final conv layer outputs a $2\times2$ feature map for channel 1: $A^1 = \begin{pmatrix} 1.0 & 2.0 \\ 0.0 & 1.0 \end{pmatrix}$. The calculated gradients are $\frac{\partial y^c}{\partial A^1} = \begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}$.
   - **Expected Output:** The channel weight is $\alpha_1^c = \frac{0.5 + 0.5 + 0.5 + 0.5}{4} = 0.5$. This indicates that Channel 1 has a moderate positive contribution to class $c$.
2. **Example 2: Negative Feature Suppression (ReLU)**
   - **Input/Scenario:** The weighted sum of feature maps yields a local coordinate value of $-3.2$.
   - **Expected Output:** Applying the ReLU function evaluates $\max(0, -3.2) = 0.0$. The negative activation is zeroed out, preventing features that decrease the class score from appearing on the heatmap.
3. **Example 3: Multi-Class Heatmap Extraction**
   - **Input/Scenario:** An image contains both a dog and a cat. The model outputs scores for both classes.
   - **Expected Output:** By calculating Grad-CAM using $y^{\text{dog}}$, the heatmap highlights the dog's location. Re-calculating with $y^{\text{cat}}$ highlights the cat's location, proving that the model separates the two concepts spatially.
4. **Example 4: Up-sampling via Interpolation**
   - **Input/Scenario:** The raw Grad-CAM heatmap has size $7\times7$. The input image has size $224\times224$.
   - **Expected Output:** The heatmap is upsampled to $224\times224$ using bilinear interpolation, smoothing the coarse grid and allowing it to be overlaid on the original image.

> **Metacognitive Checkpoint:** Why do we apply the ReLU activation function to the weighted sum of feature maps in the final step of the Grad-CAM algorithm? What would the heatmap depict if we omitted the ReLU function?

---

## Topic 3: Visualizing and Debugging Models: Overcoming Spurious Correlations

### Rationale and Mechanics
Once the raw Grad-CAM heatmap is generated and upsampled, we normalize its values to the range $[0, 1]$:
$$L_{\text{norm}}^c = \frac{L^c - \min(L^c)}{\max(L^c) - \min(L^c) + \epsilon}$$

We then apply a colormap (such as Jet, where high values are red and low values are blue) and overlay it on the original image:
$$\text{Overlay} = \alpha \times \text{Heatmap} + (1 - \alpha) \times \text{Original}$$
where $\alpha \approx 0.4$ controls the blending transparency.

This visualization is a powerful debugging tool. It helps identify **"Clever Hans"** models—networks that achieve high accuracy on validation sets by exploiting spurious background shortcuts rather than learning the actual visual concept.

```
       Original Image              Saturated Overlay (Cheat)          Correct Overlay
       
        [ Husky on Snow ]           [ Husky on SNOW ]                  [ HUSKY on Snow ]
                                       ^ (Red Heatmap on snow)            ^ (Red Heatmap on face)
```

For example, if a model trained to classify wolves vs. huskies always sees wolves in snowy backgrounds and huskies in grass, it will learn to classify based on the background. Grad-CAM reveals this immediately by showing a red heatmap over the snow rather than the animal's face.

### Trade-offs
Visualizing heatmaps is critical for auditing models before production deployment. It builds trust with stakeholders and identifies failure modes that standard validation metrics (like accuracy) miss.

The trade-off is that heatmap inspection is qualitative and subjective. Inspecting hundreds of images manually is time-consuming. Additionally, Grad-CAM can be misleading if the gradients are noisy, sometimes requiring quantitative metrics (like pixel perturbation tests) to verify that the highlighted regions are truly driving the predictions.

### Real-World Applications (Rule of 4)

1. **Example 1: Husky vs. Wolf Classifier Debugging**
   - **Input/Scenario:** A developer trains a husky vs. wolf classifier. It achieves 98% validation accuracy.
   - **Expected Output:** The Grad-CAM heatmap on test images highlights the background snow rather than the husky. The developer realizes the dataset is biased and must add images of huskies in the snow and wolves in the grass to balance the training.
2. **Example 2: Skin Lesion Classifier Audit**
   - **Input/Scenario:** An AI model flags melanomas. The clinic notices that the model performs poorly on images taken at other hospitals.
   - **Expected Output:** Grad-CAM reveals that the model highlights black marker circles drawn by doctors on the skin. The model had learned to associate marker ink with melanoma. The developer must retrain the model with data augmentation to ignore ink marks.
3. **Example 3: Document Parser Error Isolation**
   - **Input/Scenario:** An invoice parser extracts incorrect totals.
   - **Expected Output:** Grad-CAM shows the model highlights the currency symbol (e.g., "$") rather than the digits of the total, leading to classification errors. The developer adjusts the layout preprocessing.
4. **Example 4: Image Captioning Alignment**
   - **Input/Scenario:** An image captioning model generates the word "ball."
   - **Expected Output:** Grad-CAM highlights the coordinate region of the ball in the image, verifying that the model's text generation is aligned with visual objects.

> **Metacognitive Checkpoint:** How does Grad-CAM help identify dataset bias and prevent models from deploying with "cheating" shortcuts? Describe a scenario where a model might achieve 100% accuracy on training data but fail in production due to background cues.

---

## Summary & Next Steps

- **XAI Solves the Black Box Problem:** Attribution methods generate heatmaps indicating which pixel regions contributed most to a classification decision, building trust.
- **Grad-CAM Uses Feature Gradients:** Grad-CAM calculates the gradient of the class score with respect to the final conv feature maps, averaging them to weight the feature channels.
- **Visualizations Detect Bias:** Normalizing and overlaying heatmaps allows developers to detect spurious background correlations, ensuring models learn generalized features.

In the next module, we will transition to chronological data, starting **Module 4: Sequential Data (Time-Series & RNNs)** with **Lesson 15: Time-Series Preparation & Vanilla RNNs** to learn how to structure 3D tensors and build recurrent architectures.
