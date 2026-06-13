# Lesson 10: The Spatial Hierarchy & Convolutions

## Introduction & The "Why"

In the previous module, we built feedforward neural networks (Multi-Layer Perceptrons) using dense layers. While dense layers are highly versatile for tabular datasets, they fail when applied to grid-like spatial data, such as images.

An image is not just a collection of independent features; it contains structured **spatial hierarchies**. Nearby pixels are highly correlated, forming local structures like edges, corners, and textures, which combine to form complex objects. If we flatten an image into a 1D vector to feed it to a dense layer, we discard this spatial coordinate structure. Furthermore, dense layers assign a unique weight to every pixel coordinate, leading to a massive parameter explosion that causes overfitting.

To solve this, deep learning relies on **Convolutional Neural Networks (CNNs)**. CNNs use convolutional layers that preserve the 2D spatial coordinate structure and dramatically reduce parameter counts through local connectivity and parameter sharing. This lesson explains why MLPs fail on images, details the mechanics of 2D convolutions, and explains how padding and stride control the dimensions of intermediate feature maps.

---

## Topic 1: Why MLPs Fail on Images: The Parameter Explosion & Loss of Spatial Structure

### Rationale and Mechanics
In classical machine learning pipelines, raw images are flattened into 1D vectors before being passed to classifiers like Support Vector Machines or Logistic Regression. This approach treats each pixel coordinate as an independent variable, ignoring the fact that pixels derive their meaning from their neighbors.

When we apply a fully connected (Dense) layer to an image, every neuron in the hidden layer is connected to every pixel in the input image. Let's analyze the mathematical consequences.

Under the hood, let's consider a standard color image with a resolution of $256 \times 256$ pixels. Since it is a color image, it contains 3 channels (Red, Green, Blue). The total number of input features (pixels) is:
$$\text{Input Features} = H \times W \times C = 256 \times 256 \times 3 = 196,608$$

If we connect this input to a single, modest hidden layer containing $1,000$ units, the weight matrix $\mathbf{W}$ must store:
$$\text{Params} = 196,608 \times 1,000 = 196,608,000 \text{ weights}$$
Over 196 million parameters for just the first layer! 

```
        Dense Layer (MLP): Parameter Explosion
        
        Input Image (256x256x3)                         Hidden Layer
        [ Pixel 1 ] -----------------------------\----> [ Neuron 1 ]
        [ Pixel 2 ] -----------------------\----/-----> [ Neuron 2 ]
        [  ...    ]                         \  /
        [ Pixel 196608 ] --------------------\--------> [ Neuron 1000 ]
        
        Total connections: 196,608,000 (every pixel connected to every neuron)
```

This parameter explosion causes three major failures:
1. **Overfitting:** The model has so much capacity that it will easily memorize the exact pixel locations of the training images rather than learning general shapes.
2. **Computational Overhead:** Storing and updating 196 million weights requires massive GPU memory and slows down training.
3. **Loss of Translation Invariance:** If the model learns to recognize a cat in the top-left corner, it will not recognize the same cat in the bottom-right corner. The model must learn the cat's features all over again for every pixel location because the weights connected to the top-left pixels are independent of the weights connected to the bottom-right pixels.

### Trade-offs
To process spatial data, we must abandon global connectivity in favor of **local connectivity** and **parameter sharing**:
- **Local Connectivity:** A hidden unit should only look at a small, localized region of the image (e.g., a $3\times3$ pixel patch).
- **Parameter Sharing:** The feature extractor (weights) should be slid across the image, applying the same weights at every location.

This shift reduces parameter counts by orders of magnitude and forces the model to learn translation-invariant features, preserving spatial coordinates.

### Real-World Applications (Rule of 4)

1. **Example 1: Dense Parameter Count (MNIST)**
   - **Input/Scenario:** A small greyscale image from the MNIST dataset has size $28 \times 28 \times 1 = 784$ pixels. We connect it to a hidden layer of 100 units.
   - **Expected Output:** The weight matrix has size $784 \times 100 = 78,400$ parameters. For small images, an MLP is manageable, but it still scales quadratically if the image resolution increases.
2. **Example 2: Dense Parameter Count (HD Image)**
   - **Input/Scenario:** A high-definition color image has size $1024 \times 1024 \times 3 = 3,145,728$ pixels. We connect it to a hidden layer of 1,000 units.
   - **Expected Output:** The first layer requires $3,145,728 \times 1000 \approx 3.14$ billion weights, exceeding the memory capacity of standard GPUs and causing training failure.
3. **Example 3: Translation Failure in MLPs**
   - **Input/Scenario:** A dense classifier is trained to detect vertical lines using images where the line is always in the center column. During testing, the vertical line is shifted 3 pixels to the right.
   - **Expected Output:** The MLP fails to detect the line because the weights connected to the new coordinates were never updated during training, demonstrating a lack of translation invariance.
4. **Example 4: Structural Scrambling**
   - **Input/Scenario:** We randomly shuffle the pixel columns of our training dataset before training a Dense network.
   - **Expected Output:** The MLP achieves the exact same training accuracy because flattening the image treats all pixel coordinates symmetrically, proving that the dense network does not utilize spatial relationships.

> **Metacognitive Checkpoint:** Why does flattening an image discard its spatial coordinate structure? Explain how the spatial relationship between two adjacent pixels is lost when they are mapped to a 1D vector.

---

## Topic 2: 2D Convolutional Layers: Spatial Filters & Shared Weights

### Rationale and Mechanics
In classical image processing, features are extracted using hand-crafted **kernels** (small coefficient matrices). For example, a Sobel filter detects edges by calculating horizontal and vertical gradients, while a Gaussian kernel blurs the image.

In deep learning, we use **2D Convolutional Layers (Conv2D)**. Instead of using hand-crafting kernels, the network learns the kernel coefficients automatically via backpropagation.

Under the hood, a convolutional layer slides a small window of weights called a **kernel** (typically of size $3\times3$ or $5\times5$) across the input image. At each step, it performs element-wise multiplication and summation:
$$Z(i, j) = (X * K)(i, j) = \sum_{m=1}^k \sum_{n=1}^k X(i + m - 1, j + n - 1) K(m, n) + b$$
where:
- $X$ is the input image patch.
- $K$ is the learnable kernel of size $k \times k$.
- $b$ is the learnable bias.
- $Z(i, j)$ is the output coordinate in the resulting **feature map**.

```
       Input Image (5x5)                   Kernel (3x3)              Feature Map (3x3)
       [ 1  0  1  0  0 ]                   [ 1  0 -1 ]
       [ 0  1  1  1  0 ] * Convolution     [ 1  0 -1 ]   =======>    [ -1  1  1 ]
       [ 0  0  1  0  1 ]                   [ 1  0 -1 ]
       [ 1  0  0  0  0 ]
       [ 0  1  1  0  1 ]
       
       Calculation for top-left output:
       (1*1) + (0*0) + (1*-1) + (0*1) + (1*0) + (1*-1) + (0*1) + (0*0) + (1*-1) = 1 - 1 - 1 - 1 = -2
```

For color images with $C$ channels, the kernel has depth $C$ (shape: $k \times k \times C$). The kernel performs a 3D dot product across all channels, summing the results into a single output feature channel.

To detect multiple features (e.g., edges, textures, colors), a Conv2D layer uses $F$ distinct kernels. The output of the layer is a stack of $F$ feature maps, forming a tensor of shape:
$$\text{Output Shape} = (H_{\text{out}}, W_{\text{out}}, F)$$

The parameter count for a Conv2D layer with $F$ filters of size $k \times k$ receiving $C$ input channels is:
$$\text{Parameters} = F \times \left( k \times k \times C + 1 \right)$$

For example, a Conv2D layer with 64 filters of size $3\times3$ receiving an RGB input ($C=3$) has:
$$\text{Params} = 64 \times (3 \times 3 \times 3 + 1) = 64 \times 28 = 1,792 \text{ parameters}$$
This parameter count is tiny compared to a dense layer, preventing overfitting and enabling fast training.

### Trade-offs
- **Advantages:** Conv2D layers enforce **local connectivity** (each output unit only depends on a local input region) and **parameter sharing** (the same kernel weights are used at all locations). This makes the feature extractor translation-invariant.
- **Disadvantages:** Sliding window operations are computationally intensive. However, modern GPUs parallelize these calculations using matrix multiplication (Im2Col algorithm) and specialized hardware.

### Real-World Applications (Rule of 4)

1. **Example 1: Parameter Reduction Comparison**
   - **Input/Scenario:** We process an image of size $256 \times 256 \times 3$. Layer 1 is a Conv2D layer with 32 filters of size $3\times3$.
   - **Expected Output:** The Conv2D layer requires $32 \times (3 \cdot 3 \cdot 3 + 1) = 896$ parameters. This is $220,000\times$ fewer parameters than the equivalent Dense layer, preventing overfitting.
2. **Example 2: Edge Detection (Sobel Analogy)**
   - **Input/Scenario:** An early Conv2D layer has a filter initialized with weights $K = \begin{pmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{pmatrix}$.
   - **Expected Output:** Sliding this kernel across an image calculates the difference between pixels on the left and right, outputting high values along vertical boundaries, detecting edges.
3. **Example 3: Channel Integration**
   - **Input/Scenario:** A Conv2D layer receives a 3-channel (RGB) feature map and uses a $3\times3$ filter.
   - **Expected Output:** The filter contains 27 weights ($3\times3\times3$). It computes a weighted sum across all three color channels simultaneously, combining color information into a single feature map.
4. **Example 4: Feature Map Dimension Change**
   - **Input/Scenario:** A Conv2D layer with 64 filters processes an input of shape $(128, 128, 3)$.
   - **Expected Output:** The output tensor has shape $(H_{\text{out}}, W_{\text{out}}, 64)$. The number of channels has increased from 3 to 64, representing 64 distinct spatial features.

> **Metacognitive Checkpoint:** Why does parameter sharing in convolutional layers guarantee translation invariance? Explain how using the same weights at every location ensures the network can recognize a feature regardless of where it appears in the image.

---

## Topic 3: Stride & Padding: Controlling Feature Map Dimensions

### Rationale and Mechanics
As a kernel slides across an input image, the spatial dimensions of the output feature map shrink. If we process a $5\times5$ image with a $3\times3$ kernel, the kernel can only fit in 3 horizontal and 3 vertical positions, yielding a $3\times3$ output. 

If we build a deep network with many layers, the spatial size will shrink to $0$ after a few layers. To prevent this shrinkage and control the output size, we use **Padding** and **Stride**.

Under the hood, let the input width be $W_{\text{in}}$, the kernel size be $k$, the padding be $P$, and the stride be $S$. The output width $W_{\text{out}}$ is:
$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - k + 2P}{S} \right\rfloor + 1$$

- **Padding ($P$):** Adds border pixels around the input image before applying convolution.
  - **Valid Padding ($P = 0$):** No padding is added. The output size shrinks.
  - **Same Padding:** Adds enough padding so that when stride $S = 1$, the output size matches the input size ($W_{\text{out}} = W_{\text{in}}$). For a kernel of size $k$, this requires:
    $$P = \frac{k - 1}{2}$$
- **Stride ($S$):** The step size of the kernel as it slides. A stride of $S=1$ moves the kernel 1 pixel at a time. A stride of $S=2$ moves the kernel 2 pixels at a time, downsampling the spatial dimensions by approximately half.

```
       Valid Padding (P=0): Shrinks Output     Same Padding (P=1): Preserves Size
       
            [x x x x x]                             [0 0 0 0 0 0 0]
            [x x x x x]                             [0 x x x x x 0]
            [x x x x x]                             [0 x x x x x 0]
            [x x x x x]                             [0 x x x x x 0]
            [x x x x x]                             [0 x x x x x 0]
                                                    [0 x x x x x 0]
                                                    [0 0 0 0 0 0 0]
```

### Trade-offs
- **Padding Trade-off:** "Same" padding preserves spatial dimensions, preventing loss of boundary information. However, adding zero-value pixels introduces artificial boundary values at the borders, which can affect learning at the edges.
- **Stride Trade-off:** A stride of $S \ge 2$ downsamples the feature map, reducing parameter counts and memory usage in downstream layers. However, downsampling too quickly can discard detailed spatial information that is useful for task resolution.

### Real-World Applications (Rule of 4)

1. **Example 1: Same Padding Calculation**
   - **Input/Scenario:** We process an input of width $W_{\text{in}} = 28$ using a kernel of size $k = 5$ with stride $S = 1$. We use same padding ($P = \frac{5-1}{2} = 2$).
   - **Expected Output:** The output width is:
     $$W_{\text{out}} = \left\lfloor \frac{28 - 5 + 2(2)}{1} \right\rfloor + 1 = (28 - 5 + 4) + 1 = 28$$
     The spatial dimension is preserved.
2. **Example 2: Valid Padding Shrinkage**
   - **Input/Scenario:** The same input ($W_{\text{in}} = 28, k = 5, S = 1$) is processed using valid padding ($P = 0$).
   - **Expected Output:** The output width is:
     $$W_{\text{out}} = \left\lfloor \frac{28 - 5 + 0}{1} \right\rfloor + 1 = 23 + 1 = 24$$
     The feature map has shrunk by 4 pixels.
3. **Example 3: Stride 2 Downsampling**
   - **Input/Scenario:** An input of size $W_{\text{in}} = 128$ is processed using a $3\times3$ kernel ($k=3$), padding $P=1$, and stride $S=2$.
   - **Expected Output:** The output width is:
     $$W_{\text{out}} = \left\lfloor \frac{128 - 3 + 2(1)}{2} \right\rfloor + 1 = \left\lfloor \frac{127}{2} \right\rfloor + 1 = 63 + 1 = 64$$
     The spatial dimension has been halved.
4. **Example 4: Dimensionality Collapse Prevention**
   - **Input/Scenario:** A developer designs a 5-layer CNN with $3\times3$ kernels and no padding ($P=0$). The input image size is $10\times10$.
   - **Expected Output:**
     - Layer 1: $10 - 2 = 8$
     - Layer 2: $8 - 2 = 6$
     - Layer 3: $6 - 2 = 4$
     - Layer 4: $4 - 2 = 2$
     - Layer 5: $2 - 2 = 0$.
     The spatial dimension collapses before Layer 5, causing a code crash. The developer must add `padding='same'` to keep spatial coordinates alive.

> **Metacognitive Checkpoint:** Given an input image of shape $(224, 224, 3)$, a convolutional layer with $64$ filters of size $7\times7$, stride $S=2$, and padding $P=3$. Calculate the exact shape of the output feature map. Show your steps.

---

## Summary & Next Steps

- **MLPs Scale Poorly on Images:** Dense layers ignore spatial coordinates and suffer from parameter explosions when scaling to high-resolution images.
- **Convolutions Share Parameters:** 2D convolutional layers slide small kernels across images, enforcing local connectivity and parameter sharing to extract translation-invariant features.
- **Stride and Padding Control Shapes:** Padding prevents spatial shrinkage, while stride downsamples feature maps to manage computational complexity.

In the next lesson, we will explore **Pooling & Translational Invariance**, learning how pooling layers reduce dimensionality and help the network recognize features regardless of minor shifts.
