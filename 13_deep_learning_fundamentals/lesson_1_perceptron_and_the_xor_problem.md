# Lesson 1: The Perceptron & The XOR Problem

## Introduction & The "Why"

When we want a computer to make decisions, we often start with simple rules. In classical machine learning, we use models like **Logistic Regression** to draw a straight boundary between two categories—like sorting emails into "spam" or "inbox." 

However, real-world decisions are rarely that simple. A single decision maker (like a single neuron in a brain or a single node in a computer network) is extremely limited. To build complex systems that can recognize faces, translate languages, and drive cars, we need to stack simple decision makers together.

This lesson introduces the **Perceptron**, the absolute first building block of deep learning. We will learn how a single artificial neuron makes decisions, how it determines its decision boundaries, and how it updates its parameters when it makes a mistake. Finally, we will mathematically prove why a single neuron is unable to solve a simple non-linear problem called the **XOR gate**. This fundamental limit is the exact reason why we must build multi-layer networks.

---

## Topic 1: From Logistic Regression to the Perceptron

### Rationale and Mechanics
Imagine you are deciding whether to go to an outdoor music festival. You might consider three factors:
1. Is the weather good? ($x_1$)
2. Is the ticket cheap? ($x_2$)
3. Is your favorite band playing? ($x_3$)

In classical Logistic Regression, you assign a level of importance (**Weight**) to each factor and add a baseline preference (**Bias**). You calculate a score:
$$\text{Score} = (\text{Weather} \times w_1) + (\text{Ticket} \times w_2) + (\text{Band} \times w_3) + \text{Bias}$$
You then pass this score through a smooth curve (the Sigmoid function) to get a probability between $0\%$ and $100\%$.

The **Perceptron** is a simplified, historical version of this decision maker. Instead of outputting a smooth probability, it makes a hard, "yes or no" decision ($1$ or $0$) using a **Step Function**. If your total score is $0$ or higher, you go ($1$). If the score is negative, you stay home ($0$).

Under the hood, here is the formula for the net input $z$:
$$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$
where:
- $x_i$ are the inputs (features).
- $w_i$ are the weights (the importance of each input).
- $b$ is the bias (how eager the neuron is to fire. A positive bias makes it easy to fire; a negative bias makes it hard).

The output $\hat{y}$ is determined by the step function:
$$\hat{y} = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

### Python Code Implementation
Here is how to build a Perceptron from scratch in Python:

```python
import numpy as np

def step_function(z):
    return 1 if z >= 0 else 0

def perceptron(x, w, b):
    # Calculate the weighted sum: z = w1*x1 + w2*x2 + ... + wn*xn + b
    z = np.dot(w, x) + b
    # Apply the step function to get a hard 0 or 1 decision
    return step_function(z)

# Example: Music festival decision
# Inputs: Weather is good (1), Ticket is expensive (0), Favorite band is playing (1)
inputs = np.array([1, 0, 1])
# Weights: Weather is important (2.0), Cheap ticket is moderate (0.5), Band is important (1.5)
weights = np.array([2.0, 0.5, 1.5])
# Bias: Hard to please (-3.0)
bias = -3.0

decision = perceptron(inputs, weights, bias)
print("Decision (1 = Go, 0 = Stay):", decision)
# Output: z = (1*2.0) + (0*0.5) + (1*1.5) - 3.0 = 0.5 >= 0 => Output: 1
```

### Trade-offs
The Perceptron is extremely simple and fast. However, because it uses a step function, its slope (derivative) is flat everywhere (zero) and undefined at $z = 0$. This means we cannot calculate gradients, making **Gradient Descent** and backpropagation mathematically impossible. Modern neural networks replace the step function with smooth curves (like Sigmoid or ReLU) to allow the model to learn via gradients.

### Real-World Applications (Rule of 4)

1. **Example 1: E-commerce Fraud Detection**
   - **Input/Scenario:** A transaction is evaluated based on `is_new_device` ($x_1 \in \{0, 1\}$) and `transaction_amount_usd` ($x_2$, scaled to $[0, 10]$). The weights are $w_1 = 2.0$, $w_2 = 0.5$, and the bias is $b = -3.0$. A transaction arrives from a new device ($x_1 = 1$) with a scaled transaction amount of $4.0$ ($x_2 = 4$).
   - **Expected Output:** The net input is $z = (2.0 \cdot 1) + (0.5 \cdot 4) - 3.0 = 1.0$. Since $z \ge 0$, the activation function outputs $\hat{y} = 1$, indicating the transaction is blocked/flagged as fraud.
2. **Example 2: Clinical Risk Screening**
   - **Input/Scenario:** A hospital system uses a simple neuron to flag patients at high risk for diabetes based on normalized `fasting_blood_sugar` ($x_1$, scaled to $[0, 2]$) and `body_mass_index` ($x_2$, scaled to $[0, 3]$). The weights are $w_1 = 1.5$, $w_2 = 0.8$, and $b = -2.5$. A patient has a normalized blood sugar of $1.0$ and a normalized BMI of $1.0$.
   - **Expected Output:** The net input is $z = (1.5 \cdot 1.0) + (0.8 \cdot 1.0) - 2.5 = -0.2$. Since $z < 0$, the output is $\hat{y} = 0$, meaning the patient is classified as low risk.
3. **Example 3: Financial Credit Pre-Screening**
   - **Input/Scenario:** A bank pre-approves credit cards using binary inputs `has_stable_income` ($x_1 \in \{0, 1\}$) and `has_no_defaults` ($x_2 \in \{0, 1\}$). The weights are $w_1 = 1.2$, $w_2 = 1.5$, and bias $b = -1.0$. A customer has stable income ($x_1 = 1$) but has default history ($x_2 = 0$).
   - **Expected Output:** The net input is $z = (1.2 \cdot 1) + (1.5 \cdot 0) - 1.0 = 0.2$. Since $z \ge 0$, the model outputs $\hat{y} = 1$, approving the application for manual review.
4. **Example 4: Industrial Sensor Maintenance**
   - **Input/Scenario:** A manufacturing sensor flags an assembly robot for maintenance based on normalized `vibration_amplitude` ($x_1$) and `operating_temperature` ($x_2$). The weights are $w_1 = 3.0$, $w_2 = 2.5$, and bias $b = -4.0$. The sensor reports $x_1 = 0.5$ and $x_2 = 0.8$.
   - **Expected Output:** The net input is $z = (3.0 \cdot 0.5) + (2.5 \cdot 0.8) - 4.0 = -0.5$. Since $z < 0$, the output is $\hat{y} = 0$, meaning no maintenance is scheduled.

> **Metacognitive Checkpoint:** Why does standard gradient descent fail when training a network of perceptrons that use the Heaviside step function? Explain your answer in terms of the derivative of the activation function and how it affects the chain rule.

---

## Topic 2: The Geometric Interpretation of Decision Boundaries

### Rationale and Mechanics
Imagine you are looking at a scatter plot of houses, where the horizontal axis ($x_1$) is size and the vertical axis ($x_2$) is price. Green dots are "luxury" houses, and red dots are "budget" houses. If you can draw a single straight line that splits the green dots from the red dots, the classes are **Linearly Separable**.

For a Perceptron, the decision boundary is the line where the score is exactly zero:
$$w_1 x_1 + w_2 x_2 + b = 0$$

Under the hood, we can rearrange this equation to solve for $x_2$, which gives us the standard equation of a line ($y = mx + c$):
$$x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$$
where:
- The slope of the line is $-\frac{w_1}{w_2}$ (controlled by the weights).
- The vertical intercept is $-\frac{b}{w_2}$ (controlled by the bias).

Geometrically, the weight vector $\mathbf{w} = [w_1, w_2]^T$ points perpendicular to the decision boundary line. The bias $b$ determines how far the line is shifted from the origin $(0, 0)$. If $b = 0$, the line must pass directly through the origin.

### Python Code Implementation
Here is how to calculate and plot the decision boundary of a Perceptron:

```python
import numpy as np
import matplotlib.pyplot as plt

# Perceptron parameters
w1, w2 = 2.0, -1.0
b = -1.0

# Generate coordinates for plotting the line: w1*x1 + w2*x2 + b = 0
# Rearranged: x2 = -(w1/w2)*x1 - (b/w2)
x1_values = np.linspace(-2, 5, 100)
x2_values = -(w1 / w2) * x1_values - (b / w2)

# Verify points
# Point (2, 1): z = 2*(2) - 1*(1) - 1 = 2 >= 0 => Class 1
# Point (0, 0): z = 2*(0) - 1*(0) - 1 = -1 < 0 => Class 0

plt.plot(x1_values, x2_values, label="Decision Boundary", color="black")
plt.scatter([2], [1], color="green", label="Class 1")
plt.scatter([0], [0], color="red", label="Class 0")
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.legend()
plt.grid(True)
plt.show()
```

### Trade-offs
Drawing a straight line is computationally instant and easy to understand. However, if your data points cannot be separated by a straight line—for example, if the green dots are surrounded by a ring of red dots—a single Perceptron will fail completely. In deep learning, instead of inventing complex curves ourselves, we stack layers of neurons. The hidden layers warp the coordinate space so that the classes become linearly separable by the final layer.

### Real-World Applications (Rule of 4)

1. **Example 1: Property Class Classifier**
   - **Input/Scenario:** A real estate model classifies properties as "Premium" ($1$) or "Standard" ($0$) using features `square_footage` ($x_1$, in thousands of sq ft) and `lot_size` ($x_2$, in acres). The weights are $w_1 = 4.0$, $w_2 = 1.5$, and $b = -10.0$. The decision boundary is the line $4.0x_1 + 1.5x_2 - 10.0 = 0$.
   - **Expected Output:** Rearranging gives $x_2 = -2.67x_1 + 6.67$. Any property lying above this line (e.g., $x_1 = 2.5, x_2 = 2.0 \implies z = 3.0 \ge 0$) is classified as Premium ($1$). Properties below this boundary line are classified as Standard ($0$).
2. **Example 2: Industrial Quality Assurance**
   - **Input/Scenario:** A rod manufacturer classifies components as "Defective" ($1$) or "Acceptable" ($0$) based on deviations in length ($x_1$, in mm) and diameter ($x_2$, in mm). The weights are $w_1 = 0.8$, $w_2 = 1.2$, and $b = -0.6$. The decision boundary is $0.8x_1 + 1.2x_2 - 0.6 = 0$.
   - **Expected Output:** The boundary line is $x_2 = -0.67x_1 + 0.5$. If a component exhibits deviations $x_1 = 0.2$ and $x_2 = 0.4$, the net input is $z = 0.8(0.2) + 1.2(0.4) - 0.6 = 0.04$. Because $z \ge 0$, it lies in the positive half-space and is classified as Defective ($1$).
3. **Example 3: Agricultural Drone Target Selection**
   - **Input/Scenario:** A crop-monitoring drone classifies field patches as "Needs Irrigation" ($1$) or "Sufficient Moisture" ($0$) using `infrared_reflectance` ($x_1$) and `soil_humidity` ($x_2$). The decision boundary is $-2.0x_1 - 0.05x_2 + 4.0 = 0$.
   - **Expected Output:** The boundary line is $x_2 = -40.0x_1 + 80.0$. For a patch with reflectance $x_1 = 1.5$ and soil humidity $x_2 = 50\%$, the net input is $z = -2.0(1.5) - 0.05(50) + 4.0 = -1.5$. Since $z < 0$, the patch falls below the boundary line and is classified as having Sufficient Moisture ($0$).
4. **Example 4: Network Security Filtering**
   - **Input/Scenario:** A packet filter classifies requests as "Malicious" ($1$) or "Benign" ($0$) using features `request_rate_per_sec` ($x_1$, scaled) and `payload_size_kb` ($x_2$, scaled). The decision boundary is $5.0x_1 + 0.1x_2 - 50.0 = 0$.
   - **Expected Output:** The boundary line is $x_2 = -50.0x_1 + 500.0$. A request with $x_1 = 8.0$ and $x_2 = 150.0$ yields $z = 5.0(8.0) + 0.1(150.0) - 50.0 = 5.0$. Since $z \ge 0$, the request lies in the malicious half-space and is classified as Malicious ($1$).

> **Metacognitive Checkpoint:** If you scale all weights $\mathbf{w}$ and the bias $b$ by a positive constant factor $k > 0$ (i.e., new weights are $k\mathbf{w}$ and new bias is $kb$), how does this affect the geometric decision boundary in space? How does it affect the magnitude of $z$ for any arbitrary input $\mathbf{x}$?

---

## Topic 3: The Perceptron Learning Rule

### Rationale and Mechanics
If a model makes a mistake, it must adjust its weights. In smooth models, we use gradient calculus. For the non-differentiable Perceptron, Frank Rosenblatt designed a simple heuristic called the **Perceptron Learning Rule**. It is an error-driven rule: it only adjusts weights when it makes a mistake.

Under the hood, for each training sample with input features $x_i$, true label $y \in \{0, 1\}$, and model prediction $\hat{y} \in \{0, 1\}$:
1. Compute the error: $e = y - \hat{y}$.
   - If the prediction is correct, the error is $0$, and nothing changes.
   - If the model predicted $0$ but the target was $1$ (False Negative), the error is $+1$.
   - If the model predicted $1$ but the target was $0$ (False Positive), the error is $-1$.
2. Update each weight and the bias:
   $$w_i \leftarrow w_i + \eta \cdot (y - \hat{y}) \cdot x_i$$
   $$b \leftarrow b + \eta \cdot (y - \hat{y})$$
   where $\eta$ is the **Learning Rate** (a speed limiter, typically between $0.01$ and $1.0$).

This adjustment is intuitive:
- If the model missed a positive sample ($error = +1$), we increase the weights of active inputs ($x_i > 0$) and the bias, making the neuron more likely to output $1$ next time.
- If the model falsely triggered on a negative sample ($error = -1$), we decrease the weights and bias.

The **Perceptron Convergence Theorem** guarantees that if the training dataset is linearly separable, this rule will find a perfect boundary line in a finite number of steps.

### Python Code Implementation
Here is how to train a Perceptron using the learning rule:

```python
def train_perceptron(X, y, epochs=10, lr=0.1):
    # Initialize weights and bias to zero
    num_features = X.shape[1]
    weights = np.zeros(num_features)
    bias = 0.0
    
    for epoch in range(epochs):
        for i in range(len(y)):
            # Predict
            z = np.dot(weights, X[i]) + bias
            pred = 1 if z >= 0 else 0
            
            # Calculate error
            error = y[i] - pred
            
            # Update weights and bias if there was an error
            if error != 0:
                weights += lr * error * X[i]
                bias += lr * error
                print(f"Update: weights={weights}, bias={bias}")
    return weights, bias

# Tiny training set (AND logic gate)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1])

w, b = train_perceptron(X_train, y_train)
```

### Trade-offs
This learning rule requires no calculus, derivatives, or matrix inversion, making it extremely lightweight. However, if the dataset is *not* linearly separable, the algorithm will loop indefinitely, oscillating back and forth as it tries to resolve conflicting points. In practice, we must set a limit on training epochs to prevent infinite loops.

### Real-World Applications (Rule of 4)

1. **Example 1: Spam Filter Update (No Error Case)**
   - **Input/Scenario:** A perceptron classifies emails as Spam ($1$) or Normal ($0$) using binary features `has_prize_words` ($x_1$) and `has_urgent_subject` ($x_2$). Current weights are $w_1 = 0.5$, $w_2 = 0.2$, and bias $b = -0.6$. A training email arrives with $x_1 = 1$, $x_2 = 1$, and true label $y = 1$. The learning rate is $\eta = 0.1$.
   - **Expected Output:** The net input is $z = 0.5(1) + 0.2(1) - 0.6 = 0.1 \ge 0 \implies \hat{y} = 1$. Since $y = \hat{y}$, the error is $0$. No weights change, and the parameters remain $w = [0.5, 0.2]$ and $b = -0.6$.
2. **Example 2: Spam Filter Update (False Positive Case)**
   - **Input/Scenario:** Using the same email classifier ($w_1 = 0.5$, $w_2 = 0.2$, $b = -0.6$), a legitimate notification email arrives with $x_1 = 1$, $x_2 = 1$, but true label $y = 0$. The learning rate is $\eta = 0.2$.
   - **Expected Output:** The prediction is $\hat{y} = 1$ (since $z = 0.1 \ge 0$), resulting in a false positive error. The error is $y - \hat{y} = -1$. The updates are:
     $\Delta w_1 = 0.2 \cdot (-1) \cdot 1 = -0.2 \implies w_1 \leftarrow 0.3$
     $\Delta w_2 = 0.2 \cdot (-1) \cdot 1 = -0.2 \implies w_2 \leftarrow 0.0$
     $\Delta b = 0.2 \cdot (-1) = -0.2 \implies b \leftarrow -0.8$
     The updated parameters are $w = [0.3, 0.0]$ and $b = -0.8$.
3. **Example 3: Credit Scoring Adjustment (False Negative Case)**
   - **Input/Scenario:** A perceptron screens applicants for high credit risk ($1$) using binary indicators `credit_score_low` ($x_1$) and `debt_high` ($x_2$). Current parameters are $w_1 = 0.1$, $w_2 = 0.1$, and $b = -0.25$. A high-risk applicant ($y = 1$) applies with $x_1 = 1$ and $x_2 = 0$. The learning rate is $\eta = 0.5$.
   - **Expected Output:** The net input is $z = 0.1(1) + 0.1(0) - 0.25 = -0.15$. The prediction is $\hat{y} = 0$, causing a false negative. The error is $1$. The updates are:
     $\Delta w_1 = 0.5 \cdot (1) \cdot 1 = 0.5 \implies w_1 \leftarrow 0.6$
     $\Delta w_2 = 0.5 \cdot (1) \cdot 0 = 0.0 \implies w_2 \leftarrow 0.1$
     $\Delta b = 0.5 \cdot (1) = 0.5 \implies b \leftarrow 0.25$
     The updated parameters are $w = [0.6, 0.1]$ and $b = 0.25$.
4. **Example 4: Factory Temperature Alert Adjustment**
   - **Input/Scenario:** A sensor node classifies boiler status as "Overheating" ($1$) using a binary indicator `temp_critical` ($x_1$). Initial parameters are $w_1 = -0.5$ and $b = 0.1$. An overheating event occurs ($y = 1$) and the sensor reports $x_1 = 1$. The learning rate is $\eta = 1.0$.
   - **Expected Output:** The net input is $z = -0.5(1) + 0.1 = -0.4$. The prediction is $\hat{y} = 0$ (false negative). The error is $1$. The updates are:
     $\Delta w_1 = 1.0 \cdot (1) \cdot 1 = 1.0 \implies w_1 \leftarrow 0.5$
     $\Delta b = 1.0 \cdot (1) = 1.0 \implies b \leftarrow 1.1$
     The updated parameters are $w_1 = 0.5$ and $b = 1.1$.

> **Metacognitive Checkpoint:** Suppose you have a dataset that is not linearly separable. If you run the Perceptron learning rule with a very small learning rate ($\eta = 10^{-6}$), will the model eventually converge to a stable decision boundary? Explain why or why not.

---

## Topic 4: The XOR Problem & The Death of the Single Neuron

### Rationale and Mechanics
While AND and OR logic gates can be split by a single line, the **Exclusive OR (XOR)** logic gate cannot. The XOR gate outputs $1$ if the inputs are different, and $0$ if they are identical:
- $(0, 0) \to 0$
- $(1, 0) \to 1$
- $(0, 1) \to 1$
- $(1, 1) \to 0$

If we plot these four coordinates on a grid, the positive points $(1,0)$ and $(0,1)$ sit diagonally opposite to the negative points $(0,0)$ and $(1,1)$. It is geometrically impossible to draw a single straight line that separates them.

```
       XOR Grid: Cannot separate with a single line
       
              x2
              ^
            1 |  (0,1) [Class 1]       (1,1) [Class 0]
              |
            0 |  (0,0) [Class 0]       (1,0) [Class 1]
              +----------------------------------------> x1
                 0                      1
```

Let's prove this contradiction mathematically. For a single Perceptron to solve XOR, there must exist weights $w_1, w_2$ and a bias $b$ that satisfy the following four inequalities:
1. For $(0, 0)$: $b < 0$
2. For $(1, 0)$: $w_1 + b \geq 0$
3. For $(0, 1)$: $w_2 + b \geq 0$
4. For $(1, 1)$: $w_1 + w_2 + b < 0$

Let's prove the contradiction:
From inequality (1), we establish that $b$ is negative ($b < 0$).
Let's add inequalities (2) and (3):
$$(w_1 + b) + (w_2 + b) \geq 0 \implies w_1 + w_2 + 2b \geq 0$$
We can rewrite this expression as:
$$(w_1 + w_2 + b) + b \geq 0$$
However, inequality (4) states that $(w_1 + w_2 + b)$ is negative, and inequality (1) states that $b$ is negative. Adding two negative numbers must yield a negative number. Thus, we have:
$$(w_1 + w_2 + b) + b < 0$$
This is a direct mathematical contradiction. Thus, no single linear classifier can ever represent the XOR logic function. In 1969, Marvin Minsky and Seymour Papert published this proof, causing funding for neural networks to dry up (initiating the first "AI Winter").

### Python Code Implementation
Here is a demonstration of a single Perceptron failing to learn the XOR gate:

```python
# XOR training data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Try training a single perceptron
w_xor, b_xor = train_perceptron(X_xor, y_xor, epochs=20, lr=0.1)

# Test accuracy
correct = 0
for i in range(len(y_xor)):
    pred = 1 if np.dot(w_xor, X_xor[i]) + b_xor >= 0 else 0
    if pred == y_xor[i]:
        correct += 1
print(f"XOR Training Accuracy: {correct/4 * 100}%")
# Output: Will fluctuate and fail to reach 100% (typically stabilizes at 50% or 75%)
```

### Trade-offs
Understanding this limitation is critical. It shows that single linear neurons are fundamentally limited. To solve XOR, we must add a **Hidden Layer**. The hidden layer maps the data to a new representation where the points *are* linearly separable. Stacking layers allows the network to build complex decision boundaries, moving from a single Perceptron to a Multi-Layer Perceptron (MLP).

### Real-World Applications (Rule of 4)

1. **Example 1: Digital XOR Logic Gate**
   - **Input/Scenario:** Building a neural circuit that implements the XOR logic gate where $x_1, x_2 \in \{0, 1\}$.
   - **Expected Output:** A single perceptron yields 0% training accuracy or infinite training oscillations. A two-layer neural network solves the problem: the hidden layer computes two intermediate features: $h_1 = \text{AND}(x_1, \text{NOT}(x_2))$ and $h_2 = \text{AND}(\text{NOT}(x_1), x_2)$, and the output layer computes $y = \text{OR}(h_1, h_2)$.
2. **Example 2: E-commerce Transaction Verification**
   - **Input/Scenario:** A system flags transactions as "Requires MFA" ($1$) using binary inputs `is_international_ip` ($x_1$) and `is_domestic_shipping_address` ($x_2$). If both are true (international IP shipping to domestic) or both are false (domestic IP shipping to international), the transaction is high risk. If both match (domestic/domestic or international/international), it is low risk.
   - **Expected Output:** This represents a classic XOR distribution in feature space. A single-layer model fails to classify it. A multi-layer neural network with a hidden layer successfully warps the coordinate space to separate the risk categories.
3. **Example 3: Bioinformatics (Synthetic Lethality)**
   - **Input/Scenario:** A biological model predicts cancer cell death ($1$) based on the knockout status of Gene A ($x_1$) and Gene B ($x_2$). The cell dies if Gene A is knocked out OR Gene B is knocked out, but survives if both are active (no effect) or both are knocked out (the cell adapts via alternative pathways).
   - **Expected Output:** Because cell death only occurs with one knockout (an XOR relationship), a single neuron cannot model this interaction. A multi-layer perceptron with non-linear activations is required to map the gene-gene interaction.
4. **Example 4: Sentiment Analysis (Sarcasm Detection)**
   - **Input/Scenario:** A natural language processor detects sarcasm ($1$) based on the presence of `positive_words` ($x_1$) and `negative_punctuation_patterns` ($x_2$, e.g., sarcastic exclamation marks). Sarcasm occurs when positive words are paired with negative punctuation, or vice versa, but not when they align.
   - **Expected Output:** Sarcasm detection represents a non-linear feature interaction that a single linear model cannot separate. A Multi-Layer Perceptron (MLP) learns to project the features into a latent space where sarcasm can be linearly classified.

> **Metacognitive Checkpoint:** If you add a hidden layer to a network but use *linear* activation functions for all hidden neurons, can this network solve the XOR problem? Prove why or why not using matrix multiplication.

---

## Summary & Next Steps

- **The Perceptron is a Linear Decision Maker:** It calculates a weighted sum of inputs and applies a step function to output a hard binary decision.
- **Decision Boundaries slice Space:** A single neuron is restricted to drawing a straight line, plane, or hyperplane, separating linearly separable categories.
- **XOR requires Depth:** A single neuron cannot solve the XOR gate. To resolve non-linear patterns, we must stack neurons into hidden layers.

In the next lesson, we will explore the **Multi-Layer Perceptron (MLP)**, learning how stacking layers and mapping features mathematically resolves the XOR bottleneck.
