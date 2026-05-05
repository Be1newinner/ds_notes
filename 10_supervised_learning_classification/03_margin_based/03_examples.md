# Examples: Support Vector Machines

Here is a breakdown of the Python examples provided in the `code/` directory.

## 1. Basic Linear SVM (`example-01-basic-svm-linear.py`)
- **Goal:** Introduce the mechanics of a hard-margin vs. soft-margin SVM on linearly separable data.
- **Dataset:** A synthetic 2D dataset that can be cleanly separated with a straight line.
- **Key Concepts Shown:** 
  - Using `SVC(kernel='linear')`.
  - Extracting and visualizing the `support_vectors_`.
  - Demonstrating how changing the `C` parameter alters the margin size.
- **Takeaway:** The concept of the margin and how Support Vectors are the only points that dictate the boundary.

## 2. SVM Kernels (`example-02-svm-kernels.py`)
- **Goal:** Show why the Kernel Trick is the defining feature of SVMs.
- **Dataset:** The `make_circles` dataset—data points arranged in two concentric circles (impossible to separate with a straight line).
- **Key Concepts Shown:** 
  - Training a `linear` kernel (which completely fails).
  - Training an `rbf` (Radial Basis Function) kernel (which perfectly separates the circles).
- **Takeaway:** Kernels allow SVMs to draw complex, non-linear boundaries by mathematically projecting the data into higher dimensions.

## 3. Real-World Scenario: Medical Diagnosis (`example-03-real-world-medical.py`)
- **Goal:** Apply SVM to a complex dataset with many features.
- **Dataset:** The classic Breast Cancer Wisconsin dataset (predicting Malignant vs. Benign).
- **Key Concepts Shown:** 
  - The absolute necessity of `StandardScaler` for SVMs.
  - Setting `probability=True` to get probability scores (which SVMs don't do by default).
  - Interpreting the classification report.
- **Takeaway:** SVMs are incredibly powerful for high-dimensional medical data, provided the data is scaled properly.
