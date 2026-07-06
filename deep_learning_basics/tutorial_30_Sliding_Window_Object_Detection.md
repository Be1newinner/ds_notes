# Sliding Window Object Detection: From Theory to Practice (2026 Tutorial)

This tutorial turns the sliding window object detection concept into a complete, modern, and practical guide, building on the notes from the Codebasics “Sliding Window Object Detection” video but updated and systematized for June 2026.

You’ll learn:

- What sliding window detection is and why it matters historically.
- How to implement it step by step in Python with TensorFlow/Keras.
- How to use an image pyramid, sliding windows, and a CNN classifier.
- How to clean detections using Non-Maximum Suppression (NMS).
- Where this fits relative to YOLO / Faster R-CNN and how you might still use it today.
- How to wrap it into a real-life workflow (e.g., FastAPI backend or ML service).

Your default language is TypeScript, but since this is core CV/ML work, we’ll use Python for the model-side and discuss integration patterns afterwards.

---

## 1. Conceptual Foundations

### 1.1 What Is Sliding Window Object Detection?

Sliding window object detection is an early but foundational technique for finding objects in an image:

- You define a fixed-size rectangular window (e.g., 64×64 pixels).
- You slide that window across the image at regular steps (stride).
- Each window patch is fed into a classifier (often a CNN) that predicts:
  - Does this patch contain the object of interest?
  - Optionally, what class is it (for multi-class detection)?

In its simplest form, you only detect one object category (e.g., “face” or “car”).

### 1.2 Why It’s Important Historically

Before YOLO, SSD, and Faster R-CNN, detection was mostly done via:

- Handcrafted features (HOG, SIFT, etc.).
- Sliding windows over the image.
- A classifier like SVM or boosted decision trees.

This paved the way for:

- The idea of scanning an image spatially.
- The role of “region proposals” (instead of brute-force windows).
- The realization that feature maps can be reused (Fast/Faster R-CNN).

Understanding sliding windows gives you intuition for:

- Why detection is computationally expensive.
- Why modern architectures try to reuse computation.
- How region-based and single-shot detectors improve over brute force.

### 1.3 Why It’s Rarely Used in Production in 2026

In 2026, you almost never ship a sliding-window-based detector to production because:

- It’s very slow:
  - Thousands of windows per image.
  - Each window goes through a CNN.
- Modern detectors (YOLOv8/YOLOv9, DETR variants, Faster R-CNN) are:
  - Faster.
  - More accurate.
  - Easier to use with existing tooling.

But sliding windows are still useful:

- For teaching object detection fundamentals.
- For toy problems or constrained environments.
- For understanding the “region scanning” mindset when reading research papers.

---

## 2. Core Building Blocks

We’ll build a minimal sliding window detector with:

- A simple CNN binary classifier.
- A sliding window generator.
- An image pyramid generator.
- A detection function that:
  - Slides windows across scales.
  - Classifies each window.
  - Collects detections above a threshold.
- Optional Non-Maximum Suppression to clean duplicates.

Assumptions:

- You want to detect a single object category (e.g., “cat”).
- You can prepare training patches (positive and negative).
- You use Python 3.10+ and TensorFlow 2.16+ (or any recent 2.x).

---

## 3. Environment Setup

### 3.1 Python Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow numpy opencv-python matplotlib
```

If you’re on GPU:

```bash
pip install "tensorflow[and-cuda]"
```

(Ensure CUDA/cuDNN compatibility based on TensorFlow’s latest docs for 2026.)

### 3.2 Project Structure

A simple project layout:

```text
sliding_window_detector/
  data/
    train/
      positives/   # patches with object
      negatives/   # patches without object
    val/
      positives/
      negatives/
  models/
  scripts/
    train_cnn.py
    detect.py
  images/
    test_image.jpg
  requirements.txt
  README.md
```

You can adapt to your dataset and structure preferences.

---

## 4. Step 1 – Training a Simple CNN Classifier

The sliding window algorithm relies on a classifier that, given a patch, predicts if the object is present. Let’s define a simple CNN.

### 4.1 Model Definition

We’ll assume input patches of size 64×64×3 (RGB).

```python
# scripts/train_cnn.py
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

PATCH_SIZE = (64, 64)

def load_patches(folder, label):
    X = []
    y = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, PATCH_SIZE)
        X.append(img)
        y.append(label)
    return np.array(X), np.array(y)

def build_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # binary
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    pos_dir = "data/train/positives"
    neg_dir = "data/train/negatives"

    X_pos, y_pos = load_patches(pos_dir, label=1)
    X_neg, y_neg = load_patches(neg_dir, label=0)

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    X = X.astype('float32') / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(input_shape=(64, 64, 3))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/patch_classifier.h5")
    print("Model saved to models/patch_classifier.h5")

if __name__ == "__main__":
    main()
```

### 4.2 Real-Life Example: Simple Car Detector

Imagine you want to detect cars in CCTV images:

- Positive patches:
  - Manually crop 64×64 car regions from road images.
- Negative patches:
  - Randomly crop 64×64 patches from parts of the scene without cars (road, buildings, sky, background).

This mirrors how classic face detection datasets were built (faces vs non-faces).

---

## 5. Step 2 – Implementing Sliding Window and Image Pyramid

### 5.1 Sliding Window Function

The sliding window yields patches across the image:

```python
# scripts/detect.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

PATCH_SIZE = (64, 64)

def sliding_window(image, step_size, window_size):
    """
    Slide a window across the image.

    image: (H, W, C)
    step_size: stride in pixels
    window_size: (width, height)
    """
    h, w = image.shape[:2]
    win_w, win_h = window_size

    for y in range(0, h - win_h + 1, step_size):
        for x in range(0, w - win_w + 1, step_size):
            patch = image[y:y + win_h, x:x + win_w]
            yield x, y, patch
```

Key points:

- If stride is too small (e.g., 4), it’s very slow but precise.
- If stride is too large (e.g., 64), you might miss objects.

A practical middle ground often is 16 or 32 pixels for 64×64 windows.

### 5.2 Image Pyramid Function

To handle objects of different sizes:

```python
def image_pyramid(image, scale=1.5, min_size=(64, 64)):
    """
    Generate image pyramid.

    scale: downscale factor > 1 (e.g., 1.5)
    min_size: stop when width or height below this size
    """
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))

        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        yield image
```

For each scaled version:

- The fixed window size (64×64) will correspond to different real-world sizes in the original image.
- Large objects are detected at coarser scales; small objects at finer scales.

---

## 6. Step 3 – Running Detection Across Scales

Now we combine:

- The trained CNN.
- The image pyramid.
- The sliding window.

### 6.1 Detection Function

```python
def detect_objects(
    image,
    model,
    window_size=(64, 64),
    step_size=32,
    pyramid_scale=1.5,
    threshold=0.9
):
    """
    Returns: list of (x, y, w, h, score) in original image coordinates.
    """
    detections = []

    orig_h, orig_w = image.shape[:2]

    for resized in image_pyramid(image, scale=pyramid_scale, min_size=window_size):
        scale_ratio_x = orig_w / float(resized.shape[1])
        scale_ratio_y = orig_h / float(resized.shape[0])

        for x, y, window in sliding_window(resized, step_size, window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue

            patch = cv2.resize(window, window_size)
            patch = patch.astype('float32') / 255.0
            patch = np.expand_dims(patch, axis=0)

            prob = model.predict(patch, verbose=0)[0][0]

            if prob >= threshold:
                x_orig = int(x * scale_ratio_x)
                y_orig = int(y * scale_ratio_y)
                w_orig = int(window_size[0] * scale_ratio_x)
                h_orig = int(window_size[1] * scale_ratio_y)

                detections.append((x_orig, y_orig, w_orig, h_orig, float(prob)))

    return detections
```

Parameters:

- `threshold`:
  - High threshold (0.9–0.95): fewer false positives.
  - Lower threshold (0.6–0.7): more detections, more noise.
- `pyramid_scale`:
  - 1.5 means each level is 2/3 the width and height of the previous.

### 6.2 Visualization Utility

```python
def draw_detections(image, detections, color=(0, 255, 0)):
    out = image.copy()
    for (x, y, w, h, score) in detections:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out,
            f"{score:.2f}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    return out
```

### 6.3 Running a Test

```python
def run_example():
    model = load_model("models/patch_classifier.h5")

    image = cv2.imread("images/test_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections = detect_objects(
        image,
        model,
        window_size=PATCH_SIZE,
        step_size=32,
        pyramid_scale=1.5,
        threshold=0.9
    )

    print(f"Raw detections: {len(detections)}")

    output = draw_detections(image, detections)

    plt.figure(figsize=(10, 8))
    plt.imshow(output)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    run_example()
```

Real-life scenario:

- Suppose you have a parking lot camera.
- You run this function on a frame from the video to find cars.
- You’ll likely see many overlapping boxes around the same car—that’s why we need Non-Maximum Suppression.

---

## 7. Step 4 – Non-Maximum Suppression (NMS)

NMS merges overlapping detections into a single box per object.

### 7.1 NMS Implementation

```python
def non_max_suppression(boxes, overlap_thresh=0.3):
    """
    boxes: list of (x, y, w, h, score)

    returns: indices of boxes to keep
    """
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2]
    y2 = boxes_np[:, 1] + boxes_np[:, 3]
    scores = boxes_np[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= overlap_thresh)[0]
        order = order[inds + 1]

    return keep
```

### 7.2 Applying NMS to Detections

```python
def run_example_with_nms():
    model = load_model("models/patch_classifier.h5")

    image = cv2.imread("images/test_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections = detect_objects(
        image,
        model,
        window_size=PATCH_SIZE,
        step_size=32,
        pyramid_scale=1.5,
        threshold=0.9
    )

    print(f"Raw detections: {len(detections)}")

    keep_idx = non_max_suppression(detections, overlap_thresh=0.3)
    filtered = [detections[i] for i in keep_idx]

    print(f"Detections after NMS: {len(filtered)}")

    output = draw_detections(image, filtered)

    plt.figure(figsize=(10, 8))
    plt.imshow(output)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    run_example_with_nms()
```

Real-life example:

- Retail analytics:
  - You want to count how many customers are in a store.
  - Without NMS, each person might get 3–5 boxes.
  - After NMS, each person gets one box, making counting feasible.

---

## 8. Real-Life Use Cases (and Why You’d Still Care in 2026)

Even though modern detectors dominate, sliding window is still useful in some real scenarios:

### 8.1 Legacy Systems Modernization

Suppose a client has:

- A legacy C++ system using HOG + SVM + sliding windows for pedestrian detection.
- They want incremental improvements without a full rewrite.

You can:

- Replace the SVM with a lightweight CNN classifier.
- Keep the sliding window logic.
- Gain accuracy while preserving deployment constraints.

### 8.2 Edge Devices with Strict Constraints

Some small MCUs or edge devices:

- Don’t support large modern models.
- Have very limited compute and storage.

You could:

- Train a tiny CNN (perhaps quantized) to classify a small region.
- Use sliding windows only over a very small region of interest.
  - For example, checking a specific gate or door area.

### 8.3 Educational and Interview Preparation

For you as an SDE aiming at ML-heavy roles:

- Knowing sliding window mechanics is valuable when:
  - Explaining how Faster R-CNN built on R-CNN.
  - Discussing computational trade-offs between “scan everything” and “propose first”.
  - Designing simplified detection pipelines in interviews or whiteboard discussions.

---

## 9. Comparison with Modern Detectors (2026 Context)

Here’s a conceptual comparison for understanding, not implementation-level detail:

| Aspect               | Sliding Window + CNN                         | Faster R-CNN / YOLOv8 (Modern)                  |
| -------------------- | -------------------------------------------- | ----------------------------------------------- |
| Computation reuse    | None; each window recomputed                 | High; shared backbone feature maps              |
| Speed on 1080p image | Typically very slow                          | Real-time or near-real-time on GPU              |
| Code complexity      | Conceptually simple, but verbose in practice | More complex, but libraries handle most details |
| Accuracy             | Limited, depends heavily on training patches | High; trained on large datasets (COCO, etc.)    |
| Multi-class support  | Harder to scale cleanly                      | Native multi-class detection                    |
| Deployment ecosystem | Minimal tooling                              | Strong ecosystem, pre-trained models, docs      |

In production (June 2026), you would almost always choose:

- YOLOv8/YOLOv9 (Ultralytics) or YOLOv10 derivatives.
- DETR variants (for research/advanced use).
- Faster R-CNN / RetinaNet via frameworks like Detectron2 or MMDetection.

You’d use sliding window primarily for:

- Teaching.
- Very constrained bespoke systems.
- Domain-specific experiments where simple patches suffice.

---

## 10. Integrating with a Modern Backend (FastAPI / TypeScript Frontend)

Given your stack, a practical setup might be:

### 10.1 FastAPI Backend (Python)

- Expose an endpoint `/detect_sliding_window` that:
  - Accepts an uploaded image.
  - Runs the sliding window detection pipeline.
  - Returns bounding boxes and scores.

Sketch:

```python
# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("models/patch_classifier.h5")

@app.post("/detect_sliding_window")
async def detect_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = detect_objects(
        img,
        model,
        window_size=(64, 64),
        step_size=32,
        pyramid_scale=1.5,
        threshold=0.9
    )

    keep_idx = non_max_suppression(detections, overlap_thresh=0.3)
    filtered = [detections[i] for i in keep_idx]

    result = [
        {"x": x, "y": y, "width": w, "height": h, "score": score}
        for (x, y, w, h, score) in filtered
    ]

    return JSONResponse(content={"detections": result})
```

### 10.2 TypeScript Frontend (Next.js)

- Upload image from Next.js.
- Call the FastAPI endpoint.
- Draw boxes on canvas.

Pseudo-code for the front-end:

```ts
// Example in Next.js (client component)
async function handleImageUpload(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://localhost:8000/detect_sliding_window", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  // data.detections -> draw on canvas or overlay
}
```

This gives you an end-to-end educational demo:

- Model: Python, TensorFlow.
- Service: FastAPI.
- UI: Next.js + TypeScript.

---

## 11. Practical Advice and Next Steps

- For learning:
  - Implement the full pipeline on a tiny dataset (e.g., 100–200 images).
  - Play with stride, threshold, and pyramid scale.
  - Measure how detection quality and speed change.

- For serious work:
  - Move to YOLO or Faster R-CNN:
    - Use Ultralytics YOLO CLI or Python API for quick results.
    - Compare YOLO’s detections and speed with this sliding window implementation.

- For AI/ML depth:
  - Explore why sharing convolutional feature maps (as in modern detectors) solves performance issues.
  - Study how anchor boxes and feature pyramid networks relate conceptually to sliding windows.
