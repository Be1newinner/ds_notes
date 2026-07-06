## 1. What You’ll Learn

You’ll learn the differences between the three core computer vision tasks:

- Image classification
- Object detection
- Image segmentation (semantic and instance)

We’ll go beyond the original video and:

- Connect each concept to modern (2026) deep-learning models
- Show real-world use cases
- Sketch code-oriented workflows (Python backend + TypeScript/frontend)

You can treat this as a conceptual foundation before jumping into actual code.

---

## 2. Image Classification

Image classification is the simplest and most common CV task.

### 2.1 What is Image Classification?

- You give the model an image.
- The model outputs a single label (or distribution over labels).
- It answers: “What is in this image (overall)?”

The model does NOT tell you where in the image the object is; just _what_ is present.

### 2.2 Typical Outputs

- A class label, for example:
  - `dog`, `cat`, `car`, `pneumonia`, `benign`, `spam`, etc.
- A probability or confidence score per class, for example:
  - `{"dog": 0.92, "cat": 0.06, "other": 0.02}`

### 2.3 Real-Life Examples (2026)

- Medical diagnosis support
  - Given a chest X-ray, classify as `normal` or `pneumonia`.
  - Used as a “second reader” for radiologists, not a replacement.

- E‑commerce product tagging
  - Auto-tag product photos as `t-shirt`, `jeans`, `sneakers`.
  - Additional labels like `formal`, `casual`, `sport`.

- Content moderation
  - Classify images as `safe` vs `adult` vs `graphic`.
  - Used by social media platforms to triage content.

- Document sorting
  - Classify camera-scanned pages into categories like `invoice`, `resume`, `ID card`, etc.

### 2.4 Modern Architectures (mid‑2026)

Common families you’d use today:

- CNN-based (still relevant but less dominant):
  - ResNet, EfficientNet, MobileNet.

- Vision Transformers (ViT) and variants:
  - ViT (Google), DeiT, Swin Transformer, ConvNeXt-style hybrids.

- Foundation models / multimodal models:
  - CLIP-like models that embed image + text, used for zero-shot classification.

For production, you often:

- Start from a pretrained model (ImageNet or larger).
- Replace the last layer with a new classifier head.
- Fine-tune on your dataset.

### 2.5 Implementation Sketch

Typical Python backend (PyTorch) workflow:

- Load a pretrained model (e.g., ResNet50).
- Replace the final FC layer with your own `nn.Linear` mapping to your number of classes.
- Fine-tune on labeled images.
- Expose prediction via REST API.

High-level pseudo-code (Python / PyTorch):

```python
from torchvision import models, transforms
import torch.nn as nn
import torch

# 1. Load pretrained model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)  # custom number of classes

# 2. Training: standard supervised training loop (omitted here)

# 3. Inference transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

On the TypeScript side (e.g., Next.js):

- Upload image to backend.
- Call `/predict` endpoint.
- Display predicted label and confidence.

---

## 3. From Classification to Localization

Before full object detection, there’s an intermediate concept: **classification with localization**.

### 3.1 Concept

- There is typically one main object in the image.
- You want both:
  - The class of the object.
  - Its location (bounding box).

It answers: “What is in this image, and where is it?”

### 3.2 Output Format

- Class label: `dog`.
- Bounding box: often `x_min, y_min, x_max, y_max` or `x_center, y_center, width, height`.

This is conceptually similar to object detection but restricted to one object.

### 3.3 Real-Life Example

- Retail shelf monitoring:
  - One main product in each image, you want to crop it accurately.

- QR/Barcode scanning:
  - Detect the main code region first, then decode only that region.

### 3.4 Models and Practice

You can train a simple model that outputs both:

- Class logits for category prediction.
- 4 regression outputs for bounding box coordinates.

This kind of network is a gentle intro to the multi-task learning idea used in full detection models.

---

## 4. Object Detection

Object detection generalizes classification + localization to **multiple objects** in an image.

### 4.1 What is Object Detection?

- You input an image.
- The model outputs multiple detections, each with:
  - A bounding box.
  - A class label.
  - A confidence score.

It answers: “Which objects are present, and where is each one?”

### 4.2 Typical Output

- A list like:

```json
[
  { "label": "person", "confidence": 0.97, "bbox": [x1, y1, x2, y2] },
  { "label": "car",    "confidence": 0.93, "bbox": [x3, y3, x4, y4] }
]
```

### 4.3 Real-Life Examples

- Self-driving / ADAS
  - Detect cars, motorbikes, pedestrians, traffic lights, stop signs.
  - Used for both perception and risk estimation.

- Smart CCTV and analytics
  - Count number of people, detect safety helmets, detect suspicious objects.

- Warehouse automation
  - Detect boxes, pallets, forklifts.
  - Used for inventory management and collision avoidance.

- Retail checkout / smart carts
  - Detect items in a cart or on a counter for automated checkout.

### 4.4 Modern Architectures (mid‑2026)

Historically:

- Two-stage detectors: Faster R‑CNN, Mask R‑CNN.
- One-stage detectors: YOLO family, SSD, RetinaNet.

As of 2026, common choices:

- YOLOv8 / YOLOv9 variants (Ultralytics ecosystem, many forks).
- DETR-style transformer detectors:
  - DETR, Deformable DETR, DINO, and more recent efficient transformer-based detectors.
- Open-source general-purpose models:
  - Meta’s Segment Anything derivatives (for segmentation, with detection-like capabilities via boxes/masks).
  - Google’s and OpenAI’s multimodal models that can do “describe objects in this image,” etc.

For practical work:

- If you want a fast, deployable detector: **YOLOv8/YOLOv9** family (PyTorch, ONNX, TFLite, etc.).
- If you want research-grade or transformer-based: **DETR variants**.

### 4.5 Implementation Sketch: YOLO-style

High-level YOLO workflow:

- Define dataset with annotations (bounding boxes + labels).
- Use a YOLO training script with a YAML config describing:
  - Dataset paths.
  - Class names.
- Train or fine-tune.
- Export model to ONNX or TensorRT for deployment.

At inference:

- You pass an image.
- The model outputs bounding boxes, labels, scores.
- You draw boxes on the image or send the raw data to frontend.

In a TypeScript frontend, you might:

- Call a `/detect` endpoint that returns detection JSON.
- Draw rectangles on a `<canvas>` overlay using returned coordinates.

---

## 5. Image Segmentation

Image segmentation provides the **most detailed** scene understanding: labeling every pixel.

### 5.1 What is Image Segmentation?

- For each pixel, the model predicts a class.
- It answers: “Exactly which pixels belong to which object/class?”

Instead of bounding boxes, you get pixel-precise masks.

### 5.2 Semantic vs Instance Segmentation

There are two major types:

- Semantic segmentation
  - Pixels are labeled by class.
  - Instances of the same class are not distinguished.
  - Example: Two dogs in the image → every dog pixel is labeled `dog`, but Dog A and Dog B are not separated.

- Instance segmentation
  - Pixels are grouped into _instances_ for each class.
  - Example: Two dogs → Dog 1 mask and Dog 2 mask, each separate.
  - It combines detection and segmentation: “where is each individual thing and which pixels does it occupy?”

The example in the original video corresponds to **instance segmentation**, where pixels of dog and cat are differently colored, each representing a separate object.

### 5.3 Typical Outputs

- Semantic segmentation
  - A 2D array (H × W) where each value is a class ID.

- Instance segmentation
  - A set of objects, each with:
    - A mask (binary or probabilistic).
    - A label.
    - Optional bounding box and score.

### 5.4 Real-Life Examples

- Medical imaging
  - Segment tumors, organs, lesions in MRI/CT images.
  - Used to estimate volumes, shapes, and treatment planning.

- Autonomous driving
  - Segment road, lane markings, sidewalks, vehicles, pedestrians, vegetation.
  - Helps planning path and understanding drivable area.

- Photo editing and AR
  - Remove background by segmenting people vs background.
  - Apply filters only to foreground objects.

- Agriculture
  - Count plants, measure leaf area, detect disease regions in crops.

### 5.5 Modern Architectures (mid‑2026)

Historically:

- Semantic segmentation: U‑Net, FCN, DeepLabv3(+), PSPNet.
- Instance segmentation: Mask R‑CNN.

Currently:

- Transformer-based segmenters, often built on top of DETR-like frameworks.
- Meta’s Segment Anything Model (SAM) and follow-ups:
  - You provide prompts (points, boxes, text) and get high-quality masks.
  - Very useful as a general segmentation engine.

For most practical projects:

- If you have labels and need a task-specific segmenter:
  - U‑Net-like variants for medical and scientific imaging.
  - DeepLab/HRNet for general semantic segmentation.

- If you want flexible segmentation with minimal training:
  - Use SAM (or similar models) and adapt via prompt-engineering or light fine-tuning.

---

## 6. Conceptual Comparison

Here’s a compact mental model to differentiate the three tasks:

| Task                 | Question Answered                             | Output Type                                        | Example Use Case                      |
| -------------------- | --------------------------------------------- | -------------------------------------------------- | ------------------------------------- |
| Image Classification | What is in this image?                        | Single label (and probability)                     | Is this image a dog or a cat?         |
| Object Detection     | What objects are here, and where?             | Boxes + labels + scores                            | Find all people and cars in the image |
| Image Segmentation   | For each pixel, what class does it belong to? | Pixel-wise labels, masks (optionally per instance) | Precisely outline organs in an MRI    |

A quick intuitive analogy:

- Image classification: “Title of the photo.”
- Object detection: “Tagged boxes around each thing in the photo.”
- Image segmentation: “Color each pixel according to what it belongs to.”

---

## 7. Choosing the Right Task for Your Project

As an SDE / full-stack dev, the choice depends on the **business question**.

### 7.1 When to Use Image Classification

Use classification if:

- You only care about a **global decision**.
- The image has one main object, or localization is irrelevant.
- Examples:
  - Flagging medical images with possible disease.
  - Predicting whether an image is suitable as a thumbnail.
  - Detecting if an industrial component is faulty (yes/no).

Advantages:

- Models are smaller and simpler.
- Easier and cheaper to label (image-level labels).

### 7.2 When to Use Object Detection

Use detection if:

- You need to know **what** is in the image and **where** each object is.
- You need to count objects or interact with them in a UI.
- Examples:
  - Counting people in a store.
  - Detecting objects on a conveyor belt.
  - Detecting vehicles and pedestrians for a parking or traffic system.

Advantages:

- Richer information (locations, count).
- Good tradeoff between complexity and usefulness for many apps.

### 7.3 When to Use Segmentation

Use segmentation if:

- You need **pixel-level** accuracy.
- Object shape and boundaries matter.
- Examples:
  - Measuring tumor volume in 3D scans.
  - Estimating area of farmland or damage from satellite images.
  - High-quality background removal for photos/videos.

Disadvantages:

- More complex models.
- Harder and more expensive to label (pixel-level masks).
- Higher compute costs.

---

## 8. Integrating These Tasks into a Full-Stack Project

Given your stack (TypeScript, Next.js, FastAPI, etc.), here’s a typical architecture.

### 8.1 Backend (Python / FastAPI)

- Serve models (classification/detection/segmentation) as REST or WebSocket APIs.
- Use GPU-accelerated inference where possible (NVIDIA, Apple Silicon, etc.).

Example FastAPI endpoint structure:

- `POST /classify` → returns top class and probabilities.
- `POST /detect` → returns bounding boxes and labels.
- `POST /segment` → returns masks (either as binary images, RLE, or polygon coordinates).

### 8.2 Frontend (Next.js, React, TypeScript)

- Use `<input type="file" />` or drag-and-drop to upload images.
- Display predictions:
  - Classification: show label + confidence.
  - Detection: draw boxes on `<canvas>` overlay on top of the image.
  - Segmentation: overlay masks as semi-transparent colored layers.

For detection overlay, a minimal approach:

- Use HTML `<canvas>` and the original image dimensions.
- Use bounding box coordinates from backend and draw rectangles using the Canvas API.

For segmentation:

- Either return a pre-rendered PNG mask from backend and overlay with CSS.
- Or return polygons and use SVG `<path>` or `<polygon>` elements.

### 8.3 Example: Simple Workflow

Suppose you build an app to analyze street scenes:

- Backend uses a YOLOv8 detector trained on COCO-like data.
- Frontend:
  - User uploads a dashcam photo.
  - Frontend sends POST to `/detect`.
  - Backend responds with detections.
  - Frontend draws colored boxes over the image and lists counts:
    - “2 pedestrians, 3 cars, 1 traffic light.”

Later, you add segmentation:

- Replace or complement YOLO with a segmentation model (e.g., DeepLabv3 or a SAM-based pipeline).
- Backend returns segmentation masks.
- Frontend overlays the drivable area in green, obstacles in red.

---

## 9. Practical Tips and Common Pitfalls

### 9.1 Data Quality

- Garbage in, garbage out.
- Ensure labels are:
  - Consistent (class definitions clear).
  - Sufficiently numerous for each class.
  - Representative of real deployment conditions (lighting, angles, noise).

### 9.2 Dataset Size and Transfer Learning

- For classification:
  - You can start getting decent models with a few thousand labeled images per class using transfer learning.

- For detection/segmentation:
  - Usually need more labeled data per class.
  - Consider starting with pretrained models on COCO or similar and fine-tuning.

### 9.3 Evaluation Metrics

- Classification:
  - Accuracy, precision, recall, F1, AUC.

- Detection:
  - mAP (mean Average Precision) at various IoU thresholds (e.g., mAP@0.5, mAP@0.5:0.95).

- Segmentation:
  - IoU (Intersection over Union), Dice coefficient, pixel accuracy.

### 9.4 Deployment Considerations

- Latency:
  - For real-time systems (e.g., surveillance, robotics), prioritize lightweight models.

- Hardware:
  - Edge devices: use quantized or distilled models (TFLite, TensorRT).
  - Server-side: use GPU or inference-optimized instances.

- Privacy:
  - For sensitive data (medical, faces), consider on-device or private-cloud inference.

---

## 10. How to Practice and Go Further

As of June 2026, good ways to deepen your skills:

- Use open datasets:
  - Classification: CIFAR-10/100, ImageNet subsets, Domain-specific datasets (medical, fashion).
  - Detection: COCO, Pascal VOC, Open Images.
  - Segmentation: Cityscapes, ADE20K, medical segmentation datasets.

- Hands-on exercises:
  - Build a simple classifier for a niche dataset (e.g., Indian traffic signs).
  - Implement a YOLO-based detection system for helmet detection on CCTV frames.
  - Implement a U‑Net-style model for segmenting handwritten digits or simple shapes first, then move to more complex images.

- Integrate into full-stack apps:
  - Create a small Next.js dashboard for visualizing predictions.
  - Add authentication and project-specific logic (e.g., storing annotation corrections to continuously improve your model).
