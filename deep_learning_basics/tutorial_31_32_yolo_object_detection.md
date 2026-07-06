# End‑to‑End Guide to YOLO Object Detection (2026 Edition)

I’ll keep the flow:

1. Conceptual understanding of YOLO (v1–v4 level).
2. Installing and running YOLOv4 (darknet).
3. Practical usage patterns and real‑world examples.
4. How this relates to modern YOLO variants (YOLOv5–YOLOv9) as of 2026.
5. Suggestions for integrating with your own apps.

---

## 1. What Problem Does YOLO Solve?

Object detection has to answer two questions for an image:

- What objects are present?
- Where are they in the image (bounding boxes)?

Historically, the pipeline looked like this:

- Run a **sliding window** (crop) over many positions and scales.
- Classify each crop using a CNN (or HOG+SVM etc.).
- Aggregate positive windows as detections.

This is:

- Slow (thousands of windows per image).
- Hard to tune.
- Complex to implement.

Region‑based methods (R‑CNN, Fast R‑CNN, Faster R‑CNN) improved speed and accuracy but still needed multiple steps:

- Region proposal.
- Feature extraction.
- Classification and box regression.

YOLO’s core philosophy:

- Treat detection as a **single regression problem** from image pixels to bounding boxes and class probabilities.
- Run the network **once per image** (hence “You Only Look Once”).

Real‑life example:

- Safety camera on a factory floor:
  - Needs to detect humans near machines in real time.
  - Sliding window or multi‑stage systems may be too slow.
  - YOLO allows running 30–60 FPS on a GPU, enabling live alarms.

---

## 2. From Classification to Single‑Object Localization

Before multi‑object detection, let’s understand the simpler step.

### 2.1 Pure Image Classification

Given an image, say 224×224×3, a classifier outputs:

- A label: dog or person.
- Possibly a probability distribution across classes.

Example output:

- Dog: 0.93
- Person: 0.07

There is **no information** about where the dog or person is in the image.

### 2.2 Single Object Localization (One Object Per Image)

For localization, we want:

- Class.
- Bounding box.

A convenient representation for **one** object is a 7‑dimensional vector:

- \(p_c\): probability that an object is present at all.
- \(b_x, b_y\): center of bounding box (relative to image or some normalized scale).
- \(b_w, b_h\): width and height of bounding box.
- \(c_1, c_2,\dots\): one‑hot encoded class indicators.

For a 2‑class example (dog / person):

- \(c_1\): dog.
- \(c_2\): person.

Examples:

1. Image with a dog:
   - \(p_c = 1\).
   - \(b_x, b_y, b_w, b_h\) = bounding box around dog.
   - \(c_1 = 1, c_2 = 0\).

2. Image with a person:
   - \(p_c = 1\).
   - \(b_x, b_y, b_w, b_h\) = bounding box around person.
   - \(c_1 = 0, c_2 = 1\).

3. Image with no relevant object:
   - \(p_c = 0\).
   - Remaining outputs irrelevant.

The network is trained on many such pairs \((x^{(i)}, y^{(i)})\).

Limitation:

- Only works if **there is at most one object** in the image.

Real‑life example:

- Medical imaging: one tumor per scan.
  - Single object localization might be sufficient.

But most everyday scenes have **multiple** objects.

---

## 3. Multi‑Object Detection and the YOLO Trick

The challenge:

- Number of objects per image is variable and unknown in advance.
- You cannot simply fix the output as “10 objects × 7 numbers” and expect that to handle all cases cleanly.

### 3.1 YOLO’s Grid Idea

YOLO divides the image into an \(S \times S\) grid.

- Typical original YOLO: 7×7.
- YOLOv3/v4: feature maps like 13×13, 26×26, 52×52 at different scales.
- Conceptually: let’s imagine 4×4 for simplicity.

Rules:

- Each spatial cell is responsible for detecting objects whose **center** falls inside that cell.

Each cell outputs a small vector:

- \(p_c\): objectness (probability that any object is present).
- \(b_x, b_y, b_w, b_h\): bounding box of the object (relative to the cell).
- \(c_1, c_2, ..., c_K\): class probabilities or one‑hot encoding across K classes.

So per cell we have (for K = 2):

- 1 + 4 + 2 = 7 values.

The entire network output becomes a tensor:

- Shape: \(S \times S \times (5+K)\) (e.g., 4×4×7).

Interpretation:

- For each cell:
  - If \(p_c\) is low (≈0): no object.
  - If \(p_c\) is high: cell believes there is an object with the given box and class.

Why this solves multi‑object?

- Multiple objects → multiple centers.
- Each center falls into a different cell.
- Network outputs **one object prediction per cell** (per anchor, more on this soon).

Real‑life example:

- Traffic monitoring camera:
  - Cars, bikes, pedestrians.
  - Each object’s center lies inside some cell.
  - YOLO’s output grid contains one or more confident boxes per cell, which you can convert into the final detections.

---

## 4. Network Output, Anchors, and Multiple Objects in a Cell

### 4.1 Basic Per‑Cell Output

For a cell with **one** object:

- \(p_c = 1\).
- \(b_x, b_y\): offsets within the cell, often in. [youtube](https://www.youtube.com/watch?v=ag3DLKsl2vk)
- \(b_w, b_h\): width and height relative to image or anchor box.
- \(c_i\): probability of each class.

Cells without any object:

- \(p_c = 0\).
- Box coordinates ignored in loss.

This is enough if:

- At most one object’s center falls into each cell.

But reality is messy.

### 4.2 Multiple Objects in a Single Cell

If two objects’ centers fall into the same cell:

- A single 7‑dimensional vector is not enough.

YOLO solution:

- **Anchor boxes** (a.k.a. priors).
- Each cell predicts **B** boxes (e.g., B=3).
- For each anchor:
  - \(p_c^{(b)}\)
  - \(b_x^{(b)}, b_y^{(b)}, b_w^{(b)}, b_h^{(b)}\)
  - class scores

- Now per cell output size is:
  - \(B \times (5 + K)\).

Why anchors?

- Real objects vary in shape/size.
- You pre‑define shapes (anchors) based on clustering of training dataset boxes.
- Network learns residual adjustments over these anchors.

So multi‑object in one cell is handled by:

- Multiple anchors.
- Each anchor ideally learning to specialize in certain shapes/aspect ratios.

Real‑life example:

- Retail shelf monitoring:
  - Different product sizes (bottles, boxes).
  - Anchors approximate typical shapes, the network refines.

---

## 5. Non‑Max Suppression (NMS) and IOU

Once you have predictions per cell × anchors, you’ll often have multiple overlapping boxes for the **same** object.

You must:

- Keep the best one.
- Remove duplicates.

### 5.1 Intersection over Union (IOU)

IOU between two boxes A and B:

- Intersection area: overlapping region.
- Union area: total area covered by A ∪ B.
- IOU = (Intersection area) / (Union area).

Properties:

- IOU = 1 → boxes identical.
- IOU = 0 → no overlap.
- Larger IOU → more similar.

### 5.2 Non‑Max Suppression (per class)

For each class independently:

1. Collect all predicted boxes with confidence above a certain threshold.
2. Sort them by confidence descending.
3. Pick the top box, add to final list.
4. Remove all remaining boxes that have IOU > threshold (e.g., 0.5) with the selected box.
5. Repeat until no boxes left.

This ensures:

- One box per object instance.
- Duplicate boxes for the same object are suppressed.

Real‑life example:

- Vehicle counting:
  - Raw predictions may contain many overlapped boxes per car.
  - NMS collapses these into one box per car → stable counting.

---

## 6. YOLO Versions in Context (Up to 2026)

High‑level timeline and what matters in practice:

- YOLOv1 (2016): first version, single scale, moderate accuracy.
- YOLOv2 (2017): better accuracy, 67 FPS demo, multi‑scale training.
- YOLOv3 (2018):
  - FPN‑like multi‑scale outputs.
  - Very popular baseline for years.
- YOLOv4 (2020, AlexeyAB):
  - Many “bag of freebies” improvements: Mish / CSPDarknet, CIoU loss, data augmentation, etc.
  - Implemented in the **darknet** framework.
- YOLOv5 (2020–…):
  - Implemented in PyTorch by Ultralytics.
  - Very popular due to ease of use, ecosystem.
- YOLOv6, YOLOv7, YOLOv8, YOLOv9 (2022–2025+):
  - Multiple implementations (Meituan, community, Ultralytics).
  - YOLOv8/v9 (Ultralytics) widely used for practical projects: easier training, export (ONNX, TensorRT, Web), multi‑task (segmentation, pose).

Where YOLOv4 still shines:

- Understanding **classical YOLO** concepts.
- Learning how a pure C/C++ GPU program (darknet) works.
- Good baseline when GPU support is already tuned for darknet.

Where newer YOLOs are often better (for a production developer in 2026):

- Pythonic workflows (Ultralytics YOLOv8/YOLOv9).
- Easier custom training.
- Better export and integration (e.g., ONNX, TFLite, WebGPU).
- More model variants (nano, small, medium, large).

For this tutorial we’ll **use YOLOv4** (to stay consistent with the videos) but I’ll add remarks on how you’d translate this knowledge to YOLOv8/YOLOv9.

---

## 7. Setting Up YOLOv4 (Darknet) – Modernized Flow

I’ll describe a Windows + NVIDIA GPU setup first, then high‑level Linux notes.

### 7.1 Prerequisites (Windows, 2026)

- Windows 10/11 64‑bit.
- NVIDIA GPU with recent drivers.
- **Visual Studio 2019/2022 Community** (Desktop development with C++).
- **CUDA Toolkit**:
  - Choose a version compatible with both your GPU and the specific darknet branch.
  - Historically YOLOv4 was usually built with CUDA 10.x–11.x. In 2026, check AlexeyAB repo for recommended CUDA version.
- **CMake** (latest).
- **Git**.

Note:

- The vcpkg approach from the video is still valid.
- Today, some people also directly set up OpenCV/CUDA manually, but vcpkg remains convenient.

### 7.2 Install/Clone vcpkg

From an elevated PowerShell:

```powershell
# Choose a workspace
mkdir C:\dev
cd C:\dev

# Clone vcpkg
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat
```

Install darknet dependencies:

```powershell
.\vcpkg.exe install darknet[opencv,cuda,cudnn]:x64-windows
```

- This may take a while.
- Ensure no errors; if there are, fix them (e.g., CUDA version, MSVC toolchain).

### 7.3 Clone the Darknet Repository

In `C:\dev`:

```powershell
cd C:\dev
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
```

### 7.4 Configure and Build Darknet

There are two main ways:

1. Via `build.ps1` script (as in the video).
2. Via CMake + Visual Studio.

#### Option 1 – Using build.ps1

Try:

```powershell
.\build.ps1
```

If you get execution policy errors:

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

Wait for the build to complete. You should end up with:

- `darknet.exe` in the `darknet` directory.

#### Option 2 – Using CMake (more standard in 2026)

From PowerShell:

```powershell
cd C:\dev\darknet
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake -A x64 -DENABLE_CUDA=ON -DENABLE_CUDNN=ON -DENABLE_OPENCV=ON
cmake --build . --config Release
```

Adjust the `vcpkg.cmake` path as needed.

After build, `darknet.exe` should be in `build` or a subdirectory like `build\Release`.

---

## 8. Downloading YOLOv4 Pre‑Trained Weights

From the official documentation or mirrors:

- Download `yolov4.weights`.
- Place it in the same folder as `darknet.exe`.

Directory structure example:

```text
C:\dev\darknet
  ├─ darknet.exe
  ├─ yolov4.weights
  ├─ cfg\
  ├─ data\
  └─ ...
```

The `cfg` folder contains `yolov4.cfg` and `coco.data` (or similar).

---

## 9. Running YOLOv4 on Sample Images

### 9.1 Basic Command

From PowerShell in the `darknet` folder:

```powershell
.\darknet.exe detector test cfg\coco.data cfg\yolov4.cfg yolov4.weights data\dog.jpg
```

What each argument means:

- `detector test`: run detection in test mode (no training).
- `cfg\coco.data`: dataset description (path to names file, train/test sets, etc.).
- `cfg\yolov4.cfg`: network architecture.
- `yolov4.weights`: pre‑trained weights.
- `data\dog.jpg`: input image.

### 9.2 Outputs

The program will:

- Print detection results in console:
  - `dog: 96%`
  - `bicycle: 92%`
  - `truck: 88%`
  - etc.

- Save an output image (often `predictions.jpg`) with:
  - Bounding boxes around detected objects.
  - Labels and confidence scores.

Real‑life example:

- You can replace `dog.jpg` with a frame extracted from a warehouse CCTV, and see detections for pallets, forklifts, people, etc. (restricted to COCO classes).

---

## 10. Running YOLOv4 on Your Own Images

### 10.1 Prepare Your Image

- Copy your image to `data/` directory.
  - Example: `data\factory_floor.jpg`.

### 10.2 Detection Command

```powershell
.\darknet.exe detector test cfg\coco.data cfg\yolov4.cfg yolov4.weights data\factory_floor.jpg
```

Outputs:

- Predictions printed in console.
- Annotated image `predictions.jpg` with bounding boxes (people, trucks, chairs, etc. as per COCO).

Real‑life example (for you):

- At Seekho Computer:
  - Take a classroom photo.
  - Run detection to show students the bounding boxes for “person”, “chair”, “laptop”, etc.
- At Shipsar Developers:
  - Show a demo to a client with their shop floor images.

---

## 11. Understanding Misclassifications and COCO Labels

The pre‑trained YOLOv4 we’re using is trained on COCO.

- COCO includes 80 object categories (for detection):
  - person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, …
  - laptop, cell phone, tv, bottle, etc.

If your object is **not** among these 80:

- The network will still try to “guess” some class from this set.
- Example from the video:
  - Image with a hammer and laptop.
  - Laptop is detected correctly.
  - Hammer is misclassified as “toothbrush”.

Why?

- COCO might not have a dedicated “hammer” label for detection.
- The shape/texture resembles some toothbrush examples in the training data.
- The model has no choice but to produce one of its known labels.

Real‑life implication:

- Using COCO pre‑trained models:
  - Fine for generic objects in daily scenes.
  - Not good for domain‑specific items (industrial tools, medical instruments, Indian snacks, etc.).

To handle specific classes (e.g., types of factory tools, brand‑specific products):

- You must **train on a custom dataset** with your desired labels.

---

## 12. Training YOLO on a Custom Dataset (Conceptual Overview)

While the video only hints at this, here’s a modern high‑level flow.

Steps:

1. **Collect images** with your target objects (e.g., helmets, safety vests).
2. **Annotate** bounding boxes:
   - Tools: LabelImg, CVAT, Roboflow, Label Studio.
   - Output format: typically YOLO text files or VOC XML or COCO JSON.
3. **Organize data**:
   - `images/train`, `images/val`.
   - `labels/train`, `labels/val`.
4. **Create config files**:
   - `.data` file referencing:
     - Train list.
     - Val list.
     - Names file.
   - `.names` file listing class names line by line.
   - Custom `.cfg` based on `yolov4.cfg` with modified:
     - Filters, classes in [yolo] layers.
5. **Train** with darknet:

   ```bash
   ./darknet detector train custom.data custom.cfg yolov4.conv.137
   ```

6. **Evaluate**:
   - Check mAP.
   - Inspect predictions on validation images.

In 2026, if you use Ultralytics YOLOv8/YOLOv9 instead, training is even simpler:

- `yolo task=detect mode=train data=custom.yaml model=yolov8s.pt`

But conceptually, it’s the same YOLO philosophy.

---

## 13. Real‑Life Use Cases and Considerations

### 13.1 Traffic and Road Safety

- Use YOLO to:
  - Detect cars, bikes, pedestrians, traffic lights.
  - Count vehicles.
  - Detect red‑light violations.

Key considerations:

- Camera mounting and perspective.
- Latency requirements.
- Night‑time performance (may need custom training).

### 13.2 Retail Analytics

- Count customers entering/exiting.
- Detect product placement and stock levels.
- Trigger alerts if shelf is empty.

Challenges:

- Occlusion (crowded scenes).
- Tiny objects (small products).
- Privacy (blur faces, anonymize data).

### 13.3 Industrial Safety

- Detect:
  - People near dangerous zones.
  - Missing helmets or safety vests (with custom training).
- Integrate with:
  - Alarm systems.
  - Access control (e.g., automatic door closure).

### 13.4 Education and Demos

At Seekho Computer, you can:

- Show the entire path:
  - Image → grid → per‑cell predictions → NMS → final boxes.
- Have students modify:
  - Confidence thresholds.
  - IOU thresholds.
- Compare:
  - YOLOv4 (darknet) vs YOLOv8 (Ultralytics PyTorch).

---

## 14. How YOLOv4 Knowledge Transfers to Modern YOLO (v8/v9)

While YOLOv8/YOLOv9 are implemented differently (PyTorch, modular), the core ideas are the same:

- Grid/feature map predictions.
- Anchors (or anchor‑free variants).
- Objectness scores.
- Class probabilities.
- NMS post‑processing.

For practical projects in 2026:

- YOLOv8/v9 are generally recommended for:
  - Easier Python API.
  - Built‑in training and evaluation.
  - Better export (ONNX, TensorRT, CoreML, WebGPU).

Your YOLOv4 understanding is still vital because:

- It teaches the underlying math and logic.
- Helps debug any modern YOLO system.
- Makes it easier to adapt or write custom layers.

---

## 15. Integrating YOLO into Your Applications

As a full‑stack TS/Python dev, here’s how you might wire things up.

### 15.1 Python Service (FastAPI)

- Wrap YOLO inference as a REST API:
  - Endpoint: `POST /detect` with an image.
  - Response: list of detected boxes with class, confidence.

Conceptual steps:

1. Load model on startup (YOLOv4 via darknet Python bindings, or YOLOv8 via Ultralytics).
2. For each request:
   - Decode image.
   - Run detection.
   - Return JSON of bounding boxes.

### 15.2 TypeScript Frontend / Next.js App

- Build FE that:
  - Lets users upload images.
  - Sends them to the detection API.
  - Draws bounding boxes in a `<canvas>` over the image.

You could also:

- Build a dashboard for:
  - Live camera feeds.
  - Historical detection logs.
  - Analytics (counts per hour/day, heatmaps).

### 15.3 Mobile (React Native + Expo)

- Use a native module or run ONNX/TFLite model on‑device for simple cases.
- Or stream frames to a backend model service.

Real‑life example:

- A mobile app for field technicians:
  - They point the camera at equipment.
  - YOLO recognizes components, shows checklists, safety instructions.
