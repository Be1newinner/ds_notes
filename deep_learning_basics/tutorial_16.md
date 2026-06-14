# Tutorial 16: TensorBoard Devtools

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=9SdLOcGnebU)

## Executive Summary

TensorBoard is not just a graphing tool—it is a full-stack debugging dashboard. It allows you to peer into the high-dimensional space of your weights, profile hardware bottlenecks, and compare hundreds of experiments side-by-side.

## Advanced Observability Capabilities

### 1. Hardware Profiler

Identifies if your CPU is bottlenecking your GPU. Shows "Input Pipeline" delays and kernel execution times.

### 2. Embedding Projector

Visualizes high-dimensional vectors (like word embeddings) in 3D using PCA or t-SNE algorithms.

### 3. HParams Dashboard

Parallel coordinate plots that correlate hyperparameter choices (LR, Batch Size) with final accuracy.

## Technical Implementation & Comparisons

### Tracking Multiple Experiments

```python
# Import the datetime module to generate unique timestamp strings for log folders
import datetime

# Construct a unique log directory path using the current date and time to avoid overwriting log sessions
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Instantiate a TensorBoard callback object to track metrics, distributions, and profiling details
tensorboard_callback = tf.keras.callbacks.TensorBoard(
   # Specify the destination directory where log files will be written
   log_dir=log_dir,
   # Compute weight histograms after every single training epoch to monitor layer convergence
   histogram_freq=1,
   # Profile hardware execution times specifically from step 500 to 520 to trace bottlenecks
   profile_batch='500,520'
)

# Fit the model on the training data while passing the TensorBoard callback to log training metrics in real-time
model.fit(train_data, callbacks=[tensorboard_callback])
```

### 💡 Beginner's Guide: Launching & Reading TensorBoard

After writing your TensorFlow/Keras code to log details, how do you actually use TensorBoard?

1. **How to Launch it**:
   Open your terminal, navigate to your project directory containing the `logs/` folder, and run:
   ```bash
   tensorboard --logdir logs/fit/
   ```
   This will output a local URL (usually `http://localhost:6006/`). Open this URL in your web browser.

2. **What to Look at First**:
   * **Scalars Tab**: This is where your **Loss** and **Accuracy** curves are plotted epoch-by-epoch.
     * *Healthy Training*: The training loss decreases smoothly, and the validation loss decreases alongside it.
     * *Overfitting*: The training loss keeps going down, but the validation loss starts curving back UP.
     * *High Learning Rate*: The loss line will bounce up and down violently like a zig-zag.
   * **Graphs Tab**: Shows a visual flowchart of your model architecture, letting you verify how your input flows through the layers.

---

### 💡 Supplementary Notes

* **TensorBoard Profiler**: The built-in Profiler is crucial for identifying execution bottlenecks, such as slow data loading pipelines (input bound) or suboptimal kernel execution on the GPU.

## Active Recall Checkpoint

The VRAM Question

Why can high-frequency histogram logging (`histogram_freq=1`) sometimes crash a training job or slow it down significantly?

Optimization Diagnosis

If your Loss curve in TensorBoard is oscillating wildly (massive zig-zags), which hyperparameter should you likely adjust first?