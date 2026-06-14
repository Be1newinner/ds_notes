# Tutorial 6: Deep Learning Frameworks

> Study Guide

[Watch Video Tutorial](https://www.youtube.com/watch?v=z-ZR_8BZ1wQ)

## Executive Summary

The choice of framework is no longer about syntax, but about **Execution Models**. We compare PyTorch's "Define-by-Run" (Eager) approach with TensorFlow's "Define-and-Run" (Graph) approach, and how Keras 3.0 bridges this gap by targeting JAX, PyTorch, and TF.

---

## 🚗 The Engine vs. Dashboard Analogy

To understand deep learning frameworks, imagine building or driving a car:
*   **TensorFlow & PyTorch are the Engines:** They are heavy-duty mathematical engines. They handle the hard math (like matrix multiplications and computing gradients) and talk to the GPU/TPU hardware. Working directly with them can feel like tuning engine pistons by hand.
*   **Keras is the Dashboard & Steering Wheel:** You don't want to manually trigger fuel injectors to accelerate; you just want to press the gas pedal. Keras is a high-level user interface that sits *on top* of the engine (TensorFlow, PyTorch, or JAX) and lets you control it with clean, simple commands.

---

## Technical Deep Dive: Architectural Paradigms

### 1. Imperative (Eager) vs. Declarative (Graph)

How the mathematical engine handles computations determines your debugging experience and production speed.

*   **Imperative (PyTorch):** Calculations happen immediately as lines of code run. Easy to debug with standard Python tools (`pdb`). Often referred to as **"Define-by-Run."**
*   **Declarative (TensorFlow 1.x / JAX):** Builds a static "blueprint" or "computational graph" first, then executes it with data. Harder to debug, but highly optimizable for hardware execution. Often referred to as **"Define-and-Run."**

### 2. The Multi-Backend Revolution (Keras 3.0)

Keras 3.0 is a complete rewrite that abstracts the backend engine. Write code in Keras, and choose your engine:
*   Use the **JAX** backend for XLA compiler speed.
*   Use the **PyTorch** backend to leverage the Torch ecosystem.
*   Use the **TensorFlow** backend to deploy to mobile/web via TFLite/TF.js.

---

## 💡 Beginner's Framework Guide

| Feature / Aspect | PyTorch | TensorFlow | Keras | JAX |
| :--- | :--- | :--- | :--- | :--- |
| **Role** | Engine | Engine | Dashboard / API | Engine |
| **Primary Style** | Imperative (Eager, like standard Python) | Hybrid (Eager by default, Graph for speed) | High-level API (multi-backend) | Pure Functional (Stateless, math-heavy) |
| **Ease of Learning** | **High** (Feels like writing NumPy) | **Medium** (Larger API surface, complex details) | **Very High** (Extremely intuitive Lego-like building) | **Medium-Low** (Requires functional programming mindset) |
| **Debugging** | **Easy** (Standard Python debuggers work) | **Medium** (Easy in eager mode, hard inside compiled graphs) | **Very Easy** (Very clear errors for architecture mismatches) | **Medium** (Compilation errors can be cryptic) |
| **Best Used For** | Research, custom layers, academic papers | Enterprise production, deployment pipelines | Quick prototyping, standard models, beginner projects | High-performance scientific computing, TPU-based training |

### Key Framework Breakdowns

*   **Keras:** Minimizes boilerplate code. Stacking layers is as simple as calling `Sequential([Dense(64), Dense(10)])`. Ideal if you want a working prototype quickly without worrying about raw math details.
*   **PyTorch:** Highly customizable. You define exactly how data flows through your layers inside a custom Python class. Preferred by researchers because it behaves like standard Python.
*   **TensorFlow:** Built with production in mind. It has the most mature suite of deployment tools (TensorFlow Serving, TensorFlow Lite, TensorFlow.js) for shipping models to servers, mobile devices, and browsers.
*   **JAX:** Built for high-performance computing. It treats neural networks as pure mathematical functions and uses an XLA compiler for maximum GPU/TPU acceleration.

**Rule of Thumb for Beginners:** 
*   If you want the **absolute easiest entry point**, start with **Keras** (using TensorFlow or PyTorch backend).
*   If you want to **read academic code** and build custom architectures from scratch, learn **PyTorch**.
*   If you need to **deploy models to production** at scale (especially web/mobile), learn **TensorFlow**.

---

### 💡 Supplementary Notes

* **PyTorch 2.0 vs JAX**: PyTorch 2.0 introduces `torch.compile()` which enables XLA/Triton graph execution on top of its eager model, minimizing compile overhead. JAX operates on purely functional programming principles (stateless functions), which makes it highly parallelizable and popular in TPU-based large model pre-training.

## Active Recall Checkpoint
1

Engine vs. API

Explain the analogy: "Keras is to TensorFlow as the User Interface is to the Operating System."
2

JAX's Edge

Why would a developer use Keras 3.0 with a JAX backend for training, but perhaps switch back to PyTorch for research experimentation?