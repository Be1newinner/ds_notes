# Containerization with Docker

## Learning Objective
Students should understand the problem of environment inconsistency ("it works on my machine") and how Docker containers solve this by packaging an application with all its dependencies.

## What Is This Topic?
Containerization is the process of packaging software code along with all its necessary components—libraries, frameworks, and other dependencies—so that it can run uniformly and consistently on any infrastructure. Docker is the most popular tool for this.

## Why This Topic Matters
Data Science relies heavily on specific library versions (e.g., pandas 1.5 vs pandas 2.0). If you deploy a model to a server that has a different version of scikit-learn than the one you trained on, the model might break. Containerization ensures that the exact environment you used to build the model is replicated perfectly in production.

## Core Intuition
Imagine moving to a new house. Instead of packing your furniture in boxes, disassembling your kitchen, and trying to rebuild it in the new house (hoping it fits), containerization is like moving your entire house, walls and all, via a giant helicopter. When you drop it in the new location, everything inside is exactly as you left it.

## Key Concepts
- **Image:** A read-only template containing instructions for creating a Docker container. It's like a blueprint or a snapshot of an environment.
- **Container:** A runnable instance of an image. If the image is the class definition, the container is the object instance.
- **Dockerfile:** A simple text file containing a list of commands (instructions) that Docker uses to build an image automatically.
- **Requirements.txt:** A Python convention; a text file listing all the Python libraries (and their specific versions) needed for the project.

## Step-by-Step Explanation
1. Write your machine learning code (e.g., an API using FastAPI).
2. Generate a `requirements.txt` file listing the dependencies (e.g., `fastapi`, `uvicorn`, `scikit-learn==1.2.2`).
3. Create a `Dockerfile` in the same directory.
4. In the Dockerfile, specify a "base image" (e.g., an official Python image).
5. Add instructions to copy your code and requirements into the image.
6. Add instructions to install the dependencies (`pip install -r requirements.txt`).
7. Add instructions on what command to run when the container starts.
8. Build the Docker Image using the command line (`docker build`).
9. Run the Docker Container (`docker run`).

## Output / Result Interpretation
When you run a containerized API, the output is a running web server accessible via your browser, but isolated from your host operating system.

## Real-World Uses
- Deploying a model API to cloud platforms like AWS Fargate or Google Cloud Run, which require Docker images as input.
- Sharing a complex deep learning project with a colleague. Instead of them struggling to install CUDA and PyTorch locally, they just run your Docker container.

## Advantages
- **Consistency:** Guarantee that the code runs identically everywhere.
- **Isolation:** The container's dependencies don't interfere with the host machine's software.
- **Scalability:** It is very easy for cloud providers to spin up 100 identical containers if web traffic to your model increases.

## Limitations
- **Learning Curve:** Requires learning Docker syntax and command-line tools.
- **Resource Overhead:** Containers use fewer resources than Virtual Machines, but more than running a script natively.

## Common Mistakes
- **Not pinning versions:** If your `requirements.txt` just says `pandas` instead of `pandas==2.0.0`, Docker will grab the newest version when building, which might break your code.
- **Including unnecessary files:** Copying huge datasets or virtual environments into the Docker image makes it bloated. Use a `.dockerignore` file.

## Code References
- `code/Dockerfile-basic` — A simple Dockerfile for a basic Python script.
- `code/Dockerfile-api` — A Dockerfile specifically for running a FastAPI application.
