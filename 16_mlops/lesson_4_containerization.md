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


---

## Methods, Options, and Properties: Docker

This document explains the common commands used inside a `Dockerfile` and the command-line interface for Docker.

### 1. Dockerfile Commands

A `Dockerfile` is a text file that contains the instructions to build an image.

- **`FROM`**: The starting point. Specifies the base image. For Python projects, this is usually an official Python image.
  - *Example:* `FROM python:3.9-slim` (The `-slim` version is a smaller, optimized version).
- **`WORKDIR`**: Sets the working directory inside the container. All subsequent commands will be run from this folder.
  - *Example:* `WORKDIR /app`
- **`COPY`**: Copies files from your host computer into the container.
  - *Example:* `COPY requirements.txt .` (Copies `requirements.txt` to the current `WORKDIR`).
  - *Example:* `COPY . .` (Copies everything from the host directory to the `WORKDIR`).
- **`RUN`**: Executes a command during the *build* process (used for installing things).
  - *Example:* `RUN pip install -r requirements.txt`
- **`EXPOSE`**: Documents which port the container will listen on at runtime. It does not actually publish the port; it's mostly for documentation.
  - *Example:* `EXPOSE 8000`
- **`CMD`**: Provides the default command to run when a container starts. There can only be one `CMD`.
  - *Example:* `CMD ["python", "my_script.py"]`
  - *Example (for APIs):* `CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]`

---

### 2. Docker CLI Commands

These are the commands you type into your terminal to interact with Docker.

#### Building an Image
- **`docker build -t image_name .`**: Builds an image using the `Dockerfile` in the current directory (`.`). The `-t` flag tags the image with a readable name.
  - *Example:* `docker build -t my_ml_api .`

#### Running a Container
- **`docker run image_name`**: Starts a container from the image.
- **`-p host_port:container_port`**: Maps a port on your local machine to a port inside the container. Essential for APIs.
  - *Example:* `docker run -p 8000:8000 my_ml_api` (Forwards your computer's port 8000 to the container's port 8000).
- **`-d`**: Runs the container in "detached" mode (in the background) so it doesn't block your terminal.
  - *Example:* `docker run -d -p 8000:8000 my_ml_api`

#### Managing Containers and Images
- **`docker ps`**: Lists all running containers.
- **`docker ps -a`**: Lists all containers (running and stopped).
- **`docker images`**: Lists all downloaded/built images on your machine.
- **`docker stop container_id`**: Stops a running container.
- **`docker rm container_id`**: Removes a stopped container.
- **`docker rmi image_name`**: Removes an image.

### Typical Workflow
1. Write python code and `requirements.txt`.
2. Write `Dockerfile`.
3. Terminal: `docker build -t my_model_app .`
4. Terminal: `docker run -p 8080:8080 my_model_app`
5. Test the app in browser (`localhost:8080`).

---

## Examples: Containerization

This document outlines the practical examples provided for learning Docker.

### Code References

- `code/example-01-basic.py` & `code/Dockerfile-basic`: 
  - **Concept:** Containerizing a simple Python script that runs and exits.
  - **How to use:** Navigate to the `code` folder. Run `docker build -f Dockerfile-basic -t basic-script .` followed by `docker run basic-script`.
- `code/example-02-api.py` & `code/Dockerfile-api` & `code/requirements.txt`:
  - **Concept:** Containerizing a FastAPI web server. This requires exposing ports and handling dependencies.
  - **How to use:** Navigate to the `code` folder. Run `docker build -f Dockerfile-api -t my-fastapi-app .` followed by `docker run -p 8000:8000 my-fastapi-app`. Then open `localhost:8000` in your browser.

### Important Note on Context
When running `docker build`, the `.` at the end tells Docker to use the current directory as the "build context". The Dockerfile will only be able to `COPY` files that exist inside this directory. Therefore, always run `docker build` from the directory where your project files live.

---

## Practice: Containerization

These exercises will test your ability to write Dockerfiles and build images.

### Exercise 1: Containerize a Script
1. Write a very simple Python script (`hello.py`) that prints "Hello from inside the container!" and imports the `numpy` library to print an array of zeros.
2. Create a `requirements.txt` containing `numpy`.
3. Write a `Dockerfile` that:
   - Uses `python:3.9-slim`
   - Copies `requirements.txt` and installs it.
   - Copies `hello.py`.
   - Runs `python hello.py` as the CMD.
4. Build the image and tag it `hello-container`.
5. Run the container. It should print the text and the array, and then stop.

### Exercise 2: Containerize a Streamlit App
1. Go back to `01_model_deployment` and copy the `example-01-basic.py` (the Streamlit app) into a new folder.
2. Create a `requirements.txt` containing `streamlit` and `pandas`.
3. Write a `Dockerfile`.
   - *Hint:* Streamlit runs on port 8501 by default.
   - *Hint:* The command to run Streamlit is `streamlit run filename.py`. You will need to structure your CMD instruction as `CMD ["streamlit", "run", "example-01-basic.py", "--server.port", "8501"]`.
4. Build the image as `my-streamlit-app`.
5. Run the container, making sure to map port 8501 on your host to port 8501 on the container (`-p 8501:8501`).
6. Access the app in your browser.

### Exercise 3: Cleaning Up
1. Use `docker ps -a` to see all the containers you've run today.
2. Use `docker rm` to delete them.
3. Use `docker images` to see the images you built.
4. Note how large the images are (often several hundred megabytes).

---

## Interview Questions: Containerization

### Beginner Questions
1. **What is the difference between an Image and a Container in Docker?**
   - *Answer concept:* An image is a static blueprint or file (like a class definition). A container is a running instance of that image (like an object).
2. **What does a `Dockerfile` do?**
   - *Answer concept:* It contains the step-by-step instructions needed to assemble a Docker image.
3. **Why do we need a `requirements.txt` file when building a Python Docker image?**
   - *Answer concept:* The base Python image comes empty. We need to tell Docker exactly which third-party libraries (like pandas, scikit-learn) to install so our code can run.

### Conceptual Questions
4. **How does Docker solve the "it works on my machine" problem?**
   - *Answer concept:* Code often breaks on different machines due to different operating systems, Python versions, or library versions. Docker packages the OS environment, Python runtime, libraries, and code together into one isolated unit.
5. **What is the purpose of port mapping (`-p 8000:8000`) when running a container?**
   - *Answer concept:* Containers are isolated from the host machine. If a web API is running on port 8000 *inside* the container, it is invisible to your browser. Port mapping creates a bridge between a port on your computer and the port inside the container.

### Practical Questions
6. **You built a Docker image yesterday, but today you changed your Python code. If you run the container again, will it use the new code?**
   - *Answer concept:* No. An image is a static snapshot taken at the time of the `docker build` command. You must rebuild the image (`docker build`) to include the new code changes before running the container.
7. **Your Docker image size is 2GB, which is too large. What are two ways you might reduce it?**
   - *Answer concept:* 1. Use a smaller base image (e.g., `python:3.9-slim` instead of just `python:3.9`). 2. Use a `.dockerignore` file to ensure you aren't copying large training datasets or local virtual environment folders into the image.

### Comparison Questions
8. **What is the difference between a Virtual Machine (VM) and a Docker Container?**
   - *Answer concept:* A VM emulates an entire hardware system and runs a full, heavy guest Operating System. A container shares the host machine's OS kernel, making it much more lightweight, faster to start, and less resource-intensive.

---

## Python Code Examples

### `example-01-basic.py`

```python
import time

print("Hello! I am a Python script running inside a Docker Container.")
print("I don't need any special web frameworks to run.")

for i in range(5):
    print(f"Working... step {i+1}/5")
    time.sleep(1)

print("Job finished! The container will now stop because my process is complete.")
```

### `example-02-api.py`

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from a containerized FastAPI application!"}

@app.get("/predict")
def dummy_predict():
    return {"prediction": "This is a dummy prediction from inside Docker"}

# Note: We do not call uvicorn.run() here.
# Instead, we will start uvicorn from the Dockerfile's CMD instruction.
```
