# Interview Questions: Containerization

## Beginner Questions
1. **What is the difference between an Image and a Container in Docker?**
   - *Answer concept:* An image is a static blueprint or file (like a class definition). A container is a running instance of that image (like an object).
2. **What does a `Dockerfile` do?**
   - *Answer concept:* It contains the step-by-step instructions needed to assemble a Docker image.
3. **Why do we need a `requirements.txt` file when building a Python Docker image?**
   - *Answer concept:* The base Python image comes empty. We need to tell Docker exactly which third-party libraries (like pandas, scikit-learn) to install so our code can run.

## Conceptual Questions
4. **How does Docker solve the "it works on my machine" problem?**
   - *Answer concept:* Code often breaks on different machines due to different operating systems, Python versions, or library versions. Docker packages the OS environment, Python runtime, libraries, and code together into one isolated unit.
5. **What is the purpose of port mapping (`-p 8000:8000`) when running a container?**
   - *Answer concept:* Containers are isolated from the host machine. If a web API is running on port 8000 *inside* the container, it is invisible to your browser. Port mapping creates a bridge between a port on your computer and the port inside the container.

## Practical Questions
6. **You built a Docker image yesterday, but today you changed your Python code. If you run the container again, will it use the new code?**
   - *Answer concept:* No. An image is a static snapshot taken at the time of the `docker build` command. You must rebuild the image (`docker build`) to include the new code changes before running the container.
7. **Your Docker image size is 2GB, which is too large. What are two ways you might reduce it?**
   - *Answer concept:* 1. Use a smaller base image (e.g., `python:3.9-slim` instead of just `python:3.9`). 2. Use a `.dockerignore` file to ensure you aren't copying large training datasets or local virtual environment folders into the image.

## Comparison Questions
8. **What is the difference between a Virtual Machine (VM) and a Docker Container?**
   - *Answer concept:* A VM emulates an entire hardware system and runs a full, heavy guest Operating System. A container shares the host machine's OS kernel, making it much more lightweight, faster to start, and less resource-intensive.
