# Methods, Options, and Properties: Docker

This document explains the common commands used inside a `Dockerfile` and the command-line interface for Docker.

## 1. Dockerfile Commands

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

## 2. Docker CLI Commands

These are the commands you type into your terminal to interact with Docker.

### Building an Image
- **`docker build -t image_name .`**: Builds an image using the `Dockerfile` in the current directory (`.`). The `-t` flag tags the image with a readable name.
  - *Example:* `docker build -t my_ml_api .`

### Running a Container
- **`docker run image_name`**: Starts a container from the image.
- **`-p host_port:container_port`**: Maps a port on your local machine to a port inside the container. Essential for APIs.
  - *Example:* `docker run -p 8000:8000 my_ml_api` (Forwards your computer's port 8000 to the container's port 8000).
- **`-d`**: Runs the container in "detached" mode (in the background) so it doesn't block your terminal.
  - *Example:* `docker run -d -p 8000:8000 my_ml_api`

### Managing Containers and Images
- **`docker ps`**: Lists all running containers.
- **`docker ps -a`**: Lists all containers (running and stopped).
- **`docker images`**: Lists all downloaded/built images on your machine.
- **`docker stop container_id`**: Stops a running container.
- **`docker rm container_id`**: Removes a stopped container.
- **`docker rmi image_name`**: Removes an image.

## Typical Workflow
1. Write python code and `requirements.txt`.
2. Write `Dockerfile`.
3. Terminal: `docker build -t my_model_app .`
4. Terminal: `docker run -p 8080:8080 my_model_app`
5. Test the app in browser (`localhost:8080`).
