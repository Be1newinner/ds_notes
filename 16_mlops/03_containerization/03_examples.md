# Examples: Containerization

This document outlines the practical examples provided for learning Docker.

## Code References

- `code/example-01-basic.py` & `code/Dockerfile-basic`: 
  - **Concept:** Containerizing a simple Python script that runs and exits.
  - **How to use:** Navigate to the `code` folder. Run `docker build -f Dockerfile-basic -t basic-script .` followed by `docker run basic-script`.
- `code/example-02-api.py` & `code/Dockerfile-api` & `code/requirements.txt`:
  - **Concept:** Containerizing a FastAPI web server. This requires exposing ports and handling dependencies.
  - **How to use:** Navigate to the `code` folder. Run `docker build -f Dockerfile-api -t my-fastapi-app .` followed by `docker run -p 8000:8000 my-fastapi-app`. Then open `localhost:8000` in your browser.

## Important Note on Context
When running `docker build`, the `.` at the end tells Docker to use the current directory as the "build context". The Dockerfile will only be able to `COPY` files that exist inside this directory. Therefore, always run `docker build` from the directory where your project files live.
