# Practice: Containerization

These exercises will test your ability to write Dockerfiles and build images.

## Exercise 1: Containerize a Script
1. Write a very simple Python script (`hello.py`) that prints "Hello from inside the container!" and imports the `numpy` library to print an array of zeros.
2. Create a `requirements.txt` containing `numpy`.
3. Write a `Dockerfile` that:
   - Uses `python:3.9-slim`
   - Copies `requirements.txt` and installs it.
   - Copies `hello.py`.
   - Runs `python hello.py` as the CMD.
4. Build the image and tag it `hello-container`.
5. Run the container. It should print the text and the array, and then stop.

## Exercise 2: Containerize a Streamlit App
1. Go back to `01_model_deployment` and copy the `example-01-basic.py` (the Streamlit app) into a new folder.
2. Create a `requirements.txt` containing `streamlit` and `pandas`.
3. Write a `Dockerfile`.
   - *Hint:* Streamlit runs on port 8501 by default.
   - *Hint:* The command to run Streamlit is `streamlit run filename.py`. You will need to structure your CMD instruction as `CMD ["streamlit", "run", "example-01-basic.py", "--server.port", "8501"]`.
4. Build the image as `my-streamlit-app`.
5. Run the container, making sure to map port 8501 on your host to port 8501 on the container (`-p 8501:8501`).
6. Access the app in your browser.

## Exercise 3: Cleaning Up
1. Use `docker ps -a` to see all the containers you've run today.
2. Use `docker rm` to delete them.
3. Use `docker images` to see the images you built.
4. Note how large the images are (often several hundred megabytes).
