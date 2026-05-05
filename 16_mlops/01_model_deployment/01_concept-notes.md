# Model Deployment

## Learning Objective
Students should understand what model deployment means, why it is necessary, and how to create basic REST APIs using FastAPI and interactive web applications using Streamlit to serve machine learning models.

## What Is This Topic?
Model deployment is the process of integrating a machine learning model into an existing production environment to make practical business decisions based on data. It takes the model out of the data scientist's notebook and makes it available to other software applications or end-users.

## Why This Topic Matters
A model sitting in a Jupyter Notebook provides zero value to a business. To be useful, it must be deployed so that web applications, mobile apps, or internal tools can send it new data and receive predictions in real-time.

## Core Intuition
Think of a machine learning model like a chef in a kitchen. The chef (model) knows how to cook (predict). But without a waiter (API) to take orders (data) from the customers (users/apps) and bring back the food (predictions), the chef's skills are useless. Deployment is setting up the waiter and the restaurant interface.

## Key Concepts
- **API (Application Programming Interface):** A way for different software programs to communicate.
- **REST API:** A specific architectural style for APIs that uses standard HTTP methods (GET, POST, etc.).
- **Serialization:** Converting a trained model into a file format (like `.pkl` using `joblib` or `pickle`) so it can be saved and loaded later.
- **Deserialization:** Loading the saved model file back into memory to make predictions.
- **Endpoint:** A specific URL where your API can be accessed.

## Step-by-Step Explanation (Deploying an API)
1. Train and evaluate your model in a notebook.
2. Serialize (save) the trained model using `joblib` or `pickle`.
3. Create a Python script using a framework like FastAPI.
4. In the script, load the serialized model.
5. Define an endpoint (e.g., `/predict`) that accepts incoming data.
6. Process the incoming data, pass it to the model, and return the prediction as a JSON response.
7. Run the API server.

## Important Parameters / Options / Settings
- **Host & Port:** When running an API server (like Uvicorn for FastAPI), you specify a host (e.g., `0.0.0.0` to make it publicly accessible on the network) and a port (e.g., `8000`).
- **HTTP Methods:**
  - `GET`: Retrieve data (e.g., checking if the API is alive).
  - `POST`: Send data to be processed (e.g., sending patient data to get a disease prediction).

## Output / Result Interpretation
The output of a model deployment API is typically a JSON object containing the prediction.
Example: `{"prediction": "churn", "probability": 0.85}`. The consuming application reads this JSON and updates its UI accordingly.

## Real-World Uses
- A banking app sending transaction details to an API to get a "Fraud" or "Not Fraud" prediction in real-time.
- An e-commerce website querying a recommendation engine API to display "Products you might like" on a user's homepage.
- A Streamlit dashboard used by a marketing team to input campaign budgets and predict expected sales.

## Advantages
- **Scalability:** APIs can handle multiple requests from different applications simultaneously.
- **Separation of Concerns:** The data science team can update the model behind the API without the software engineering team needing to change the web application code.
- **Accessibility:** Models can be accessed from any programming language capable of making HTTP requests.

## Limitations
- **Latency:** Making network requests to an API adds a small delay compared to running the model locally.
- **Infrastructure:** Requires servers to host the API, adding cost and maintenance overhead.

## Common Mistakes
- **Forgetting Data Preprocessing:** The API must apply the *exact same* scaling, encoding, and imputation steps to incoming data as were applied during model training.
- **Not Handling Errors:** If the API receives unexpected data (e.g., text instead of a number), it should return a helpful error message, not crash.
- **Exposing Sensitive Data:** APIs need security measures if they are handling sensitive information.

## Related Methods
- **Batch Prediction:** Instead of real-time API predictions, the model runs on a large batch of data overnight (e.g., generating recommendations for all users at 2 AM).
- **Edge Deployment:** Deploying models directly onto devices (like mobile phones or IoT sensors) instead of servers.

## Code References
- `code/example-01-basic.py` — Simple Streamlit web application.
- `code/example-02-intermediate.py` — Basic FastAPI REST API.
- `code/example-03-real-world.py` — FastAPI with input validation and a trained model.
