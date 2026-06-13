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


---

## Methods, Options, and Properties: Model Deployment

This document explains the common frameworks and methods used for deploying Machine Learning models in Python.

### 1. Streamlit

Streamlit is a framework for building interactive web applications for data science and machine learning quickly, without needing frontend experience (HTML/CSS/JS).

#### Core Functions:
- `st.title("My App")`: Sets the main title of the web page.
- `st.header("Section 1")`: Creates a sub-header.
- `st.write(data)`: A versatile function that can print text, dataframes, or charts.

#### Input Widgets:
- `st.text_input("Label")`: Creates a single-line text input box.
- `st.number_input("Label", min_value=0)`: Creates a numeric input box.
- `st.selectbox("Label", options=["A", "B"])`: Creates a dropdown menu.
- `st.slider("Label", min_value=0, max_value=100)`: Creates a slider.

#### Output/Interaction:
- `st.button("Predict")`: Creates a button. Usually used in an `if` statement: `if st.button("Predict"): do_something()`
- `st.dataframe(df)`: Displays a pandas DataFrame as an interactive table.

#### Typical Workflow:
1. Load model using `joblib`.
2. Create input widgets to collect user data.
3. When the user clicks a button, format the inputs into a format the model expects (e.g., a 2D NumPy array or DataFrame).
4. Call `model.predict()`.
5. Display the result using `st.write()` or `st.success()`.

---

### 2. FastAPI

FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.

#### Core Concepts:
- **`FastAPI()`**: The main application instance.
  ```python
  from fastapi import FastAPI
  app = FastAPI()
  ```
- **Decorators**: Used to map URLs to Python functions.
  ```python
  @app.get("/")
  def read_root():
      return {"Hello": "World"}
  ```

#### Data Validation (Pydantic):
FastAPI uses Pydantic models to validate incoming data. This ensures the API receives the correct data types.
```python
from pydantic import BaseModel

class PredictionInput(BaseModel):
    feature1: float
    feature2: int
    category: str
```

#### HTTP Methods:
- `@app.get("/endpoint")`: Used for retrieving data. Parameters are usually passed in the URL.
- `@app.post("/endpoint")`: Used for sending data to the server (e.g., sending feature data to get a prediction). The data is sent in the request body.

#### Typical Workflow:
1. Define Pydantic classes for the expected input data.
2. Initialize the FastAPI app.
3. Load the pre-trained model (usually done when the app starts).
4. Create a `@app.post("/predict")` endpoint.
5. In the endpoint function, extract data from the Pydantic model, preprocess it, and run `model.predict()`.
6. Return the prediction as a standard Python dictionary (FastAPI automatically converts this to JSON).

#### Running FastAPI:
FastAPI requires an ASGI server like Uvicorn to run.
**Command line:** `uvicorn filename:app --reload`
- `filename`: The name of your Python script (without `.py`).
- `app`: The name of the FastAPI instance variable.
- `--reload`: Automatically restarts the server when code changes (useful for development).

---

## Examples: Model Deployment

This document outlines the practical examples provided for learning model deployment.

### Code References

- `code/example-01-basic.py` — **Basic Streamlit App**: A simple interactive web application that takes user input and performs a basic calculation (simulating a prediction). Run with `streamlit run example-01-basic.py`.
- `code/example-02-intermediate.py` — **Basic FastAPI**: Introduces the FastAPI framework, routing, and creating a simple GET endpoint. Run with `uvicorn example-02-intermediate:app --reload`.
- `code/example-03-real-world.py` — **FastAPI with ML Model**: A realistic example demonstrating how to load a pre-trained scikit-learn model, validate input using Pydantic, and return predictions via a POST endpoint.

### How to use these examples

1. Start by running the Streamlit app to understand how easily a UI can be built around Python logic.
2. Move to the intermediate FastAPI example to understand the structure of an API and how to interact with it using a browser or tools like Postman/cURL.
3. Finally, examine the real-world FastAPI example to see how a trained model is integrated into the API structure. Note the importance of data validation.

---

## Practice: Model Deployment

These exercises will test your ability to build web applications and APIs for machine learning models.

### Exercise 1: Build a Streamlit Salary Predictor
1. Assume you have a simple linear regression model that predicts salary based on years of experience (`Salary = 25000 + (Experience * 10000)`).
2. Create a Streamlit app (`salary_app.py`).
3. Add a title "Employee Salary Predictor".
4. Add a number input or slider for "Years of Experience" (0 to 50).
5. Add a "Predict Salary" button.
6. When the button is clicked, calculate the predicted salary and display it using `st.success()`.

### Exercise 2: Basic FastAPI Setup
1. Create a FastAPI application.
2. Define a root GET endpoint (`/`) that returns a welcome message: `{"message": "Welcome to the ML API"}`.
3. Define another GET endpoint (`/health`) that returns the status: `{"status": "API is running smoothly"}`.
4. Run the API using Uvicorn and test both endpoints in your web browser.

### Exercise 3: Post Endpoint with Validation
1. Extend the FastAPI application from Exercise 2.
2. Create a Pydantic model named `HouseData` that expects:
   - `sqft` (integer)
   - `bedrooms` (integer)
   - `location_type` (string, e.g., "urban" or "suburban")
3. Create a POST endpoint `/predict_price` that accepts `HouseData`.
4. Inside the endpoint, return a mock prediction based on the inputs (e.g., `sqft * 150 + bedrooms * 10000`).
5. Test this endpoint using the interactive docs provided by FastAPI (navigate to `http://localhost:8000/docs`).

### Exercise 4: Full Deployment (Challenge)
1. Train a simple Logistic Regression model on the Iris dataset and save it as `iris_model.pkl`.
2. Create a FastAPI app that loads this model.
3. Create a POST endpoint that takes the 4 sepal/petal measurements and returns the predicted flower class.
4. Test the API thoroughly.

---

## Interview Questions: Model Deployment

### Beginner Questions
1. **What is the difference between a machine learning model and an API?**
   - *Answer concept:* A model is an algorithm trained on data to make predictions. An API is the software interface that allows other applications to send data to the model and receive those predictions.
2. **Why do we need to serialize models (e.g., using pickle)?**
   - *Answer concept:* To save the trained state (weights, learned parameters) of the model to disk so it can be loaded into memory later without retraining.
3. **What is JSON and why is it commonly used in REST APIs?**
   - *Answer concept:* JSON (JavaScript Object Notation) is a lightweight data-interchange format. It is easy for humans to read/write and easy for machines to parse, making it the standard format for sending data to and from web APIs.

### Conceptual Questions
4. **Explain the difference between GET and POST requests in the context of ML deployment.**
   - *Answer concept:* GET is used to retrieve data (e.g., checking API health). POST is used to submit data to the server for processing (e.g., sending feature data to a model for prediction).
5. **Why is data validation important before passing inputs to a model in production?**
   - *Answer concept:* Models expect specific data types and shapes. If an API receives unexpected data (e.g., a string instead of an integer), it will crash the application. Validation (like Pydantic in FastAPI) ensures the API handles errors gracefully before the model breaks.

### Practical Questions
6. **You have trained a scikit-learn pipeline that includes scaling and an SVM classifier. How do you deploy this?**
   - *Answer concept:* You must serialize the *entire pipeline*, not just the SVM model. The incoming API data must be passed through the pipeline so it is scaled exactly as the training data was before prediction.
7. **If your deployed model API is running too slowly, what are some potential bottlenecks?**
   - *Answer concept:* Network latency, the time taken to deserialize the model (if done per request instead of on startup), inefficient preprocessing code, or the model algorithm itself being computationally heavy.

### Comparison Questions
8. **When would you choose Streamlit over FastAPI?**
   - *Answer concept:* Streamlit is best for building internal dashboards or quick interactive prototypes where you need a visual interface quickly. FastAPI is best for building robust, scalable backend services intended to be consumed by other software (not directly by humans).

---

## Python Code Examples

### `example-01-basic.py`

```python
import streamlit as st
import pandas as pd

# Title of the application
st.title("Customer Spending Predictor (Mock)")

st.write("""
This is a simple interactive web application built with Streamlit.
It demonstrates how to take user inputs and display a simulated prediction.
""")

# Create an input form
st.header("Enter Customer Details")

# Various input widgets
age = st.slider("Customer Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=50000)
membership = st.selectbox("Membership Tier", options=["Basic", "Premium", "VIP"])

# Prediction Button
if st.button("Predict Expected Spending"):
    # Mock prediction logic (In reality, this would be model.predict())
    base_spend = income * 0.1
    
    if membership == "Premium":
        base_spend *= 1.2
    elif membership == "VIP":
        base_spend *= 1.5
        
    if age < 30:
        base_spend *= 0.9
        
    predicted_spend = round(base_spend, 2)
    
    # Display the result
    st.success(f"The predicted annual spending for this customer is: ${predicted_spend}")
    
    # Show the inputted data as a table
    st.write("Based on the following data:")
    data = pd.DataFrame({"Age": [age], "Income": [income], "Tier": [membership]})
    st.dataframe(data)

# To run this app, open your terminal and type:
# streamlit run example-01-basic.py
```

### `example-02-intermediate.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the FastAPI application
app = FastAPI(title="Basic ML API")

# 1. Basic GET Endpoint
# Used to check if the API is running (Health Check)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Basic ML API. The server is running!"}

# 2. Defining the expected input data structure using Pydantic
# This ensures the API only accepts data in this format
class SimpleInput(BaseModel):
    value_a: int
    value_b: int

# 3. Basic POST Endpoint
# Used to send data to the server and get a result back
@app.post("/calculate")
def calculate_sum(data: SimpleInput):
    # 'data' is automatically parsed into the SimpleInput structure
    result = data.value_a + data.value_b
    
    # Return the result as a dictionary (FastAPI converts to JSON)
    return {
        "operation": "addition",
        "input_a": data.value_a,
        "input_b": data.value_b,
        "result": result
    }

# To run this API:
# 1. Install required packages: pip install fastapi uvicorn
# 2. Run the server: uvicorn example-02-intermediate:app --reload
# 3. View the interactive documentation: http://localhost:8000/docs
```

### `example-03-real-world.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# --- MOCK MODEL SETUP ---
# In a real scenario, you would load a trained model using joblib:
# import joblib
# model = joblib.load("my_trained_model.pkl")

class MockModel:
    def predict(self, features):
        # A simple fake logic for demonstration: 
        # If the sum of features is > 150, return Class 1, else Class 0
        total = np.sum(features)
        if total > 150:
            return [1]
        else:
            return [0]

model = MockModel()
# ------------------------

app = FastAPI(title="Real Estate Predictor API")

# Define the expected input schema
class HouseFeatures(BaseModel):
    square_feet: float
    num_bedrooms: int
    age_of_house_years: int
    
@app.get("/")
def health_check():
    return {"status": "API is online"}

@app.post("/predict")
def predict_house_category(data: HouseFeatures):
    try:
        # 1. Extract data from the Pydantic model
        features = np.array([[
            data.square_feet,
            data.num_bedrooms,
            data.age_of_house_years
        ]])
        
        # 2. (Optional but common) Preprocessing would go here
        # features_scaled = scaler.transform(features)
        
        # 3. Make prediction
        prediction = model.predict(features)
        
        # 4. Format the output
        predicted_class = int(prediction[0])
        category = "Premium" if predicted_class == 1 else "Standard"
        
        return {
            "prediction_class": predicted_class,
            "category_label": category,
            "inputs_received": data.dict()
        }
        
    except Exception as e:
        # Return a clean HTTP 500 error if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))

# To run this API:
# uvicorn example-03-real-world:app --reload
```
