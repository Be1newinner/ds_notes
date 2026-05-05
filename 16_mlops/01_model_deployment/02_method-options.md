# Methods, Options, and Properties: Model Deployment

This document explains the common frameworks and methods used for deploying Machine Learning models in Python.

## 1. Streamlit

Streamlit is a framework for building interactive web applications for data science and machine learning quickly, without needing frontend experience (HTML/CSS/JS).

### Core Functions:
- `st.title("My App")`: Sets the main title of the web page.
- `st.header("Section 1")`: Creates a sub-header.
- `st.write(data)`: A versatile function that can print text, dataframes, or charts.

### Input Widgets:
- `st.text_input("Label")`: Creates a single-line text input box.
- `st.number_input("Label", min_value=0)`: Creates a numeric input box.
- `st.selectbox("Label", options=["A", "B"])`: Creates a dropdown menu.
- `st.slider("Label", min_value=0, max_value=100)`: Creates a slider.

### Output/Interaction:
- `st.button("Predict")`: Creates a button. Usually used in an `if` statement: `if st.button("Predict"): do_something()`
- `st.dataframe(df)`: Displays a pandas DataFrame as an interactive table.

### Typical Workflow:
1. Load model using `joblib`.
2. Create input widgets to collect user data.
3. When the user clicks a button, format the inputs into a format the model expects (e.g., a 2D NumPy array or DataFrame).
4. Call `model.predict()`.
5. Display the result using `st.write()` or `st.success()`.

---

## 2. FastAPI

FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.

### Core Concepts:
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

### Data Validation (Pydantic):
FastAPI uses Pydantic models to validate incoming data. This ensures the API receives the correct data types.
```python
from pydantic import BaseModel

class PredictionInput(BaseModel):
    feature1: float
    feature2: int
    category: str
```

### HTTP Methods:
- `@app.get("/endpoint")`: Used for retrieving data. Parameters are usually passed in the URL.
- `@app.post("/endpoint")`: Used for sending data to the server (e.g., sending feature data to get a prediction). The data is sent in the request body.

### Typical Workflow:
1. Define Pydantic classes for the expected input data.
2. Initialize the FastAPI app.
3. Load the pre-trained model (usually done when the app starts).
4. Create a `@app.post("/predict")` endpoint.
5. In the endpoint function, extract data from the Pydantic model, preprocess it, and run `model.predict()`.
6. Return the prediction as a standard Python dictionary (FastAPI automatically converts this to JSON).

### Running FastAPI:
FastAPI requires an ASGI server like Uvicorn to run.
**Command line:** `uvicorn filename:app --reload`
- `filename`: The name of your Python script (without `.py`).
- `app`: The name of the FastAPI instance variable.
- `--reload`: Automatically restarts the server when code changes (useful for development).
