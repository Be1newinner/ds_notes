# Practice: Model Deployment

These exercises will test your ability to build web applications and APIs for machine learning models.

## Exercise 1: Build a Streamlit Salary Predictor
1. Assume you have a simple linear regression model that predicts salary based on years of experience (`Salary = 25000 + (Experience * 10000)`).
2. Create a Streamlit app (`salary_app.py`).
3. Add a title "Employee Salary Predictor".
4. Add a number input or slider for "Years of Experience" (0 to 50).
5. Add a "Predict Salary" button.
6. When the button is clicked, calculate the predicted salary and display it using `st.success()`.

## Exercise 2: Basic FastAPI Setup
1. Create a FastAPI application.
2. Define a root GET endpoint (`/`) that returns a welcome message: `{"message": "Welcome to the ML API"}`.
3. Define another GET endpoint (`/health`) that returns the status: `{"status": "API is running smoothly"}`.
4. Run the API using Uvicorn and test both endpoints in your web browser.

## Exercise 3: Post Endpoint with Validation
1. Extend the FastAPI application from Exercise 2.
2. Create a Pydantic model named `HouseData` that expects:
   - `sqft` (integer)
   - `bedrooms` (integer)
   - `location_type` (string, e.g., "urban" or "suburban")
3. Create a POST endpoint `/predict_price` that accepts `HouseData`.
4. Inside the endpoint, return a mock prediction based on the inputs (e.g., `sqft * 150 + bedrooms * 10000`).
5. Test this endpoint using the interactive docs provided by FastAPI (navigate to `http://localhost:8000/docs`).

## Exercise 4: Full Deployment (Challenge)
1. Train a simple Logistic Regression model on the Iris dataset and save it as `iris_model.pkl`.
2. Create a FastAPI app that loads this model.
3. Create a POST endpoint that takes the 4 sepal/petal measurements and returns the predicted flower class.
4. Test the API thoroughly.
