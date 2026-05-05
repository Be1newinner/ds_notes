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
