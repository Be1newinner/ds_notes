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
