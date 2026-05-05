# Examples: Model Deployment

This document outlines the practical examples provided for learning model deployment.

## Code References

- `code/example-01-basic.py` — **Basic Streamlit App**: A simple interactive web application that takes user input and performs a basic calculation (simulating a prediction). Run with `streamlit run example-01-basic.py`.
- `code/example-02-intermediate.py` — **Basic FastAPI**: Introduces the FastAPI framework, routing, and creating a simple GET endpoint. Run with `uvicorn example-02-intermediate:app --reload`.
- `code/example-03-real-world.py` — **FastAPI with ML Model**: A realistic example demonstrating how to load a pre-trained scikit-learn model, validate input using Pydantic, and return predictions via a POST endpoint.

## How to use these examples

1. Start by running the Streamlit app to understand how easily a UI can be built around Python logic.
2. Move to the intermediate FastAPI example to understand the structure of an API and how to interact with it using a browser or tools like Postman/cURL.
3. Finally, examine the real-world FastAPI example to see how a trained model is integrated into the API structure. Note the importance of data validation.
