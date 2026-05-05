# Interview Questions: Model Deployment

## Beginner Questions
1. **What is the difference between a machine learning model and an API?**
   - *Answer concept:* A model is an algorithm trained on data to make predictions. An API is the software interface that allows other applications to send data to the model and receive those predictions.
2. **Why do we need to serialize models (e.g., using pickle)?**
   - *Answer concept:* To save the trained state (weights, learned parameters) of the model to disk so it can be loaded into memory later without retraining.
3. **What is JSON and why is it commonly used in REST APIs?**
   - *Answer concept:* JSON (JavaScript Object Notation) is a lightweight data-interchange format. It is easy for humans to read/write and easy for machines to parse, making it the standard format for sending data to and from web APIs.

## Conceptual Questions
4. **Explain the difference between GET and POST requests in the context of ML deployment.**
   - *Answer concept:* GET is used to retrieve data (e.g., checking API health). POST is used to submit data to the server for processing (e.g., sending feature data to a model for prediction).
5. **Why is data validation important before passing inputs to a model in production?**
   - *Answer concept:* Models expect specific data types and shapes. If an API receives unexpected data (e.g., a string instead of an integer), it will crash the application. Validation (like Pydantic in FastAPI) ensures the API handles errors gracefully before the model breaks.

## Practical Questions
6. **You have trained a scikit-learn pipeline that includes scaling and an SVM classifier. How do you deploy this?**
   - *Answer concept:* You must serialize the *entire pipeline*, not just the SVM model. The incoming API data must be passed through the pipeline so it is scaled exactly as the training data was before prediction.
7. **If your deployed model API is running too slowly, what are some potential bottlenecks?**
   - *Answer concept:* Network latency, the time taken to deserialize the model (if done per request instead of on startup), inefficient preprocessing code, or the model algorithm itself being computationally heavy.

## Comparison Questions
8. **When would you choose Streamlit over FastAPI?**
   - *Answer concept:* Streamlit is best for building internal dashboards or quick interactive prototypes where you need a visual interface quickly. FastAPI is best for building robust, scalable backend services intended to be consumed by other software (not directly by humans).
