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
