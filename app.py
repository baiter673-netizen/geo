from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime

# Load your trained model
model = joblib.load("risk_model.pkl")  # <-- replace with your model file name

app = FastAPI(title="Risk Prediction API")

# Input data format
class InputData(BaseModel):
    timestamp: str
    latitude: float
    longitude: float

# Extract hour from timestamp
def extract_hour(timestamp: str):
    dt = datetime.fromisoformat(timestamp)
    return dt.hour

@app.post("/predict")
def predict(data: InputData):
    # Get hour from timestamp
    hour = extract_hour(data.timestamp)

    # Features must match your training setup
    features = np.array([[data.latitude, data.longitude, hour]])

    # Predict probabilities
    probas = model.predict_proba(features)[0]
    risk_class = model.predict(features)[0]

    # Risk score = max probability
    risk_score = float(round(probas.max(), 3))

    # Risk levels
    labels = {0: "Low", 1: "Medium", 2: "High"}
    risk_level = labels.get(risk_class, "Unknown")

    return {
        "timestamp": data.timestamp,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "risk_score": risk_score,
        "risk_level": risk_level
    }
