from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

# Load trained model and encoder
model = joblib.load("geofence_risk_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI(title="Geofence Risk Prediction API")

# Input format
class RiskInput(BaseModel):
    timestamp: str
    latitude: float
    longitude: float

@app.get("/")
def root():
    return {"message": "Geofence Risk Prediction API is running"}

@app.post("/predict")
def predict_risk(data: RiskInput):
    # Convert timestamp to hour
    try:
        hour = datetime.fromisoformat(data.timestamp).hour
    except ValueError:
        return {"error": "Invalid timestamp format. Use ISO format (YYYY-MM-DDTHH:MM:SS.mmmmmm)"}

    # Features
    features = pd.DataFrame([{
        "latitude": data.latitude,
        "longitude": data.longitude,
        "hour_of_day": hour
    }])

    # Predict
    risk_class = model.predict(features)[0]
    risk_score = model.predict_proba(features).max()

    # Decode label
    risk_level = label_encoder.inverse_transform([risk_class])[0]

    return {
        "timestamp": data.timestamp,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "risk_score": round(float(risk_score), 3),
        "risk_level": risk_level
    }


