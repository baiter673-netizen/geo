import pandas as pd
import joblib
from datetime import datetime

# Load trained model + label encoder
model = joblib.load("geofence_risk_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Feature order (must match training)
FEATURE_ORDER = [
    "latitude", "longitude", "hour_of_day",
    "crime_rate", "geo_risk", "crowd_density",
    "restricted_zone"
]

def predict_risk(timestamp, latitude, longitude):
    # 1. Parse timestamp → extract hour
    dt = datetime.fromisoformat(timestamp)
    hour_of_day = dt.hour   # value 0–23

    # 2. Build feature dataframe
    features = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "hour_of_day": hour_of_day,
        "crime_rate": 1.0,       # placeholder (can be replaced with real values)
        "geo_risk": 1.0,
        "crowd_density": 1.0,
        "restricted_zone": 0
    }])

    # Ensure same column order as training
    features = features[FEATURE_ORDER]

    # 3. Predict
    probas = model.predict_proba(features)[0]
    pred_class = model.predict(features)[0]
    risk_label = label_encoder.inverse_transform([pred_class])[0]

    return {
        "timestamp": timestamp,
        "latitude": latitude,
        "longitude": longitude,
        #"hour_of_day": hour_of_day,
        "risk_score": round(probas[pred_class], 3),
        "risk_level": risk_label
    }

# Example usage
if __name__ == "__main__":
    print(predict_risk("2025-09-19T15:22:03.988992", 10.9360826, 76.9544039))
