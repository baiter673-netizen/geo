import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 1. Load dataset (ensure you generated geofence_risk_hourly.csv earlier)
df = pd.read_csv("geofence_risk_hourly.csv")

# 2. Define features and target
FEATURES = [
    "latitude", "longitude", "hour_of_day",
    "crime_rate", "geo_risk", "crowd_density",
    "restricted_zone"
]
TARGET = "risk_level"

X = df[FEATURES]
y = df[TARGET]

# 3. Encode target labels (Low, Medium, High → 0,1,2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 7. Save model + encoder
joblib.dump(model, "geofence_risk_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model training complete. Files saved: geofence_risk_model.pkl, label_encoder.pkl")
