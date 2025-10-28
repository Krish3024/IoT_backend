from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import pickle
import numpy as np
import uvicorn

# ----------------- MongoDB Setup -----------------
MONGO_URI = "mongodb+srv://krishsah5216_db_user:1234@cluster0.qkdm0zl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# client = MongoClient(MONGO_URI)
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsAllowInvalidCertificates=True
)
db = client["PlantMonitoring"]
collection = db["SensorData"]

# ----------------- FastAPI Setup -----------------
app = FastAPI(title="Plant Water Prediction API")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Load Trained Model -----------------
import joblib
model = joblib.load("plant_model.pkl")

# ----------------- Request Schema -----------------
class SensorData(BaseModel):
    plant_name: str
    temperature: float
    humidity: float
    light_intensity: float
    soil_moisture: float

# ----------------- Routes -----------------

@app.get("/")
def root():
    return {"message": "🌱 Plant Water Prediction API is running!"}


# @app.post("/predict")
# def predict(data: SensorData):
#     try:
#         # Convert input data to model format
#         features = np.array([[data.temperature, data.humidity, data.light_intensity, data.soil_moisture]])

#         # Make prediction
#         prediction = model.predict(features)[0]

#         # Save to MongoDB
#         record = data.dict()
#         record["needs_water"] = bool(prediction)
#         collection.insert_one(record)
#         proba = model.predict_proba(features)[0][1]
#         return {"prediction": "Needs Water" if prediction == 1 else "No Water Needed", "confidence": round(float(proba), 3)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict")
def predict(data: SensorData):
    try:
        # ✅ Convert input data to numeric format (ensure soil_moisture is float)
        features = np.array([
            [float(data.temperature), float(data.humidity), float(data.light_intensity), float(data.soil_moisture)]
        ])

        # ✅ Make prediction safely
        prediction = int(model.predict(features)[0])

        # ✅ Compute confidence (if model supports it)
        confidence = (
            float(model.predict_proba(features)[0][1])
            if hasattr(model, "predict_proba")
            else 0.85  # fallback confidence
        )

        # ✅ Save to MongoDB
        record = data.dict()
        record["needs_water"] = bool(prediction)
        record["confidence"] = confidence
        collection.insert_one(record)

        # ✅ Return proper JSON for frontend
        return {
            "prediction": "Needs Water" if prediction == 1 else "No Water Needed",
            "confidence": round(confidence, 3),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_reading")
def add_reading(data: SensorData):
    """Store sensor readings in MongoDB."""
    try:
        record = data.dict()
        collection.insert_one(record)
        return {"message": "Sensor reading saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_readings")
def get_readings():
    """Fetch all saved sensor readings."""
    try:
        readings = list(collection.find({}, {"_id": 0}))
        return {"readings": readings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- Run Server -----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
