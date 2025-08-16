from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from src.models.predict import Predictor
import os

app = FastAPI(title="NYC Taxi Trip Duration Predictor")

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb.joblib")
_predictor = None

class TripRequest(BaseModel):
    pickup_datetime: str
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: Optional[int] = Field(1, ge=0)

@app.on_event("startup")
def load_model():
    global _predictor
    try:
        _predictor = Predictor(MODEL_PATH)
        app.state.model_loaded = True
    except Exception as e:
        app.state.model_loaded = False
        app.state.load_error = str(e)

@app.get("/")
def root():
    return {"service": "nyc-taxi-duration", "model_loaded": getattr(app.state, "model_loaded", False)}

@app.post("/predict")
def predict(req: TripRequest):
    if not getattr(app.state, "model_loaded", False):
        return {"error": "model not loaded", "detail": getattr(app.state, "load_error", None)}
    pred = _predictor.predict_row(req.dict())
    return {"predicted_duration_seconds": pred}