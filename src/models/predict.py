import joblib
import numpy as np
import pandas as pd
from src.features.build_features import build_features

class Predictor:
    def __init__(self, model_path: str):
        blob = joblib.load(model_path)
        self.model = blob['model']
        self.features = blob['features']

    def predict_row(self, row: dict):
        df = pd.DataFrame([row])
        df, feat_cols = build_features(df)
        X = df[self.features].fillna(0)
        pred_log = self.model.predict(X)[0]
        # model trained on log1p targets
        return float(np.expm1(pred_log))

if __name__ == "__main__":
    import sys
    mp = sys.argv[1] if len(sys.argv) > 1 else "models/xgb.joblib"
    p = Predictor(mp)
    sample = {
        "pickup_datetime": "2016-03-15 08:10:00",
        "pickup_longitude": -73.982154,
        "pickup_latitude": 40.767937,
        "dropoff_longitude": -73.964630,
        "dropoff_latitude": 40.765602,
        "passenger_count": 1
    }
    print("Predicted duration (s):", p.predict_row(sample))