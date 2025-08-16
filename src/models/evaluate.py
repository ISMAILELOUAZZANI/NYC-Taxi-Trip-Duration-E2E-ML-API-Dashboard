import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from src.features.build_features import build_features

def evaluate(model_path: str, data_path: str):
    blob = joblib.load(model_path)
    model = blob['model']
    features = blob['features']
    df = pd.read_csv(data_path)
    df, _ = build_features(df, target_col='trip_duration')
    df = df.dropna(subset=features + ['target_log1p'])
    X = df[features]
    y_true = np.expm1(df['target_log1p'])
    y_pred = np.expm1(model.predict(X))
    rmsle = mean_squared_log_error(y_true, y_pred) ** 0.5
    print(f"RMSLE on {data_path}: {rmsle:.6f}")
    return rmsle

if __name__ == "__main__":
    import sys
    mp = sys.argv[1] if len(sys.argv) > 1 else "models/xgb.joblib"
    dp = sys.argv[2] if len(sys.argv) > 2 else "data/raw/train.csv"
    evaluate(mp, dp)