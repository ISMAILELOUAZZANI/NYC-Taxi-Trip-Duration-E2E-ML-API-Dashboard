"""
Simple training script using XGBoost. Supports optuna tuning (lightweight example).
Usage:
    python src/models/train.py --data-path data/raw/train.csv --output models/xgb.joblib
"""
import argparse
import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb

from src.features.build_features import build_features

def train_main(data_path, output_path, config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    df = pd.read_csv(data_path)
    df, features = build_features(df, target_col='trip_duration')
    # drop missing
    df = df.dropna(subset=features + ['target_log1p'])
    X = df[features]
    y = df['target_log1p']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg['train']['test_size'], random_state=cfg['train']['seed']
    )

    model = xgb.XGBRegressor(
        n_estimators=cfg['train']['xgb']['n_estimators'],
        learning_rate=cfg['train']['xgb']['learning_rate'],
        max_depth=cfg['train']['xgb']['max_depth'],
        subsample=cfg['train']['xgb']['subsample'],
        colsample_bytree=cfg['train']['xgb']['colsample_bytree'],
        random_state=cfg['train']['xgb']['random_state'],
        n_jobs=-1,
        verbosity=1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=10)

    # validate
    y_pred_val = model.predict(X_val)
    rmsle = (mean_squared_log_error((np.expm1(y_val)), (np.expm1(y_pred_val)))) ** 0.5
    print(f"Validation RMSLE: {rmsle:.6f}")

    joblib.dump({'model': model, 'features': features}, output_path)
    print(f"Saved model to {output_path}")

if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/xgb.joblib")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    train_main(args.data_path, args.output, args.config)