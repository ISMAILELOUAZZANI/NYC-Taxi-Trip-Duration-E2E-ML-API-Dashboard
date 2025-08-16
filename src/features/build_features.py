import pandas as pd
import numpy as np
from .geo import haversine_distance

def build_features(df: pd.DataFrame, target_col: str = "trip_duration"):
    """
    Adds common features:
    - pickup_datetime -> datetime features
    - Haversine distance (km)
    - log1p(target) if target exists
    Returns DataFrame with features + optionally target_log.
    """
    df = df.copy()
    # parse datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['weekday'] = df['pickup_datetime'].dt.weekday
    df['month'] = df['pickup_datetime'].dt.month
    df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)

    # distance in km
    df['haversine_km'] = df.apply(
        lambda r: haversine_distance(
            r['pickup_latitude'], r['pickup_longitude'],
            r['dropoff_latitude'], r['dropoff_longitude']
        ), axis=1
    )

    # avoid zero distances
    df['haversine_km'] = df['haversine_km'].replace(0, 0.0)

    # target transform
    if target_col in df.columns:
        df['target_log1p'] = np.log1p(df[target_col])

    # keep a simple feature list
    feature_cols = [
        'passenger_count', 'haversine_km', 'hour', 'day', 'weekday', 'month', 'is_weekend'
    ]
    # ensure cols exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols