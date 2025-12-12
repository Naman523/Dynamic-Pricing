"""
Train and save demand forecasting model (XGBoost).
Compatible with older XGBoost + sklearn versions.
"""

import sys
import os

# --- FIX: Ensure project root is importable ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from src.data import load_raw
from src.features import create_features, get_feature_columns



def _supports_early_stopping():
    """
    Check if installed XGBoost version supports early_stopping_rounds in .fit().
    Older XGBoost versions (<1.7) do NOT support it.
    """
    try:
        ver = xgb.__version__
        major, minor = ver.split(".")[:2]
        return (int(major), int(minor)) >= (1, 7)
    except:
        return False



def train_model(
    input_path="data/raw/sim_retail.csv",
    model_out="models/xgb_model.pkl",
    test_size=0.2,
    random_state=42
):
    """Train XGBoost model and save (model, features)."""

    # create models directory if missing
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    # load + feature engineering
    df = load_raw(input_path)
    df = create_features(df)

    features = get_feature_columns()
    X = df[features]
    y = df["target"]

    # simple random split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )

    # --- SAFE TRAINING FOR OLD XGBOOST VERSIONS ---
    try:
        if _supports_early_stopping():
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=30,
                verbose=False,
            )
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
    except TypeError:
        # fallback for very old versions
        model.fit(X_train, y_train)



    # --- SAFE RMSE (works on all sklearn versions) ---
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    print(f"Validation RMSE: {rmse:.4f}")



    # Save (model, features) because Streamlit expects tuple
    joblib.dump((model, features), model_out)
    print(f"Saved model → {model_out}")

    return model, features



if __name__ == "__main__":
    train_model()
