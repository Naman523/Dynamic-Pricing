"""
Estimate local price elasticity using model-based finite difference.
"""
import numpy as np
import pandas as pd

def estimate_local_elasticity(model, features, row_series, price_col='price', delta=0.01):
    """
    row_series: pandas Series with feature columns present
    model: trained model object with predict method
    features: list of feature names in order
    delta: relative perturbation (e.g., 0.01 = 1% price increase)
    """
    base_price = float(row_series[price_col])
    x = row_series[features].values.reshape(1, -1)
    base_pred = model.predict(x)[0]

    if base_pred <= 0 or base_price == 0:
        return 0.0

    dp = base_price * delta
    x2 = row_series[features].copy()
    # set price to base + dp
    x2[features.index(price_col)] = base_price + dp
    pred_p = model.predict(x2.values.reshape(1, -1))[0]

    pct_change_d = (pred_p - base_pred) / base_pred
    pct_change_p = dp / base_price
    elasticity = pct_change_d / pct_change_p
    return float(elasticity)

def estimate_elasticity_for_df(model, features, df, price_col='price', delta=0.01):
    results = []
    for idx, row in df.iterrows():
        e = estimate_local_elasticity(model, features, row, price_col=price_col, delta=delta)
        results.append(e)
    return np.array(results)
