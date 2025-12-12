"""
Simple pricing optimizer using grid search on candidate multipliers.
"""
import numpy as np
import pandas as pd

def find_optimal_price_row(model, features, row_series, price_col='price', pct_grid=None):
    """
    For a single row (Series), evaluate candidate prices and return best.
    pct_grid: list of multipliers e.g. [0.8, 0.9, 1.0, 1.05, 1.1, 1.2]
    Returns dict with best_price, best_revenue, predicted_demand
    """
    if pct_grid is None:
        pct_grid = [0.8, 0.9, 1.0, 1.05, 1.1, 1.2]

    base_price = float(row_series[price_col])
    best = None
    for pct in pct_grid:
        p = base_price * pct
        new_row = row_series.copy()
        new_row[price_col] = p
        x = new_row[features].values.reshape(1, -1)
        pred = float(model.predict(x)[0])
        revenue = p * max(pred, 0)
        if (best is None) or (revenue > best['best_revenue']):
            best = {'best_price': float(round(p, 2)), 'best_revenue': float(round(revenue, 2)), 'predicted_demand': float(round(pred, 2)), 'multiplier': pct}
    return best

def find_optimal_prices_for_df(model, features, df, price_col='price', pct_grid=None):
    results = []
    for idx, row in df.iterrows():
        res = find_optimal_price_row(model, features, row, price_col=price_col, pct_grid=pct_grid)
        results.append(res)
    return pd.DataFrame(results)
