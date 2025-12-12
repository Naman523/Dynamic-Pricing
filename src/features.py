"""
Feature engineering pipeline.
Takes raw dataframe and returns dataframe with ML features and target.
"""
import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['product_id', 'date']).reset_index(drop=True)

    # time features
    df['dow'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

    # seasonality (sin/cos for annual cycle)
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_yr'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_yr'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    # lag / rolling features per product
    df['lag1'] = df.groupby('product_id')['sales'].shift(1).fillna(0)
    df['lag7_mean'] = df.groupby('product_id')['sales'].shift(1).rolling(7, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
    df['lag30_mean'] = df.groupby('product_id')['sales'].shift(1).rolling(30, min_periods=1).mean().reset_index(0, drop=True).fillna(0)

    # price related features
    df['price_disc_pct'] = (df['base_price'] - df['price']) / df['base_price']
    df['price_x_promo'] = df['price'] * df['promo']

    # target
    df['target'] = df['sales']

    # fill any remaining NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df

def get_feature_columns():
    return [
        'price', 'promo', 'lag1', 'lag7_mean', 'lag30_mean', 'price_disc_pct',
        'price_x_promo', 'sin_yr', 'cos_yr', 'dow', 'month'
    ]
