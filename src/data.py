"""
Data loading and simple cleaning utilities.
"""
import pandas as pd

def load_raw(path="data/raw/sim_retail.csv"):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    # basic cleaning
    df = df.dropna(subset=['date', 'product_id', 'price', 'sales'])
    # normalize types
    df['price'] = df['price'].astype(float)
    df['sales'] = df['sales'].astype(int)
    df['promo'] = df['promo'].astype(int)
    return df
