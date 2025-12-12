"""
Small helper utilities for plotting and saving.
"""
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_sales_time_series(df, product_id, save_path=None):
    p_df = df[df['product_id'] == product_id].sort_values('date')
    plt.figure(figsize=(10,4))
    plt.plot(p_df['date'], p_df['sales'], marker='.', linestyle='-')
    plt.title(f"Sales for {product_id}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

def ensure_dirs():
    for d in ['models', 'data/raw', 'data/processed', 'reports']:
        os.makedirs(d, exist_ok=True)
