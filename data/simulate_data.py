"""
Generate a synthetic retail dataset with price-demand relationships,
promotions, seasonality and per-product elasticity.

Saves: data/raw/sim_retail.csv
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

def simulate_data(n_products=50, n_days=365, start_date="2023-01-01", out_path="data/raw/sim_retail.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for pid in range(n_products):
        base_price = np.random.uniform(10, 100)
        base_demand = np.random.uniform(20, 200)
        elasticity = np.random.uniform(-2.5, -0.2)  # negative elasticity
        season_amp = np.random.uniform(0.1, 1.0)
        promo_prob = np.random.choice([0.02, 0.05, 0.08])

        for day in range(n_days):
            date = pd.Timestamp(start_date) + pd.Timedelta(days=day)
            season = 1 + season_amp * np.sin(2 * np.pi * (day % 365) / 365)
            promo = np.random.binomial(1, promo_prob)
            price_noise = np.random.normal(1, 0.02)
            price = base_price * (1 - 0.15 * promo) * price_noise
            # expected demand follows power law with price elasticity
            expected = base_demand * (price / base_price) ** elasticity * season
            # ensure non-negative and create Poisson observed sales
            observed = np.random.poisson(max(expected, 0.1))
            rows.append([
                date.strftime("%Y-%m-%d"),
                f"P{pid:03d}",
                round(price, 2),
                int(observed),
                int(promo),
                round(base_price, 2),
                round(elasticity, 3)
            ])

    df = pd.DataFrame(rows, columns=[
        "date", "product_id", "price", "sales", "promo", "base_price", "true_elasticity"
    ])
    df.to_csv(out_path, index=False)
    print(f"Saved simulated dataset → {out_path}  (rows={len(df)})")
    return df

if __name__ == "__main__":
    simulate_data()
