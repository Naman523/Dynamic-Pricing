"""
Upgraded Streamlit App:
- Tabs (Overview, Pricing Recommendation, Simulation Grid, Insights)
- Demand Curve, Revenue Curve
- Correlation Heatmap
- Price Elasticity Curve
- Competitor Pricing Module
"""

import sys
import os

# Ensure project root is importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_raw
from src.features import create_features, get_feature_columns
from src.optimize import find_optimal_price_row


# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="Dynamic Pricing Engine")
st.title("🔮 Dynamic Pricing — Price Recommendation Demo")


# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
@st.cache_data
def load_model(path="models/xgb_model.pkl"):
    model, features = joblib.load(path)
    return model, features


# -----------------------------------------------------------
# SIDEBAR — DATA INPUT
# -----------------------------------------------------------
st.sidebar.markdown("## Upload / Data")

uploaded = st.sidebar.file_uploader(
    "Upload CSV (date,product_id,price,sales,promo,base_price)",
    type=['csv']
)

use_sample = st.sidebar.checkbox("Use sample simulated data", value=True)

if uploaded:
    df = pd.read_csv(uploaded)
elif use_sample:
    try:
        df = load_raw("data/raw/sim_retail.csv")
    except Exception:
        st.error("Sample data not found. Run data/simulate_data.py first.")
        st.stop()
else:
    st.info("Upload a CSV or enable sample data.")
    st.stop()


# Prepare features
df_feat = create_features(df)

# Product selection
st.sidebar.markdown("## Select product")
product_list = df_feat["product_id"].unique().tolist()
pid = st.sidebar.selectbox("Product", product_list)

product_df = df_feat[df_feat["product_id"] == pid].sort_values("date")

# Get last row
last_row = product_df.iloc[-1].copy()


# -----------------------------------------------------------
# COMPETITOR CONTROLS
# -----------------------------------------------------------
st.sidebar.markdown("## Competitor Controls")

competitor_price = st.sidebar.number_input(
    "Competitor Price",
    value=float(max(1, last_row["price"] - 5))
)

sensitivity = st.sidebar.slider(
    "Competitor Influence Strength",
    0.0, 1.0, 0.3
)


# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
try:
    model, features = load_model()
except Exception:
    st.error("Trained model not found. Run: python src/model.py")
    st.stop()

# Predict current demand
X_last = last_row[features].values.reshape(1, -1)
pred_current = float(model.predict(X_last)[0])


# -----------------------------------------------------------
# TABS
# -----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "💰 Pricing Recommendation",
    "📈 Simulation Grid",
    "🔍 Insights"
])


# -----------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------
with tab1:
    st.header(f"Product: {pid}")

    st.subheader("Historical Sales")
    st.line_chart(product_df.set_index("date")["sales"])

    st.subheader("Current Snapshot")
    st.write(last_row[["date", "price", "sales", "promo", "base_price"]])

    st.metric("Predicted demand (current price)", f"{pred_current:.1f}")

    # Competitor-adjusted demand
    competitor_adjust = (competitor_price - last_row["price"]) * sensitivity
    pred_compet_adjusted = pred_current + competitor_adjust

    st.metric("Competitor-Adjusted Demand", f"{pred_compet_adjusted:.1f}")


# -----------------------------------------------------------
# TAB 2: PRICING RECOMMENDATION
# -----------------------------------------------------------
with tab2:
    st.header("Recommended Price")

    # Base optimal price (no competitor adjustment)
    base_opt = find_optimal_price_row(model, features, last_row)

    col1, col2 = st.columns(2)
    col1.metric("Current Price", f"{last_row['price']:.2f}")
    col2.metric("Recommended Price", f"{base_opt['best_price']:.2f}")

    st.metric("Expected Revenue (recommended)", f"{base_opt['best_revenue']:.2f}")

    # Generate price range for curves
    pct_grid = [0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3]
    prices = [last_row["price"] * pct for pct in pct_grid]

    # Demand & Revenue Curves
    demands = []
    revenues = []
    demands_adj = []
    revenues_adj = []

    for p in prices:
        new_row = last_row.copy()
        new_row["price"] = p
        pred = float(model.predict(new_row[features].values.reshape(1, -1))[0])

        # competitor adjusted prediction
        pred_adj = pred + (competitor_price - p) * sensitivity

        demands.append(pred)
        revenues.append(pred * p)
        demands_adj.append(pred_adj)
        revenues_adj.append(pred_adj * p)

    # Demand Curve
    st.subheader("Demand Curve")
    st.line_chart({"Base Demand": demands, "Competitor-Adjusted Demand": demands_adj})

    # Revenue Curve
    st.subheader("Revenue Curve")
    st.line_chart({"Base Revenue": revenues, "Adjusted Revenue": revenues_adj})

    # Competitor-adjusted recommended price
    best_idx = revenues_adj.index(max(revenues_adj))
    best_price_comp = prices[best_idx]

    st.subheader("Competitor-Adjusted Recommended Price")
    st.metric("Best Price (with competition)", f"{best_price_comp:.2f}")


# -----------------------------------------------------------
# TAB 3: SIMULATION GRID
# -----------------------------------------------------------
with tab3:
    st.header("Price Grid Simulation")

    pct_grid_ui = st.multiselect(
        "Select multipliers",
        [0.8, 0.9, 1.0, 1.05, 1.1, 1.2],
        default=[0.8, 0.9, 1.0, 1.05, 1.1, 1.2]
    )

    sim = []
    for pct in pct_grid_ui:
        p = last_row["price"] * pct
        new_row = last_row.copy()
        new_row["price"] = p

        pred = float(model.predict(new_row[features].values.reshape(1, -1))[0])
        pred_adj = pred + (competitor_price - p) * sensitivity

        sim.append({
            "price": round(p, 2),
            "predicted_demand": round(pred, 2),
            "expected_revenue": round(pred * p, 2),
            "adj_demand_competitor": round(pred_adj, 2),
            "adj_revenue_competitor": round(pred_adj * p, 2)
        })

    st.dataframe(pd.DataFrame(sim))


# -----------------------------------------------------------
# TAB 4: INSIGHTS (Heatmap + Elasticity Curve)
# -----------------------------------------------------------
with tab4:
    st.header("Insights & Analytics")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df_feat[features + ["sales"]].corr()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=False, cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Elasticity curve
    st.subheader("Price Elasticity Curve")

    elasticity = []
    for i in range(len(prices)-1):
        dp = prices[i+1] - prices[i]
        dd = demands[i+1] - demands[i]
        elasticity.append(dd / dp if dp != 0 else 0)

    st.line_chart({"elasticity": elasticity})


# Footer
st.markdown("---")
st.markdown(
    "**Notes:** This engine uses a model trained on simulated data. "
    "For production deployment, validate with holdout/backtesting and A/B testing."
)
