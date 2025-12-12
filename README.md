Dynamic Pricing Optimization System
AI-Powered Price Recommendation Engine with Competitor Simulation & Revenue Optimization

This project implements a complete Dynamic Pricing Engine using Machine Learning, Price Elasticity Modeling, and Competitor-Aware Optimization.
It helps businesses predict demand, determine optimal product prices, maximize revenue, and simulate market competition—all in real time.

The system includes:

🧠 XGBoost Demand Forecasting Model

💰 Optimal Price Recommendation Engine

📊 Interactive Streamlit Dashboard

⚔️ Competitor Price Simulation Module

📈 Demand, Revenue & Elasticity Curves

🔥 Automatic Feature Engineering Pipeline

🎨 Modern UI with Tabs, Heatmaps & Insights


📸 Live Demo Screenshots








⭐ Features
🔮 1. Machine Learning–Driven Price Optimization

Trains an XGBoost model on historical sales

Predicts demand for any price

Recommends revenue-maximizing price

Supports grid search & demand elasticity

⚔️ 2. Competitor-Aware Pricing Module

Allows input of competitor price

Adjusts demand prediction dynamically

Recalculates revenue & suggests competitor-adjusted optimal price

📊 3. Interactive Dashboard (Streamlit)

Historical sales visualization

Price & revenue curves

Elasticity curve

Simulation grid (price → demand → revenue)

Product selector

Upload your own dataset

🔍 4. Insights & Analytics

Correlation heatmap of features

Elasticity analysis

Demand sensitivity

Revenue behavior under price changes

🧱 5. Modular ML Pipeline

Synthetic dataset generation

Feature engineering

Model training

Price optimization

Real-time inference










📂 Project Structure



dynamic_pricing_project/
│
├── app/
│   └── streamlit_app.py        # Full UI dashboard
│
├── data/
│   ├── raw/
│   │   └── sim_retail.csv      # Sample dataset
│   └── simulate_data.py        # Synthetic data generator
│
├── models/
│   └── xgb_model.pkl           # Trained ML model
│
├── src/
│   ├── data.py                 # Data loading functions
│   ├── features.py             # Feature engineering
│   ├── model.py                # ML model training
│   ├── optimize.py             # Price optimization logic
│   ├── elasticity.py           # Elasticity utilities
│   └── utils.py                # Helpers
│
├── README.md
└── requirements.txt




🛠️ Tech Stack


Languages & Frameworks

Python 3.10+

Streamlit

XGBoost

Scikit-learn

Pandas / NumPy

Matplotlib & Seaborn

Key ML Techniques

Gradient Boosted Decision Trees

Price Elasticity Modeling

Revenue Maximization

Competitor-Aware Demand Adjustment



