# Macro-Driven Sector Intelligence Platform

A production-grade, end-to-end macro-economic sector analysis platform for Indian equity markets. This system ingests 15+ years of macro and market data, performs statistical hypothesis testing, detects economic regimes using unsupervised ML, forecasts sector returns using XGBoost and Prophet, and serves the results through a FastAPI backend, a React frontend, and a legacy Streamlit prototype.

---

## Architecture Overview

The platform is organized into 7 core phases:

1. **Data Pipeline**: Ingests Nifty sectoral indices via `yfinance`, US Macro/Commodities via `fredapi`, and Indian Macro via `wbgapi` (`wbgapi` used in place of deprecated `wbdata`). Handles forward-filling and monthly resampling.
2. **Exploratory Data Analysis**: Generates correlation heatmaps and event-based performance distributions (e.g., 2008 GFC, 2020 COVID).
3. **Statistical Analysis**: Computes Granger causality tests and extracts time-varying rolling OLS betas.
4. **Regime Detection**: Engineers features (MoM, YoY, Rolling Z-Scores), reduces dimensions via PCA, and applies K-Means clustering to classify historical months into 4 regimes: Expansion, Peak, Contraction, Recovery.
5. **Predictive Modeling**: Trains XGBoost classifiers via strict walk-forward backtesting to predict next-month sectoral direction. Employs Facebook Prophet for absolute index forecasting.
6. **Risk Metrics**: Calculates annualized Sharpe Ratios, Maximum Drawdowns, and Historical VaR segmented by regime.
7. **Application Layer**: FastAPI APIs, a React frontend for the main product interface, and a legacy Streamlit prototype kept for analyst-style exploration.

---

## Setup Instructions

1. **Clone the repository** and navigate to the project root:
   ```bash
   cd macro-sector-intelligence
   ```

2. **Set up a Python Virtual Environment** (Python 3.10+ recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**:
   Create a `.env` file in the root directory and add your FRED API key:
   ```env
   FRED_API_KEY=your_fred_api_key_here
   ```

---

## Running the Pipeline

Execute the phases sequentially from the project root. The scripts are designed to gracefully handle API rate limits or missing data columns.

```bash
# Phase 1: Build the Master Dataset
python src/data_pipeline.py

# Phase 2: Exploratory Data Analysis
python src/eda.py

# Phase 3: Statistical Analysis
python src/statistical_analysis.py

# Phase 4: Regime Detection (PCA + K-Means)
python src/regime_detection.py

# Phase 5: Predictive Models (XGBoost + Prophet)
python src/ml_models.py

# Phase 6: Risk Metrics
python src/risk_metrics.py
```

*All generated datasets (CSV/JSON) are saved to `data/processed/` and charts are saved to `data/processed/charts/`.*

---

## Running the App

Once all phases have been executed and the processed artifacts exist, start the backend and frontend:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

The main app is available at `http://127.0.0.1:5173`, and the API runs at `http://127.0.0.1:8000`.

### Main Interface
- **React Frontend**: Primary product interface for regime views, model outputs, risk analytics, tuning, and live updates.
- **FastAPI Backend**: Provides prediction, regime, simulation, sentiment, HMM, tuning, and email-report endpoints.

## Legacy Streamlit Prototype

This repository also includes an earlier Streamlit-based interface for analyst-oriented exploration. It is not the primary product surface, but it is kept in the repo as a prototype and reference implementation.

```bash
streamlit run legacy_streamlit/app.py
```

### Prototype Tabs
- **Macro Panel**: KPIs and historical trends.
- **Regime Heatmap**: Sector rotation matrix and timeline of detected macro regimes.
- **Predictive Engine**: Next-month XGBoost directional forecasts per sector with feature importance breakdown.
- **Risk Management**: Segmented Sharpe Ratio, Max Drawdown, and VaR table.
- **What-If Simulator**: Interactive sliders to simulate macro shocks and view the predicted impact on sector probabilities.

---

## Key Analytical Insights
- **Interest Rates vs. IT/Pharma**: Defensive sectors like IT and Pharma historically show distinct rolling beta behaviors during contraction regimes compared to high-beta sectors like Realty and Metals.
- **Regime Classification**: The K-Means clustering effectively identified the 2008 GFC and 2020 COVID crash as "Contraction" regimes purely based on unsupervised macro features (VIX spikes, negative GDP momentum).
- **Predictive Power**: Rate-of-change features (MoM, YoY) on DXY and Brent Crude proved highly influential in the XGBoost models for predicting FMCG and Auto sector directions.
