"""
Phase 10 — Interactive Streamlit Dashboard (Institutional UX).

Provides a professional, institutional-grade sidebar interface to interact
with the output of the Macro-Driven Sector Intelligence Platform.
Focuses on clear UX, lack of gimmicks, and high readability.
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Add root to path so we can import from src
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))

from src.utils import (
    get_data_paths,
    SECTOR_INDICES,
    REGIME_COLORS,
    MACRO_COLS,
    PALETTE,
)

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    HAS_AGGRID = True
except ImportError:
    HAS_AGGRID = False


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Macro Sector Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Institutional Dark Mode CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-color: #0d1117;
        --card-bg: #161b22;
        --card-bg-hover: #1c2128;
        --text-color: #c9d1d9;
        --text-muted: #8b949e;
        --accent: #58a6ff;
        --accent-green: #3fb950;
        --accent-red: #f85149;
        --accent-orange: #d29922;
        --border: #30363d;
    }

    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter', -apple-system, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 500 !important;
        letter-spacing: -0.5px;
    }

    /* Metric Cards */
    .metric-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 20px;
        text-align: left;
        transition: background-color 0.2s ease;
    }
    .metric-card:hover {
        background-color: var(--card-bg-hover);
        border-color: #484f58;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #ffffff;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 12px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    .metric-delta.positive { color: var(--accent-green); font-size: 13px; font-weight: 500;}
    .metric-delta.negative { color: var(--accent-red); font-size: 13px; font-weight: 500;}

    /* Sentiment / Prediction Card */
    .highlight-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 6px;
        padding: 24px;
        text-align: left;
    }

    /* News Item */
    .news-item {
        padding: 12px 0;
        border-bottom: 1px solid var(--border);
    }
    .news-item:last-child {
        border-bottom: none;
    }
    .news-title {
        font-size: 14px;
        font-weight: 500;
        color: var(--text-color);
        margin-bottom: 4px;
    }
    .news-meta {
        font-size: 12px;
        color: var(--text-muted);
    }

    /* General */
    div[data-testid="stSidebar"] {
        background-color: var(--card-bg) !important;
        border-right: 1px solid var(--border);
    }
    
    /* Clean up default Streamlit elements */
    .stSelectbox label, .stSlider label {
        font-weight: 500 !important;
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data():
    """Load all necessary datasets from disk."""
    paths = get_data_paths()

    try:
        master = pd.read_csv(
            paths["processed"] / "master_dataset.csv",
            parse_dates=["Date"], index_col="Date",
        )
    except Exception:
        master = pd.DataFrame()

    try:
        regime = pd.read_csv(
            paths["processed"] / "regime_labels.csv",
            parse_dates=["Date"], index_col="Date",
        )
    except Exception:
        regime = pd.DataFrame()

    try:
        risk = pd.read_csv(paths["processed"] / "risk_metrics.csv")
    except Exception:
        risk = pd.DataFrame()

    try:
        rotation = pd.read_csv(
            paths["processed"] / "sector_rotation_matrix.csv", index_col=0,
        )
    except Exception:
        rotation = pd.DataFrame()

    models = {}
    if paths["models"].exists():
        for file in paths["models"].glob("xgb_*.joblib"):
            sector_name = file.stem.replace("xgb_", "")
            try:
                models[sector_name] = joblib.load(file)
            except Exception:
                pass

    try:
        with open(paths["processed"] / "backtest_results.json") as f:
            backtest = json.load(f)
    except Exception:
        backtest = {}

    return master, regime, risk, rotation, models, backtest


master_df, regime_df, risk_df, rotation_df, xgb_models, backtest_results = load_data()


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Macro Intelligence")
    st.markdown("Sector Analytics Platform")
    st.markdown("---")
    
    navigation = st.radio(
        "Navigation",
        [
            "Platform Overview",
            "Market Overview",
            "Economic Environments",
            "Future Predictions",
            "AI Strategy Performance",
            "Risk & Safety",
            "Scenario Tester",
            "Live News Sentiment"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")

    if not master_df.empty:
        min_date = master_df.index.min().to_pydatetime()
        max_date = master_df.index.max().to_pydatetime()

        selected_dates = st.slider(
            "Global Date Filter",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="MMM YYYY",
            help="Filters the historical data shown across all relevant panels."
        )

        df = master_df[
            (master_df.index >= selected_dates[0])
            & (master_df.index <= selected_dates[1])
        ]
        if not regime_df.empty:
            r_df = regime_df[
                (regime_df.index >= selected_dates[0])
                & (regime_df.index <= selected_dates[1])
            ]
        else:
            r_df = pd.DataFrame()
    else:
        df = master_df
        r_df = regime_df
        st.warning("Data not found. Run the data pipeline.")

    st.markdown("---")
    st.caption("Institutional v2.1")


# ---------------------------------------------------------------------------
# Page: Platform Overview
# ---------------------------------------------------------------------------
if navigation == "Platform Overview":
    st.title("Platform Overview")
    st.markdown("""
    Welcome to the **Macro-Driven Sector Intelligence Platform**. This system analyzes macroeconomic data to identify structural market environments and predict equity sector performance.

    ### How to Use This Platform

    Use the sidebar navigation to access different analytical modules:

    - **Market Overview**: View current macro indicators (like Interest Rates and Inflation) and historical trends.
    - **Economic Environments**: See how an unsupervised machine learning algorithm has classified historical periods into four economic regimes (Expansion, Peak, Contraction, Recovery) and how different sectors performed in each.
    - **Future Predictions**: View next-month directional forecasts for each sector generated by our trained XGBoost models.
    - **AI Strategy Performance**: Review the historical backtest results of a trading strategy that follows the AI's predictions compared to a standard Buy & Hold strategy.
    - **Risk & Safety**: Analyze the historical risk metrics (like Drawdown and Volatility) for each sector across different economic regimes.
    - **Scenario Tester**: Manually adjust macroeconomic inputs to simulate a market shock and see how the AI adjusts its sector predictions.
    - **Live News Sentiment**: View real-time market headlines processed through a Natural Language Processing (NLP) model to gauge current market sentiment.

    *Use the Global Date Filter in the sidebar to restrict historical analysis to a specific timeframe.*
    """)

# ---------------------------------------------------------------------------
# Page: Market Overview
# ---------------------------------------------------------------------------
elif navigation == "Market Overview":
    st.title("Market Overview")
    st.markdown("Current macro-economic indicators and historical asset trends.")

    if not df.empty:
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        metrics = [
            ("Nifty_50", "Nifty 50 Index", col1, True, "Benchmark Indian equity index."),
            ("India_VIX", "India VIX", col2, False, "Measures market expectation of near-term volatility (fear gauge)."),
            ("US_Fed_Funds_Rate", "Fed Funds Rate (%)", col3, False, "Key US interest rate. Drives global capital flows."),
            ("Brent_Crude", "Brent Crude ($)", col4, False, "Global benchmark for oil prices. Impacts Indian inflation heavily."),
            ("DXY", "US Dollar Index", col5, False, "Measures the value of the US dollar relative to a basket of foreign currencies."),
            ("US_10Y_Yield", "US 10Y Yield (%)", col6, False, "Yield on the 10-year US Treasury note. A benchmark for global borrowing costs."),
        ]

        for col_id, title, streamlit_col, is_higher_better, tooltip in metrics:
            if col_id in df.columns and not pd.isna(latest.get(col_id, np.nan)):
                val = latest[col_id]
                delta = val - prev[col_id]
                pct_delta = (delta / prev[col_id]) * 100 if prev[col_id] != 0 else 0

                if "Rate" in col_id or "Yield" in col_id:
                    val_str = f"{val:.2f}%"
                    delta_str = f"{delta * 100:.1f} bps"
                else:
                    val_str = f"{val:,.2f}"
                    delta_str = f"{pct_delta:.2f}%"

                color_class = ""
                if delta > 0:
                    color_class = "positive" if is_higher_better else "negative"
                    delta_str = "▲ " + delta_str
                elif delta < 0:
                    color_class = "negative" if is_higher_better else "positive"
                    delta_str = "▼ " + str(delta_str).replace("-", "")

                html = f"""
                <div class="metric-card" title="{tooltip}">
                    <div class="metric-label">{title}</div>
                    <div class="metric-value">{val_str}</div>
                    <div class="metric-delta {color_class}">{delta_str} (MoM)</div>
                </div>
                """
                streamlit_col.markdown(html, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        st.subheader("Historical Trends")
        available_cols = [
            c for c in MACRO_COLS + list(SECTOR_INDICES.keys())
            if c in df.columns
        ]
        plot_col = st.selectbox("Select Indicator:", available_cols)

        fig = px.line(
            df.reset_index(), x="Date", y=plot_col,
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#161b22",
            paper_bgcolor="#0d1117",
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="",
            yaxis_title=plot_col.replace('_', ' ')
        )
        fig.update_traces(line_color="#58a6ff", line_width=2)

        if not r_df.empty:
            for regime in r_df["Regime"].unique():
                dates = r_df[r_df["Regime"] == regime].index
                fig.add_scatter(
                    x=dates,
                    y=[df[plot_col].min()] * len(dates),
                    mode="markers",
                    marker=dict(
                        color=REGIME_COLORS.get(regime, "gray"),
                        size=6, symbol="square",
                    ),
                    name=regime,
                )

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Economic Environments
# ---------------------------------------------------------------------------
elif navigation == "Economic Environments":
    st.title("Economic Environments")
    st.markdown("Analysis of historical macroeconomic regimes and resulting sectoral performance.")

    if not rotation_df.empty:
        st.subheader("Sector Rotation Matrix")
        st.markdown(
            "Average historical annualized returns for sectors across different mathematically defined macro environments.",
            help="This matrix calculates the average return for each sector when the economy meets specific conditions (e.g., Rising Interest Rates)."
        )

        fig = px.imshow(
            rotation_df.values,
            x=rotation_df.columns,
            y=rotation_df.index,
            text_auto=".1%",
            color_continuous_scale="RdYlGn",
            aspect="auto",
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#161b22",
            paper_bgcolor="#0d1117",
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sector rotation matrix not found.")

    st.markdown("---")

    if not r_df.empty and not df.empty:
        st.subheader("Unsupervised Regime Timeline")
        st.markdown(
            "The Nifty 50 equity index mapped against economic regimes identified by our K-Means clustering algorithm.",
            help="The machine learning algorithm grouped historical months into 4 clusters based on inflation, growth, and volatility data, without being told what the stock market did."
        )
        merged = df[["Nifty_50"]].join(r_df)
        fig = px.line(
            merged.reset_index(), x="Date", y="Nifty_50", color="Regime",
            color_discrete_map=REGIME_COLORS,
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#161b22",
            paper_bgcolor="#0d1117",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="",
            yaxis_title="Nifty 50 Index"
        )
        fig.update_traces(mode="lines+markers", marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Future Predictions
# ---------------------------------------------------------------------------
elif navigation == "Future Predictions":
    st.title("Future Predictions")
    st.markdown("Next-month directional forecasts generated by sector-specific Machine Learning (XGBoost) models.")

    if xgb_models and not df.empty:
        sector_choice = st.selectbox(
            "Target Sector:", list(xgb_models.keys()),
            format_func=lambda x: x.replace('_Return', '').replace('_', ' ')
        )
        model = xgb_models[sector_choice]

        avail_macros = [c for c in MACRO_COLS if c in df.columns]
        latest_data = pd.DataFrame(index=[df.index[-1]])

        for m in avail_macros:
            filled_macro = df[m].ffill().bfill()
            latest_data[m] = filled_macro.iloc[-1]
            latest_data[f"{m}_MoM"] = filled_macro.pct_change().iloc[-1]
            latest_data[f"{m}_YoY"] = filled_macro.pct_change(12).iloc[-1]

        if not r_df.empty:
            latest_regime = r_df["Regime"].iloc[-1]
            for r in ["Contraction", "Expansion", "Peak", "Recovery"]:
                latest_data[f"Regime_{r}"] = 1.0 if r == latest_regime else 0.0

        try:
            model_features = model.feature_names_in_
            X_pred = latest_data.reindex(columns=model_features).fillna(0)
            prob = model.predict_proba(X_pred)[0]
            pred = model.predict(X_pred)[0]

            col1, col2 = st.columns([1, 1.5])

            with col1:
                direction = "Positive / Upward" if pred == 1 else "Negative / Downward"
                color = "#3fb950" if pred == 1 else "#f85149"
                
                st.markdown(f"""
                <div class="highlight-card" style="border-left-color: {color}; height: 100%;">
                    <div class="metric-label">Next Month Forecast</div>
                    <div style="font-size: 32px; font-weight: 600; color: {color}; margin: 16px 0;">
                        {direction}
                    </div>
                    <div style="color: var(--text-muted); font-size: 14px;">
                        Model Confidence: <strong>{prob[pred] * 100:.1f}%</strong>
                    </div>
                    <p style="margin-top: 20px; font-size: 13px; color: #8b949e;">
                        This forecast is based purely on historical macroeconomic data and patterns. It does not account for fundamental company news or unpredictable black swan events.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                imp = pd.DataFrame({
                    "Feature": model_features,
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=True).tail(5)

                fig = px.bar(
                    imp, x="Importance", y="Feature",
                    orientation="h", title="Top Predictive Drivers for this Sector",
                )
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#161b22",
                    paper_bgcolor="#0d1117",
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                fig.update_traces(marker_color="#58a6ff")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not generate prediction: {e}")

    else:
        st.info("Predictive models not found. Ensure the pipeline has been executed.")

# ---------------------------------------------------------------------------
# Page: AI Strategy Performance
# ---------------------------------------------------------------------------
elif navigation == "AI Strategy Performance":
    st.title("Strategy Performance")
    st.markdown(
        "Historical backtest results of a trading strategy executing the AI's monthly predictions, compared against a passive Buy & Hold approach.",
        help="The strategy goes Long when the model predicts UP, and moves to Cash when the model predicts DOWN. Assumes zero transaction costs."
    )

    if backtest_results:
        bt_sectors = list(backtest_results.keys())
        bt_choice = st.selectbox(
            "Select Sector:", bt_sectors, key="bt_sector",
            format_func=lambda x: x.replace('_', ' ')
        )
        stats = backtest_results[bt_choice]

        c1, c2, c3, c4 = st.columns(4)

        ai_ret = stats.get("strategy_return", 0)
        bh_ret = stats.get("buyhold_return", 0)
        ai_color = "positive" if ai_ret > 0 else "negative"
        bh_color = "positive" if bh_ret > 0 else "negative"

        c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AI Strategy Return</div>
            <div class="metric-value" style="color: {'#3fb950' if ai_ret > 0 else '#f85149'};">{ai_ret:+.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Buy & Hold Return</div>
            <div class="metric-value" style="color: {'#3fb950' if bh_ret > 0 else '#f85149'};">{bh_ret:+.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div class="metric-card" title="Percentage of months where the model correctly predicted the direction.">
            <div class="metric-label">Prediction Accuracy</div>
            <div class="metric-value">{stats.get('win_rate', 0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        c4.markdown(f"""
        <div class="metric-card" title="The largest peak-to-trough drop in portfolio value. Lower is better.">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value" style="color: #f85149;">{stats.get('max_drawdown', 0):.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        paths = get_data_paths()
        chart_path = paths["charts"] / "backtest" / f"backtest_{bt_choice}.png"

        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)
        else:
            st.info("Detailed equity curve chart not found.")

        st.markdown("---")
        st.subheader("Cross-Sector Comparison")

        comparison_data = []
        for sector, s in backtest_results.items():
            comparison_data.append({
                "Sector": sector.replace('_', ' '),
                "AI Return": s.get("strategy_return", 0),
                "B&H Return": s.get("buyhold_return", 0),
            })

        comp_df = pd.DataFrame(comparison_data)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="AI Strategy", x=comp_df["Sector"], y=comp_df["AI Return"], marker_color="#3fb950",
        ))
        fig.add_trace(go.Bar(
            name="Buy & Hold", x=comp_df["Sector"], y=comp_df["B&H Return"], marker_color="#58a6ff",
        ))
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            plot_bgcolor="#161b22",
            paper_bgcolor="#0d1117",
            title="Cumulative Return Comparison",
            yaxis_tickformat=".0%",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No backtest results found in the database.")


# ---------------------------------------------------------------------------
# Page: Risk Management
# ---------------------------------------------------------------------------
elif navigation == "Risk & Safety":
    st.title("Risk & Safety Metrics")
    st.markdown("Examine the historical risk profile of different sectors.")

    if not risk_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            regime_filter = st.selectbox(
                "Filter by Economic Environment:",
                ["All Regimes"] + list(risk_df["Regime"].unique()),
            )

        if regime_filter != "All Regimes":
            filtered_risk = risk_df[risk_df["Regime"] == regime_filter]
        else:
            filtered_risk = risk_df[risk_df["Regime"] == "All Regimes"]

        if HAS_AGGRID and not filtered_risk.empty:
            display_df = filtered_risk.copy()

            for col_name in ["Sharpe_Ratio"]:
                if col_name in display_df.columns:
                    display_df[col_name] = display_df[col_name].round(3)
            for col_name in ["Max_Drawdown", "VaR_95", "VaR_99"]:
                if col_name in display_df.columns:
                    display_df[col_name] = (display_df[col_name] * 100).round(2)

            # Rename columns for clarity
            display_df.rename(columns={
                'Sharpe_Ratio': 'Risk-Adjusted Return (Sharpe)',
                'Max_Drawdown': 'Max Loss % (Drawdown)',
                'VaR_95': '95% Value at Risk %'
            }, inplace=True)
            
            # Drop the confusing 99% VaR to keep it simple
            if 'VaR_99' in display_df.columns:
                display_df.drop('VaR_99', axis=1, inplace=True)

            gb = GridOptionsBuilder.from_dataframe(display_df)
            gb.configure_default_column(filterable=True, sortable=True, resizable=True)
            grid_options = gb.build()

            AgGrid(
                display_df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                theme="alpine",
                height=400,
                fit_columns_on_grid_load=True,
            )
            st.caption(
                "💡 **Risk-Adjusted Return (Sharpe)**: Higher is better. It measures how much return you get for the risk you take.<br>"
                "💡 **Max Loss (Drawdown)**: Lower is better. The biggest historical drop from a peak to a trough.<br>"
                "💡 **95% Value at Risk**: The expected maximum loss in a normal month with 95% confidence.",
                unsafe_allow_html=True
            )
        else:
            st.dataframe(filtered_risk, use_container_width=True)

    else:
        st.info("Risk metrics not found.")

# ---------------------------------------------------------------------------
# Page: Scenario Tester
# ---------------------------------------------------------------------------
elif navigation == "Scenario Tester":
    st.title("Macro Scenario Tester")
    st.markdown("Adjust key economic indicators to simulate a hypothetical market shock. The AI will recalculate its sector predictions based on your custom inputs.")

    if xgb_models and not df.empty:
        latest = df.iloc[-1].copy()

        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.subheader("Adjust Economic Inputs")

            sim_rate = st.slider(
                "US Interest Rate (%)", 0.0, 10.0,
                float(latest.get("US_Fed_Funds_Rate", 5.0)), 0.25,
            )
            sim_crude = st.slider(
                "Oil Price (Brent Crude $)", 20.0, 150.0,
                float(latest.get("Brent_Crude", 80.0)), 5.0,
            )
            sim_dxy = st.slider(
                "US Dollar Strength (DXY)", 70.0, 130.0,
                float(latest.get("DXY", 100.0)), 1.0,
            )
            sim_vix = st.slider(
                "Market Fear (India VIX)", 10.0, 80.0,
                float(latest.get("India_VIX", 15.0)), 1.0,
            )

            st.markdown("---")
            run_sim = st.button("Run Simulation", type="primary", use_container_width=True)

        with col2:
            st.subheader("AI Output: Probability of Positive Return")

            if run_sim:
                sim_data = pd.DataFrame(index=[0])

                for m in [c for c in MACRO_COLS if c in df.columns]:
                    if m == "US_Fed_Funds_Rate": val = sim_rate
                    elif m == "Brent_Crude": val = sim_crude
                    elif m == "DXY": val = sim_dxy
                    elif m == "India_VIX": val = sim_vix
                    else: val = latest[m]

                    sim_data[m] = val
                    prev_val = df[m].iloc[-2]
                    sim_data[f"{m}_MoM"] = (val - prev_val) / prev_val if prev_val != 0 else 0

                    if len(df) >= 12:
                        yoy_val = df[m].iloc[-12]
                        sim_data[f"{m}_YoY"] = (val - yoy_val) / yoy_val if yoy_val != 0 else 0
                    else:
                        sim_data[f"{m}_YoY"] = 0

                if not r_df.empty:
                    latest_regime = r_df["Regime"].iloc[-1]
                    for r in ["Contraction", "Expansion", "Peak", "Recovery"]:
                        sim_data[f"Regime_{r}"] = 1.0 if r == latest_regime else 0.0

                results = []
                for sector, mdl in xgb_models.items():
                    model_features = mdl.feature_names_in_
                    X_sim = sim_data.reindex(columns=model_features).fillna(0)
                    p = mdl.predict_proba(X_sim)[0][1]
                    results.append({
                        "Sector": sector.replace("_Return", "").replace("_", " "),
                        "Probability": p,
                    })

                res_df = pd.DataFrame(results).sort_values("Probability", ascending=True)

                fig = px.bar(
                    res_df, x="Probability", y="Sector",
                    orientation="h", color="Probability",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 1],
                )
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#161b22",
                    paper_bgcolor="#0d1117",
                    xaxis_tickformat=".0%",
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                fig.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="#8b949e")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Adjust the sliders on the left and click 'Run Simulation' to view how the AI reacts to the new environment.")

    else:
        st.info("Predictive models required. Run the pipeline.")


# ---------------------------------------------------------------------------
# Page: Live News Sentiment
# ---------------------------------------------------------------------------
elif navigation == "Live News Sentiment":
    st.title("Live News Sentiment")
    st.markdown("Real-time sentiment analysis of market news to gauge current investor mood.")

    try:
        from src.sentiment import get_market_pulse
        sentiment_available = True
    except ImportError:
        sentiment_available = False

    if sentiment_available:
        with st.spinner("Fetching and analyzing latest headlines..."):
            pulse = get_market_pulse()

        score = pulse["score"]
        label = pulse["label"]
        news_df = pulse["df"]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"""
            <div class="highlight-card" style="margin-bottom: 20px;">
                <div class="metric-label">Overall Market Mood</div>
                <div style="font-size: 36px; font-weight: 600; color: {'#3fb950' if score > 0 else ('#f85149' if score < 0 else '#d29922')}; margin: 10px 0;">
                    {label}
                </div>
                <div style="color: var(--text-muted); font-size: 14px;">
                    Algorithm Score: <strong>{score:+.3f}</strong> (Range: -1 to +1)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption("This score is calculated by an NLP (Natural Language Processing) algorithm that reads recent news headlines and mathematically scores their tone.")

        with col2:
            if not news_df.empty:
                st.subheader("Analyzed Headlines")
                for _, row in news_df.iterrows():
                    color = "#3fb950" if row['label'] == "Positive" else ("#f85149" if row['label'] == "Negative" else "#8b949e")
                    st.markdown(f"""
                    <div class="news-item">
                        <div class="news-title">{row['title']}</div>
                        <div class="news-meta">
                            <span style="color: {color}; font-weight: 600;">{row['label']} ({row['compound']:+.2f})</span>
                            &nbsp;&nbsp;|&nbsp;&nbsp; {row['publisher']} &nbsp;&nbsp;|&nbsp;&nbsp; {row['published']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No news headlines available at the moment.")
    else:
        st.error("Sentiment module unavailable.")
