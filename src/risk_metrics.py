"""
Phase 6 — Risk Metrics.

Computes annualized Sharpe Ratio, Maximum Drawdown, and Value at Risk (VaR)
for each sector, segmented by the 4 detected economic regimes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    load_master_dataset,
    load_regime_labels,
    setup_plotting_style,
    save_chart,
    get_data_paths,
    SECTOR_RETURN_COLS,
    REGIME_COLORS,
)

def compute_max_drawdown(returns):
    """Compute maximum drawdown from a series of returns."""
    if len(returns) == 0:
        return np.nan
    # Convert returns to wealth index
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns.min()

def compute_risk_metrics(df, regime_df):
    """Compute risk metrics per sector per regime."""
    print("  Computing Risk Metrics...")
    
    avail_sectors = [c for c in SECTOR_RETURN_COLS if c in df.columns]
    if not avail_sectors:
        return None
        
    merged = df.join(regime_df["Regime"], how="inner")
    
    # Risk-free rate proxy
    if "US_Fed_Funds_Rate" in merged.columns:
        # Convert annual percentage to monthly decimal
        merged["Risk_Free_Monthly"] = (merged["US_Fed_Funds_Rate"] / 100) / 12
    else:
        merged["Risk_Free_Monthly"] = 0.02 / 12

    results = []

    # 1. Overall Metrics
    for sector in avail_sectors:
        returns = merged[sector].dropna()
        rf = merged.loc[returns.index, "Risk_Free_Monthly"]
        
        if len(returns) < 12:
            continue
            
        excess_returns = returns - rf
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(12) if returns.std() != 0 else 0
        max_dd = compute_max_drawdown(returns)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        results.append({
            "Sector": sector.replace("_Return", ""),
            "Regime": "All Regimes",
            "Sharpe_Ratio": sharpe,
            "Max_Drawdown": max_dd,
            "VaR_95": var_95,
            "VaR_99": var_99,
            "Months": len(returns)
        })

    # 2. Regime-specific Metrics
    for regime in merged["Regime"].unique():
        if pd.isna(regime):
            continue
            
        regime_data = merged[merged["Regime"] == regime]
        
        for sector in avail_sectors:
            returns = regime_data[sector].dropna()
            rf = regime_data.loc[returns.index, "Risk_Free_Monthly"]
            
            if len(returns) < 6:
                # Insufficient data
                results.append({
                    "Sector": sector.replace("_Return", ""),
                    "Regime": regime,
                    "Sharpe_Ratio": np.nan,
                    "Max_Drawdown": np.nan,
                    "VaR_95": np.nan,
                    "VaR_99": np.nan,
                    "Months": len(returns)
                })
                continue
                
            excess_returns = returns - rf
            sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(12) if returns.std() != 0 else 0
            max_dd = compute_max_drawdown(returns)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            results.append({
                "Sector": sector.replace("_Return", ""),
                "Regime": regime,
                "Sharpe_Ratio": sharpe,
                "Max_Drawdown": max_dd,
                "VaR_95": var_95,
                "VaR_99": var_99,
                "Months": len(returns)
            })

    results_df = pd.DataFrame(results)
    
    paths = get_data_paths()
    out_path = paths["processed"] / "risk_metrics.csv"
    results_df.to_csv(out_path, index=False)
    print(f"  Saved risk metrics table: {out_path.name}")
    
    return results_df

def plot_risk_metrics(results_df):
    """Generate visualizations for risk metrics."""
    print("  Generating risk charts...")
    
    # Filter out 'All Regimes' and NaNs for plotting regime comparisons
    regime_only = results_df[(results_df["Regime"] != "All Regimes")].dropna()
    all_only = results_df[results_df["Regime"] == "All Regimes"].dropna()
    
    if regime_only.empty:
         return
         
    # 1. Sharpe Ratio Bar Chart
    plt.figure(figsize=(16, 8))
    sns.barplot(
        data=regime_only,
        x="Sector",
        y="Sharpe_Ratio",
        hue="Regime",
        palette=REGIME_COLORS,
        hue_order=["Expansion", "Peak", "Contraction", "Recovery"]
    )
    plt.title("Annualized Sharpe Ratio by Sector and Regime", pad=20)
    plt.ylabel("Sharpe Ratio")
    plt.axhline(0, color="white", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    save_chart(plt.gcf(), "sharpe_ratio_comparison", subfolder="risk")
    
    # 2. Max Drawdown Heatmap
    pivot_dd = regime_only.pivot(index="Regime", columns="Sector", values="Max_Drawdown")
    if not pivot_dd.empty:
        plt.figure(figsize=(14, 6))
        sns.heatmap(
            pivot_dd,
            annot=True,
            fmt=".1%",
            cmap="Reds_r", # Negative values, red is worse
            center=pivot_dd.values.mean(),
            linewidths=0.5,
            linecolor="#21262D"
        )
        plt.title("Maximum Drawdown by Sector and Regime", pad=20)
        plt.ylabel("Macro Regime")
        plt.xlabel("Sector")
        plt.xticks(rotation=45, ha='right')
        save_chart(plt.gcf(), "max_drawdown_heatmap", subfolder="risk")
        
    # 3. Overall VaR
    if not all_only.empty:
        var_data = pd.melt(
            all_only, 
            id_vars=["Sector"], 
            value_vars=["VaR_95", "VaR_99"],
            var_name="Confidence", 
            value_name="VaR"
        )
        
        plt.figure(figsize=(14, 7))
        sns.barplot(
            data=var_data,
            x="Sector",
            y="VaR",
            hue="Confidence",
            palette=["#FFA657", "#F85149"]
        )
        plt.title("Historical Value at Risk (Monthly) Across All Regimes", pad=20)
        plt.ylabel("Value at Risk (%)")
        plt.axhline(0, color="white", linestyle="-", alpha=0.2)
        plt.xticks(rotation=45, ha='right')
        
        # Format y-axis as percentage
        from matplotlib.ticker import PercentFormatter
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        
        save_chart(plt.gcf(), "var_comparison", subfolder="risk")


def run_risk_metrics():
    """Execute Phase 6 Risk Metrics."""
    print("=" * 60)
    print("  PHASE 6 — RISK METRICS")
    print("=" * 60)
    
    setup_plotting_style()
    df = load_master_dataset()
    
    if df.empty:
        print("  Master dataset is empty. Run Phase 1 first.")
        return
        
    try:
        regime_df = load_regime_labels()
    except Exception:
        print("  Regime labels not found. Run Phase 4 first.")
        return
        
    results_df = compute_risk_metrics(df, regime_df)
    if results_df is not None:
        plot_risk_metrics(results_df)
        
    print("=" * 60)
    print("  PHASE 6 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_risk_metrics()
