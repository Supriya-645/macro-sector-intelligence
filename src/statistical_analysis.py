"""
Phase 3 — Statistical Analysis & Hypothesis Testing.

Computes Granger causality tests, runs 36-month rolling OLS regressions
to extract time-varying betas, computes correlation p-values, and builds
a Sector Rotation Matrix based on macro environments.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    load_master_dataset,
    setup_plotting_style,
    save_chart,
    get_data_paths,
    SECTOR_RETURN_COLS,
    MACRO_COLS,
)

# Remove Gold_Price from MACRO_COLS as it failed to download in Phase 1
if "Gold_Price" in MACRO_COLS:
    MACRO_COLS.remove("Gold_Price")


def compute_granger_causality(df, max_lags=3):
    """Compute Granger causality between macro indicators and sectors.
    
    Tests if past values of a macro indicator help predict future values
    of a sector return.
    """
    print("  Running Granger causality tests...")
    
    avail_sectors = [c for c in SECTOR_RETURN_COLS if c in df.columns]
    avail_macros = [c for c in MACRO_COLS if c in df.columns]
    
    results = []
    
    # Needs non-null continuous data
    clean_df = df[avail_sectors + avail_macros].dropna()
    
    if len(clean_df) < max_lags + 10:
        print("  Insufficient data for Granger causality.")
        return None

    # Matrix for heatmap (min p-value across lags)
    p_value_matrix = pd.DataFrame(index=avail_macros, columns=avail_sectors)

    for macro in avail_macros:
        for sector in avail_sectors:
            # We want to test if Macro granger-causes Sector
            # The grangercausalitytests function expects [y, x] where we test if x causes y
            data = clean_df[[sector, macro]]
            
            try:
                # We suppress the output printing of statsmodels
                import contextlib
                import os
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    gc_res = grangercausalitytests(data, maxlag=max_lags, verbose=False)
                
                min_p_val = 1.0
                best_lag = 1
                
                for lag in range(1, max_lags + 1):
                    # using SSR based F-test
                    p_val = gc_res[lag][0]['ssr_ftest'][1]
                    f_stat = gc_res[lag][0]['ssr_ftest'][0]
                    
                    if p_val < min_p_val:
                        min_p_val = p_val
                        best_lag = lag
                        
                    results.append({
                        "Macro": macro,
                        "Sector": sector,
                        "Lag": lag,
                        "F_Statistic": f_stat,
                        "P_Value": p_val
                    })
                    
                p_value_matrix.loc[macro, sector] = min_p_val
                
            except Exception as e:
                # Can happen if variance is zero or colinearity issues
                pass

    results_df = pd.DataFrame(results)
    
    paths = get_data_paths()
    results_path = paths["processed"] / "granger_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  Saved Granger results: {results_path.name}")
    
    # Plot heatmap of min p-values
    p_value_matrix = p_value_matrix.astype(float)
    plt.figure(figsize=(14, 10))
    # We use a custom colormap where low p-values are red/hot, high are cool
    sns.heatmap(
        p_value_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap="YlOrRd_r", # Reverse so smaller is darker/redder
        linewidths=0.5,
        linecolor="#21262D",
        vmin=0, vmax=0.1 # Cap at 0.1 to highlight significance
    )
    plt.title("Granger Causality (Minimum P-Value across 3 lags)", pad=20)
    plt.xlabel("Sector Returns")
    plt.ylabel("Macro Indicators")
    plt.xticks(rotation=45, ha='right')
    save_chart(plt.gcf(), "granger_pvalues", subfolder="statistical")
    
    return results_df


def compute_rolling_ols(df, window=36):
    """Run 36-month rolling OLS regressions to find time-varying betas."""
    print(f"  Running {window}-month rolling OLS regressions...")
    
    avail_sectors = [c for c in SECTOR_RETURN_COLS if c in df.columns]
    key_macros = ["US_Fed_Funds_Rate", "US_CPI", "Brent_Crude", "US_10Y_Yield", "DXY"]
    avail_macros = [m for m in key_macros if m in df.columns]
    
    if not avail_sectors or not avail_macros:
        return
        
    clean_df = df[avail_sectors + avail_macros].dropna()
    
    if len(clean_df) < window + 5:
         print("  Insufficient data for rolling OLS.")
         return

    # Add a constant for the intercept
    X = clean_df[avail_macros]
    X = sm.add_constant(X)
    
    for sector in avail_sectors:
        y = clean_df[sector]
        
        # We need a rolling OLS
        from statsmodels.regression.rolling import RollingOLS
        try:
            model = RollingOLS(y, X, window=window)
            rolling_res = model.fit()
            
            # rolling_res.params is a DataFrame of the coefficients over time
            params = rolling_res.params
            
            plt.figure(figsize=(14, 8))
            for macro in avail_macros:
                plt.plot(params.index, params[macro], label=macro, linewidth=1.5)
                
            plt.title(f"{window}-Month Rolling OLS Betas for {sector.replace('_Return', '')}", pad=15)
            plt.axhline(0, color="white", linestyle="--", alpha=0.5)
            plt.ylabel("Beta Coefficient")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
            save_chart(plt.gcf(), f"rolling_ols_betas_{sector}", subfolder="statistical")
            
        except Exception as e:
            print(f"    Failed rolling OLS for {sector}: {e}")


def compute_correlation_pvalues(df):
    """Compute Pearson and Spearman correlations with exact p-values."""
    print("  Computing correlation p-values...")
    
    avail_sectors = [c for c in SECTOR_RETURN_COLS if c in df.columns]
    avail_macros = [c for c in MACRO_COLS if c in df.columns]
    
    clean_df = df[avail_sectors + avail_macros].dropna()
    
    results = []
    
    for macro in avail_macros:
        for sector in avail_sectors:
            x = clean_df[macro]
            y = clean_df[sector]
            
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_rho, spearman_p = spearmanr(x, y)
            
            results.append({
                "Macro": macro,
                "Sector": sector,
                "Pearson_r": pearson_r,
                "Pearson_p": pearson_p,
                "Spearman_rho": spearman_rho,
                "Spearman_p": spearman_p
            })
            
    results_df = pd.DataFrame(results)
    paths = get_data_paths()
    results_path = paths["processed"] / "correlation_pvalues.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  Saved correlation p-values: {results_path.name}")


def build_sector_rotation_matrix(df):
    """Build a summary matrix of sector performance across macro environments."""
    print("  Building Sector Rotation Matrix...")
    
    avail_sectors = [c for c in SECTOR_RETURN_COLS if c in df.columns]
    if not avail_sectors:
        return
        
    # We need to define macro environments based on month-over-month changes
    environments = {}
    
    if "US_Fed_Funds_Rate" in df.columns:
        fed_mom = df["US_Fed_Funds_Rate"].diff()
        environments["Rising Rates"] = df[fed_mom > 0.05] # Add a small threshold
        environments["Falling Rates"] = df[fed_mom < -0.05]
        
    if "Brent_Crude" in df.columns:
        oil_mom = df["Brent_Crude"].pct_change()
        environments["Rising Oil"] = df[oil_mom > 0.02]
        environments["Falling Oil"] = df[oil_mom < -0.02]
        
    if "India_VIX" in df.columns:
        vix_median = df["India_VIX"].median()
        environments["High VIX"] = df[df["India_VIX"] > vix_median * 1.1] # Top tier
        environments["Low VIX"] = df[df["India_VIX"] < vix_median * 0.9] # Bottom tier

    if not environments:
        print("  Missing indicators for rotation matrix environments.")
        return

    # Calculate average returns
    rotation_data = {}
    counts = {}
    for env_name, env_df in environments.items():
        if not env_df.empty:
            # Calculate annualized mean return (mean monthly * 12)
            mean_returns = env_df[avail_sectors].mean() * 12
            rotation_data[env_name] = mean_returns
            counts[env_name] = len(env_df)

    if not rotation_data:
        return
        
    rotation_matrix = pd.DataFrame(rotation_data).T
    
    # Format sector names
    rotation_matrix.columns = [c.replace('_Return', '') for c in rotation_matrix.columns]
    
    # Sort columns by overall mean to make it look nicer
    rotation_matrix = rotation_matrix[rotation_matrix.mean().sort_values(ascending=False).index]

    # Save to CSV
    paths = get_data_paths()
    results_path = paths["processed"] / "sector_rotation_matrix.csv"
    rotation_matrix.to_csv(results_path)
    print(f"  Saved sector rotation matrix: {results_path.name}")
    
    # Plot Heatmap
    plt.figure(figsize=(14, 8))
    
    # Create custom labels showing N count
    y_labels = [f"{env} (n={counts[env]})" for env in rotation_matrix.index]
    
    sns.heatmap(
        rotation_matrix,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="#21262D"
    )
    plt.title("Sector Rotation Matrix: Annualized Returns by Macro Environment", pad=20)
    plt.xlabel("Sectors")
    plt.ylabel("Macro Environments")
    plt.yticks(ticks=np.arange(len(y_labels))+0.5, labels=y_labels, rotation=0)
    plt.xticks(rotation=45, ha='right')
    save_chart(plt.gcf(), "sector_rotation_matrix", subfolder="statistical")


def run_statistical_analysis():
    """Execute Phase 3 Statistical Analysis."""
    print("=" * 60)
    print("  PHASE 3 — STATISTICAL ANALYSIS")
    print("=" * 60)
    
    setup_plotting_style()
    df = load_master_dataset()
    
    if df.empty:
        print("  Master dataset is empty. Run Phase 1 first.")
        return

    compute_granger_causality(df)
    compute_rolling_ols(df)
    compute_correlation_pvalues(df)
    build_sector_rotation_matrix(df)
    
    print("=" * 60)
    print("  PHASE 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_statistical_analysis()
