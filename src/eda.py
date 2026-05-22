"""
Phase 2 — Exploratory Data Analysis.

Generates visualizations to understand relationships between macro indicators
and sector returns, including correlation heatmaps, rolling correlations,
and performance distributions during historical events.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    load_master_dataset,
    setup_plotting_style,
    save_chart,
    get_data_paths,
    SECTOR_RETURN_COLS,
    MACRO_COLS,
    EVENT_WINDOWS,
)

# Remove Gold_Price from MACRO_COLS as it failed to download in Phase 1
if "Gold_Price" in MACRO_COLS:
    MACRO_COLS.remove("Gold_Price")

def plot_correlation_heatmap(df):
    """Plot static and interactive correlation heatmaps."""
    print("  Generating correlation heatmaps...")
    
    # Filter columns that exist in the dataframe
    avail_sectors = [c for c in SECTOR_RETURN_COLS if c in df.columns]
    avail_macros = [c for c in MACRO_COLS if c in df.columns]
    
    if not avail_sectors or not avail_macros:
        print("  Missing required columns for correlation.")
        return

    # Calculate correlation
    corr_df = df[avail_macros + avail_sectors].corr(method="pearson")
    macro_sector_corr = corr_df.loc[avail_macros, avail_sectors]

    # Static Matplotlib/Seaborn heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        macro_sector_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="#21262D",
        cbar_kws={"shrink": .8}
    )
    plt.title("Macro Indicators vs. Sector Returns Correlation", pad=20)
    plt.xlabel("Sector Returns")
    plt.ylabel("Macro Indicators")
    plt.xticks(rotation=45, ha='right')
    save_chart(plt.gcf(), "correlation_heatmap", subfolder="eda")

    # Interactive Plotly heatmap
    fig = px.imshow(
        macro_sector_corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Interactive Macro-Sector Correlation"
    )
    fig.update_layout(template="plotly_dark", height=800)
    
    paths = get_data_paths()
    eda_charts_dir = paths["charts"] / "eda"
    eda_charts_dir.mkdir(parents=True, exist_ok=True)
    html_path = eda_charts_dir / "correlation_heatmap.html"
    fig.write_html(str(html_path))
    print(f"  Saved interactive heatmap: {html_path.relative_to(paths['processed'].parent.parent)}")


def plot_rolling_correlations(df, window=12):
    """Plot 12-month rolling correlations for top macro indicators vs sectors."""
    print(f"  Generating {window}-month rolling correlations...")
    
    top_macros = ["US_Fed_Funds_Rate", "Brent_Crude", "DXY", "India_VIX", "US_10Y_Yield"]
    avail_macros = [m for m in top_macros if m in df.columns]
    avail_sectors = [c for c in SECTOR_RETURN_COLS if c in df.columns]

    for macro in avail_macros:
        plt.figure(figsize=(14, 8))
        for sector in avail_sectors:
            rolling_corr = df[macro].rolling(window).corr(df[sector])
            # Only plot if we have valid correlation data
            if not rolling_corr.dropna().empty:
                plt.plot(df.index, rolling_corr, label=sector.replace("_Return", ""), alpha=0.7, linewidth=1.5)
        
        plt.title(f"{window}-Month Rolling Correlation: {macro} vs Sectors", pad=15)
        plt.axhline(0, color="white", linestyle="--", alpha=0.5)
        plt.ylabel("Pearson Correlation")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
        save_chart(plt.gcf(), f"rolling_corr_{macro}", subfolder="eda")


def plot_event_distributions(df):
    """Plot box plots of sector returns during specific historical events."""
    print("  Generating event-based return distributions...")
    
    avail_sectors = [c for c in SECTOR_RETURN_COLS if c in df.columns]
    if not avail_sectors:
        return

    # Prepare data for plotting
    event_data = []
    for event_name, (start, end) in EVENT_WINDOWS.items():
        # Sub-select date range
        mask = (df.index >= start) & (df.index <= end)
        event_df = df.loc[mask, avail_sectors].copy()
        
        # Melt to long format for Seaborn
        if not event_df.empty:
            melted = event_df.melt(var_name="Sector", value_name="Return")
            melted["Event"] = event_name
            # Format sector names for display
            melted["Sector"] = melted["Sector"].str.replace("_Return", "")
            event_data.append(melted)
            
    if not event_data:
        print("  No data available during specified event windows.")
        return
        
    combined_event_data = pd.concat(event_data, ignore_index=True)
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    sns.boxplot(
        data=combined_event_data, 
        x="Event", 
        y="Return", 
        hue="Sector",
        palette="husl"
    )
    plt.title("Sector Return Distributions During Historical Market Events", pad=20)
    plt.ylabel("Monthly Return")
    plt.axhline(0, color="white", linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Sector", frameon=False)
    save_chart(plt.gcf(), "event_return_distributions", subfolder="eda")


def plot_trend_overview(df):
    """Plot multi-panel time series overview."""
    print("  Generating trend overview...")
    
    cols_to_plot = ["Nifty_50", "India_VIX", "US_Fed_Funds_Rate", "Brent_Crude"]
    avail_cols = [c for c in cols_to_plot if c in df.columns]
    
    if not avail_cols:
        return

    fig, axes = plt.subplots(len(avail_cols), 1, figsize=(14, 3 * len(avail_cols)), sharex=True)
    if len(avail_cols) == 1:
        axes = [axes]

    for i, col in enumerate(avail_cols):
        ax = axes[i]
        ax.plot(df.index, df[col], color="#58A6FF", linewidth=1.5)
        ax.set_ylabel(col.replace("_", " "))
        
        # Shade event windows
        for event_name, (start, end) in EVENT_WINDOWS.items():
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            if start_date in df.index or end_date in df.index or (start_date > df.index.min() and end_date < df.index.max()):
                 ax.axvspan(start_date, end_date, color="#F85149", alpha=0.15)
                 
                 # Only add text to the top plot
                 if i == 0:
                     mid_date = start_date + (end_date - start_date) / 2
                     # Ensure text is within bounds before plotting
                     if df.index.min() <= mid_date <= df.index.max():
                         ax.text(mid_date, ax.get_ylim()[1]*0.95, event_name, 
                                 horizontalalignment='center', rotation=90, 
                                 color="#F85149", alpha=0.8, fontsize=9)

    axes[0].set_title("Macro Trend Overview with Crisis Events", pad=20)
    save_chart(fig, "macro_trend_overview", subfolder="eda")


def run_eda():
    """Execute Phase 2 Exploratory Data Analysis."""
    print("=" * 60)
    print("  PHASE 2 — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    setup_plotting_style()
    df = load_master_dataset()
    
    if df.empty:
        print("  Master dataset is empty. Run Phase 1 first.")
        return

    plot_correlation_heatmap(df)
    plot_rolling_correlations(df)
    plot_event_distributions(df)
    plot_trend_overview(df)
    
    print("=" * 60)
    print("  PHASE 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
