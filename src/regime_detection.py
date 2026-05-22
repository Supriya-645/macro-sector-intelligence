"""
Phase 4 — Regime Detection via Unsupervised ML.

Uses PCA and K-Means clustering on macro-economic features to classify
market months into 4 historical regimes (Expansion, Peak, Contraction, Recovery).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    load_master_dataset,
    setup_plotting_style,
    save_chart,
    get_data_paths,
    SECTOR_RETURN_COLS,
    MACRO_COLS,
    REGIME_COLORS,
    EVENT_WINDOWS,
)

# Remove Gold_Price from MACRO_COLS as it failed to download in Phase 1
if "Gold_Price" in MACRO_COLS:
    MACRO_COLS.remove("Gold_Price")


def engineer_features(df):
    """Engineer features from macro indicators for clustering."""
    print("  🔧 Engineering macro features...")
    
    avail_macros = [c for c in MACRO_COLS if c in df.columns]
    features = pd.DataFrame(index=df.index)
    
    for macro in avail_macros:
        # MoM Change
        features[f"{macro}_MoM"] = df[macro].pct_change()
        # YoY Change (12-month)
        features[f"{macro}_YoY"] = df[macro].pct_change(12)
        # 6-month Rolling Z-score
        rolling_mean = df[macro].rolling(6).mean()
        rolling_std = df[macro].rolling(6).std()
        features[f"{macro}_ZScore_6M"] = (df[macro] - rolling_mean) / rolling_std
        # 3-month momentum (difference)
        features[f"{macro}_Mom_3M"] = df[macro].diff(3)
        
    return features.dropna()


def perform_pca(features_df):
    """Standardize features and apply PCA to retain >=85% variance."""
    print("  📉 Applying PCA for dimensionality reduction...")
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    # Fit full PCA to get explained variance ratio
    pca_full = PCA()
    pca_full.fit(scaled_features)
    
    # Cumulative variance
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Number of components needed for 85% variance
    n_components = np.argmax(cumulative_variance >= 0.85) + 1
    print(f"    ➡️ Using {n_components} components to explain {cumulative_variance[n_components-1]*100:.1f}% of variance")
    
    # Plot scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color="#58A6FF")
    plt.axhline(y=0.85, color='r', linestyle='--', label='85% Threshold')
    plt.axvline(x=n_components, color='r', linestyle='--')
    plt.title('PCA Cumulative Explained Variance', pad=15)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    save_chart(plt.gcf(), "pca_scree_plot", subfolder="regime")
    
    # Transform data
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(scaled_features)
    
    return pd.DataFrame(pca_features, index=features_df.index)


def detect_regimes(pca_df, original_df):
    """Cluster PCA features and label regimes."""
    print("  🤖 Clustering via K-Means (k=4)...")
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_df)
    
    # Create a dataframe for labeling
    labels_df = pd.DataFrame(index=pca_df.index)
    labels_df["Cluster"] = clusters
    
    # We need to map clusters 0,1,2,3 to readable names based on Nifty 50 returns
    # A simple heuristic: sort clusters by median Nifty return
    if "Nifty_50_Return" in original_df.columns:
        merged = labels_df.join(original_df[["Nifty_50_Return"]])
        median_returns = merged.groupby("Cluster")["Nifty_50_Return"].median().sort_values()
        
        # Order: Contraction, Recovery, Peak, Expansion
        # Contraction is worst, Expansion is best.
        # This is a heuristic mapping
        mapping = {
            median_returns.index[0]: "Contraction",
            median_returns.index[1]: "Recovery",
            median_returns.index[2]: "Peak",
            median_returns.index[3]: "Expansion"
        }
    else:
        mapping = {0: "Contraction", 1: "Recovery", 2: "Peak", 3: "Expansion"}
        
    labels_df["Regime"] = labels_df["Cluster"].map(mapping)
    
    # Save regime labels
    paths = get_data_paths()
    labels_path = paths["processed"] / "regime_labels.csv"
    labels_df[["Regime"]].to_csv(labels_path)
    print(f"  💾 Saved regime labels: {labels_path.name}")
    
    return labels_df


def plot_regime_validation(labels_df, df):
    """Plot Nifty 50 with background regime colors."""
    print("  📊 Plotting regime validation...")
    
    if "Nifty_50" not in df.columns:
        return
        
    # Join Nifty 50
    plot_df = labels_df.join(df[["Nifty_50"]]).dropna()
    
    plt.figure(figsize=(16, 6))
    plt.plot(plot_df.index, plot_df["Nifty_50"], color="white", linewidth=1.5, zorder=5)
    
    # Shade backgrounds based on regime
    for i in range(len(plot_df) - 1):
        regime = plot_df["Regime"].iloc[i]
        color = REGIME_COLORS.get(regime, "gray")
        plt.axvspan(plot_df.index[i], plot_df.index[i+1], color=color, alpha=0.3, lw=0)
        
    # Custom legend for regimes
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=color, alpha=0.5, label=regime) for regime, color in REGIME_COLORS.items()]
    plt.legend(handles=handles, loc="upper left")
    
    plt.title("Nifty 50 Index with Detected Macro Regimes", pad=20)
    plt.ylabel("Nifty 50 Level")
    
    save_chart(plt.gcf(), "regime_timeline", subfolder="regime")


def plot_regime_sector_distributions(labels_df, df):
    """Box plots of sector returns by regime."""
    print("  📊 Plotting sector return distributions by regime...")
    
    avail_sectors = [c for c in SECTOR_RETURN_COLS if c in df.columns]
    if not avail_sectors:
        return
        
    merged = labels_df.join(df[avail_sectors]).dropna()
    
    # Melt for seaborn
    melted = merged.melt(id_vars=["Regime"], value_vars=avail_sectors, var_name="Sector", value_name="Return")
    melted["Sector"] = melted["Sector"].str.replace("_Return", "")
    
    plt.figure(figsize=(16, 10))
    sns.boxplot(
        data=melted,
        x="Regime",
        y="Return",
        hue="Sector",
        palette="husl",
        order=["Expansion", "Peak", "Contraction", "Recovery"] # Logical cycle order
    )
    plt.title("Sector Return Distributions by Macro Regime", pad=20)
    plt.ylabel("Monthly Return")
    plt.axhline(0, color="white", linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Sector", frameon=False)
    
    save_chart(plt.gcf(), "regime_sector_distributions", subfolder="regime")


def run_regime_detection():
    """Execute Phase 4 Regime Detection."""
    print("=" * 60)
    print("  PHASE 4 — REGIME DETECTION")
    print("=" * 60)
    
    setup_plotting_style()
    df = load_master_dataset()
    
    if df.empty:
        print("  ❌ Master dataset is empty. Run Phase 1 first.")
        return
        
    features_df = engineer_features(df)
    
    if features_df.empty:
        print("  ❌ Could not generate features.")
        return
        
    pca_df = perform_pca(features_df)
    labels_df = detect_regimes(pca_df, df)
    
    plot_regime_validation(labels_df, df)
    plot_regime_sector_distributions(labels_df, df)

    print("=" * 60)
    print("  ✅ PHASE 4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_regime_detection()
