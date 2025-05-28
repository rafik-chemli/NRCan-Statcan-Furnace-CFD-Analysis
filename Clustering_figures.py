#!/usr/bin/env python3
"""
CFD Figure Export Tool

This script exports publication-quality figures from the CFD analysis data.
It follows journal publication guidelines for figure format, resolution,
and appearance.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, Birch
import umap

# Set up accessible color schemes
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

COLORBLIND_PALETTE = sns.color_palette("colorblind", 10)
sns.set_palette(COLORBLIND_PALETTE)

SINGLE_COLUMN_SIZE = (3.5, 3.5)
FULL_WIDTH_SIZE = (7.5, 5)

def load_data(data_path):
    df = pd.read_csv(data_path)
    def rename_column(col):
        if col.startswith("proportion_"): return col.replace("proportion_", "proportion ")
        return col
    df = df.rename(columns=rename_column)
    state_mapping = {'penetrating': 'penetrating', 'oscillating': 'oscillating', 'bouncing': 'bouncing'}
    df["state"] = df["state"].map(state_mapping)
    return df

def perform_clustering(X_scaled, algorithm, n_clusters):
    if algorithm == "KMeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    elif algorithm == "Spectral":
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors', n_neighbors=100)
    elif algorithm == "BIRCH":
        clusterer = Birch(n_clusters=n_clusters)
    print(f"Performing {algorithm} clustering with n_clusters={n_clusters}, random_state=42")
    return clusterer.fit_predict(X_scaled)

def export_state_distribution(df, output_dir, width='single', figure_num=1):
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    state_counts = df['state'].value_counts()
    fig, ax = plt.subplots(figsize=fig_size)
    ax.pie(state_counts, labels=state_counts.index, autopct='%1.1f%%',
           colors=[COLORBLIND_PALETTE[i] for i in range(len(state_counts))],
           startangle=90, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    ax.set_title('Distribution of Particle States')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure_1_State_Distribution.png"), format='png', dpi=1000, bbox_inches='tight')
    plt.close()
    caption_file = os.path.join(output_dir, "figure_captions.txt")
    with open(caption_file, 'w' if figure_num == 1 else 'a') as f:
        if figure_num == 1: f.write("Figure Captions\n==============\n\n")
        f.write(f"Figure {figure_num}: Distribution of particle states in the CFD simulation. "
                f"The dataset contains {len(df)} particles distributed across "
                f"{len(state_counts)} states ({', '.join(state_counts.index)}).\n\n")

def export_feature_correlations(df, features, output_dir, width='single', feature_labels=None, figure_num=2):
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    state_dummies = pd.get_dummies(df['state'], prefix='state')
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    combined_data = pd.concat([pd.DataFrame(X_scaled, columns=features), state_dummies], axis=1)
    corr = pd.DataFrame(np.zeros((len(features), len(state_dummies.columns))), index=features, columns=state_dummies.columns)
    for f in features:
        for s in state_dummies.columns:
            corr.loc[f, s] = combined_data[f].corr(combined_data[s])
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=sns.diverging_palette(220, 10, as_cmap=True), center=0, square=True,
                linewidths=.5, cbar_kws={"shrink": .8, "label": "Correlation Coefficient"}, ax=ax)
    if feature_labels and len(feature_labels) == len(corr.index): ax.set_yticklabels(feature_labels, rotation=0)
    ax.set_title('Feature-State Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Figure_2_Feature_State_Correlations.png"), format='png', dpi=1000, bbox_inches='tight')
    plt.close()
    with open(os.path.join(output_dir, "figure_captions.txt"), 'a') as f:
        f.write(f"Figure {figure_num}: Correlation heatmap between particle features and states. "
                "Positive values (red) indicate positive correlation, while negative values (blue) "
                "indicate inverse correlation.\n\n")

def export_cluster_state_proportions(df, output_dir, width='full', feature_labels=None, figure_num=3, n_clusters=None):
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    cluster_state = df.groupby(['cluster', 'state']).size().unstack(fill_value=0)
    cluster_state_props = cluster_state.div(cluster_state.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=fig_size)
    bottom = np.zeros(len(cluster_state))
    for state in cluster_state.columns:
        values = cluster_state_props[state].values
        ax.bar(cluster_state.index, values, bottom=bottom, label=state, edgecolor='white', linewidth=1)
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 5: ax.text(i, b + v/2, f'{v:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        bottom += values
    total_sizes = cluster_state.sum(axis=1)
    for i, total in enumerate(total_sizes): ax.text(i, 105, f'n={total}', ha='center', fontsize=9)
    ax.set_title('Cluster Analysis: State Proportions and Counts')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Proportion (%)')
    ax.set_ylim(0, 110)
    ax.set_xticks(range(len(cluster_state)))
    ax.set_xticklabels(cluster_state.index)
    ax.legend(title='State', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(output_dir, f"Figure_3_Cluster_State_Proportions.png"), format='png', dpi=1000, bbox_inches='tight')
    plt.close()
    with open(os.path.join(output_dir, "figure_captions.txt"), 'a') as f:
        f.write(f"Figure {figure_num}: Proportion of particle states within each of the {n_clusters} clusters. "
                f"The stacked bar chart shows how particles with different states "
                f"({', '.join(cluster_state.columns)}) are distributed across the "
                f"{len(cluster_state)} identified clusters.\n\n")

def export_umap_visualization(df, features, output_dir, width='full', feature_labels=None, figure_num=4):
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, n_jobs=-1)
    embedding = reducer.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=fig_size)
    for i, state in enumerate(df['state'].unique()):
        mask = df['state'] == state
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=30, alpha=0.7, label=state, edgecolors='none')
    ax.set_title('UMAP Projection of Particle Features')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.legend(title='State', markerscale=1.5, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(output_dir, f"Figure_4_UMAP_Visualization.png"), format='png', dpi=1000, bbox_inches='tight')
    plt.close()
    with open(os.path.join(output_dir, "figure_captions.txt"), 'a') as f:
        f.write(f"Figure {figure_num}: UMAP dimensionality reduction visualization of the particle feature space. "
                "Points are colored by particle state.\n\n")

def export_feature_distributions(df, features, output_dir, width='full', feature_labels=None, figure_num=5):
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    scaled_df = df.copy()
    for f in features: scaled_df[f] = (df[f] - df[f].min()) / (df[f].max() - df[f].min())
    avg_by_state = scaled_df.groupby('state')[features].mean()
    std_by_state = scaled_df.groupby('state')[features].std()
    fig, ax = plt.subplots(figsize=fig_size)
    n_states = len(avg_by_state)
    width_bar = 0.8 / len(features)
    for i, f in enumerate(features):
        positions = np.arange(n_states) + (i - len(features)/2 + 0.5) * width_bar
        ax.bar(positions, avg_by_state[f], width=width_bar, yerr=std_by_state[f], label=f, capsize=3)
    ax.set_title('Standardized Feature Values by State')
    ax.set_ylabel('Standardized Value (0-1)')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(n_states))
    ax.set_xticklabels(avg_by_state.index)
    if feature_labels and len(feature_labels) == len(features): ax.legend(handles=ax.get_legend_handles_labels()[0], labels=feature_labels, title='Feature', loc='center left', bbox_to_anchor=(1, 0.5))
    else: ax.legend(title='Feature', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(output_dir, f"Figure_5_Feature_Distributions.png"), format='png', dpi=1000, bbox_inches='tight')
    plt.close()
    with open(os.path.join(output_dir, "figure_captions.txt"), 'a') as f:
        f.write(f"Figure {figure_num}: Standardized feature values by particle state. "
                "Bars show the mean value for each feature across states.\n\n")

def export_feature_patterns(df, features, output_dir, width='full', feature_labels=None, figure_num=6, n_clusters=None):
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    cluster_stats = df.groupby('cluster')[features].mean()
    cluster_stats_std = cluster_stats.copy()
    for f in features: cluster_stats_std[f] = (cluster_stats[f] - cluster_stats[f].min()) / (cluster_stats[f].max() - cluster_stats[f].min())
    cluster_stats_std = cluster_stats_std.T
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(cluster_stats_std, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    if feature_labels and len(feature_labels) == len(cluster_stats_std.index): ax.set_yticklabels(feature_labels, rotation=0)
    ax.set_title('Feature Patterns Across Clusters (Standardized)')
    ax.set_xlabel('Cluster')
    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(output_dir, f"Figure_6_Feature_Patterns.png"), format='png', dpi=1000, bbox_inches='tight')
    plt.close()
    with open(os.path.join(output_dir, "figure_captions.txt"), 'a') as f:
        f.write(f"Figure {figure_num}: Feature patterns across the {n_clusters} clusters. "
                "The heatmap shows the standardized mean values for each feature.\n\n")

def export_parallel_plot(df, features, output_dir, width='full', feature_labels=None, figure_num=7):
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    scaled_df = df.copy()
    for f in features: scaled_df[f] = (df[f] - df[f].min()) / (df[f].max() - df[f].min())
    fig, ax = plt.subplots(figsize=(fig_size[0] * 1.5, fig_size[1]))
    states = df['state'].unique()
    color_dict = {state: COLORBLIND_PALETTE[i] for i, state in enumerate(states)}
    for state in states:
        state_data = scaled_df[scaled_df['state'] == state]
        for _, row in state_data.iterrows():
            xs = np.arange(len(features))
            ys = [row[f] for f in features]
            ax.plot(xs, ys, alpha=0.4, c=color_dict[state], linewidth=1)
        state_means = state_data[features].mean()
        ax.plot(xs, [state_means[f] for f in features], c=color_dict[state], linewidth=3, label=state)
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(feature_labels if feature_labels else features, rotation=45, ha='right')
    ax.set_title('Parallel Coordinates Plot of Features by State')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Standardized Value (0-1)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='State', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(output_dir, f"Figure_7_Parallel_Plot.png"), format='png', dpi=1000, bbox_inches='tight')
    plt.close()
    with open(os.path.join(output_dir, "figure_captions.txt"), 'a') as f:
        f.write(f"Figure {figure_num}: Parallel coordinates plot of standardized feature values by state.\n\n")

def main():
    parser = argparse.ArgumentParser(description='Export journal-quality figures from CFD data')
    parser.add_argument('--data', type=str, default='final_states_clean.csv', help='Path to the CSV data file')
    parser.add_argument('--output', type=str, default='figures', help='Directory to save output figures')
    parser.add_argument('--features', type=str, nargs='+', default=['diameter', 'density', 'mass', 'vm', 'pressure', 'primary_flow', 'particle_feed'], help='Feature columns to use for analysis')
    parser.add_argument('--labels', type=str, nargs='+', default=None, help='Custom labels for features (same order as features)')
    parser.add_argument('--clustering', type=str, default='Spectral', choices=['KMeans', 'Spectral', 'BIRCH'], help='Clustering algorithm to use')
    parser.add_argument('--n_clusters', type=int, nargs='+', default=[6], help='List of cluster numbers to create')
    parser.add_argument('--width', type=str, default='full', choices=['single', 'full'], help='Figure width: single column or full page width')
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}")
    df = load_data(args.data)
    X = df[args.features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for n_clusters in args.n_clusters:
        print(f"\nProcessing {n_clusters} clusters...")
        print()
        df['cluster'] = perform_clustering(X_scaled, args.clustering, n_clusters)
        print("Cluster distribution:", df['cluster'].value_counts().sort_index().to_dict())
        cluster_dir = os.path.join(args.output, f"{n_clusters}_clusters")
        os.makedirs(cluster_dir, exist_ok=True)
        
        export_state_distribution(df, cluster_dir, args.width, figure_num=1)
        print("✓ Exported Figure 1: State Distribution")
        export_feature_correlations(df, args.features, cluster_dir, args.width, args.labels, figure_num=2)
        print("✓ Exported Figure 2: Feature-State Correlations")
        export_cluster_state_proportions(df, cluster_dir, args.width, args.labels, figure_num=3, n_clusters=n_clusters)
        print("✓ Exported Figure 3: Cluster State Proportions")
        export_umap_visualization(df, args.features, cluster_dir, args.width, args.labels, figure_num=4)
        print("✓ Exported Figure 4: UMAP Visualization")
        export_feature_distributions(df, args.features, cluster_dir, args.width, args.labels, figure_num=5)
        print("✓ Exported Figure 5: Feature Distributions")
        export_feature_patterns(df, args.features, cluster_dir, args.width, args.labels, figure_num=6, n_clusters=n_clusters)
        print("✓ Exported Figure 6: Feature Patterns Across Clusters")
        export_parallel_plot(df, args.features, cluster_dir, args.width, args.labels, figure_num=7)
        print("✓ Exported Figure 7: Parallel Coordinates Plot")
        print(f"Figures for {n_clusters} clusters saved to {cluster_dir}")

if __name__ == "__main__":
    main()