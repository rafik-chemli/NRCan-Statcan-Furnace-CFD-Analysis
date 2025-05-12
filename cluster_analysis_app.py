import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Define feature names and labels
FEATURES = ['primary_flow', 'pressure', 'density', 'particle_feed', 'diameter']
LABELS = [
    "Primary gas flow (kg/s)",
    "Pressure (bar)",
    "Particle density (kg/m³)",
    "Particle feed rate (kg/s)",
    "Particle diameter (m)"
]

# Load data
data = pd.read_csv('final_states_with_clusters.csv')

# Scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[FEATURES])
for i, feature in enumerate(FEATURES):
    data[f'scaled_{feature}'] = scaled_features[:, i]

# Set up plot styling to match publication quality
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
FIGURE_SIZE = (10, 5)  # Wider figure size

# Streamlit app
st.title("Cluster Feature Analysis")

# Create tabs
tab1, tab2 = st.tabs(["State-Based Analysis", "Custom Cluster Selection"])

with tab1:
    # Create layout columns
    col1, col2 = st.columns(2)

    # Cluster selection
    with col1:
        st.header("Cluster Selection")
        selected_state = st.selectbox("Select state", options=['penetrating', 'oscillating', 'bouncing'])

    # Sort options
    with col2:
        st.header("Sort Clusters")
        sort_options = ["State Proportion"] + LABELS
        sort_by = st.selectbox("Sort clusters by", options=sort_options, index=0)

    # Number of clusters slider
    num_clusters = st.slider("Number of top clusters", 1, len(data['cluster'].unique()), 3)

    # Calculate cluster proportions
    cluster_proportions = data.groupby('cluster')['state'].value_counts(normalize=True).unstack(fill_value=0) * 100
    top_clusters = cluster_proportions.sort_values(selected_state, ascending=False).head(num_clusters).index.tolist()
    st.write(f"Top clusters for {selected_state}: {', '.join(map(str, top_clusters))}")

    # Filter data
    filtered_data = data[data['cluster'].isin(top_clusters)]

    # Calculate mean scaled features
    feature_stats = filtered_data.groupby('cluster')[[f'scaled_{feature}' for feature in FEATURES]].mean()
    feature_stats = feature_stats.reindex(top_clusters).T
    feature_stats.index = LABELS

    # Calculate state proportions for filtered data
    state_proportions = filtered_data.groupby('cluster')['state'].value_counts(normalize=True).unstack(fill_value=0) * 100

    # Sort clusters
    if sort_by == "State Proportion":
        sorted_clusters = state_proportions.sort_values(selected_state, ascending=False).index
    else:
        feature_idx = LABELS.index(sort_by)
        feature_name = FEATURES[feature_idx]
        sort_values = feature_stats.loc[sort_by]
        sorted_clusters = sort_values.sort_values(ascending=False).index

    # Reorder data
    feature_stats_sorted = feature_stats[sorted_clusters]
    state_proportions_sorted = state_proportions.loc[sorted_clusters]

    # Create cluster labels
    cluster_labels = [f"Cluster {c}" for c in sorted_clusters]

    # Create heatmap without numbers
    st.subheader(f"Mean Scaled Features Sorted by {sort_by}")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=150)
    sns.heatmap(
        feature_stats_sorted,
        annot=False,
        cmap="YlGnBu",
        linewidths=.5,
        cbar_kws={"shrink": .8},
        ax=ax
    )
    ax.set_title(f'Top {num_clusters} {selected_state} Clusters Sorted by {sort_by}')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Scaled Feature')
    plt.tight_layout(pad=1.2)
    st.pyplot(fig)

    # Display state proportions
    st.subheader(f"State Proportions (%) Sorted by {sort_by}")
    st.dataframe(state_proportions_sorted.style.format("{:.2f}"))

    # Calculate pure clusters with adjustable threshold
    st.subheader("Number of Pure Clusters by State")
    purity_threshold = st.slider("Purity threshold (%)", min_value=50, max_value=100, value=95, step=1)
    all_cluster_proportions = data.groupby('cluster')['state'].value_counts(normalize=True).unstack(fill_value=0) * 100
    states = ['penetrating', 'oscillating', 'bouncing']
    pure_counts = {state: (all_cluster_proportions[state] >= purity_threshold).sum() for state in states}

    # Create bar plot for pure clusters
    fig2, ax2 = plt.subplots(figsize=FIGURE_SIZE, dpi=150)
    sns.barplot(
        x=states,
        y=[pure_counts[state] for state in states],
        palette=COLORBLIND_PALETTE[:3],
        ax=ax2
    )
    ax2.set_title(f'Count of Pure Clusters (>={purity_threshold}%) by State')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Number of Clusters')
    for i, v in enumerate([pure_counts[state] for state in states]):
        ax2.text(i, v + 0.1, str(v), ha='center', va='bottom')
    plt.tight_layout(pad=1.2)
    st.pyplot(fig2)

    # Add bar plot for mean feature values by state with standard deviation
    st.subheader("Standardized Feature Values by State")
    # Calculate mean and std of scaled features by state
    avg_by_state = data.groupby('state')[[f'scaled_{feature}' for feature in FEATURES]].mean()
    std_by_state = data.groupby('state')[[f'scaled_{feature}' for feature in FEATURES]].std()

    # Rename columns to match LABELS
    avg_by_state.columns = LABELS
    std_by_state.columns = LABELS

    # Create bar plot
    fig3, ax3 = plt.subplots(figsize=FIGURE_SIZE, dpi=150)
    n_states = len(avg_by_state)
    width_bar = 0.8 / len(FEATURES)  # Width of each bar
    for i, feature in enumerate(LABELS):
        positions = np.arange(n_states) + (i - len(FEATURES)/2 + 0.5) * width_bar
        ax3.bar(
            positions,
            avg_by_state[feature],
            width=width_bar,
            yerr=std_by_state[feature],
            label=feature,
            capsize=3,
            color=COLORBLIND_PALETTE[i]
        )

    ax3.set_title('Standardized Feature Values by State')
    ax3.set_ylabel('Standardized Value (0-1)')
    ax3.set_ylim(0, 1.2)  # Extend y-axis to accommodate error bars
    ax3.set_xticks(range(n_states))
    ax3.set_xticklabels(avg_by_state.index)
    ax3.legend(title='Feature', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=1.2)
    st.pyplot(fig3)

    # Add explanatory text
    st.markdown("""
    This plot shows the mean standardized feature values for each state, with error bars representing the standard deviation. 
    The large variability within each state highlights why clustering is necessary to better understand how features interact together.
    """)

with tab2:
    st.header("Custom Cluster Selection")
    st.markdown("Select specific clusters to compare their mean scaled features")

    # Get all unique clusters
    all_clusters = sorted(data['cluster'].unique())

    # Multi-select for clusters
    selected_clusters = st.multiselect("Select clusters to compare:", options=all_clusters, default=all_clusters[:3])

    if selected_clusters:
        # Calculate mean scaled features for selected clusters
        custom_feature_stats = data[data['cluster'].isin(selected_clusters)].groupby('cluster')[[f'scaled_{feature}' for feature in FEATURES]].mean()
        custom_feature_stats = custom_feature_stats.reindex(selected_clusters).T
        custom_feature_stats.index = LABELS
        
        # Create heatmap for selected clusters
        st.subheader("Mean Scaled Features for Selected Clusters")
        fig4, ax4 = plt.subplots(figsize=FIGURE_SIZE, dpi=150)
        sns.heatmap(
            custom_feature_stats,
            annot=True,  # Show values in cells
            fmt=".2f",   # Format to 2 decimal places
            cmap="YlGnBu",
            linewidths=.5,
            cbar_kws={"shrink": .8},
            ax=ax4
        )
        ax4.set_title('Mean Scaled Features for Selected Clusters')
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Scaled Feature')
        plt.tight_layout(pad=1.2)
        st.pyplot(fig4)
        
        # Display state proportions for selected clusters
        st.subheader("State Proportions (%) for Selected Clusters")
        custom_state_proportions = data[data['cluster'].isin(selected_clusters)].groupby('cluster')['state'].value_counts(normalize=True).unstack(fill_value=0) * 100
        custom_state_proportions = custom_state_proportions.reindex(selected_clusters)
        st.dataframe(custom_state_proportions.style.format("{:.2f}"))
    else:
        st.info("Please select at least one cluster to view the analysis")