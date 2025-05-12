
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gc
import math

# Clusters to visualize
clusters_to_plot = [75, 39, 96]
# File paths
particle_path = '/content/drive/MyDrive/cfd/new data/aggregated_data.parquet' # full aggregated data with appropriate 
cluster_path = '/content/drive/MyDrive/cfd/new data/final_states_with_clusters_toplot.csv' # file used for clustering with tracker id


def predict_fate(df):
    rho_L = 3440
    gamma = 0.568
    turnonv = 1
    fate = []
    current_y = df['ParticleYPosition m'].iloc[0]
    current_v = df['ParticleYVelocity m/s'].iloc[0]
    current_id = df['tracker_id'].iloc[0]

    for index, row in df.iterrows():
        prev_y = current_y
        prev_v = current_v
        prev_id = current_id
        current_y = row['ParticleYPosition m']
        current_id = row['tracker_id']
        current_v = row['ParticleYVelocity m/s']

        if prev_id != current_id:
            turnonv = 1

        if current_y == -3.50525 and turnonv == 1:
            mass = row['ParticleMass kg']
            diam = row['ParticleDiameter m']
            vel = row['ParticleYVelocity m/s']
            rho = row['ParticleDensity kg/m3']
            We = rho_L * vel * vel * diam / gamma
            Bo = rho_L * 9.81 * diam * diam / gamma
            lambda_squared = (rho / rho_L) * (rho / rho_L)
            WeBo = We * math.pow(Bo * Bo * Bo, 0.5)
            if WeBo >= 12 / lambda_squared:
                fate_v = 2  # Penetrating
            elif WeBo < 12 / lambda_squared and WeBo >= 6 / lambda_squared:
                fate_v = 1  # Bouncing
            else:
                fate_v = 0  # Floating
            fate.append(fate_v)
            turnonv = 0
        elif current_y > prev_y and current_id == prev_id and turnonv == 1:
            fate_v = 1  # Bouncing
            fate.append(fate_v)
            turnonv = 0
        else:
            fate.append(None)  # No fate assigned yet

    return fate

def load_cluster_subset(particle_path, cluster_path, cluster_k, max_ids=30):
    """
    Load and merge data for a specific cluster, returning a subset of up to max_ids tracker_ids.

    Parameters:
    - particle_path: Path to aggregated_data.parquet
    - cluster_path: Path to final_states_cluster_tracker.csv
    - cluster_k: Cluster number to process
    - max_ids: Maximum number of tracker_ids to select
    """
    # Load only necessary columns from cluster data
    cluster_cols = ['tracker_id', 'cluster']
    df_clusters = pd.read_csv(cluster_path, usecols=cluster_cols)

    # Filter for the specified cluster and get unique tracker_ids
    df_clusters = df_clusters[df_clusters['cluster'] == cluster_k]
    tracker_ids = df_clusters['tracker_id'].unique()

    # Sample up to max_ids tracker_ids
    if len(tracker_ids) > max_ids:
        selected_ids = np.random.choice(tracker_ids, size=max_ids, replace=False)
    else:
        selected_ids = tracker_ids

    # Free memory
    del df_clusters
    gc.collect()

    # Load necessary columns for plotting and fate prediction
    particle_cols = ['tracker_id', 'ParticleXPosition m', 'ParticleYPosition m',
                     'ParticleResidenceTime s', 'ParticleYVelocity m/s',
                     'ParticleMass kg', 'ParticleDiameter m', 'ParticleDensity kg/m3']
    df_particles = pd.read_parquet(particle_path, columns=particle_cols)

    # Filter for selected tracker_ids
    df_subset = df_particles[df_particles['tracker_id'].isin(selected_ids)]

    # Free memory
    del df_particles
    gc.collect()

    return df_subset

def plot_cluster_trajectories(df, cluster_k, output_dir='/content'):
    """
    Plot 2D trajectories (X vs. Y) for a cluster’s tracker_ids, colored by fate, and display inline.

    Parameters:
    - df: DataFrame with particle data for selected tracker_ids
    - cluster_k: Cluster number for title
    - output_dir: Directory to save plots
    """
    # Add turnon column required by predict_fate
    df['turnon'] = 1

    # Compute fates
    fates = predict_fate(df)
    df['fate'] = fates

    # Create mapping of tracker_id to fate (use first non-null fate per tracker_id)
    fate_map = {}
    for tracker_id in df['tracker_id'].unique():
        tracker_fate = df[df['tracker_id'] == tracker_id]['fate'].dropna().iloc[0] if not df[df['tracker_id'] == tracker_id]['fate'].dropna().empty else None
        fate_map[tracker_id] = tracker_fate

    # Define state info with colors from previous code
    state_info = {
        0: {'label': 'floating', 'color': (231/255, 176/255, 79/255)},
        1: {'label': 'bouncing', 'color': (165/255, 220/255, 205/255)},
        2: {'label': 'penetrating', 'color': (50/255, 120/255, 163/255)}
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectories one tracker_id at a time
    for tracker_id in df['tracker_id'].unique():
        group = df[df['tracker_id'] == tracker_id]
        group = group.sort_values('ParticleResidenceTime s')
        fate_val = fate_map.get(tracker_id)
        if fate_val is not None and fate_val in state_info:
            color = state_info[fate_val]['color']
            ax.plot(group['ParticleXPosition m'], group['ParticleYPosition m'],
                    color=color, alpha=0.5, label=state_info[fate_val]['label'])

        # Free memory
        del group
        gc.collect()

    # Set labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Trajectories for Cluster {cluster_k} (Up to 30 Tracker IDs)')

    # Create legend
    legend_patches = [mpatches.Patch(color=info['color'], label=info['label'])
                      for fate_val, info in state_info.items()]
    ax.legend(handles=legend_patches, loc='upper right', title='Particle Fate')

    # Save plot
    plt.savefig(f'{output_dir}/cluster_{cluster_k}_trajectories.png', dpi=300, bbox_inches='tight')

    # Display plot inline
    plt.show()

    # Close figure to free memory
    plt.close(fig)
    gc.collect()




# Process each cluster one at a time
for cluster_k in clusters_to_plot:
    # Load subset for current cluster
    df_subset = load_cluster_subset(particle_path, cluster_path, cluster_k, max_ids=30)

    # Check if subset is non-empty
    if not df_subset.empty:
        print(f"Plotting cluster {cluster_k} with {len(df_subset['tracker_id'].unique())} tracker_ids")
        plot_cluster_trajectories(df_subset, cluster_k)
    else:
        print(f"Cluster {cluster_k} not found or has no matching tracker_ids")

    # Free memory
    del df_subset
    gc.collect()