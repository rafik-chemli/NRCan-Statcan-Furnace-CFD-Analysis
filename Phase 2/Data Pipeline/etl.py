#!/usr/bin/env python3
"""
ETL Pipeline for CFD Simulation Data Processing

This module provides functions for extracting, transforming, and loading
particle simulation data from CFD simulation files, including:
1. Extracting parameters from filenames
2. Assigning unique tracker IDs to particles
3. Processing and aggregating multiple simulation files
4. Converting data between formats (CSV and Parquet)
5. Calculating physical states of particles
6. Clustering particles based on their features

Author: NRCan-Statcan Collaboration
"""

import os
import re
import gc
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import SpectralClustering
from tqdm.auto import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Constants
RHO_L = 997  # Density of liquid (kg/m^3)
GAMMA = 0.072  # Surface tension (N/m)
STATES = {0: 'penetrating', 1: 'oscillating', 2: 'bouncing'}
CROSSING_Y_POSITION = -3.50525  # Y-position defining the crossing point


class FileProcessor:
    """Handles extraction and processing of CFD simulation files."""
    
    @staticmethod
    def extract_params_from_filename(filename: str) -> Optional[Dict[str, float]]:
        """
        Extract parameters from filename with pattern matching.
        
        Args:
            filename: The filename to parse
            
        Returns:
            Dictionary of extracted parameters or None if pattern doesn't match
        """
        pattern = r'(\d+)kgm3_(\d+\.\d+)vm_(\d+\.\d+)bar_primaryflow=(\d+\.\d+)kgs_particlefeed=(\d+(?:\.\d+)?)kgs'
        
        match = re.match(pattern, filename)
        if match:
            density, vm, pressure, primary_flow, particle_feed = match.groups()
            return {
                'density': float(density),
                'vm': float(vm),
                'pressure': float(pressure),
                'primary_flow': float(primary_flow),
                'particle_feed': float(particle_feed)
            }
        else:
            logger.warning(f"Could not parse parameters from filename: {filename}")
            return None
    
    @staticmethod
    def assign_tracker_ids(df: pd.DataFrame, start_tracker_id: int) -> Tuple[pd.DataFrame, int]:
        """
        Assign unique tracker IDs to particles using vectorized operations.
        
        Args:
            df: DataFrame with particle tracking data
            start_tracker_id: The starting ID to assign
            
        Returns:
            Tuple of (DataFrame with tracker_id column, last tracker ID used)
        """
        logger.info(f"Starting tracker ID assignment with {len(df)} rows")
        
        # Create conditions for new tracker IDs
        particle_change = (df['ParticleID -'] != df['ParticleID -'].shift()).astype(int)
        time_reset = ((df['ParticleResidenceTime s'] < df['ParticleResidenceTime s'].shift()) &
                     (df['ParticleID -'] == df['ParticleID -'].shift())).astype(int)
        
        # Combine conditions and create tracker IDs
        new_tracker = particle_change | time_reset
        tracker_ids = start_tracker_id + new_tracker.cumsum()
        
        df['tracker_id'] = tracker_ids
        last_tracker_id = tracker_ids.max()
        
        logger.info(f"Created {df['tracker_id'].nunique()} unique tracker IDs")
        return df, last_tracker_id
    
    @staticmethod
    def process_single_file(
        filename: str, 
        folder_path: str, 
        start_tracker_id: int, 
        output_path: str, 
        first_file: bool = False
    ) -> Tuple[Optional[Dict], int]:
        """
        Process a single data file and append to output CSV.
        
        Args:
            filename: Name of the file to process
            folder_path: Directory containing the file
            start_tracker_id: Starting tracker ID to use
            output_path: Path to write output
            first_file: Whether this is the first file (to write headers)
            
        Returns:
            Tuple of (stats dictionary, last tracker ID used)
        """
        try:
            file_path = os.path.join(folder_path, filename)
            
            # Read file
            df = pd.read_csv(file_path, index_col=False)
            
            # Assign tracker IDs
            df, last_tracker_id = FileProcessor.assign_tracker_ids(df, start_tracker_id)
            
            # Extract parameters from filename
            params = FileProcessor.extract_params_from_filename(filename)
            if params:
                for key, value in params.items():
                    df[key] = value
            
            # Add source filename column
            df['source_file'] = filename
            
            # Write to CSV
            df.to_csv(output_path, mode='w' if first_file else 'a',
                     header=first_file, index=False)
            
            # Collect statistics before deleting DataFrame
            stats = {
                'rows': len(df),
                'unique_particles': df['ParticleID -'].nunique(),
                'unique_trackers': df['tracker_id'].nunique(),
                'params': params
            }
            
            del df
            gc.collect()
            
            return stats, last_tracker_id
        
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            return None, start_tracker_id


class DataAggregator:
    """Handles aggregation of multiple data files."""
    
    @staticmethod
    def process_all_files(path: str) -> str:
        """
        Process all CSV files in a directory, writing to a single output file.
        
        Args:
            path: Directory containing CSV files to process
            
        Returns:
            Path to the output file
        """
        # Get list of all CSV files
        file_list = [f for f in os.listdir(path) if f.endswith('.csv')]
        logger.info(f"Processing {len(file_list)} CSV files...")
        
        output_path = os.path.join(path, "aggregated_data.csv")
        
        current_tracker_id = 1
        total_rows = 0
        param_ranges = defaultdict(set)
        
        # Process each file
        for i, filename in enumerate(tqdm(file_list, desc="Processing files")):
            stats, current_tracker_id = FileProcessor.process_single_file(
                filename, path, current_tracker_id,
                output_path, first_file=(i==0)
            )
            
            if stats:
                total_rows += stats['rows']
                if stats['params']:
                    for key, value in stats['params'].items():
                        param_ranges[key].add(value)
                current_tracker_id += 1
            
            gc.collect()
        
        # Print summary using accumulated statistics
        logger.info("\nProcessing Summary")
        logger.info("=================")
        logger.info(f"Total rows processed: {total_rows:,}")
        logger.info(f"Files processed: {len(file_list)}")
        logger.info("\nParameter ranges:")
        for key, values in param_ranges.items():
            logger.info(f"{key}: {sorted(values)}")
        logger.info(f"\nOutput saved to: {output_path}")
        
        return output_path


class FormatConverter:
    """Handles conversion between data formats and column inspection."""
    
    @staticmethod
    def csv_to_parquet(csv_path: str, parquet_path: str, chunk_size: int = 1000000) -> None:
        """
        Convert CSV file to Parquet format in chunks.
        
        Args:
            csv_path: Path to source CSV file
            parquet_path: Path to destination Parquet file
            chunk_size: Size of chunks to process at once
        """
        logger.info(f"Converting CSV to Parquet: {csv_path} -> {parquet_path}")
        
        # Read CSV in chunks and write to Parquet
        first_chunk = True
        for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size), desc="Converting to Parquet"):
            if first_chunk:
                chunk.to_parquet(parquet_path, compression='snappy', index=False)
                first_chunk = False
            else:
                chunk.to_parquet(parquet_path, compression='snappy', mode='append', index=False)
        
        logger.info(f"Conversion complete. Parquet file saved to {parquet_path}")
    
    @staticmethod
    def csv_to_parquet_arrow(csv_path: str, parquet_path: str, chunk_size: int = 1000000) -> None:
        """
        Convert CSV file to Parquet format using PyArrow for better performance.
        
        Args:
            csv_path: Path to source CSV file
            parquet_path: Path to destination Parquet file
            chunk_size: Size of chunks to process at once
        """
        logger.info(f"Converting CSV to Parquet using PyArrow: {csv_path} -> {parquet_path}")
        
        pq_writer = None
        df_iterator = pd.read_csv(csv_path, chunksize=chunk_size)
        
        for chunk in tqdm(df_iterator, desc="Converting to Parquet"):
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(
                    parquet_path,
                    pa.Table.from_pandas(chunk).schema,
                    compression='snappy'
                )
            table = pa.Table.from_pandas(chunk)
            pq_writer.write_table(table)
        
        if pq_writer:
            pq_writer.close()
            
        logger.info(f"Conversion complete. Parquet file saved to {parquet_path}")
    
    @staticmethod
    def inspect_columns(csv_path: str, parquet_path: str) -> List[str]:
        """
        Convert CSV to Parquet if needed and return column names.
        
        Args:
            csv_path: Path to source CSV file
            parquet_path: Path to destination Parquet file
            
        Returns:
            List of column names in the dataset
        """
        # Check if Parquet file exists
        if not os.path.exists(parquet_path):
            logger.info(f"Parquet file not found at {parquet_path}. Converting CSV to Parquet...")
            FormatConverter.csv_to_parquet(csv_path, parquet_path)
        else:
            logger.info(f"Parquet file found at {parquet_path}. Proceeding with existing file.")
        
        # Read the Parquet file metadata to get column names
        parquet_file = pq.ParquetFile(parquet_path)
        columns = parquet_file.schema.names
        
        logger.info("\nColumns in the dataset:")
        for col in columns:
            logger.info(f"- {col}")
            
        return columns


class StateCalculator:
    """Handles calculation of physical states for particles."""
    
    @staticmethod
    def calculate_states_vectorized(
        df: pd.DataFrame, 
        rho_L: float = RHO_L, 
        gamma: float = GAMMA
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized calculations for Weber number, Bond number, and lambda squared.
        
        Args:
            df: DataFrame with particle data
            rho_L: Liquid density (kg/m^3)
            gamma: Surface tension (N/m)
            
        Returns:
            Tuple of (states, Weber numbers, Bond numbers, lambda_squared values)
        """
        # Calculate velocity magnitude vectorized
        velocity_magnitude = np.sqrt(
            df['ParticleXVelocity m/s']**2 +
            df['ParticleYVelocity m/s']**2 +
            df['ParticleZVelocity m/s']**2
        )
        
        # Calculate parameters vectorized
        We = rho_L * velocity_magnitude**2 * df['ParticleDiameter m'] / gamma
        Bo = rho_L * 9.81 * df['ParticleDiameter m']**2 / gamma
        lambda_squared = (df['ParticleDensity kg/m3'] / rho_L) ** 2
        
        # Calculate WeBo
        WeBo = We * np.sqrt(Bo**3)
        
        # Determine states vectorized
        states = np.zeros(len(df), dtype=int)
        states[(WeBo >= 12 / lambda_squared)] = 0  # penetrating
        states[(WeBo >= 6 / lambda_squared) & (WeBo < 12 / lambda_squared)] = 1  # oscillating
        states[(WeBo < 6 / lambda_squared)] = 2  # bouncing
        
        return states, We, Bo, lambda_squared
    
    @staticmethod
    def process_tracker_states(
        input_path: str, 
        output_path: str, 
        rho_L: float = RHO_L, 
        gamma: float = GAMMA, 
        chunk_size: int = 1000000
    ) -> pd.DataFrame:
        """
        Process particle tracker states from aggregated data.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output Parquet file
            rho_L: Liquid density (kg/m^3)
            gamma: Surface tension (N/m)
            chunk_size: Size of chunks to process at once
            
        Returns:
            DataFrame with final states
        """
        # Convert CSV to Parquet if needed
        parquet_path = input_path.replace('.csv', '.parquet')
        if not os.path.exists(parquet_path):
            logger.info("Converting CSV to Parquet for better performance...")
            FormatConverter.csv_to_parquet_arrow(input_path, parquet_path, chunk_size)
        
        logger.info("Processing data in chunks...")
        parquet_file = pq.ParquetFile(parquet_path)
        
        all_last_states = []
        tracker_info = {}  # Store first time and crossing time for each tracker
        
        # First pass: find first times and crossing times for each tracker
        logger.info("Finding first times and crossing times for each tracker...")
        for chunk in tqdm(parquet_file.iter_batches(batch_size=chunk_size)):
            df_chunk = chunk.to_pandas()
            
            for tracker_id, group in df_chunk.groupby('tracker_id'):
                if tracker_id not in tracker_info:
                    # Get first time for this tracker
                    first_time = group['ParticleResidenceTime s'].min()
                    
                    # Find first crossing time if it exists
                    crossing_mask = group['ParticleYPosition m'] <= CROSSING_Y_POSITION
                    if crossing_mask.any():
                        crossing_time = group[crossing_mask]['ParticleResidenceTime s'].min()
                    else:
                        # If no crossing, use the last time
                        crossing_time = group['ParticleResidenceTime s'].max()
                    
                    tracker_info[tracker_id] = {
                        'first_time': first_time,
                        'last_time': crossing_time,
                        'crossed': crossing_mask.any()
                    }
                else:
                    # Update last time if we find a crossing point
                    if not tracker_info[tracker_id]['crossed']:
                        crossing_mask = group['ParticleYPosition m'] <= CROSSING_Y_POSITION
                        if crossing_mask.any():
                            crossing_time = group[crossing_mask]['ParticleResidenceTime s'].min()
                            tracker_info[tracker_id]['last_time'] = crossing_time
                            tracker_info[tracker_id]['crossed'] = True
                        else:
                            # If still no crossing, update the last time
                            last_time = group['ParticleResidenceTime s'].max()
                            if last_time > tracker_info[tracker_id]['last_time']:
                                tracker_info[tracker_id]['last_time'] = last_time
        
        # Second pass: process first and last states
        logger.info("Processing first and last states...")
        for chunk in tqdm(parquet_file.iter_batches(batch_size=chunk_size)):
            df_chunk = chunk.to_pandas()
            
            # Find first and last states in this chunk
            states_mask = df_chunk.apply(
                lambda row: (tracker_info[row['tracker_id']]['first_time'] == row['ParticleResidenceTime s']) or
                           (tracker_info[row['tracker_id']]['last_time'] == row['ParticleResidenceTime s']),
                axis=1
            )
            
            if states_mask.any():
                states_chunk = df_chunk[states_mask].copy()
                
                # Add state type (first/last)
                states_chunk['state_type'] = states_chunk.apply(
                    lambda row: 'first' if row['ParticleResidenceTime s'] == tracker_info[row['tracker_id']]['first_time'] else 'last',
                    axis=1
                )
                
                # Calculate states
                states, We, Bo, lambda_squared = StateCalculator.calculate_states_vectorized(states_chunk, rho_L, gamma)
                
                states_chunk['state'] = states
                states_chunk['Weber_number'] = We
                states_chunk['Bond_number'] = Bo
                states_chunk['lambda_squared'] = lambda_squared
                
                all_last_states.append(states_chunk)
            
            del df_chunk
            gc.collect()
        
        # Combine results
        logger.info("Combining results...")
        final_states = pd.concat(all_last_states, ignore_index=True)
        
        # Save results
        logger.info("Saving results...")
        final_states.to_parquet(output_path, compression='snappy')
        
        # Print summary
        logger.info("\nSummary Statistics:")
        logger.info("-----------------")
        logger.info(f"Total trackers processed: {final_states['tracker_id'].nunique():,}")
        logger.info("\nState distribution by type:")
        logger.info(str(final_states.groupby(['state_type', 'state']).size().unstack(fill_value=0)))
        logger.info(f"\nAverage residence time (last states): {final_states[final_states['state_type'] == 'last']['ParticleResidenceTime s'].mean():.2f} s")
        logger.info(f"Average residence time (first states): {final_states[final_states['state_type'] == 'first']['ParticleResidenceTime s'].mean():.2f} s")
        
        return final_states


class DataTransformer:
    """Handles data transformation, cleaning, and renaming."""
    
    @staticmethod
    def process_and_rename(input_path: str, output_path: str) -> pd.DataFrame:
        """
        Process and rename dataset variables, creating separate first/last state columns.
        
        Args:
            input_path: Path to input Parquet file
            output_path: Path to output Parquet file
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing and renaming data from {input_path}")
        
        # Read the parquet file
        df = pd.read_parquet(input_path)
        
        # Select base variables with new shorter names mapping
        var_mapping = {
            'ParticleDiameter m': 'diameter',
            'ParticleDensity kg/m3': 'density',
            'ParticleMass kg': 'mass',
            'ParticleXPosition m': 'x_pos',
            'ParticleYVelocity m/s': 'y_vel',
            'ParticleTemperature K': 'temperature',
            'ParticleVelocityMagnitude m/s': 'vel_mag',
            'vm': 'vm',
            'density': 'density_field',  # added '_field' to distinguish from particle density
            'pressure': 'pressure',
            'primary_flow': 'primary_flow',
            'particle_feed': 'particle_feed',
            'state_type': 'state_type',
            'state': 'state',
            'source_file': 'source_file',
            'tracker_id': 'tracker_id'
        }
        
        # Select and rename base variables
        selected_vars = list(var_mapping.keys())
        df_renamed = df[selected_vars].rename(columns=var_mapping)
        
        # Create two separate dataframes for first and last states
        first_df = df_renamed[df['state_type'] == 'first'].copy()
        last_df = df_renamed[df['state_type'] == 'last'].copy()
        
        # Rename columns with prefixes
        first_df.columns = ['first_' + col for col in first_df.columns]
        last_df.columns = ['last_' + col for col in last_df.columns]
        
        # Join the dataframes
        new_df = pd.concat([first_df.reset_index(drop=True),
                         last_df.reset_index(drop=True)], axis=1)
        
        # Save the new dataset
        new_df.to_parquet(output_path, compression='snappy')
        
        logger.info("New dataset columns:")
        for col in new_df.columns:
            logger.info(col)
        
        return new_df
    
    @staticmethod
    def clean_data(input_path: str, output_path: str) -> pd.DataFrame:
        """
        Create a clean dataset with selected variables.
        
        Args:
            input_path: Path to input Parquet file
            output_path: Path to output Parquet file
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data from {input_path}")
        
        # Read the parquet file
        df = pd.read_parquet(input_path)
        
        # Select variables
        selected_vars = [
            'first_diameter',
            'first_density',
            'first_mass',
            'first_vm',
            'first_pressure',
            'first_primary_flow',
            'first_particle_feed',
            'last_state',
            'first_source_file',  # Keep track of the source file
            'first_tracker_id'
        ]
        
        # Select only needed columns and remove NaN rows
        df_clean = df[selected_vars].dropna()
        
        logger.info(f"Original shape: {df.shape}")
        logger.info(f"Shape after removing NaNs: {df_clean.shape}")
        
        # Save the cleaned dataset
        df_clean.to_parquet(output_path, compression='snappy')
        
        logger.info(f"\nCleaned dataset saved to: {output_path}")
        logger.info("\nColumns in cleaned dataset:")
        for col in df_clean.columns:
            logger.info(f"- {col}")
        
        return df_clean


class ClusteringProcessor:
    """Handles clustering of particle data."""
    
    @staticmethod
    def load_data(data_path: str) -> pd.DataFrame:
        """
        Load and prepare data for clustering.
        
        Args:
            data_path: Path to input Parquet file
            
        Returns:
            Prepared DataFrame
        """
        df = pd.read_parquet(data_path)
        
        # Rename columns by removing prefixes
        def rename_column(col):
            if col.startswith("first_"): 
                return col[len("first_"):]
            elif col.startswith("last_"): 
                return col[len("last_"):]
            return col.replace("proportion", "%")
        
        df = df.rename(columns=rename_column)
        
        # Map state numbers to names
        state_mapping = {0: 'penetrating', 1: 'oscillating', 2: 'bouncing'}
        if df["state"].dtype == np.int64 or df["state"].dtype == np.int32:
            df["state"] = df["state"].map(state_mapping)
            
        return df
    
    @staticmethod
    def perform_clustering(X_scaled: np.ndarray, n_clusters: int = 100) -> np.ndarray:
        """
        Perform spectral clustering on scaled features.
        
        Args:
            X_scaled: Scaled feature matrix
            n_clusters: Number of clusters to create
            
        Returns:
            Array of cluster assignments
        """
        logger.info(f"Performing Spectral clustering with n_clusters={n_clusters}, random_state=42")
        
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=100,
            random_state=42,
            assign_labels='kmeans'
        )
        
        return clustering.fit_predict(X_scaled)
    
    @staticmethod
    def calculate_proportions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the proportion of each state for each source file.
        
        Args:
            df: DataFrame with state data
            
        Returns:
            DataFrame with added proportion columns
        """
        # Group by source file and calculate state proportions
        proportions = df.groupby('source_file')['state'].value_counts(normalize=True).unstack(fill_value=0)
        proportions = proportions.reset_index()
        
        # Rename columns to match the expected format
        proportion_cols = {state: f'proportion_{state}' for state in ['penetrating', 'oscillating', 'bouncing']}
        proportions = proportions.rename(columns=proportion_cols)
        
        # Merge proportions back to the main dataframe
        df = df.merge(proportions, on='source_file')
        
        return df
    
    @staticmethod
    def run_clustering(input_path: str, output_path: str, features: List[str], n_clusters: int = 100) -> pd.DataFrame:
        """
        Run the complete clustering process.
        
        Args:
            input_path: Path to input Parquet file
            output_path: Path to output CSV file
            features: List of feature names to use for clustering
            n_clusters: Number of clusters to create
            
        Returns:
            DataFrame with clustering results
        """
        # Set seeds for reproducibility
        np.random.seed(42)
        
        # Load and prepare data
        logger.info(f"Loading data from {input_path}")
        df = ClusteringProcessor.load_data(input_path)
        
        # Verify features exist in the dataframe
        for feature in features:
            if feature not in df.columns:
                logger.warning(f"Column '{feature}' not found in the dataframe")
        
        logger.info(f"Using features: {features}")
        
        # Extract features
        X = df[features].values
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        df['cluster'] = ClusteringProcessor.perform_clustering(X_scaled, n_clusters)
        
        # Calculate state proportions per source file and add to dataframe
        df = ClusteringProcessor.calculate_proportions(df)
        
        # Save the result to CSV
        logger.info(f"Saving results to {output_path}")
        df.to_csv(output_path, index=False)
        
        # Print cluster distribution
        logger.info("Cluster distribution:")
        logger.info(str(df['cluster'].value_counts().sort_index().to_dict()))
        
        logger.info(f"Completed! Data saved to {output_path}")
        
        return df


def main():
    """Main execution function to run the complete ETL pipeline."""
    # Define paths
    base_path = '/workspaces/NRCan-Statcan-Furnace-CFD-Analysis'
    data_path = os.path.join(base_path, 'data')
    
    # Ensure data directory exists
    os.makedirs(data_path, exist_ok=True)
    
    # Define file paths
    raw_data_path = os.path.join(data_path, 'raw')
    processed_path = os.path.join(data_path, 'processed')
    
    # Ensure directories exist
    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)
    
    # Step 1: Process all raw data files (if needed)
    logger.info("Step 1: Processing raw data files")
    aggregated_csv = os.path.join(processed_path, "aggregated_data.csv")
    
    if not os.path.exists(aggregated_csv) and os.listdir(raw_data_path):
        aggregated_csv = DataAggregator.process_all_files(raw_data_path)
    
    # Step 2: Convert to Parquet format if needed
    logger.info("Step 2: Converting to Parquet format")
    aggregated_parquet = aggregated_csv.replace('.csv', '.parquet')
    
    if not os.path.exists(aggregated_parquet) and os.path.exists(aggregated_csv):
        FormatConverter.csv_to_parquet_arrow(aggregated_csv, aggregated_parquet)
    
    # Step 3: Calculate particle states
    logger.info("Step 3: Calculating particle states")
    final_states_path = os.path.join(processed_path, "final_states.parquet")
    
    if not os.path.exists(final_states_path) and os.path.exists(aggregated_parquet):
        final_states = StateCalculator.process_tracker_states(
            aggregated_parquet,
            final_states_path,
            rho_L=RHO_L,
            gamma=GAMMA
        )
    
    # Step 4: Process and rename data
    logger.info("Step 4: Processing and renaming data")
    prefixed_path = os.path.join(processed_path, "final_states_prefixed.parquet")
    
    if not os.path.exists(prefixed_path) and os.path.exists(final_states_path):
        DataTransformer.process_and_rename(final_states_path, prefixed_path)
    
    # Step 5: Clean data
    logger.info("Step 5: Cleaning data")
    clean_path = os.path.join(processed_path, "final_states_clean.parquet")
    
    if not os.path.exists(clean_path) and os.path.exists(prefixed_path):
        DataTransformer.clean_data(prefixed_path, clean_path)
    
    # Step 6: Perform clustering
    logger.info("Step 6: Performing clustering")
    clustered_path = os.path.join(processed_path, "final_states_with_clusters.csv")
    
    if not os.path.exists(clustered_path) and os.path.exists(clean_path):
        features = ['diameter', 'density', 'pressure', 'primary_flow', 'particle_feed']
        ClusteringProcessor.run_clustering(clean_path, clustered_path, features, n_clusters=100)
    
    logger.info("ETL pipeline completed successfully!")


if __name__ == "__main__":
    main()