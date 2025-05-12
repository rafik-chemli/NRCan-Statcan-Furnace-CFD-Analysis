# CFD Data Processing Pipeline

This directory contains the ETL (Extract, Transform, Load) pipeline for processing Computational Fluid Dynamics (CFD) simulation data of particle behavior in furnaces.

## Overview

The `etl.py` script provides a complete data processing pipeline for:
1. Extracting parameters from simulation filenames
2. Assigning unique tracker IDs to particles
3. Aggregating multiple simulation files
4. Converting between data formats (CSV to Parquet)
5. Calculating physical states of particles based on physical models
6. Clustering particles based on their features

## Pipeline Outputs

The pipeline creates the following output files in the `/data/processed/` directory:

| File | Description | Use Case |
|------|-------------|----------|
| `aggregated_data.csv` | Initial aggregated data from all input files | Raw data exploration |
| `aggregated_data.parquet` | Parquet version of the aggregated CSV | Faster data access |
| `final_states.parquet` | Calculated particle states with first/last tracking points | Physics-based analysis |
| `final_states_prefixed.parquet` | Data with columns renamed and prefixed (first_/last_) | Intermediate analysis |
| `final_states_clean.parquet` | Cleaned dataset with selected variables | Statistical analysis |
| `final_states_with_clusters.csv` | Final data with cluster assignments | Visualization & clustering analysis |

## Using with Visualization Tools

For visualization and analysis with tools like `plot_clusters.py` or `cluster_analysis_app.py`, you can use:

1. `final_states_clean.parquet` - For pre-clustering analysis and visualization
2. `final_states_with_clusters.csv` - For visualizing clusters and their characteristics

## Running the Pipeline

```bash
# Make sure you're in the correct directory
cd /path/to/NRCan-Statcan-Furnace-CFD-Analysis/Phase\ 2/Data\ Pipeline/

# Run the ETL pipeline
python etl.py
```

## Pipeline Steps

1. **Data Aggregation**: Combines data from multiple simulation files
2. **Format Conversion**: Converts data to efficient Parquet format
3. **State Calculation**: Computes physical states based on Weber and Bond numbers
4. **Data Transformation**: Renames and restructures data for analysis
5. **Data Cleaning**: Selects relevant features and removes NaN values
6. **Clustering**: Groups particles with similar characteristics

## Customization

The main parameters you might want to modify:

- `features`: List of features used for clustering in the `main()` function
- `n_clusters`: Number of clusters to create (default: 100)
- `RHO_L` and `GAMMA`: Physical constants for state calculation
- `raw_data_path`: Location of your raw input files

## Dependencies

- pandas
- numpy
- pyarrow
- scikit-learn
- tqdm
- seaborn (for visualization)
- matplotlib (for visualization)

## Troubleshooting

If you encounter memory issues with large datasets:
1. Reduce the `chunk_size` parameter in processing functions
2. Ensure enough disk space for intermediate files
3. Use the PyArrow implementation for better memory efficiency