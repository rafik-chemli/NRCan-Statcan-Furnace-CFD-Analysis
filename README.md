# NRCan-Statcan Furnace CFD Analysis

This repository contains the code and analysis for a collaborative project between Natural Resources Canada (NRCan) and Statistics Canada, focusing on Computational Fluid Dynamics (CFD) analysis of bio-particles in industrial steel furnaces.

## Project Overview

The primary goal of this project is to leverage NRCan's comprehensive furnace datasets for advanced analytics to identify the optimal conditions for introducing bio-particles in arc steel furnaces, ultimately reducing greenhouse gas (GHG) emissions in the steel industry.

## Repository Structure

- **Phase 1**: Initial data exploration and modeling
  - `NRCan_Statcan_CFD_Exploration.ipynb`: Exploratory data analysis of furnace simulation data
  - `NRCan_Statcan_CFD_Modeling.ipynb`: Machine learning models to predict particle behavior
  - `aggregated_processed_data.csv`: Processed dataset used for modeling
  - `best_model.pkl`: Serialized best-performing machine learning model

- **Phase 2**: Advanced clustering and pattern analysis
  - `NRCan_Statcan_CFD_Clustering.ipynb`: Detailed clustering analysis of particle behaviors
  - `Data Pipeline/etl.py`: Complete ETL pipeline for processing raw CFD data through to cluster analysis
  - `Data Pipeline/README.md`: Detailed documentation of the ETL process and output files

- **Applications**:
  - `cfd_clustering_app.py`: Streamlit application for interactive cluster analysis of CFD data
  - `cluster_analysis_app.py`: Streamlit application for exploring silhouette scores and cluster feature analysis
  - `Clustering_figures.py`: Command-line tool for generating publication-quality figures
  - `final_states_clean.parquet`: Clean dataset of particle states for analysis

## Applications

### ETL Pipeline
- Complete data processing pipeline for CFD simulation data
- Extracts parameters from filenames, calculates physics-based states, and performs clustering
- Creates multiple output files for different analysis stages
- Run with: `python Phase\ 2/Data\ Pipeline/etl.py`
- Outputs can be used with visualization tools and Streamlit applications

### CFD Clustering Application
- Interactive analysis of particle behavior in CFD simulations
- Features statistical analysis, clustering, parallel plots, and UMAP visualization
- Run with: `streamlit run cfd_clustering_app.py`
- A live demo of the CFD exploration tool is available at: https://cfdexploration-nrcan-statcan.streamlit.app/

### Cluster Analysis Application
- Focused analysis of cluster features and state proportions
- Visualization of feature importance and silhouette scores
- Run with: `streamlit run cluster_analysis_app.py`
- Live demo for filtering cluster results: https://cfdexploration-filtering.streamlit.app/

### Figure Generation Tool
- Command-line tool for generating publication-quality figures
- Follows journal guidelines for figure format and appearance
- Run with: `python Clustering_figures.py --output figures`


## Requirements

Requirements are specified in `requirements.txt`. Install with:
```
pip install -r requirements.txt
```

## License

This project is licensed under the terms specified in the LICENSE file.