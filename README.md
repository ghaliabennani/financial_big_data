# Project: Clustering Time Series of Market Data and Strategy Application

## Overview

This project involves clustering time series market data into distinct market states and applying trading strategies tailored to these states. By leveraging clustering techniques and actionable insights from market metrics, the strategy aims to enhance trading performance.

## Features

- **Data Preprocessing**: Cleaning, merging, and aggregating raw market data for analysis.
- **Clustering**: Identifying distinct market states using advanced techniques like Louvain clustering.
- **Strategy Development**: Creating dynamic trading strategies informed by market state analysis.
- **Visualization**: Plotting cumulative returns, market metrics, and clustering outputs.

## Installation

1. Install the required Python dependencies:
   
   `pip install -r requirements.txt`
   
2. Put the data directory in the base directory of the project.
   Get the dataset by downlaoding the directory named data via this link (https://drive.google.com/drive/folders/1nJHIpQcFF-ClLEfM0yRrnJjxuOKFoO1S?usp=sharing)

## Usage

### Running the Analysis

1. Choose between full analysis and demo workflow using the `--full` argument:

   `python main.py --full`

   Without `--full`, the script runs a demo on a smaller dataset.

2. The workflow includes:

   - Data preprocessing.
   - Clustering market data into states.
   - Applying a trading strategy based on clustered states.
   - Generating results and visualizations.

### Outputs

- Preprocessed data files.
- Clustering results and market state insights.
- Strategy performance metrics and visualizations.

## File Structure

- `src/preprocessing.py`: Data cleaning, merging, and feature creation.
- `src/clustering.py`: Implements feature aggregation, sliding window clustering, and Louvain clustering.
- `src/strategy.py`: Defines trading rules, applies the strategy, and generates visualizations.
- `requirements.txt`: Lists all required Python libraries.
- `analysis.py`: Orchestrates the entire workflow.

## Dependencies

The project uses the following Python libraries:

- `polars`
- `pandas`
- `numpy`
- `matplotlib`
- `networkx`
- `community-louvain`
- `scikit-learn`
- `seaborn`

Install all dependencies using the `requirements.txt` file.

## Contact

For any inquiries or support, please contact:

- Ghalia Bennani: [[ghalia.bennani@epfl.ch](mailto\:author1@example.com)]
- Othmane Idrissi Oudghiri: [[mohamed.idrissioudghiri@epfl.ch](mailto\:mohamed.idrissioudghiri@epfl.ch)]
- Mhamed Karim Sefrioui: [[mhamed.sefrioui@epfl.ch](mailto\:mhamed.sefrioui@epfl.ch)]