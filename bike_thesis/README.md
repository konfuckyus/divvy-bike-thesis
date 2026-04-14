# Divvy Bike Thesis Project

This repository contains the codebase for the final thesis project, focusing on data visualization and machine learning using the Chicago Divvy Bike dataset (2025). 

## Project Structure

- `data/`: Contains raw and processed data.
  - `raw/`: Raw trip CSV files from Divvy.
  - `processed/`: Cleaned datasets and generated features.
- `src/`: Python source code for data processing, analysis, and modeling.
  - `load_data.py`: Handles loading the raw trip data.
  - `preprocess.py`: Cleans the raw data (handles datetime conversions, creates ride durations, processes station-level aggregates).
  - `eda.py`: Runs Exploratory Data Analysis (EDA) and generates visualization plots.
  - `demand_dataset.py`: **[NEW]** Aggregates cleaned trips into a station-hour level dataset and adds forecasting-oriented features (lag features, rolling means, and calendar values). 
  - `baseline_model.py`: **[NEW]** Implements a historical average baseline forecasting method for station-hour bike demand. Predicts each test row using historical averages grouped chronologically.
  - `main.py`: Orchestrates the full pipeline (loading, preprocessing, building the demand dataset, exploratory data analysis, and running statistical summaries).
- `outputs/`: Saved plots and final analysis reports.

## Setup and Usage

1. Install project dependencies via:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main pipeline (loads, cleans data, creates the demand dataset, and generates EDA plots):
   ```bash
   python -m src.main
   ```
3. Run the baseline forecast model separately to evaluate predictions:
   ```bash
   python -m src.baseline_model
   ```

## Workflow 

This project operates in a sequential pipeline architecture:
1. **Data Ingestion**: Raw data is ingested.
2. **Preprocessing**: Invalid lines and outliers are dropped, basic features extracted.
3. **Structuring**: A demand dataset is generated with rich lagged features for ML.
4. **Baseline Modeling**: A historical-average model gives baseline MAE/RMSE/R2 metrics for future comparisons.
5. **EDA**: Final plots give spatial and chronological insight into Divvy riderships.
