# Predictive Modeling and Analysis of Urban Micro-Mobility Demand

This repository contains the computational framework and analytical methodologies developed to investigate and forecast urban micro-mobility patterns, specifically focusing on the Chicago Divvy Bike sharing system. The research incorporates extensive data preprocessing, exploratory spatial-temporal analysis, and the implementation of predictive machine learning models to estimate station-level bicycle demand.

## Repository Architecture

The repository is structured to ensure reproducibility and modularity across the data science lifecycle:

- `data/`: Serves as the primary data directory.
  - `raw/`: Unprocessed operational trip data acquired from the Divvy platform.
  - `processed/`: Cleansed and aggregated datasets containing engineered features for predictive modeling.
- `src/`: Core Python modules comprising the data pipeline and machine learning implementation.
  - `load_data.py`: Facilitates the programmatic ingestion of primary datasets.
  - `preprocess.py`: Executes data cleansing protocols, datetime transformations, and initial feature extraction.
  - `eda.py`: Conducts exploratory data analysis (EDA) to evaluate spatial and temporal ridership distributions.
  - `demand_dataset.py`: Aggregates transactional data into station-hour intervals and engineers predictive covariates, including temporal lags, moving averages, and calendar effects.
  - `baseline_model.py`: Establishes a baseline forecasting methodology utilizing historical chronological averages to benchmark predictive performance.
  - `train_models.py`: Handles the training, optimization, and evaluation of advanced machine learning architectures (e.g., Random Forest, Linear Regression).
  - `model_visualization.py`: Generates comparative analytical plots for model evaluation and error analysis.
  - `export_dashboard_data.py`: Extracts and formats processed datasets and model predictions for external visualization.
  - `main.py`: The central orchestrator that integrates data ingestion, processing, feature engineering, and statistical evaluations.
- `dashboard/`: A Next.js-based analytical dashboard application constructed to present interactive visual summaries of the findings, exploratory data analysis, and model forecasts.
- `outputs/`: Destination for compiled empirical results, including model performance metrics and generated analytical figures.

## Environment Configuration

To replicate the experimental environment and execute the computational pipeline, please follow the protocol outlined below:

1. **Dependency Installation**: Ensure a Python 3.x environment is active, then install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Pipeline Execution**: Execute the primary pipeline to process the raw data, construct the analytical dataset, and generate foundational exploratory analyses:
   ```bash
   python -m src.main
   ```

3. **Model Training and Evaluation**: Execute the baseline and machine learning models to assess forecasting performance metrics (MAE, RMSE, R²):
   ```bash
   python -m src.baseline_model
   python -m src.train_models
   ```

## Methodological Workflow

The analytical process is structured as a sequential computational pipeline:

1. **Data Ingestion**: Systematized retrieval of raw spatial-temporal transactional data.
2. **Preprocessing and Cleansing**: Mitigation of missing values, exclusion of statistical outliers, and standardization of feature formats.
3. **Feature Engineering**: Construction of a robust dataset incorporating lagged variables, rolling statistics, and calendar-based covariates to capture temporal dependencies.
4. **Predictive Modeling**: Implementation of baseline and advanced predictive models to estimate hourly demand at individual docking stations.
5. **Evaluation and Visualization**: Comprehensive assessment of model efficacy through rigorous statistical metrics and subsequent visualization of residual distributions and predictive accuracy.
