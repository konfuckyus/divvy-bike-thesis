# Predictive Modeling and Analysis of Urban Micro-Mobility Demand

**Author:** Ertuğrul Tetik  
**Project Type:** Undergraduate Graduation Thesis  
**Dataset:** Chicago Divvy Bike Share Data (2025)

## Project Overview

This repository contains the computational framework and analytical methodologies developed to investigate and forecast urban micro-mobility patterns, specifically focusing on the Chicago Divvy Bike sharing system. The research incorporates extensive data preprocessing, exploratory spatial-temporal analysis, and the implementation of predictive machine learning models to estimate station-level bicycle demand.

The primary objectives of this research include:
1. **Behavioral Analysis:** Quantifying operational differences between annual members and casual riders.
2. **Temporal Dynamics:** Identifying daily, weekly, and monthly seasonality alongside usage distributions.
3. **Spatial Distribution:** Evaluating high-demand nodes and network bottlenecks across the urban infrastructure.
4. **Predictive Modeling:** Constructing and evaluating machine learning architectures to forecast station-level demand and optimize resource allocation.

## Repository Architecture

```text
tez_kod/
│
├── 2025/                       # Unprocessed Divvy trip datasets
├── bike_thesis/                # Primary computational pipeline and analytical modules
│   ├── data/                   
│   │   ├── raw/                # Primary datasets for ingestion
│   │   └── processed/          # Aggregated datasets with engineered features
│   ├── outputs/
│   │   ├── figures/            # Auto-generated analytical plots and distributions
│   │   └── tables/             # Summary output tables
│   ├── src/                    # Core Python Modules
│   │   ├── load_data.py        # Data ingestion protocols
│   │   ├── preprocess.py       # Cleansing and feature engineering
│   │   ├── demand_dataset.py   # Station-hour aggregation and lag feature construction
│   │   ├── baseline_model.py   # Historical average baseline forecasting
│   │   ├── train_models.py     # Machine learning model training and evaluation
│   │   ├── eda.py              # Exploratory Data Analysis execution
│   │   ├── model_visualization.py # Performance metric visualizations
│   │   └── main.py             # Pipeline orchestrator
│   ├── dashboard/              # Next.js analytical dashboard application
│   └── requirements.txt        # Python dependency specifications
├── .gitignore                  # Repository exclusion rules
└── README.md                   # Project documentation (Current File)
```

## Environment Configuration and Execution

### 1. Prerequisites
Execution of the pipeline requires a Python 3.8+ environment. The use of a virtual environment is recommended to ensure dependency isolation.

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux environments

pip install -r bike_thesis/requirements.txt
```

### 2. Data Preparation
1. Acquire the necessary operational trip data CSV files for the target timeframe from the Divvy Data portal.
2. Place the uncompressed CSV files within the root `2025/` directory.

### 3. Pipeline Execution
Navigate to the `bike_thesis` directory to execute the primary orchestrator module. This script facilitates data loading, preprocessing, structuring, and the generation of exploratory visualizations.

```bash
cd bike_thesis
python -m src.main
```

### 4. Model Training and Evaluation
To construct the predictive models and evaluate their performance metrics (Mean Absolute Error, Root Mean Squared Error, R-squared), execute the designated modeling scripts:

```bash
cd bike_thesis
python -m src.baseline_model
python -m src.train_models
```

## Analytical Outputs
The computational pipeline autonomously generates a comprehensive suite of analytical figures, stored within `bike_thesis/outputs/figures/`, including:
- Demand distribution heatmaps
- Moving average trend analyses
- Residual distribution histograms for predictive models
- Comparative model performance charts

### 5. Dashboard Deployment (Vercel)
The interactive analytical dashboard is built with Next.js and optimized for deployment on Vercel.

1. Navigate to the dashboard directory:
   ```bash
   cd bike_thesis/dashboard
   ```
2. Deploy using the Vercel CLI or connect the repository directly via the Vercel platform:
   ```bash
   npx vercel
   ```
