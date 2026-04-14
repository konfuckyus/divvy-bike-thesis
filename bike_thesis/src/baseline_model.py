"""
baseline_model.py

This module implements a simple and thesis-friendly baseline forecasting method
for station-hour bike demand.

Baseline idea
-------------
Predict each test row using historical average demand from training data.
Primary grouping:
    - start_station_id
    - weekday
    - hour

Fallback strategy (if a group is unseen in training):
    1) start_station_id + hour
    2) start_station_id
    3) global mean demand

Why chronological split?
------------------------
For time-based forecasting, we must train on past data and test on future data.
A random split can leak future information into training and overestimate model
quality. Therefore, we split by calendar time.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from demand_dataset import get_station_hour_demand_path


def get_project_root() -> Path:
    """
    Return the project root folder.
    """
    return Path(__file__).resolve().parents[1]


def get_processed_data_folder() -> Path:
    """
    Return the processed data folder path: bike_thesis/data/processed.
    """
    return get_project_root() / "data" / "processed"


def get_baseline_predictions_path(year: int = 2025) -> Path:
    """
    Return output path for actual-vs-predicted baseline CSV.
    """
    return get_processed_data_folder() / f"baseline_predictions_{year}.csv"


def get_baseline_metrics_path(year: int = 2025) -> Path:
    """
    Return output path for baseline metrics text file.
    """
    return get_processed_data_folder() / f"baseline_metrics_{year}.txt"


def load_demand_data(year: int = 2025) -> pd.DataFrame:
    """
    Load station-hour demand dataset from data/processed/.
    """
    demand_path = get_station_hour_demand_path(year=year)
    if not demand_path.exists():
        raise FileNotFoundError(
            f"Demand dataset not found at {demand_path}. "
            f"Run demand dataset creation first."
        )

    print(f"Loading demand dataset: {demand_path}")
    df = pd.read_csv(demand_path, low_memory=False)

    # Ensure timestamp and station id have consistent types.
    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce")
    df["start_station_id"] = df["start_station_id"].astype(str)

    # Remove malformed rows only if needed for safe chronological split.
    df = df.dropna(subset=["timestamp_hour"]).copy()

    return df


def split_train_test_chronologically(
    df: pd.DataFrame,
    train_end_month: int = 10,
    year: int = 2025,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split demand data into train/test sets using calendar time.

    Default behavior for 2025:
    - Train: January to October (first 10 months)
    - Test: November to December (last 2 months)

    The logic is intentionally parameterized so you can change split boundaries
    later without rewriting the full module.
    """
    print("Applying chronological train-test split...")

    df_sorted = df.sort_values(["start_station_id", "timestamp_hour"]).copy()

    train_mask = (
        (df_sorted["timestamp_hour"].dt.year == year)
        & (df_sorted["timestamp_hour"].dt.month <= train_end_month)
    )
    test_mask = (
        (df_sorted["timestamp_hour"].dt.year == year)
        & (df_sorted["timestamp_hour"].dt.month > train_end_month)
    )

    train_df = df_sorted.loc[train_mask].copy()
    test_df = df_sorted.loc[test_mask].copy()

    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Chronological split produced an empty train or test set. "
            "Check the year/train_end_month or available data coverage."
        )

    return train_df, test_df


def fit_historical_average_baseline(train_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Fit historical-average baseline components from training data.

    Returns a dictionary containing group-level means and global mean.
    """
    print("Fitting historical-average baseline...")

    required_cols = ["start_station_id", "weekday", "hour", "demand"]
    missing_cols = [c for c in required_cols if c not in train_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required training columns: {missing_cols}")

    # Primary table: station + weekday + hour
    mean_station_weekday_hour = train_df.groupby(
        ["start_station_id", "weekday", "hour"]
    )["demand"].mean()

    # Fallback 1: station + hour
    mean_station_hour = train_df.groupby(["start_station_id", "hour"])["demand"].mean()

    # Fallback 2: station only
    mean_station = train_df.groupby("start_station_id")["demand"].mean()

    # Fallback 3: global average demand
    global_mean = float(train_df["demand"].mean())

    baseline_model = {
        "mean_station_weekday_hour": mean_station_weekday_hour,
        "mean_station_hour": mean_station_hour,
        "mean_station": mean_station,
        "global_mean": global_mean,
    }

    return baseline_model


def predict_with_baseline(
    test_df: pd.DataFrame,
    baseline_model: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Predict test demand using hierarchical historical averages with fallbacks.
    """
    print("Generating baseline predictions on test data...")

    preds = test_df.copy()

    # Merge primary level mean: station + weekday + hour
    preds = preds.merge(
        baseline_model["mean_station_weekday_hour"].rename("pred_level_1"),
        on=["start_station_id", "weekday", "hour"],
        how="left",
    )

    # Merge fallback 1 mean: station + hour
    preds = preds.merge(
        baseline_model["mean_station_hour"].rename("pred_level_2"),
        on=["start_station_id", "hour"],
        how="left",
    )

    # Merge fallback 2 mean: station only
    preds = preds.merge(
        baseline_model["mean_station"].rename("pred_level_3"),
        on=["start_station_id"],
        how="left",
    )

    # Apply fallback chain in order.
    preds["predicted_demand"] = preds["pred_level_1"]
    preds["predicted_demand"] = preds["predicted_demand"].fillna(preds["pred_level_2"])
    preds["predicted_demand"] = preds["predicted_demand"].fillna(preds["pred_level_3"])
    preds["predicted_demand"] = preds["predicted_demand"].fillna(
        baseline_model["global_mean"]
    )

    # Keep a clean output by removing intermediate helper columns.
    preds = preds.drop(columns=["pred_level_1", "pred_level_2", "pred_level_3"])

    return preds


def evaluate_regression_model(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate regression predictions with MAE, RMSE, and R^2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def run_baseline_forecast_pipeline(
    year: int = 2025,
    train_end_month: int = 10,
) -> Dict[str, float]:
    """
    Run the full baseline forecasting pipeline:
    1) Load demand dataset
    2) Chronological train-test split
    3) Fit historical-average baseline
    4) Predict test demand
    5) Evaluate and save outputs
    """
    demand_df = load_demand_data(year=year)
    train_df, test_df = split_train_test_chronologically(
        demand_df, train_end_month=train_end_month, year=year
    )

    baseline_model = fit_historical_average_baseline(train_df)
    preds_df = predict_with_baseline(test_df, baseline_model)

    metrics = evaluate_regression_model(
        y_true=preds_df["demand"],
        y_pred=preds_df["predicted_demand"],
    )

    # Save output files in processed folder.
    processed_folder = get_processed_data_folder()
    processed_folder.mkdir(parents=True, exist_ok=True)

    predictions_path = get_baseline_predictions_path(year=year)
    metrics_path = get_baseline_metrics_path(year=year)

    preds_df.to_csv(predictions_path, index=False)

    metrics_text = (
        f"Baseline Forecast Metrics ({year})\n"
        f"Train months: 1-{train_end_month}\n"
        f"Test months: {train_end_month + 1}-12\n"
        f"MAE: {metrics['mae']:.4f}\n"
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"R2: {metrics['r2']:.4f}\n"
    )
    metrics_path.write_text(metrics_text, encoding="utf-8")

    print("\n=== Baseline Forecast Evaluation ===")
    print(f"MAE : {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2  : {metrics['r2']:.4f}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")

    return metrics


if __name__ == "__main__":
    # Example direct run:
    # python -m src.baseline_model
    run_baseline_forecast_pipeline(year=2025, train_end_month=10)
