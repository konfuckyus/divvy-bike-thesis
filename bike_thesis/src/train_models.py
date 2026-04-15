"""
train_models.py

This module trains and evaluates simple machine learning regressors for
station-hour bike demand forecasting.

Design goals
------------
- Keep the workflow clear and thesis-friendly.
- Use chronological train-test split (never random split for time data).
- Compare a few standard models without hyperparameter tuning.
- Save model comparison and per-model predictions for reporting.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from demand_dataset import get_station_hour_demand_path


def get_project_root() -> Path:
    """
    Return the project root folder.
    """
    return Path(__file__).resolve().parents[1]


def get_processed_data_folder() -> Path:
    """
    Return processed data folder path: bike_thesis/data/processed.
    """
    return get_project_root() / "data" / "processed"


def get_model_comparison_path(year: int = 2025) -> Path:
    """
    Output path for model comparison table.
    """
    return get_processed_data_folder() / f"model_comparison_{year}.csv"


def get_model_predictions_path(model_name: str, year: int = 2025) -> Path:
    """
    Output path for per-model prediction file.
    """
    safe_model_name = model_name.lower().replace(" ", "_")
    return get_processed_data_folder() / f"predictions_{safe_model_name}_{year}.csv"


def load_modeling_dataset(year: int = 2025) -> pd.DataFrame:
    """
    Load the station-hour demand dataset used for forecasting models.

    The expected input file is:
        data/processed/station_hour_demand_2025.csv
    """
    demand_path = get_station_hour_demand_path(year=year)
    if not demand_path.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found at {demand_path}. "
            f"Create the demand dataset first."
        )

    print(f"Loading modeling dataset: {demand_path}")
    df = pd.read_csv(demand_path, low_memory=False)

    # Safe type handling for key columns.
    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce")
    df["start_station_id"] = df["start_station_id"].astype(str)

    # Remove rows with invalid timestamps to ensure valid chronological split.
    df = df.dropna(subset=["timestamp_hour"]).copy()

    return df


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], pd.DataFrame]:
    """
    Prepare feature matrix and target vector for ML models.

    Target:
    - demand

    Features:
    - Temporal features
    - Lag / rolling demand features
    - Station identifier (categorical)

    We keep only rows that have all required lag-based information.
    """
    print("Preparing features and target for modeling...")

    target_col = "demand"

    # Required lag/rolling features for this project stage.
    lag_feature_cols = [
        "prev_hour_demand",
        "prev_2hour_demand",
        "prev_3hour_demand",
        "prev_24hour_demand",
        "rolling_mean_3h",
        "rolling_mean_6h",
        "rolling_mean_24h",
    ]

    temporal_feature_cols = [
        "hour",
        "day_of_month",
        "month_num",
        "weekday_num",
        "is_weekend",
    ]

    categorical_feature_cols = [
        "start_station_id",
    ]

    all_feature_cols = temporal_feature_cols + lag_feature_cols + categorical_feature_cols

    required_columns = [target_col, "timestamp_hour"] + all_feature_cols
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required modeling columns: {missing_columns}")

    # Keep only rows where lag/rolling features exist.
    # These NaNs are expected at the beginning of each station time series.
    modeling_df = df.dropna(subset=lag_feature_cols).copy()

    X = modeling_df[all_feature_cols].copy()
    y = modeling_df[target_col].copy()

    print(f"Rows available for modeling after lag filtering: {len(modeling_df):,}")

    return X, y, categorical_feature_cols, temporal_feature_cols + lag_feature_cols, modeling_df


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    modeling_df: pd.DataFrame,
    year: int = 2025,
    train_end_month: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Split features/target into train and test sets chronologically.

    Why chronological split?
    ------------------------
    In forecasting, future values must not influence training. Random split
    would mix time periods and can produce unrealistic, overly optimistic scores.

    Default for 2025:
    - train months: 1..10
    - test months: 11..12
    """
    print("Applying chronological train-test split for ML models...")

    train_mask = (
        (modeling_df["timestamp_hour"].dt.year == year)
        & (modeling_df["timestamp_hour"].dt.month <= train_end_month)
    )
    test_mask = (
        (modeling_df["timestamp_hour"].dt.year == year)
        & (modeling_df["timestamp_hour"].dt.month > train_end_month)
    )

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()
    meta_train = modeling_df.loc[train_mask, ["start_station_id", "timestamp_hour"]].copy()
    meta_test = modeling_df.loc[test_mask, ["start_station_id", "timestamp_hour"]].copy()

    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows:  {len(X_test):,}")

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Split produced empty train or test set. "
            "Check year/train_end_month and dataset date coverage."
        )

    return X_train, X_test, y_train, y_test, meta_train, meta_test


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical_cols: List[str],
    numeric_cols: List[str],
    meta_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Train and evaluate multiple regression models with a shared preprocessing step.
    """
    print("Training and evaluating models...")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
            ("num", "passthrough", numeric_cols),
        ]
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=42,
        ),
    }

    results = []
    predictions_by_model: Dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name} ...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        results.append(
            {
                "model": model_name,
                "mae": float(mae),
                "rmse": rmse,
                "r2": float(r2),
            }
        )

        model_pred_df = meta_test.copy()
        model_pred_df["actual_demand"] = y_test.values
        model_pred_df["predicted_demand"] = y_pred
        model_pred_df["model"] = model_name
        predictions_by_model[model_name] = model_pred_df

        print(f"{model_name} | MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

    comparison_df = pd.DataFrame(results).sort_values("rmse", ascending=True).reset_index(
        drop=True
    )

    print("\n=== Model Ranking (best to worst by RMSE) ===")
    for i, row in comparison_df.iterrows():
        print(
            f"{i + 1}. {row['model']} | RMSE={row['rmse']:.4f} "
            f"| MAE={row['mae']:.4f} | R2={row['r2']:.4f}"
        )

    return comparison_df, predictions_by_model


def save_model_results(
    comparison_df: pd.DataFrame,
    predictions_by_model: Dict[str, pd.DataFrame],
    year: int = 2025,
) -> None:
    """
    Save model comparison and per-model predictions to CSV files.
    """
    processed_folder = get_processed_data_folder()
    processed_folder.mkdir(parents=True, exist_ok=True)

    comparison_path = get_model_comparison_path(year=year)
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Model comparison saved to: {comparison_path}")

    for model_name, pred_df in predictions_by_model.items():
        pred_path = get_model_predictions_path(model_name=model_name, year=year)
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved for {model_name}: {pred_path}")


def run_training_pipeline(
    year: int = 2025,
    train_end_month: int = 10,
) -> pd.DataFrame:
    """
    Run full model training pipeline and return comparison table.
    """
    df = load_modeling_dataset(year=year)

    X, y, categorical_cols, numeric_cols, modeling_df = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test, _meta_train, meta_test = split_train_test(
        X=X,
        y=y,
        modeling_df=modeling_df,
        year=year,
        train_end_month=train_end_month,
    )

    comparison_df, predictions_by_model = train_and_evaluate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        meta_test=meta_test,
    )

    save_model_results(
        comparison_df=comparison_df,
        predictions_by_model=predictions_by_model,
        year=year,
    )

    return comparison_df


if __name__ == "__main__":
    # Example direct run:
    # python -m src.train_models
    run_training_pipeline(year=2025, train_end_month=10)
