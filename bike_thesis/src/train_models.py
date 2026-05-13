"""
train_models.py

This module trains and evaluates thesis-level regression models for
station-hour demand forecasting.

Why this module matters in a thesis
-----------------------------------
It provides a clear first machine learning benchmark after baseline forecasting.
We keep the workflow simple and transparent:
- chronological train-test split (no random leakage),
- moderate feature set (time + lag + station),
- standard regression models for fair comparison.
"""

from pathlib import Path
from time import perf_counter
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


def get_predictions_output_folder() -> Path:
    """
    Return output folder for per-model prediction files.
    """
    return get_project_root() / "outputs" / "predictions"


def get_metrics_output_folder() -> Path:
    """
    Return output folder for metrics files.
    """
    return get_project_root() / "outputs" / "metrics"


def get_model_comparison_path() -> Path:
    """
    Canonical model comparison path for the full-dataset experiment.
    """
    return get_metrics_output_folder() / "model_comparison.csv"


def get_experiment_model_comparison_path(experiment_label: str) -> Path:
    """
    Model comparison path for named experiments.
    """
    return get_metrics_output_folder() / f"model_comparison_{experiment_label}.csv"


def get_full_vs_busy_comparison_path(top_n: int) -> Path:
    """
    Comparison table between full dataset and busy-station subset results.
    """
    return get_metrics_output_folder() / f"model_comparison_full_vs_busy_top{top_n}.csv"


def get_model_predictions_paths(experiment_label: str = "full") -> Dict[str, Path]:
    """
    Return canonical prediction output paths for each model.
    Appends experiment suffix explicitly (e.g. _full or _busy_top20)
    for clear separation.
    """
    pred_folder = get_predictions_output_folder()
    suffix = f"_{experiment_label}"
    return {
        "LinearRegression": pred_folder / f"linear_regression_predictions{suffix}.csv",
        "RandomForestRegressor": pred_folder / f"random_forest_predictions{suffix}.csv",
        "GradientBoostingRegressor": pred_folder / f"gradient_boosting_predictions{suffix}.csv",
    }


def get_model_definitions(
    fast_models_only: bool = True,
    enable_slow_models: bool = False,
) -> Dict[str, object]:
    """
    Define model set for this thesis stage.

    Notes
    -----
    - LinearRegression is the default fastest benchmark.
    - RandomForest is configured with lightweight settings for practical runs.
    - Slower models are skipped unless explicitly enabled.
    """
    models: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=30,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
    }

    # Gradient boosting can be slower on larger datasets, so keep it optional.
    if enable_slow_models and not fast_models_only:
        models["GradientBoostingRegressor"] = GradientBoostingRegressor(
            random_state=42,
        )

    return models


def load_modeling_dataset(year: int = 2025) -> pd.DataFrame:
    """
    Load the station-hour demand dataset used for forecasting models.
    """
    demand_path = get_station_hour_demand_path(year=year)
    if not demand_path.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found at {demand_path}. "
            f"Create the demand dataset first."
        )

    print(f"Loading modeling dataset: {demand_path}")
    df = pd.read_csv(demand_path, low_memory=False)

    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce")
    df["start_station_id"] = df["start_station_id"].astype(str)
    df = df.dropna(subset=["timestamp_hour"]).copy()

    return df


def define_feature_columns() -> Tuple[List[str], List[str], List[str], str]:
    """
    Define target, categorical features, temporal features, and lag features.
    """
    target_col = "demand"

    # Station identity and season can encode location-level and climate-level effects.
    categorical_cols = ["start_station_id", "season"]

    # Calendar/clock features.
    temporal_cols = [
        "hour",
        "day_of_month",
        "month_num",
        "weekday_num",
        "is_weekend",
        "is_peak_hour",
    ]

    # Historical demand memory and trend features.
    lag_cols = [
        "prev_hour_demand",
        "prev_2hour_demand",
        "prev_3hour_demand",
        "prev_24hour_demand",
        "prev_day_same_hour_demand",
        "prev_week_same_hour_demand",
        "rolling_mean_3h",
        "rolling_mean_6h",
        "rolling_mean_12h",
        "rolling_mean_24h",
        "rolling_std_24h",
    ]

    return categorical_cols, temporal_cols, lag_cols, target_col


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], pd.DataFrame]:
    """
    Prepare feature matrix and target vector for ML models.

    We keep only rows that have lag/rolling history available because
    those features are necessary for forecasting models.
    """
    print("Preparing features and target for modeling...")

    categorical_cols, temporal_cols, lag_cols, target_col = define_feature_columns()
    numeric_cols = temporal_cols + lag_cols
    all_feature_cols = numeric_cols + categorical_cols

    required_columns = [target_col, "timestamp_hour"] + all_feature_cols
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required modeling columns: {missing_columns}")

    # Drop only rows that miss lag/rolling info (expected near series starts).
    modeling_df = df.dropna(subset=lag_cols).copy()

    X = modeling_df[all_feature_cols].copy()
    y = modeling_df[target_col].copy()

    print(f"Rows available for modeling after lag filtering: {len(modeling_df):,}")

    return X, y, categorical_cols, numeric_cols, modeling_df


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


def select_busy_station_subset(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    top_n_busy_stations: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Keep only top-N busiest stations ranked by training demand.

    Ranking uses only training data to avoid future leakage.
    """
    if top_n_busy_stations <= 0:
        raise ValueError("top_n_busy_stations must be positive.")

    rank_df = meta_train.copy()
    rank_df["demand"] = y_train.values

    station_rank = (
        rank_df.groupby("start_station_id")["demand"]
        .sum()
        .sort_values(ascending=False)
    )
    selected_stations = station_rank.head(top_n_busy_stations).index.tolist()

    train_mask = X_train["start_station_id"].isin(selected_stations)
    test_mask = X_test["start_station_id"].isin(selected_stations)

    X_train_busy = X_train.loc[train_mask].copy()
    X_test_busy = X_test.loc[test_mask].copy()
    y_train_busy = y_train.loc[train_mask].copy()
    y_test_busy = y_test.loc[test_mask].copy()
    meta_train_busy = meta_train.loc[train_mask].copy()
    meta_test_busy = meta_test.loc[test_mask].copy()

    print(
        f"Busy-station subset (top {top_n_busy_stations}) -> "
        f"train rows: {len(X_train_busy):,}, test rows: {len(X_test_busy):,}"
    )

    return (
        X_train_busy,
        X_test_busy,
        y_train_busy,
        y_test_busy,
        meta_train_busy,
        meta_test_busy,
        selected_stations,
    )


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and R^2 for one model.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def train_single_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical_cols: List[str],
    numeric_cols: List[str],
    meta_test: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Train a single model with shared preprocessing, then evaluate it.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_model(y_true=y_test, y_pred=y_pred)

    pred_df = meta_test.copy()
    pred_df["actual_demand"] = y_test.values
    pred_df["predicted_demand"] = y_pred
    pred_df["model"] = model_name

    return metrics, pred_df


def save_model_outputs(
    comparison_df: pd.DataFrame,
    predictions_by_model: Dict[str, pd.DataFrame],
    comparison_path: Path,
    prediction_paths: Dict[str, Path],
) -> None:
    """
    Save model comparison and per-model predictions to CSV files.
    """
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    for p in prediction_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(comparison_path, index=False)
    print(f"Model comparison saved to: {comparison_path}")

    for model_name, pred_path in prediction_paths.items():
        if model_name in predictions_by_model:
            predictions_by_model[model_name].to_csv(pred_path, index=False)
            print(f"Predictions saved for {model_name}: {pred_path}")


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical_cols: List[str],
    numeric_cols: List[str],
    meta_test: pd.DataFrame,
    comparison_path: Path,
    prediction_paths: Dict[str, Path],
    fast_models_only: bool = True,
    enable_slow_models: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Train and evaluate multiple regression models with shared preprocessing.
    """
    print("Training and evaluating models...")

    models = get_model_definitions(
        fast_models_only=fast_models_only,
        enable_slow_models=enable_slow_models,
    )

    results = []
    predictions_by_model: Dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        print(f"\nTraining started: {model_name}")
        start_time = perf_counter()

        metrics, model_pred_df = train_single_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            meta_test=meta_test,
        )

        results.append(
            {
                "model": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )
        predictions_by_model[model_name] = model_pred_df

        elapsed_seconds = perf_counter() - start_time
        print(f"Training completed: {model_name} | elapsed={elapsed_seconds:.2f} sec")
        print(
            f"{model_name} | MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}"
        )

        # Save partial progress after each model, so long runs do not lose results.
        partial_comparison_df = (
            pd.DataFrame(results).sort_values("rmse", ascending=True).reset_index(drop=True)
        )
        save_model_outputs(
            comparison_df=partial_comparison_df,
            predictions_by_model=predictions_by_model,
            comparison_path=comparison_path,
            prediction_paths=prediction_paths,
        )

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


def run_single_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical_cols: List[str],
    numeric_cols: List[str],
    meta_test: pd.DataFrame,
    experiment_label: str,
    fast_models_only: bool,
    enable_slow_models: bool,
) -> pd.DataFrame:
    """
    Run one model experiment (full data or busy-station subset).
    """
    comparison_path = get_experiment_model_comparison_path(experiment_label)
    prediction_paths = get_model_predictions_paths(experiment_label)

    comparison_df, predictions_by_model = train_and_evaluate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        meta_test=meta_test,
        comparison_path=comparison_path,
        prediction_paths=prediction_paths,
        fast_models_only=fast_models_only,
        enable_slow_models=enable_slow_models,
    )

    save_model_outputs(
        comparison_df=comparison_df,
        predictions_by_model=predictions_by_model,
        comparison_path=comparison_path,
        prediction_paths=prediction_paths,
    )

    return comparison_df


def save_full_vs_busy_summary(
    full_df: pd.DataFrame,
    busy_df: pd.DataFrame,
    top_n_busy_stations: int,
) -> Path:
    """
    Save side-by-side metrics for full dataset vs busy-station subset.
    """
    full_cmp = full_df.copy()
    full_cmp["experiment"] = "full_dataset"

    busy_cmp = busy_df.copy()
    busy_cmp["experiment"] = f"busy_stations_top_{top_n_busy_stations}"

    merged = pd.concat([full_cmp, busy_cmp], ignore_index=True)
    output_path = get_full_vs_busy_comparison_path(top_n=top_n_busy_stations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Full vs busy comparison saved to: {output_path}")
    return output_path


def run_all_models(
    year: int = 2025,
    train_end_month: int = 10,
    fast_models_only: bool = True,
    enable_slow_models: bool = False,
    run_full_dataset_experiment: bool = False,
    run_busy_stations_experiment: bool = True,
    top_n_busy_stations: int = 20,
) -> pd.DataFrame:
    """
    Run full model training pipeline.

    By default, runs only the busy-station experiment as recommended.
    """
    df = load_modeling_dataset(year=year)
    X, y, categorical_cols, numeric_cols, modeling_df = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test, meta_train, meta_test = split_train_test(
        X=X,
        y=y,
        modeling_df=modeling_df,
        year=year,
        train_end_month=train_end_month,
    )

    full_comparison_df = pd.DataFrame()
    if run_full_dataset_experiment:
        print("\n=== Experiment 1: Full dataset ===")
        full_comparison_df = run_single_experiment(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            meta_test=meta_test,
            experiment_label="full",
            fast_models_only=fast_models_only,
            enable_slow_models=enable_slow_models,
        )

    if run_busy_stations_experiment:
        print(f"\n=== Experiment 2: Busy stations only (top {top_n_busy_stations}) ===")
        (
            X_train_busy,
            X_test_busy,
            y_train_busy,
            y_test_busy,
            _meta_train_busy,
            meta_test_busy,
            _selected_stations,
        ) = select_busy_station_subset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            meta_train=meta_train,
            meta_test=meta_test,
            top_n_busy_stations=top_n_busy_stations,
        )

        if len(X_train_busy) == 0 or len(X_test_busy) == 0:
            print("Busy-station subset is empty. Skipping busy-station experiment.")
            return full_comparison_df

        busy_label = f"busy_top{top_n_busy_stations}"
        busy_comparison_df = run_single_experiment(
            X_train=X_train_busy,
            X_test=X_test_busy,
            y_train=y_train_busy,
            y_test=y_test_busy,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            meta_test=meta_test_busy,
            experiment_label=busy_label,
            fast_models_only=fast_models_only,
            enable_slow_models=enable_slow_models,
        )

        if not full_comparison_df.empty:
            save_full_vs_busy_summary(
                full_df=full_comparison_df,
                busy_df=busy_comparison_df,
                top_n_busy_stations=top_n_busy_stations,
            )

        return busy_comparison_df

    return full_comparison_df


def save_model_results(
    comparison_df: pd.DataFrame,
    predictions_by_model: Dict[str, pd.DataFrame],
    year: int = 2025,
) -> None:
    """
    Backward-compatible alias (legacy signature).
    """
    _ = year
    save_model_outputs(
        comparison_df=comparison_df,
        predictions_by_model=predictions_by_model,
        comparison_path=get_model_comparison_path(),
        prediction_paths=get_model_predictions_paths(),
    )


def run_training_pipeline(
    year: int = 2025,
    train_end_month: int = 10,
    fast_models_only: bool = True,
    enable_slow_models: bool = False,
    run_full_dataset_experiment: bool = False,
    run_busy_stations_experiment: bool = True,
    top_n_busy_stations: int = 20,
) -> pd.DataFrame:
    """
    Backward-compatible alias for run_all_models().
    """
    return run_all_models(
        year=year,
        train_end_month=train_end_month,
        fast_models_only=fast_models_only,
        enable_slow_models=enable_slow_models,
        run_full_dataset_experiment=run_full_dataset_experiment,
        run_busy_stations_experiment=run_busy_stations_experiment,
        top_n_busy_stations=top_n_busy_stations,
    )


if __name__ == "__main__":
    # Example direct run:
    # python -m src.train_models
    run_all_models(
        year=2025,
        train_end_month=10,
        fast_models_only=True,
        enable_slow_models=False,
        run_busy_stations_experiment=True,
        top_n_busy_stations=20,
    )
