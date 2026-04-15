"""
model_visualization.py

This module creates thesis-ready visualizations for forecasting results
from baseline and machine learning prediction outputs.

It is designed to be robust:
- If one prediction file is missing, that model is skipped.
- Available files are still visualized.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def get_figures_folder() -> Path:
    """
    Return figures output folder path: bike_thesis/outputs/figures.
    """
    return get_project_root() / "outputs" / "figures"


def ensure_figures_folder() -> Path:
    """
    Ensure figure folder exists and return it.
    """
    figures_folder = get_figures_folder()
    figures_folder.mkdir(parents=True, exist_ok=True)
    return figures_folder


def set_plot_style() -> None:
    """
    Set consistent plotting style for thesis visuals.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def get_default_prediction_files(year: int = 2025) -> Dict[str, Path]:
    """
    Return default prediction file paths from baseline and ML modules.
    """
    processed = get_processed_data_folder()

    return {
        "Baseline": processed / f"baseline_predictions_{year}.csv",
        "LinearRegression": processed / f"predictions_linearregression_{year}.csv",
        "RandomForestRegressor": processed
        / f"predictions_randomforestregressor_{year}.csv",
        "GradientBoostingRegressor": processed
        / f"predictions_gradientboostingregressor_{year}.csv",
    }


def load_prediction_results(year: int = 2025) -> Dict[str, pd.DataFrame]:
    """
    Load available prediction result files and return them by model name.

    The loader standardizes target and prediction column names to:
    - actual_demand
    - predicted_demand
    """
    print("Loading prediction result files for visualization...")

    model_files = get_default_prediction_files(year=year)
    available_results: Dict[str, pd.DataFrame] = {}

    for model_name, file_path in model_files.items():
        if not file_path.exists():
            print(f"Skipping missing file for {model_name}: {file_path.name}")
            continue

        df = pd.read_csv(file_path, low_memory=False)

        # Standardize baseline output to the same schema used by ML outputs.
        if "actual_demand" not in df.columns and "demand" in df.columns:
            df = df.rename(columns={"demand": "actual_demand"})

        required_cols = ["actual_demand", "predicted_demand"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(
                f"Skipping {model_name}: missing required column(s) {missing_cols}"
            )
            continue

        # Convert timestamp if present (used in line plots).
        if "timestamp_hour" in df.columns:
            df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce")

        # Ensure station id consistency if present.
        if "start_station_id" in df.columns:
            df["start_station_id"] = df["start_station_id"].astype(str)

        available_results[model_name] = df
        print(f"Loaded {model_name}: {len(df):,} rows")

    return available_results


def plot_actual_vs_predicted_scatter(
    df: pd.DataFrame,
    model_name: str,
    save: bool = True,
) -> Optional[Path]:
    """
    Plot actual vs predicted demand scatter for one model.
    """
    if df.empty:
        return None

    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / f"forecast_scatter_actual_vs_predicted_{model_name.lower()}.png"

    plt.figure()
    sns.scatterplot(
        data=df,
        x="actual_demand",
        y="predicted_demand",
        alpha=0.25,
        s=20,
        edgecolor=None,
    )

    # Ideal diagonal reference line.
    min_val = min(df["actual_demand"].min(), df["predicted_demand"].min())
    max_val = max(df["actual_demand"].max(), df["predicted_demand"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black")

    plt.title(f"Actual vs Predicted Demand ({model_name})")
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_residual_distribution(
    df: pd.DataFrame,
    model_name: str,
    save: bool = True,
) -> Optional[Path]:
    """
    Plot residual histogram for one model.
    """
    if df.empty:
        return None

    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / f"forecast_residual_distribution_{model_name.lower()}.png"

    plot_df = df.copy()
    plot_df["residual"] = plot_df["actual_demand"] - plot_df["predicted_demand"]

    plt.figure()
    sns.histplot(plot_df["residual"], bins=50, kde=True, color="steelblue")
    plt.title(f"Residual Distribution ({model_name})")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_actual_vs_predicted_line_for_station(
    df: pd.DataFrame,
    model_name: str,
    station_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    max_points: int = 500,
    save: bool = True,
) -> Optional[Path]:
    """
    Plot actual and predicted demand over time for a selected station.

    Optional time filters:
    - start_time: e.g., "2025-11-01"
    - end_time:   e.g., "2025-11-15"
    """
    required_cols = ["start_station_id", "timestamp_hour", "actual_demand", "predicted_demand"]
    if any(col not in df.columns for col in required_cols):
        print(
            f"Skipping line plot for {model_name}: missing station/timestamp columns."
        )
        return None

    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / (
        f"forecast_line_actual_vs_predicted_{model_name.lower()}_station_{station_id}.png"
    )

    plot_df = df.copy()
    plot_df = plot_df[plot_df["start_station_id"] == str(station_id)].copy()
    plot_df = plot_df.dropna(subset=["timestamp_hour"]).sort_values("timestamp_hour")

    if start_time is not None:
        plot_df = plot_df[plot_df["timestamp_hour"] >= pd.to_datetime(start_time)]
    if end_time is not None:
        plot_df = plot_df[plot_df["timestamp_hour"] <= pd.to_datetime(end_time)]

    if plot_df.empty:
        print(f"No rows available for station {station_id} in {model_name} line plot.")
        return None

    # Keep line chart readable if interval is very long.
    if len(plot_df) > max_points:
        plot_df = plot_df.tail(max_points).copy()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=plot_df,
        x="timestamp_hour",
        y="actual_demand",
        label="Actual Demand",
        color="black",
        linewidth=1.8,
    )
    sns.lineplot(
        data=plot_df,
        x="timestamp_hour",
        y="predicted_demand",
        label="Predicted Demand",
        color="tomato",
        linewidth=1.5,
    )
    plt.title(f"Actual vs Predicted Demand Over Time ({model_name}, Station {station_id})")
    plt.xlabel("Timestamp (Hour)")
    plt.ylabel("Demand")
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_model_comparison_rmse(
    year: int = 2025,
    save: bool = True,
) -> Optional[Path]:
    """
    Plot model comparison bar chart using RMSE from model_comparison CSV.

    If the comparison file does not exist, the function returns None.
    """
    comparison_path = get_processed_data_folder() / f"model_comparison_{year}.csv"
    if not comparison_path.exists():
        print(f"Skipping RMSE comparison plot: missing {comparison_path.name}")
        return None

    comparison_df = pd.read_csv(comparison_path, low_memory=False)
    if "model" not in comparison_df.columns or "rmse" not in comparison_df.columns:
        print("Skipping RMSE comparison plot: required columns not found.")
        return None

    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / f"forecast_model_comparison_rmse_{year}.png"

    plot_df = comparison_df.sort_values("rmse", ascending=True).copy()

    plt.figure()
    sns.barplot(data=plot_df, x="model", y="rmse", color="slateblue")
    plt.title(f"Model Comparison by RMSE ({year})")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def run_all_model_visualizations(
    year: int = 2025,
    station_id_for_line_plot: Optional[str] = None,
    line_start_time: Optional[str] = None,
    line_end_time: Optional[str] = None,
) -> None:
    """
    Generate a full set of model visualization figures from available files.
    """
    print("Generating model result visualizations...")

    results_by_model = load_prediction_results(year=year)
    if not results_by_model:
        print("No compatible prediction files found. Skipping visualization step.")
        return

    # If station id is not provided, pick one from the first model that has station info.
    chosen_station_id = station_id_for_line_plot
    if chosen_station_id is None:
        for df in results_by_model.values():
            if "start_station_id" in df.columns and not df["start_station_id"].empty:
                chosen_station_id = str(df["start_station_id"].iloc[0])
                break

    for model_name, pred_df in results_by_model.items():
        print(f"Creating figures for {model_name}...")
        plot_actual_vs_predicted_scatter(pred_df, model_name=model_name, save=True)
        plot_residual_distribution(pred_df, model_name=model_name, save=True)

        if chosen_station_id is not None:
            plot_actual_vs_predicted_line_for_station(
                pred_df,
                model_name=model_name,
                station_id=str(chosen_station_id),
                start_time=line_start_time,
                end_time=line_end_time,
                save=True,
            )

    plot_model_comparison_rmse(year=year, save=True)
    print(f"Model visualization figures saved to: {get_figures_folder()}")


if __name__ == "__main__":
    # Example direct run:
    # python -m src.model_visualization
    run_all_model_visualizations(
        year=2025,
        station_id_for_line_plot=None,
        line_start_time=None,
        line_end_time=None,
    )
