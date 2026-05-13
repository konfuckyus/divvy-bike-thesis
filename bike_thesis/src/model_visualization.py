"""
model_visualization.py

Generate thesis-ready plots for baseline and machine learning forecasting results.

This module is robust by design:
- If one prediction file is missing, other available models are still plotted.
- If model comparison metrics are missing, per-model plots can still run.
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


def get_predictions_folder() -> Path:
    """
    Return prediction output folder: outputs/predictions.
    """
    return get_project_root() / "outputs" / "predictions"


def get_metrics_folder() -> Path:
    """
    Return metrics output folder: outputs/metrics.
    """
    return get_project_root() / "outputs" / "metrics"


def get_figures_folder() -> Path:
    """
    Return figures output folder: outputs/figures.
    """
    return get_project_root() / "outputs" / "figures"


def ensure_figures_folder() -> Path:
    """
    Ensure that outputs/figures exists.
    """
    folder = get_figures_folder()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def set_plot_style() -> None:
    """
    Set a clean visual style suitable for a thesis report.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def get_prediction_file_map() -> Dict[str, Path]:
    """
    Return canonical prediction file paths by model name.
    """
    pred_folder = get_predictions_folder()
    return {
        "Baseline": pred_folder / "baseline_predictions.csv",
        "LinearRegression": pred_folder / "linear_regression_predictions.csv",
        "RandomForestRegressor": pred_folder / "random_forest_predictions.csv",
        "GradientBoostingRegressor": pred_folder / "gradient_boosting_predictions.csv",
    }


def load_prediction_file(model_name: str, file_path: Path) -> Optional[pd.DataFrame]:
    """
    Load a single model prediction file and standardize expected columns.

    Returns None if file is missing or not compatible.
    """
    if not file_path.exists():
        print(f"Skipping {model_name}: missing file {file_path.name}")
        return None

    df = pd.read_csv(file_path, low_memory=False)

    # Baseline files may store actual target as "demand".
    if "actual_demand" not in df.columns and "demand" in df.columns:
        df = df.rename(columns={"demand": "actual_demand"})

    required_cols = ["actual_demand", "predicted_demand"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Skipping {model_name}: missing column(s) {missing_cols}")
        return None

    if "timestamp_hour" in df.columns:
        df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce")
    if "start_station_id" in df.columns:
        df["start_station_id"] = df["start_station_id"].astype(str)

    return df


def load_all_prediction_files() -> Dict[str, pd.DataFrame]:
    """
    Load all available prediction files from outputs/predictions.

    We start with canonical files, then also include experiment-suffixed files
    (for example busy-station subsets) if available.
    """
    print("Loading prediction files from outputs/predictions ...")

    loaded: Dict[str, pd.DataFrame] = {}

    # 1) Canonical file names
    canonical_files = get_prediction_file_map()
    for model_name, file_path in canonical_files.items():
        df = load_prediction_file(model_name=model_name, file_path=file_path)
        if df is not None and not df.empty:
            loaded[model_name] = df
            print(f"Loaded {model_name}: {len(df):,} rows")

    # 2) Additional experiment files (e.g., *_busy_top20.csv)
    pred_folder = get_predictions_folder()
    for file_path in sorted(pred_folder.glob("*_predictions_*.csv")):
        model_name = file_path.stem.replace("_predictions", "")
        display_name = model_name
        if display_name in loaded:
            continue

        df = load_prediction_file(model_name=display_name, file_path=file_path)
        if df is not None and not df.empty:
            loaded[display_name] = df
            print(f"Loaded {display_name}: {len(df):,} rows")

    return loaded


def load_model_comparison_file() -> Optional[pd.DataFrame]:
    """
    Load model comparison metrics from outputs/metrics/model_comparison.csv.
    """
    metrics_path = get_metrics_folder() / "model_comparison.csv"
    if not metrics_path.exists():
        print(f"Skipping model comparison plots: missing {metrics_path.name}")
        return None

    df = pd.read_csv(metrics_path, low_memory=False)
    required_cols = ["model", "rmse", "mae"]
    if any(col not in df.columns for col in required_cols):
        print("Skipping model comparison plots: required columns not found.")
        return None

    return df


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "rmse",
    save: bool = True,
) -> Optional[Path]:
    """
    Plot model comparison bar chart for RMSE or MAE.

    This plot helps compare overall model error levels in one figure.
    """
    if comparison_df.empty or metric not in comparison_df.columns:
        return None

    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / f"forecast_model_comparison_{metric}.png"

    plot_df = comparison_df.sort_values(metric, ascending=True).copy()

    plt.figure()
    sns.barplot(data=plot_df, x="model", y=metric, color="slateblue")
    plt.title(f"Model Comparison by {metric.upper()}")
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=20)
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_actual_vs_predicted_scatter(
    df: pd.DataFrame,
    model_name: str,
    save: bool = True,
) -> Optional[Path]:
    """
    Scatter plot of actual vs predicted values.

    This plot shows calibration quality; points closer to the diagonal are better.
    """
    if df.empty:
        return None

    set_plot_style()
    figures_folder = ensure_figures_folder()
    file_stub = model_name.lower().replace(" ", "_")
    output_path = figures_folder / f"forecast_actual_vs_predicted_scatter_{file_stub}.png"

    plt.figure()
    sns.scatterplot(
        data=df,
        x="actual_demand",
        y="predicted_demand",
        alpha=0.25,
        s=20,
        edgecolor=None,
    )

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
    Residual histogram (actual - predicted) for a model.

    This plot helps inspect bias and spread of prediction errors.
    """
    if df.empty:
        return None

    set_plot_style()
    figures_folder = ensure_figures_folder()
    file_stub = model_name.lower().replace(" ", "_")
    output_path = figures_folder / f"forecast_residual_histogram_{file_stub}.png"

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


_TRAIN_DEMAND_CACHE: Optional[pd.Series] = None

def get_train_demand_totals(year: int = 2025, train_end_month: int = 10) -> pd.Series:
    """
    Load station_hour_demand dataset, filter strictly for the train period,
    and sum demand per station to rank them purely based on train splits.
    Results are cached globally across model plots to save time.
    """
    global _TRAIN_DEMAND_CACHE
    if _TRAIN_DEMAND_CACHE is not None:
        return _TRAIN_DEMAND_CACHE

    demand_path = get_project_root() / "data" / "processed" / f"station_hour_demand_{year}.csv"
    if not demand_path.exists():
        print(f"Warning: training data not found at {demand_path}. Plot grouping might fail.")
        return pd.Series(dtype=float)

    print("Loading train subset for unbiased station classification...")
    try:
        df = pd.read_csv(demand_path, usecols=["start_station_id", "timestamp_hour", "demand"])
        df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce")
        train_mask = df["timestamp_hour"].dt.month <= train_end_month
        train_df = df[train_mask]
        
        totals = train_df.groupby("start_station_id")["demand"].sum().sort_values(ascending=False)
        _TRAIN_DEMAND_CACHE = totals
        return totals
    except Exception as e:
        print(f"Error loading train demand totals: {e}")
        return pd.Series(dtype=float)


def choose_station_examples(
    df: pd.DataFrame, 
    model_name: str, 
    used_stations: set, 
    year: int = 2025, 
    min_rows: int = 48
) -> Dict[str, str]:
    """
    Select one high-, one medium-, and one low-demand station strictly using train demand totals.
    Guarantees selected stations are distinct across groups and have appropriate test distributions.
    """
    required_cols = ["start_station_id", "actual_demand"]
    if any(col not in df.columns for col in required_cols):
        return {}

    train_totals = get_train_demand_totals(year=year, train_end_month=10)
    if train_totals.empty:
        print("Cannot build train-based station groups.")
        return {}

    # Isolate stations that also appear in this test dataset DataFrame
    test_station_ids = set(df["start_station_id"].unique())
    valid_train_stations = train_totals[train_totals.index.isin(test_station_ids)]
    
    if len(valid_train_stations) < 3:
        print(f"Not enough common stations in test subset for {model_name}.")
        return {}

    ranked_stations = valid_train_stations.index.tolist()
    total_len = len(ranked_stations)
    
    # Split into rough thirds
    third = total_len // 3
    high_candidates = ranked_stations[:third]
    medium_candidates = ranked_stations[third:2*third]
    low_candidates = ranked_stations[2*third:]

    # Count rows per station in the test dataframe
    row_counts = df.groupby("start_station_id").size()

    # Create timestamp bounds for output
    if "timestamp_hour" in df.columns:
        valid_df = df[(df["timestamp_hour"].dt.year >= 2020) & (df["timestamp_hour"].dt.year <= 2026)]
        min_ts = valid_df.groupby("start_station_id")["timestamp_hour"].min()
        max_ts = valid_df.groupby("start_station_id")["timestamp_hour"].max()
    else:
        min_ts = pd.Series()
        max_ts = pd.Series()

    def find_valid_station(candidates: list) -> Optional[str]:
        # Top-down search inside the specific slice
        for sid in candidates:
            str_sid = str(sid)
            if str_sid in used_stations:
                continue
            if row_counts.get(str_sid, 0) >= min_rows:
                return str_sid
        return None

    # Retrieve explicit, non-overlapping members
    high_station = find_valid_station(high_candidates)
    if high_station: used_stations.add(high_station)

    medium_station = find_valid_station(medium_candidates)
    if medium_station: used_stations.add(medium_station)

    low_station = find_valid_station(low_candidates)
    if low_station: used_stations.add(low_station)

    result = {}
    csv_rows = []
    
    for grp, sid in [("high", high_station), ("medium", medium_station), ("low", low_station)]:
        if sid:
            result[grp] = sid
            
            # Map station name natively if exists
            sname = "Unknown"
            if "start_station_name" in df.columns:
                matches = df[df["start_station_id"] == sid]["start_station_name"].dropna()
                if not matches.empty:
                    sname = matches.iloc[0]

            csv_rows.append({
                "experiment_name": model_name,
                "demand_group": grp,
                "station_id": sid,
                "station_name": sname,
                "total_train_demand": train_totals.get(sid, 0),
                "test_row_count": row_counts.get(sid, 0),
                "min_timestamp": min_ts.get(sid, pd.NaT),
                "max_timestamp": max_ts.get(sid, pd.NaT)
            })
        else:
            print(f"Warning: Could not find isolated valid '{grp}' station for {model_name}.")

    if csv_rows:
        try:
            csv_path = get_metrics_folder() / "selected_example_stations.csv"
            out_df = pd.DataFrame(csv_rows)
            # Append if exists to collect across models
            write_header = not csv_path.exists()
            out_df.to_csv(csv_path, mode='a', index=False, header=write_header)
            print(f"Logged {len(csv_rows)} distinct station examples to {csv_path.name}")
        except Exception as e:
            print(f"Could not save selected_example_stations.csv: {e}")

    return result


def plot_station_time_series(
    df: pd.DataFrame,
    model_name: str,
    station_id: str,
    station_group_label: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    max_points: int = 336,
    save: bool = True,
) -> Optional[Path]:
    """
    Time-series line plot of actual vs predicted demand for one station.

    A limited time window keeps the figure readable and thesis-friendly.
    """
    required_cols = ["start_station_id", "timestamp_hour", "actual_demand", "predicted_demand"]
    if any(col not in df.columns for col in required_cols):
        print(f"Skipping station line plot for {model_name}: station/time columns missing.")
        return None

    set_plot_style()
    figures_folder = ensure_figures_folder()
    file_stub = model_name.lower().replace(" ", "_")
    output_path = figures_folder / (
        f"forecast_station_timeseries_{file_stub}_{station_group_label}_station_{station_id}.png"
    )

    plot_df = df.copy()
    plot_df = plot_df[plot_df["start_station_id"] == str(station_id)].copy()
    plot_df = plot_df.dropna(subset=["timestamp_hour"]).sort_values("timestamp_hour")
    
    # Exclude entirely unrealistic timestamp artifacts
    plot_df = plot_df[(plot_df["timestamp_hour"].dt.year >= 2020) & (plot_df["timestamp_hour"].dt.year <= 2026)]

    if start_time is not None:
        plot_df = plot_df[plot_df["timestamp_hour"] >= pd.to_datetime(start_time)]
    if end_time is not None:
        plot_df = plot_df[plot_df["timestamp_hour"] <= pd.to_datetime(end_time)]

    if plot_df.empty:
        print(f"No rows for station {station_id} in model {model_name}.")
        return None

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
    plt.title(
        f"Actual vs Predicted Demand Over Time ({model_name}, {station_group_label.title()}-Demand Station)"
    )
    plt.xlabel("Timestamp (Hour)")
    plt.ylabel("Demand")
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def run_all_model_plots(
    station_id_for_line_plot: Optional[str] = None,
    line_start_time: Optional[str] = None,
    line_end_time: Optional[str] = None,
) -> None:
    """
    Generate all thesis-ready forecasting evaluation plots.
    """
    print("Generating forecasting evaluation plots...")

    csv_path = get_metrics_folder() / "selected_example_stations.csv"
    if csv_path.exists():
        try:
            csv_path.unlink()
        except OSError:
            pass

    comparison_df = load_model_comparison_file()
    # if comparison_df is not None:
    #     plot_model_comparison(comparison_df=comparison_df, metric="rmse", save=True)
    #     plot_model_comparison(comparison_df=comparison_df, metric="mae", save=True)

    # predictions_by_model = load_all_prediction_files()
    # if not predictions_by_model:
    #     print("No valid prediction files found. Visualization step finished.")
    #     return

    # for model_name, pred_df in predictions_by_model.items():
    #     print(f"Creating plots for {model_name} ...")
    #     plot_actual_vs_predicted_scatter(pred_df, model_name=model_name, save=True)
    #     plot_residual_distribution(pred_df, model_name=model_name, save=True)

    #     if station_id_for_line_plot is not None:
    #         plot_station_time_series(
    #             pred_df,
    #             model_name=model_name,
    #             station_id=station_id_for_line_plot,
    #             station_group_label="custom",
    #             start_time=line_start_time,
    #             end_time=line_end_time,
    #             save=True,
    #         )
    #     else:
    #         used_stations = set()
    #         station_examples = choose_station_examples(
    #             df=pred_df, 
    #             model_name=model_name, 
    #             used_stations=used_stations
    #         )
    #         for group_label, station_id in station_examples.items():
    #             plot_station_time_series(
    #                 pred_df,
    #                 model_name=model_name,
    #                 station_id=station_id,
    #                 station_group_label=group_label,
    #                 start_time=line_start_time,
    #                 end_time=line_end_time,
    #                 save=True,
    #             )

    # print(f"All available model figures saved to: {get_figures_folder()}")


# Backward-compatible alias for earlier pipeline code.
def run_all_model_visualizations(
    year: int = 2025,
    station_id_for_line_plot: Optional[str] = None,
    line_start_time: Optional[str] = None,
    line_end_time: Optional[str] = None,
) -> None:
    """
    Alias for run_all_model_plots().
    """
    _ = year
    run_all_model_plots(
        station_id_for_line_plot=station_id_for_line_plot,
        line_start_time=line_start_time,
        line_end_time=line_end_time,
    )


if __name__ == "__main__":
    # Example direct run:
    # python -m src.model_visualization
    run_all_model_plots(
        station_id_for_line_plot=None,
        line_start_time=None,
        line_end_time=None,
    )
