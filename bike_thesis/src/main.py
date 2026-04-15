"""
main.py

This script orchestrates both Progress Report 1 and Progress Report 2 workflows.

Available pipeline stages
-------------------------
1) Load raw data + preprocess trip-level data (with caching)
2) Run exploratory data analysis (EDA)
3) Build station-hour demand dataset (with caching)
4) Run baseline forecasting
5) Train machine learning forecasting models
6) Generate model evaluation plots

Each stage can be enabled/disabled with boolean flags in main().
"""

from pathlib import Path

import pandas as pd

from load_data import load_trip_data_for_year
from preprocess import (
    get_cleaned_data_path,
    get_station_level_data_path,
    preprocess_trips,
    save_preprocessed_data,
)
from eda import run_all_plots, print_basic_summary_statistics
from demand_dataset import (
    get_station_hour_demand_path,
    create_station_hour_demand_dataset,
    save_station_hour_demand_dataset,
)
from baseline_model import run_baseline_forecast_pipeline
from train_models import run_training_pipeline
from model_visualization import run_all_model_visualizations


def get_project_root() -> Path:
    """
    Helper function in case we need the project root here.
    """
    return Path(__file__).resolve().parents[1]


def load_or_prepare_preprocessed_data(
    year: int,
    force_preprocess: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return cleaned trip-level and station-level data.

    Caching behavior:
    - If cleaned cache exists and force_preprocess=False, load from cache.
    - Otherwise, load raw CSV files and run preprocessing.
    """
    cleaned_path = get_cleaned_data_path(year=year)
    station_path = get_station_level_data_path(year=year)

    if cleaned_path.exists() and not force_preprocess:
        print("\nStep 1: Loading cached processed dataset...")
        print(f"Using existing file: {cleaned_path}")
        trips_clean = pd.read_csv(cleaned_path, low_memory=False)

        if station_path.exists():
            station_trips = pd.read_csv(station_path, low_memory=False)
        else:
            print(
                f"Station-level cached file not found ({station_path}). "
                "Rebuilding station-level dataset from cleaned data..."
            )
            from preprocess import create_station_level_dataframe

            station_trips = create_station_level_dataframe(trips_clean)
            station_trips.to_csv(station_path, index=False)
            print(f"Station-level data saved to: {station_path}")
    else:
        print("\nStep 1: Loading raw data (needed for preprocessing)...")
        trips_raw = load_trip_data_for_year(year=year)

        print("\nStep 2: Preprocessing data...")
        trips_clean, station_trips = preprocess_trips(trips_raw)

        print("Saving preprocessed datasets to data/processed/ ...")
        save_preprocessed_data(trips_clean, station_trips, year=year)

    return trips_clean, station_trips


def main(
    year: int = 2025,
    force_preprocess: bool = False,
    force_plots: bool = False,
    run_data_preparation: bool = True,
    run_eda_stage: bool = True,
    run_demand_dataset_stage: bool = True,
    run_baseline_stage: bool = False,
    run_ml_stage: bool = False,
    run_model_visualization_stage: bool = False,
) -> None:
    """
    Run thesis data and forecasting pipeline for a given year.

    Parameters
    ----------
    year : int
        Year of Divvy data to process (default 2025).
    force_preprocess : bool
        If True, regenerate the processed dataset even if it exists.
    force_plots : bool
        If True, regenerate all figures even if they already exist.
    run_data_preparation : bool
        If True, run/load trip-level preprocessing stage.
    run_eda_stage : bool
        If True, generate EDA plots and summary statistics.
    run_demand_dataset_stage : bool
        If True, build/load station-hour demand dataset.
    run_baseline_stage : bool
        If True, run baseline forecasting evaluation.
    run_ml_stage : bool
        If True, run ML model training/evaluation.
    run_model_visualization_stage : bool
        If True, generate forecast result visualization plots.
    """
    print(f"=== Divvy Bike Thesis Pipeline for {year} ===")

    cleaned_path = get_cleaned_data_path(year=year)
    station_path = get_station_level_data_path(year=year)
    demand_path = get_station_hour_demand_path(year=year)

    trips_clean: pd.DataFrame | None = None
    station_trips: pd.DataFrame | None = None

    # Decide if trip-level data is needed by selected stages.
    needs_trip_level_data = (
        run_data_preparation
        or run_eda_stage
        or run_demand_dataset_stage
        or (
            (run_baseline_stage or run_ml_stage or run_model_visualization_stage)
            and (not demand_path.exists() or force_preprocess)
        )
    )

    if needs_trip_level_data:
        if run_data_preparation:
            trips_clean, station_trips = load_or_prepare_preprocessed_data(
                year=year,
                force_preprocess=force_preprocess,
            )
        else:
            print("\nLoading cached preprocessed datasets (run_data_preparation=False)...")
            if not cleaned_path.exists():
                raise FileNotFoundError(
                    f"Cleaned dataset not found at {cleaned_path}. "
                    "Enable run_data_preparation=True first."
                )
            trips_clean = pd.read_csv(cleaned_path, low_memory=False)

            if station_path.exists():
                station_trips = pd.read_csv(station_path, low_memory=False)
            else:
                from preprocess import create_station_level_dataframe

                station_trips = create_station_level_dataframe(trips_clean)
                station_trips.to_csv(station_path, index=False)
                print(f"Station-level data saved to: {station_path}")

    # Stage: build station-hour demand dataset (with cache)
    if run_demand_dataset_stage:
        if demand_path.exists() and not force_preprocess:
            print("\nStep 3: Using cached station-hour demand dataset...")
            print(f"Using existing file: {demand_path}")
        else:
            if trips_clean is None:
                raise ValueError(
                    "Trip-level data is required to build demand dataset. "
                    "Enable run_data_preparation=True or load cached cleaned data."
                )
            print("\nStep 3: Creating station-hour demand dataset...")
            station_hour_demand_df = create_station_hour_demand_dataset(trips_clean)
            save_station_hour_demand_dataset(station_hour_demand_df, year=year)
    else:
        # If downstream forecasting stages need demand data, ensure it exists.
        if (run_baseline_stage or run_ml_stage or run_model_visualization_stage) and (
            not demand_path.exists() or force_preprocess
        ):
            if trips_clean is None:
                raise FileNotFoundError(
                    f"Demand dataset not found at {demand_path}. "
                    "Enable run_demand_dataset_stage=True first."
                )
            print(
                "\nDemand dataset required by downstream stages. "
                "Creating it now..."
            )
            station_hour_demand_df = create_station_hour_demand_dataset(trips_clean)
            save_station_hour_demand_dataset(station_hour_demand_df, year=year)

    # Stage: EDA
    if run_eda_stage:
        if trips_clean is None or station_trips is None:
            raise ValueError(
                "EDA stage requires trip-level datasets. "
                "Enable run_data_preparation=True."
            )
        print("\nStep 4: Running exploratory data analysis (EDA)...")
        run_all_plots(trips_clean, station_trips, force_plots=force_plots)

        print("\nStep 5: Printing summary statistics...")
        print_basic_summary_statistics(trips_clean)

    # Stage: baseline forecasting
    if run_baseline_stage:
        print("\nStep 6: Running baseline forecasting...")
        run_baseline_forecast_pipeline(year=year, train_end_month=10)

    # Stage: machine learning forecasting
    if run_ml_stage:
        print("\nStep 7: Training machine learning forecasting models...")
        run_training_pipeline(year=year, train_end_month=10)

    # Stage: model evaluation plots
    if run_model_visualization_stage:
        print("\nStep 8: Generating model evaluation plots...")
        run_all_model_visualizations(
            year=year,
            station_id_for_line_plot=None,
            line_start_time=None,
            line_end_time=None,
        )

    print("=== Pipeline completed successfully. ===")


if __name__ == "__main__":
    # Default call keeps current behavior for your project:
    # - data preparation (load + preprocess with caching)
    # - demand dataset generation (cached)
    # - EDA + summary statistics
    # Forecast/model stages are optional and off by default.
    main(
        year=2025,
        force_preprocess=False,
        force_plots=False,
        run_data_preparation=True,
        run_eda_stage=True,
        run_demand_dataset_stage=True,
        run_baseline_stage=False,
        run_ml_stage=False,
        run_model_visualization_stage=False,
    )

