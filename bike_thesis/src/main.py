"""
main.py

This script orchestrates the full Progress Report 1 workflow:

1. Load and combine all 2025 Divvy trip CSV files from data/raw/.
2. Clean and preprocess the data:
   - Convert datetime columns
   - Create ride_duration_min
   - Remove invalid rides
   - Add time-based features
   - Create a station-level subset
3. Save the cleaned datasets into data/processed/.
4. Run exploratory data analysis (EDA) and generate the required plots.
5. Print basic summary statistics for the thesis report.


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


def get_project_root() -> Path:
    """
    Helper function in case we need the project root here.
    """
    return Path(__file__).resolve().parents[1]


def main(
    year: int = 2025,
    force_preprocess: bool = False,
    force_plots: bool = False,
) -> None:
    """
    Run the full data pipeline for a given year.

    Parameters
    ----------
    year : int
        Year of Divvy data to process (default 2025).
    force_preprocess : bool
        If True, regenerate the processed dataset even if it exists.
    force_plots : bool
        If True, regenerate all figures even if they already exist.
    """
    print(f"=== Divvy Bike Thesis Pipeline for {year} ===")

    cleaned_path = get_cleaned_data_path(year=year)
    station_path = get_station_level_data_path(year=year)
    demand_path = get_station_hour_demand_path(year=year)

    # Step 1 + 2 (cached): load processed data if available, otherwise preprocess raw data
    if cleaned_path.exists() and not force_preprocess:
        print("\nStep 1: Loading cached processed dataset...")
        print(f"Using existing file: {cleaned_path}")
        trips_clean = pd.read_csv(cleaned_path, low_memory=False)

        # Station-level dataset is optional for caching, but used for station plots.
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

    # Step 3: Build station-hour demand dataset (cached unless forced preprocess)
    if demand_path.exists() and not force_preprocess:
        print("\nStep 3: Using cached station-hour demand dataset...")
        print(f"Using existing file: {demand_path}")
    else:
        print("\nStep 3: Creating station-hour demand dataset...")
        station_hour_demand_df = create_station_hour_demand_dataset(trips_clean)
        save_station_hour_demand_dataset(station_hour_demand_df, year=year)

    # Step 4: Run EDA and generate plots (cached per-figure unless forced)
    print("\nStep 4: Running exploratory data analysis (EDA)...")
    run_all_plots(trips_clean, station_trips, force_plots=force_plots)

    # 5) Print basic summary statistics to the console
    print("\nStep 5: Printing summary statistics...")
    print_basic_summary_statistics(trips_clean)

    print("=== Pipeline completed successfully. ===")


if __name__ == "__main__":
    # run (python -m src.main),
    # this code will execute and run the full pipeline.
    main(year=2025, force_preprocess=False, force_plots=False)

