"""
preprocess.py

This module is responsible for:
- Converting datetime columns to proper pandas datetime type.
- Creating a ride_duration_min feature (in minutes).
- Removing clearly invalid rides:
  * missing started_at or ended_at
  * negative durations
  * extremely long durations (outliers)
- Creating time-based features:
  * date
  * hour
  * weekday (name)
  * month (name)
- Creating a station-based DataFrame (dropping missing start_station_id).

We keep the logic here so that load_data.py only worries about reading files.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """
    Same helper as in load_data.py.
    Returns the project root folder.
    """
    return Path(__file__).resolve().parents[1]


def get_processed_data_folder() -> Path:
    """
    Return the path to the processed data folder: bike_thesis/data/processed.
    """
    return get_project_root() / "data" / "processed"


def get_cleaned_data_path(year: int = 2025) -> Path:
    """
    Return the canonical cached cleaned dataset path.

    This file is used by main.py for caching:
    - If it exists, we can skip raw loading + preprocessing.
    - If it does not exist, we must preprocess and create it.
    """
    return get_processed_data_folder() / f"cleaned_{year}_data.csv"


def get_station_level_data_path(year: int = 2025) -> Path:
    """
    Return the cached station-level dataset path.

    This is not strictly required for caching behavior, but it is useful to
    avoid regenerating station-level outputs when we already have them.
    """
    return get_processed_data_folder() / f"station_level_{year}_data.csv"


def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert started_at and ended_at columns to pandas datetime.

    We assume the raw columns are named 'started_at' and 'ended_at'
    as in the standard Divvy / Lyft bike sharing data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with started_at and ended_at as strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime columns converted.
    """
    # errors='coerce' will turn unparseable values into NaT (missing datetime)
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")
    return df


def add_ride_duration_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ride_duration_min column in minutes.

    The duration is calculated as:
        (ended_at - started_at) in minutes

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # Compute duration in minutes as a float
    df["ride_duration_min"] = (
        (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0
    )

    return df


def remove_invalid_rides(
    df: pd.DataFrame,
    min_duration_min: float = 1.0,
    max_duration_min: float = 240.0,
) -> pd.DataFrame:
    """
    Remove obviously invalid rides.

    Rules:
    - Drop rows where started_at or ended_at is missing.
    - Drop rows where ride_duration_min is missing or negative.
    - Drop rows where ride_duration_min is too short or too long.

    The min and max thresholds can be tuned. Here we choose:
    - min_duration_min: 1 minute (to remove extremely short/unrealistic trips)
    - max_duration_min: 240 minutes (4 hours, to remove extreme outliers)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with started_at, ended_at, and ride_duration_min.
    min_duration_min : float
        Minimum acceptable duration in minutes.
    max_duration_min : float
        Maximum acceptable duration in minutes.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    # Start by dropping rows with missing started_at or ended_at
    df_clean = df.dropna(subset=["started_at", "ended_at"]).copy()

    # Drop missing or NaN duration
    df_clean = df_clean.dropna(subset=["ride_duration_min"])

    # Keep only rides within [min_duration_min, max_duration_min]
    mask_valid_duration = (
        (df_clean["ride_duration_min"] >= min_duration_min)
        & (df_clean["ride_duration_min"] <= max_duration_min)
    )

    df_clean = df_clean.loc[mask_valid_duration].copy()

    return df_clean


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features derived from started_at.

    Features:
    - date: calendar date (YYYY-MM-DD)
    - hour: hour of day (0-23)
    - weekday: day name (e.g., Monday)
    - month: month name (e.g., January)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a started_at datetime column.

    Returns
    -------
    pd.DataFrame
    """
    # .dt.date gives Python date objects (no time part)
    df["date"] = df["started_at"].dt.date

    # Hour of the day (0-23)
    df["hour"] = df["started_at"].dt.hour

    # Day of week (Monday, Tuesday, etc.)
    df["weekday"] = df["started_at"].dt.day_name()

    # Month name (January, February, etc.)
    df["month"] = df["started_at"].dt.month_name()

    return df


def create_station_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a station-based view for start stations.

    We:
    - Remove rows with missing start_station_id (because station-level
      analysis requires a valid station identifier).
    - Optionally, we could aggregate later in eda.py when plotting.

    Parameters
    ----------
    df : pd.DataFrame
        Clean trip-level DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only rows with a valid start_station_id.
    """
    station_df = df.dropna(subset=["start_station_id"]).copy()

    # If start_station_id is numeric in some files and string in others,
    # it is often safer to convert to string for consistency.
    station_df["start_station_id"] = station_df["start_station_id"].astype(str)

    return station_df


def preprocess_trips(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full preprocessing pipeline on the raw combined DataFrame.

    Steps:
    1. Convert datetime columns.
    2. Add ride_duration_min.
    3. Remove invalid rides.
    4. Add time-based features.
    5. Create station-level DataFrame.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Combined raw trip data as loaded from load_data.py.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - cleaned_trips_df: cleaned trip-level data
        - station_trips_df: subset with valid start_station_id
    """
    df = df_raw.copy()

    # 1) Ensure datetime columns are proper datetimes
    df = convert_datetime_columns(df)

    # 2) Create ride_duration_min column
    df = add_ride_duration_minutes(df)

    # 3) Remove invalid or extreme rides
    df = remove_invalid_rides(df)

    # 4) Add time-based features
    df = add_time_features(df)

    # 5) Create station-level subset
    station_df = create_station_level_dataframe(df)

    return df, station_df


def save_preprocessed_data(
    trips_df: pd.DataFrame,
    station_trips_df: pd.DataFrame,
    year: int = 2025,
) -> None:
    """
    Save the cleaned trip-level and station-level DataFrames
    into the processed data folder.

    Parameters
    ----------
    trips_df : pd.DataFrame
        Cleaned trip-level data.
    station_trips_df : pd.DataFrame
        Station-level subset (non-null start_station_id).
    year : int
        Year of the data, used in file names.
    """
    processed_folder = get_processed_data_folder()
    processed_folder.mkdir(parents=True, exist_ok=True)

    # Canonical filenames used for caching in main.py
    trips_path = get_cleaned_data_path(year=year)
    station_path = get_station_level_data_path(year=year)

    trips_df.to_csv(trips_path, index=False)
    station_trips_df.to_csv(station_path, index=False)

    print(f"Cleaned trip-level data saved to: {trips_path}")
    print(f"Station-level data saved to: {station_path}")


if __name__ == "__main__":
    """
    Allow this module to be run directly for quick testing.

    Example:
        python -m src.preprocess

    For this to work directly, we assume you already created a combined
    raw CSV in data/processed/ using load_data.save_combined_raw_data.
    """
    from load_data import get_project_root  # local import to avoid cycles

    year = 2025
    processed_folder = get_processed_data_folder()
    combined_raw_path = (
        processed_folder / f"divvy_trips_{year}_combined_raw.csv"
    )

    if not combined_raw_path.exists():
        raise FileNotFoundError(
            f"Combined raw file not found at {combined_raw_path}. "
            f"First run `python -m src.load_data`."
        )

    print(f"Reading combined raw data from: {combined_raw_path}")
    raw_df = pd.read_csv(combined_raw_path, low_memory=False)

    trips_clean, station_trips = preprocess_trips(raw_df)
    save_preprocessed_data(trips_clean, station_trips, year=year)

