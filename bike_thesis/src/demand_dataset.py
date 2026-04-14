"""
demand_dataset.py

This module is responsible for creating a station-hour demand dataset
from cleaned trip-level Divvy data.

For Progress Report 2, we aggregate rides by:
- start_station_id
- start_station_name
- timestamp_hour (started_at floored to hour)

Then we create additional time-based features that will be useful later
for demand analysis and prediction tasks.
"""

from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    """
    Return the project root folder.

    We assume this file lives in bike_thesis/src/demand_dataset.py,
    so the project root is one level above src.
    """
    return Path(__file__).resolve().parents[1]


def get_processed_data_folder() -> Path:
    """
    Return the processed data folder path: bike_thesis/data/processed.
    """
    return get_project_root() / "data" / "processed"


def get_station_hour_demand_path(year: int = 2025) -> Path:
    """
    Return the canonical output path for the station-hour demand dataset.
    """
    return get_processed_data_folder() / f"station_hour_demand_{year}.csv"


def add_forecasting_features(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forecasting-oriented features to the station-hour demand dataset.

    Why these features matter
    -------------------------
    For demand forecasting, the most predictive information is usually:
    - recent demand history (lag features),
    - short/medium-term trend (rolling means),
    - calendar position (weekday/month effects).

    Important modeling note
    -----------------------
    We intentionally keep NaN values created by lag/rolling operations.
    These NaNs appear at the beginning of each station time series because
    there is not enough past history yet. During model training, you may
    later drop or impute these rows depending on your strategy.
    """
    print("Adding lag and rolling forecasting features...")

    df = demand_df.copy()

    # Group by station so each station gets its own time-series history.
    station_groups = df.groupby("start_station_id")["demand"]

    # Lag features: demand from specific previous hours.
    df["prev_hour_demand"] = station_groups.shift(1)
    df["prev_2hour_demand"] = station_groups.shift(2)
    df["prev_3hour_demand"] = station_groups.shift(3)
    df["prev_24hour_demand"] = station_groups.shift(24)

    # Rolling means must use ONLY past information.
    # We first shift by 1 hour to exclude current-hour demand, then roll.
    df["rolling_mean_3h"] = (
        station_groups.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["rolling_mean_6h"] = (
        station_groups.shift(1).rolling(window=6, min_periods=1).mean()
    )
    df["rolling_mean_24h"] = (
        station_groups.shift(1).rolling(window=24, min_periods=1).mean()
    )

    # Numeric calendar features are model-friendly for many algorithms.
    df["weekday_num"] = df["timestamp_hour"].dt.weekday
    df["month_num"] = df["timestamp_hour"].dt.month

    print("Forecasting features added successfully.")

    return df


def create_station_hour_demand_dataset(trips_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a station-hour demand dataset from cleaned trip-level data.

    Required input columns
    ----------------------
    - started_at
    - start_station_id
    - start_station_name

    Steps
    -----
    1) Keep only rows with valid start station id and name.
    2) Ensure start_station_id is stored as string for consistency.
    3) Floor started_at to hour and store in timestamp_hour.
    4) Group by station + hour and count rides as demand.
    5) Add additional time-based columns for later modeling/analysis.
    """
    print("Creating station-hour demand dataset...")

    required_columns = ["started_at", "start_station_id", "start_station_name"]
    missing_columns = [col for col in required_columns if col not in trips_df.columns]
    if missing_columns:
        raise KeyError(
            "Missing required column(s) for demand aggregation: "
            f"{missing_columns}"
        )

    df = trips_df.copy()

    # Ensure started_at is datetime (safe even if it is already datetime).
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")

    # Drop rows missing required station fields or invalid started_at.
    df = df.dropna(subset=["start_station_id", "start_station_name", "started_at"]).copy()

    # Keep station identifiers as strings (important for consistency across files).
    df["start_station_id"] = df["start_station_id"].astype(str)

    # Floor ride start timestamps to the beginning of each hour.
    df["timestamp_hour"] = df["started_at"].dt.floor("H")

    # Aggregate ride counts at station-hour level.
    demand_df = (
        df.groupby(
            ["start_station_id", "start_station_name", "timestamp_hour"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "demand"})
    )

    # Add time-based features from the station-hour timestamp.
    demand_df["date"] = demand_df["timestamp_hour"].dt.date
    demand_df["hour"] = demand_df["timestamp_hour"].dt.hour
    demand_df["weekday"] = demand_df["timestamp_hour"].dt.weekday
    demand_df["month"] = demand_df["timestamp_hour"].dt.month
    demand_df["year"] = demand_df["timestamp_hour"].dt.year
    demand_df["day_of_month"] = demand_df["timestamp_hour"].dt.day

    # Weekend indicator: Saturday(5) and Sunday(6) => 1, otherwise 0.
    demand_df["is_weekend"] = (demand_df["weekday"] >= 5).astype(int)

    # Sort for readability and reproducibility.
    demand_df = demand_df.sort_values(
        by=["start_station_id", "timestamp_hour"]
    ).reset_index(drop=True)

    # Add station-wise lag/rolling features after sorting by station + time.
    # This order is essential to prevent accidental data leakage from future rows.
    demand_df = add_forecasting_features(demand_df)

    print(f"Station-hour demand dataset created with {len(demand_df):,} rows.")

    return demand_df


def save_station_hour_demand_dataset(
    df: pd.DataFrame,
    year: int = 2025,
) -> Path:
    """
    Save the station-hour demand dataset into data/processed/.
    """
    processed_folder = get_processed_data_folder()
    processed_folder.mkdir(parents=True, exist_ok=True)

    output_path = get_station_hour_demand_path(year=year)
    df.to_csv(output_path, index=False)

    print(f"Station-hour demand data saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    """
    Quick standalone run support:
        python -m src.demand_dataset

    Expects that cleaned data already exists in data/processed/.
    """
    year = 2025
    cleaned_path = get_processed_data_folder() / f"cleaned_{year}_data.csv"

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {cleaned_path}. "
            f"Run `python -m src.main` first."
        )

    print(f"Reading cleaned data from: {cleaned_path}")
    cleaned_df = pd.read_csv(cleaned_path, low_memory=False)
    station_hour_demand_df = create_station_hour_demand_dataset(cleaned_df)
    save_station_hour_demand_dataset(station_hour_demand_df, year=year)
