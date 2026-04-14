"""
load_data.py

This module is responsible for:
- Finding all Divvy CSV files for a given year in the data/raw/ folder.
- Loading each CSV file into a pandas DataFrame.
- Concatenating all monthly DataFrames into one combined DataFrame.

We keep this file focused only on data loading (not heavy cleaning),
so that the workflow is easier to understand and maintain.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd


def get_project_root() -> Path:
    """
    Return the root directory of the project.

    We assume this file lives in bike_thesis/src/load_data.py,
    so the project root is two levels above this file.
    """
    return Path(__file__).resolve().parents[1]


def get_raw_data_folder() -> Path:
    """
    Return the path to the raw data folder: bike_thesis/data/raw.
    """
    return get_project_root() / "data" / "raw"


def find_csv_files_for_year(year: int) -> List[Path]:
    """
    Find all CSV files in the raw data folder that belong to a given year.

    We do not enforce a strict filename format, but we expect that
    the year (e.g., '2025') appears somewhere in the file name.
    For example:
        - "2025-01-divvy-tripdata.csv"
        - "divvy_2025_02.csv"

    Parameters
    ----------
    year : int
        The year we want to load (e.g., 2025).

    Returns
    -------
    List[Path]
        A list of file paths to CSV files matching the year.
    """
    raw_folder = get_raw_data_folder()
    year_str = str(year)

    # Collect all CSV files under data/raw (non-recursive is usually enough).
    all_csv_files = list(raw_folder.glob("*.csv"))

    # Filter files that contain the year string in the file name.
    year_csv_files = [f for f in all_csv_files if year_str in f.name]

    return sorted(year_csv_files)


def load_single_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a single Divvy CSV file into a pandas DataFrame.

    We keep this very simple:
    - Rely on pandas to infer dtypes initially.
    - Do not parse dates here; we will convert them later in preprocess.py.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Data loaded from the CSV file.
    """
    # Using low_memory=False to prevent mixed-type warnings during parsing.
    df = pd.read_csv(file_path, low_memory=False)
    df["source_file"] = file_path.name  # helpful for debugging later
    return df


def load_trip_data_for_year(year: int = 2025) -> pd.DataFrame:
    """
    Load and combine all Divvy trip CSV files for a given year.

    Steps:
    1. Find all CSV files for the year in data/raw/.
    2. Load each CSV into a pandas DataFrame.
    3. Concatenate them into a single DataFrame.

    Parameters
    ----------
    year : int, optional
        Year to load (default is 2025).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing all trips for the specified year.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found for the given year.
    """
    csv_files = find_csv_files_for_year(year)

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found for year {year} in {get_raw_data_folder()}"
        )

    dataframes = []
    for file_path in csv_files:
        print(f"Loading file: {file_path.name}")  # simple progress information
        df = load_single_csv(file_path)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    print(
        f"Loaded {len(csv_files)} file(s) for year {year} "
        f"with a total of {len(combined_df):,} rows."
    )

    return combined_df


def save_combined_raw_data(
    df: pd.DataFrame,
    year: int = 2025,
    file_name: Optional[str] = None,
) -> Path:
    """
    Optionally save the combined raw (but not yet cleaned) data
    into the processed folder for reference.

    This is useful if loading all CSVs takes time; you can reuse the
    combined file in later runs.

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame to save.
    year : int, optional
        Year of the data (default 2025).
    file_name : str, optional
        Custom file name. If None, a default name is constructed.

    Returns
    -------
    Path
        The path to the saved CSV file.
    """
    project_root = get_project_root()
    processed_folder = project_root / "data" / "processed"
    processed_folder.mkdir(parents=True, exist_ok=True)

    if file_name is None:
        file_name = f"divvy_trips_{year}_combined_raw.csv"

    output_path = processed_folder / file_name

    # index=False because the index is not meaningful here.
    df.to_csv(output_path, index=False)
    print(f"Combined raw data saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    """
    Allow this module to be run directly for quick testing.

    Example:
        python -m src.load_data

    This will:
    - Load all 2025 trip files from data/raw/
    - Combine them
    - Save a combined raw CSV in data/processed/
    """
    year_to_load = 2025
    trips_df = load_trip_data_for_year(year=year_to_load)
    save_combined_raw_data(trips_df, year=year_to_load)

