"""
eda.py

This module is responsible for:
- Performing exploratory data analysis (EDA) on the cleaned trip data.
- Generating and saving thesis-ready plots:
  1) Daily ride counts
  2) Hourly ride counts
  3) Ride counts by weekday
  4) Member vs casual distribution
  5) Top 10 most used start stations
  6) Ride duration distribution
- Printing basic summary statistics for the report.

We keep plotting logic here so it is clearly separated from loading
and preprocessing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_project_root() -> Path:
    """
    Helper to get the project root folder.
    """
    return Path(__file__).resolve().parents[1]


def get_figures_folder() -> Path:
    """
    Return the path to the figures output folder: bike_thesis/outputs/figures.
    """
    return get_project_root() / "outputs" / "figures"


def ensure_figures_folder() -> Path:
    """
    Ensure that the figures output folder exists and return its Path.
    """
    figures_folder = get_figures_folder()
    figures_folder.mkdir(parents=True, exist_ok=True)
    return figures_folder


def should_generate_figure(output_path: Path, force: bool) -> bool:
    """
    Decide whether to generate a figure based on caching rules.

    If the PNG already exists and force=False, we skip regenerating it.
    """
    if output_path.exists() and not force:
        print(f"Skipping existing figure: {output_path.name}")
        return False
    return True


def set_plot_style() -> None:
    """
    Set a consistent visual style for all plots.

    Using seaborn and matplotlib settings to make plots look cleaner
    for an academic thesis.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def plot_daily_ride_counts(
    trips_df: pd.DataFrame, save: bool = True, force: bool = False
) -> Path:
    """
    Plot daily ride counts over time.

    Parameters
    ----------
    trips_df : pd.DataFrame
        Cleaned trip-level data with a 'date' column.
    save : bool
        Whether to save the figure to disk.

    Returns
    -------
    Path
        Path to the saved figure (if save=True), otherwise None.
    """
    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / "daily_ride_counts.png"

    if save and not should_generate_figure(output_path, force=force):
        return output_path

    daily_counts = (
        trips_df.groupby("date")["ride_id"]
        .count()
        .reset_index(name="ride_count")
    )

    # Add a 7-day moving average to make overall trends easier to interpret.
    # Using a trailing window (previous 7 days including the current day).
    daily_counts["ride_count_7day_ma"] = (
        daily_counts["ride_count"].rolling(window=7, min_periods=1).mean()
    )

    plt.figure()
    # Daily counts as a lighter line
    sns.lineplot(
        data=daily_counts,
        x="date",
        y="ride_count",
        label="Daily Rides",
        color="steelblue",
        alpha=0.35,
        linewidth=1.5,
    )

    # 7-day moving average as a smoother, more prominent line
    sns.lineplot(
        data=daily_counts,
        x="date",
        y="ride_count_7day_ma",
        label="7-Day Moving Average",
        color="steelblue",
        alpha=0.95,
        linewidth=2.5,
    )
    plt.title("Daily Ride Counts")
    plt.xlabel("Date")
    plt.ylabel("Number of Rides")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_hourly_ride_counts(
    trips_df: pd.DataFrame, save: bool = True, force: bool = False
) -> Path:
    """
    Plot ride counts by hour of day.
    """
    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / "hourly_ride_counts.png"

    if save and not should_generate_figure(output_path, force=force):
        return output_path

    hourly_counts = (
        trips_df.groupby("hour")["ride_id"]
        .count()
        .reset_index(name="ride_count")
    )

    plt.figure()
    sns.barplot(data=hourly_counts, x="hour", y="ride_count", color="steelblue")
    plt.title("Hourly Ride Counts")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Rides")
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_weekday_ride_counts(
    trips_df: pd.DataFrame, save: bool = True, force: bool = False
) -> Path:
    """
    Plot ride counts by weekday.
    """
    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / "weekday_ride_counts.png"

    if save and not should_generate_figure(output_path, force=force):
        return output_path

    # To have a consistent weekday order, we define a categorical type.
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    trips_df = trips_df.copy()
    trips_df["weekday"] = pd.Categorical(
        trips_df["weekday"], categories=weekday_order, ordered=True
    )

    weekday_counts = (
        trips_df.groupby("weekday")["ride_id"]
        .count()
        .reset_index(name="ride_count")
    )

    plt.figure()
    sns.barplot(
        data=weekday_counts,
        x="weekday",
        y="ride_count",
        color="seagreen",
        order=weekday_order,
    )
    plt.title("Ride Counts by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Number of Rides")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_monthly_ride_counts(
    trips_df: pd.DataFrame, save: bool = True, force: bool = False
) -> Path:
    """
    Plot total ride counts for each month.

    This helps summarize seasonality across the year (e.g., higher demand in
    summer months and lower demand in winter months).

    Parameters
    ----------
    trips_df : pd.DataFrame
        Cleaned trip-level data with a 'month' column (month name).
    save : bool
        Whether to save the figure to disk.
    """
    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / "monthly_rides.png"

    if save and not should_generate_figure(output_path, force=force):
        return output_path

    # Ensure months appear in calendar order for readability.
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    df = trips_df.copy()
    df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)

    monthly_counts = (
        df.groupby("month")["ride_id"]
        .count()
        .reset_index(name="ride_count")
    )

    plt.figure()
    sns.barplot(
        data=monthly_counts,
        x="month",
        y="ride_count",
        order=month_order,
        color="slateblue",
    )
    plt.title("Monthly Ride Counts")
    plt.xlabel("Month")
    plt.ylabel("Number of rides")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_member_vs_casual(
    trips_df: pd.DataFrame, save: bool = True, force: bool = False
) -> Path:
    """
    Plot the distribution of rides between member and casual users.
    """
    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / "member_vs_casual_distribution.png"

    if save and not should_generate_figure(output_path, force=force):
        return output_path

    member_counts = (
        trips_df.groupby("member_casual")["ride_id"]
        .count()
        .reset_index(name="ride_count")
    )

    plt.figure()
    sns.barplot(
        data=member_counts,
        x="member_casual",
        y="ride_count",
        palette="viridis",
    )
    plt.title("Member vs Casual Ride Distribution")
    plt.xlabel("Rider Type")
    plt.ylabel("Number of Rides")
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_avg_ride_duration_by_user_type(
    trips_df: pd.DataFrame, save: bool = True, force: bool = False
) -> Path:
    """
    Plot average ride duration (minutes) by user type (member vs casual).

    This visualization is useful because user groups often have different
    riding behavior (e.g., casual users may take longer leisure trips).

    Notes
    -----
    Our preprocessing step creates 'ride_duration_min'. Some datasets may
    call this 'ride_duration', but in this project we standardize to minutes.
    """
    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / "avg_duration_user_type.png"

    if save and not should_generate_figure(output_path, force=force):
        return output_path

    avg_duration = (
        trips_df.groupby("member_casual")["ride_duration_min"]
        .mean()
        .reset_index(name="avg_ride_duration_min")
    )

    plt.figure()
    sns.barplot(
        data=avg_duration,
        x="member_casual",
        y="avg_ride_duration_min",
        palette="mako",
    )
    plt.title("Average Ride Duration by User Type")
    plt.xlabel("User Type")
    plt.ylabel("Average Ride Duration (minutes)")
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_top_start_stations(
    station_trips_df: pd.DataFrame,
    top_n: int = 10,
    save: bool = True,
    force: bool = False,
) -> Path:
    """
    Plot the top N most used start stations.
    """
    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / "top_start_stations.png"

    if save and not should_generate_figure(output_path, force=force):
        return output_path

    station_counts = (
        station_trips_df.groupby("start_station_name")["ride_id"]
        .count()
        .reset_index(name="ride_count")
        .sort_values("ride_count", ascending=False)
        .head(top_n)
    )

    plt.figure()
    sns.barplot(
        data=station_counts,
        x="ride_count",
        y="start_station_name",
        color="cornflowerblue",
    )
    plt.title(f"Top {top_n} Most Used Start Stations")
    plt.xlabel("Number of Rides")
    plt.ylabel("Start Station Name")
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_ride_demand_heatmap_weekday_hour(
    trips_df: pd.DataFrame, save: bool = True, force: bool = False
) -> Path:
    """
    Plot a heatmap of ride demand by weekday (rows) and hour (columns).

    This is a compact way to identify peak commuting hours and differences
    between weekdays and weekends.
    """
    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / "ride_heatmap.png"

    if save and not should_generate_figure(output_path, force=force):
        return output_path

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    df = trips_df.copy()
    df["weekday"] = pd.Categorical(df["weekday"], categories=weekday_order, ordered=True)

    # Pivot table: rows=weekday, columns=hour, values=count of rides
    heatmap_data = (
        df.pivot_table(
            index="weekday",
            columns="hour",
            values="ride_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=weekday_order)
        .sort_index(axis=1)
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.25, linecolor="white")
    plt.title("Ride Demand Heatmap (Weekday vs Hour)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Weekday")
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_ride_duration_distribution(
    trips_df: pd.DataFrame,
    max_duration_min: int = 120,
    save: bool = True,
    force: bool = False,
) -> Path:
    """
    Plot the distribution of ride durations (in minutes).

    We optionally limit the x-axis to a maximum duration to make
    the histogram easier to read (e.g., 0–120 minutes).
    """
    set_plot_style()
    figures_folder = ensure_figures_folder()
    output_path = figures_folder / "ride_duration_distribution.png"

    if save and not should_generate_figure(output_path, force=force):
        return output_path

    # Optionally filter extreme durations for visualization clarity
    subset = trips_df[trips_df["ride_duration_min"] <= max_duration_min]

    plt.figure()
    sns.histplot(
        data=subset,
        x="ride_duration_min",
        bins=50,
        kde=True,
        color="darkorange",
    )
    plt.title("Ride Duration Distribution (minutes)")
    plt.xlabel("Ride Duration (minutes)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def print_basic_summary_statistics(trips_df: pd.DataFrame) -> None:
    """
    Print basic summary statistics useful for the thesis progress report.

    Examples:
    - Total number of rides
    - Time period covered
    - Average / median ride duration
    - Basic counts by rider type (member/casual)
    """
    print("\n===== BASIC SUMMARY STATISTICS =====")

    # Total rides
    total_rides = len(trips_df)
    print(f"Total number of rides: {total_rides:,}")

    # Date range
    min_date = trips_df["date"].min()
    max_date = trips_df["date"].max()
    print(f"Date range: {min_date} to {max_date}")

    # Ride duration stats
    duration_desc = trips_df["ride_duration_min"].describe()
    print("\nRide duration (minutes) summary:")
    print(duration_desc.to_string())

    # Rider type distribution
    if "member_casual" in trips_df.columns:
        print("\nRider type distribution (counts):")
        print(trips_df["member_casual"].value_counts())
        print("\nRider type distribution (percentage):")
        print(trips_df["member_casual"].value_counts(normalize=True) * 100)

    print("====================================\n")


def run_all_plots(
    trips_df: pd.DataFrame,
    station_trips_df: pd.DataFrame,
    force_plots: bool = False,
) -> None:
    """
    Convenience function to generate all required plots.

    Parameters
    ----------
    trips_df : pd.DataFrame
        Cleaned trip-level data.
    station_trips_df : pd.DataFrame
        Station-level subset (non-null start_station_id).
    """
    print("Generating daily ride counts plot...")
    plot_daily_ride_counts(trips_df, force=force_plots)

    print("Generating hourly ride counts plot...")
    plot_hourly_ride_counts(trips_df, force=force_plots)

    print("Generating weekday ride counts plot...")
    plot_weekday_ride_counts(trips_df, force=force_plots)

    print("Generating monthly ride counts plot...")
    plot_monthly_ride_counts(trips_df, force=force_plots)

    print("Generating member vs casual distribution plot...")
    plot_member_vs_casual(trips_df, force=force_plots)

    print("Generating average ride duration by user type plot...")
    plot_avg_ride_duration_by_user_type(trips_df, force=force_plots)

    print("Generating top start stations plot...")
    plot_top_start_stations(station_trips_df, force=force_plots)

    print("Generating weekday vs hour demand heatmap...")
    plot_ride_demand_heatmap_weekday_hour(trips_df, force=force_plots)

    print("Generating ride duration distribution plot...")
    plot_ride_duration_distribution(trips_df, force=force_plots)

    print("All plots saved to:", get_figures_folder())

