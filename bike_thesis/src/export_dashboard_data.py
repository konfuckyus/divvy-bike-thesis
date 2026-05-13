import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def get_export_dir() -> Path:
    export_dir = get_project_root() / "outputs" / "dashboard_data"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir

def export_summary_cards(trips_clean: pd.DataFrame, best_model_name: str = "RandomForestRegressor", experiment: str = "busy_top20"):
    export_dir = get_export_dir()
    total_rides = int(len(trips_clean))
    num_stations = int(trips_clean["start_station_name"].nunique())

    summary = {
        "total_rides": total_rides,
        "num_stations": num_stations,
        "forecast_level": "hourly",
        "best_model": best_model_name,
        "main_experiment": experiment
    }

    with open(export_dir / "summary_cards.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Exported summary_cards.json")

def export_monthly_rides(trips_clean: pd.DataFrame):
    export_dir = get_export_dir()
    monthly = trips_clean.groupby("month").size().reset_index(name="ride_count")
    monthly.to_csv(export_dir / "monthly_rides.csv", index=False)
    monthly.to_json(export_dir / "monthly_rides.json", orient="records", indent=4)
    print("Exported monthly_rides")

def export_hourly_rides(trips_clean: pd.DataFrame):
    export_dir = get_export_dir()
    hourly = trips_clean.groupby("hour").size().reset_index(name="ride_count")
    hourly.to_csv(export_dir / "hourly_rides.csv", index=False)
    hourly.to_json(export_dir / "hourly_rides.json", orient="records", indent=4)
    print("Exported hourly_rides")

def export_heatmap_data(trips_clean: pd.DataFrame):
    export_dir = get_export_dir()
    heatmap = trips_clean.groupby(["weekday", "hour"]).size().reset_index(name="ride_count")
    heatmap.to_csv(export_dir / "heatmap_data.csv", index=False)
    heatmap.to_json(export_dir / "heatmap_data.json", orient="records", indent=4)
    print("Exported heatmap_data")

def export_rider_type_distribution(trips_clean: pd.DataFrame):
    export_dir = get_export_dir()
    rider_type = trips_clean.groupby("member_casual").size().reset_index(name="ride_count")
    rider_type.to_csv(export_dir / "rider_type_dist.csv", index=False)
    rider_type.to_json(export_dir / "rider_type_dist.json", orient="records", indent=4)
    print("Exported rider_type_dist")

def export_top_start_stations(trips_clean: pd.DataFrame, top_n: int = 15):
    export_dir = get_export_dir()
    top_stations = trips_clean.groupby("start_station_name").size().reset_index(name="ride_count")
    top_stations = top_stations.sort_values("ride_count", ascending=False).head(top_n)
    top_stations.to_csv(export_dir / "top_start_stations.csv", index=False)
    top_stations.to_json(export_dir / "top_start_stations.json", orient="records", indent=4)
    print("Exported top_start_stations")

def export_station_summaries(trips_clean: pd.DataFrame):
    export_dir = get_export_dir()
    
    # Needs a complete hour range to get accurate baseline demand
    # but for a simple UI export, total rides and avg_hourly_demand can be approximated
    # Calculate total duration in hours for the dataset
    min_date = pd.to_datetime(trips_clean["started_at"]).min()
    max_date = pd.to_datetime(trips_clean["started_at"]).max()
    total_hours = max(1, (max_date - min_date).total_seconds() / 3600)

    station_stats = trips_clean.groupby(["start_station_id", "start_station_name"]).size().reset_index(name="total_rides")
    station_stats["avg_hourly_demand"] = station_stats["total_rides"] / total_hours
    
    # Assign demand groups based on quantiles to mimic our ML logic
    q33 = station_stats["total_rides"].quantile(0.33)
    q66 = station_stats["total_rides"].quantile(0.66)
    
    def assign_group(rides):
        if rides > q66: return "high"
        elif rides > q33: return "medium"
        else: return "low"
    
    station_stats["demand_group"] = station_stats["total_rides"].apply(assign_group)
    station_stats.to_csv(export_dir / "station_summaries.csv", index=False)
    station_stats.to_json(export_dir / "station_summaries.json", orient="records", indent=4)
    
    # Avg demand by hour for each station
    station_hour = trips_clean.groupby(["start_station_id", "hour"]).size().reset_index(name="total_rides")
    # approximate daily count over the dataset
    total_days = max(1, total_hours / 24)
    station_hour["avg_demand"] = station_hour["total_rides"] / total_days
    station_hour.to_csv(export_dir / "station_avg_hour.csv", index=False)
    station_hour.to_json(export_dir / "station_avg_hour.json", orient="records", indent=4)

    # Avg demand by weekday for each station
    station_weekday = trips_clean.groupby(["start_station_id", "weekday"]).size().reset_index(name="total_rides")
    total_weeks = max(1, total_days / 7)
    station_weekday["avg_demand"] = station_weekday["total_rides"] / total_weeks
    station_weekday.to_csv(export_dir / "station_avg_weekday.csv", index=False)
    station_weekday.to_json(export_dir / "station_avg_weekday.json", orient="records", indent=4)
    
    print("Exported station summaries")

def export_model_comparison():
    export_dir = get_export_dir()
    metrics_file = get_project_root() / "outputs" / "metrics" / "model_comparison_busy_top20.csv"
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        df.to_csv(export_dir / "model_metrics.csv", index=False)
        df.to_json(export_dir / "model_metrics.json", orient="records", indent=4)
        print("Exported model_metrics")
    else:
        print(f"Warning: {metrics_file} not found. Skipping export_model_comparison.")

def export_forecast_samples():
    export_dir = get_export_dir()
    preds_file = get_project_root() / "outputs" / "predictions" / "random_forest_predictions_busy_top20.csv"
    examples_file = get_project_root() / "outputs" / "metrics" / "selected_example_stations.csv"
    
    if preds_file.exists():
        preds_df = pd.read_csv(preds_file)
        if "residual" not in preds_df.columns and "actual_demand" in preds_df.columns and "predicted_demand" in preds_df.columns:
            preds_df["residual"] = preds_df["actual_demand"] - preds_df["predicted_demand"]
            
        # Overall residuals/actuals could be huge, so we sample or just provide the specific stations
        
        # Export actual vs predicted for example stations
        if examples_file.exists():
            examples_df = pd.read_csv(examples_file)
            station_ids = examples_df["station_id"].unique()
            
            sample_preds = preds_df[preds_df["station_id"].isin(station_ids)].copy()
            # formatting datetime
            if "timestamp" in sample_preds.columns:
                sample_preds["timestamp"] = sample_preds["timestamp"].astype(str)
                
            sample_preds.to_csv(export_dir / "forecast_samples.csv", index=False)
            sample_preds.to_json(export_dir / "forecast_samples.json", orient="records", indent=4)
            print("Exported forecast_samples")
        else:
            print("Warning: selected_example_stations.csv not found.")
    else:
        print(f"Warning: {preds_file} not found. Skipping export_forecast_samples.")

def export_top_routes(trips_clean: pd.DataFrame, top_n: int = 20):
    export_dir = get_export_dir()
    # Filter out circular trips if needed, but not strictly requested
    # Group by route info
    route_cols = [
        "start_station_name", "end_station_name",
        "start_lat", "start_lng", "end_lat", "end_lng"
    ]
    
    # We want average ride duration and member/casual share over these routes
    routes = trips_clean.groupby(route_cols).agg(
        route_count=("ride_id", "count"),
        avg_ride_duration=("ride_duration_min", "mean"),
        member_count=("member_casual", lambda x: (x == "member").sum())
    ).reset_index()
    
    routes["member_share"] = np.round(routes["member_count"] / routes["route_count"], 2)
    routes["casual_share"] = 1.0 - routes["member_share"]
    routes.drop(columns=["member_count"], inplace=True)
    
    top_routes = routes.sort_values(by="route_count", ascending=False).head(top_n)
    
    top_routes.to_csv(export_dir / "top_routes.csv", index=False)
    top_routes.to_json(export_dir / "top_routes.json", orient="records", indent=4)
    print("Exported top_routes")

def run_all_dashboard_exports(trips_clean: pd.DataFrame):
    print("\n--- Running Dashboard Data Exports ---")
    export_summary_cards(trips_clean)
    export_monthly_rides(trips_clean)
    export_hourly_rides(trips_clean)
    export_heatmap_data(trips_clean)
    export_rider_type_distribution(trips_clean)
    export_top_start_stations(trips_clean)
    export_station_summaries(trips_clean)
    export_model_comparison()
    export_forecast_samples()
    export_top_routes(trips_clean)
    print("--- Dashboard Data Exports Complete ---\n")

if __name__ == "__main__":
    # Test script standalone functionality
    root = get_project_root()
    cleaned_path = root / "data" / "processed" / "cleaned_2025_data.csv"
    if not cleaned_path.exists():
        cleaned_path = root / "data" / "processed" / "divvy_trips_2025_cleaned.csv"
        
    if cleaned_path.exists():
        print(f"Loading {cleaned_path} for dashboard exports...")
        trips_df = pd.read_csv(cleaned_path, low_memory=False)
        run_all_dashboard_exports(trips_df)
    else:
        print(f"Could not find cleaned data. Please run main pipeline first.")
