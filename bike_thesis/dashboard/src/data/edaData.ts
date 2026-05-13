export type EdaCategory =
  | "user_type"
  | "temporal"
  | "station_based"
  | "distribution";

export interface EdaCategoryMeta {
  key: EdaCategory;
  label: string;
}

export interface EdaFigureEntry {
  title: string;
  image: string;
  interpretation: string;
  category: EdaCategory;
}

export const EDA_CATEGORIES: EdaCategoryMeta[] = [
  { key: "user_type", label: "User Type" },
  { key: "temporal", label: "Temporal" },
  { key: "station_based", label: "Station-Based" },
  { key: "distribution", label: "Distribution" },
];

export const edaFigures: EdaFigureEntry[] = [
  {
    title: "Ride Distribution by User Type",
    image: "/figures/eda/ride_distribution_by_user_type.png",
    category: "user_type",
    interpretation:
      "This figure shows the distribution of rides between member and casual users. The result helps identify which user group contributes more strongly to overall system demand. In the selected 2025 Divvy dataset, member users represent a larger share of total rides, which suggests that regular subscribers play a major role in the demand structure of the system.",
  },
  {
    title: "Average Ride Duration by User Type",
    image: "/figures/eda/average_ride_duration_by_user_type.png",
    category: "user_type",
    interpretation:
      "This figure compares the average ride duration of member and casual users. Casual users generally tend to have longer ride durations, which may indicate more leisure-oriented or less routine usage. Member users, on the other hand, usually make shorter trips, which may be associated with commuting or practical urban mobility.",
  },
  {
    title: "Monthly Ride Counts",
    image: "/figures/eda/monthly_ride_counts.png",
    category: "temporal",
    interpretation:
      "This figure presents the total number of rides by month. The monthly pattern helps reveal seasonal changes in bike-sharing demand. Higher ride counts during warmer months and lower ride counts during colder periods indicate that seasonal mobility behavior has a strong influence on bike-sharing usage.",
  },
  {
    title: "Hourly Ride Counts",
    image: "/figures/eda/hourly_ride_counts.png",
    category: "temporal",
    interpretation:
      "This figure shows how ride demand changes by hour of day. Hourly demand is important because the forecasting task in this thesis is based on station-hour demand. The graph helps identify daily usage peaks and shows that bike-sharing activity follows a structured daily rhythm rather than a random pattern.",
  },
  {
    title: "Weekday-Hour Heatmap",
    image: "/figures/eda/weekday_hour_heatmap.png",
    category: "temporal",
    interpretation:
      "This heatmap combines weekday and hour information to show when demand becomes more concentrated. It provides a clearer view of recurring temporal patterns, such as weekday commuting peaks or broader weekend usage. This visualization supports the decision to use time-based features in the forecasting process.",
  },
  {
    title: "Top 10 Start Stations",
    image: "/figures/eda/top_10_start_stations.png",
    category: "station_based",
    interpretation:
      "This figure shows the most frequently used start stations in the dataset. The result indicates that demand is not evenly distributed across the system. Some stations function as major mobility hubs and generate substantially higher ride activity, which supports the importance of station-level forecasting.",
  },
  {
    title: "Daily Ride Counts",
    image: "/figures/eda/daily_ride_counts.png",
    category: "temporal",
    interpretation:
      "This figure presents daily ride counts across the selected dataset period. It shows short-term fluctuations in overall system usage and helps reveal how demand changes from day to day. These variations confirm that bike-sharing demand is dynamic and requires careful temporal analysis.",
  },
  {
    title: "Ride Duration Distribution",
    image: "/figures/eda/ride_duration_distribution.png",
    category: "distribution",
    interpretation:
      "This figure shows the distribution of ride durations. Most rides are relatively short, while a smaller number of rides last much longer. This pattern is consistent with the expected use of bike-sharing systems for short-distance urban transportation, while also showing the presence of longer occasional trips.",
  },
  {
    title: "Ride Counts by Weekday",
    image: "/figures/eda/ride_counts_by_weekday.png",
    category: "temporal",
    interpretation:
      "This figure summarizes total ride counts by weekday. It helps identify whether demand differs across the days of the week. Differences between weekdays and weekends may reflect commuting behavior, leisure usage, and weekly mobility routines.",
  },
];
