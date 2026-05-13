export type ModelKey = "baseline" | "linear_regression" | "random_forest";
export type DemandGroup = "low" | "medium" | "high";
export type GraphType =
  | "actual_vs_predicted"
  | "residual_distribution"
  | "low_demand_station"
  | "medium_demand_station"
  | "high_demand_station";

export interface ModelMeta {
  key: ModelKey;
  label: string;
  shortLabel: string;
  description: string;
}

export interface GraphTypeMeta {
  key: GraphType;
  label: string;
  category: "diagnostic" | "station";
}

export interface FigureEntry {
  model: ModelKey;
  graphType: GraphType;
  demandGroup?: DemandGroup;
  title: string;
  image: string;
  interpretation: string;
  // Whether the figure is featured on the main dashboard pages.
  // Figures with `usedInMainDashboard: false` are surfaced through the
  // Forecasting Appendix page as supporting evidence.
  usedInMainDashboard: boolean;
}

export const MODELS: ModelMeta[] = [
  {
    key: "baseline",
    label: "Baseline",
    shortLabel: "Baseline",
    description:
      "Simple reference model used as a minimum performance benchmark.",
  },
  {
    key: "linear_regression",
    label: "Linear Regression",
    shortLabel: "Linear Reg.",
    description: "Interpretable linear baseline for hourly demand.",
  },
  {
    key: "random_forest",
    label: "Random Forest",
    shortLabel: "Random Forest",
    description:
      "Tree ensemble trained on the busiest top-20 stations, the best performing model.",
  },
];

export const GRAPH_TYPES: GraphTypeMeta[] = [
  {
    key: "actual_vs_predicted",
    label: "Actual vs Predicted",
    category: "diagnostic",
  },
  {
    key: "residual_distribution",
    label: "Residual Distribution",
    category: "diagnostic",
  },
  {
    key: "low_demand_station",
    label: "Low-Demand Station Forecast",
    category: "station",
  },
  {
    key: "medium_demand_station",
    label: "Medium-Demand Station Forecast",
    category: "station",
  },
  {
    key: "high_demand_station",
    label: "High-Demand Station Forecast",
    category: "station",
  },
];

export const DEMAND_GROUPS: { key: DemandGroup; label: string }[] = [
  { key: "low", label: "Low Demand" },
  { key: "medium", label: "Medium Demand" },
  { key: "high", label: "High Demand" },
];

export const figures: FigureEntry[] = [
  // ----------------------------- Baseline -----------------------------
  {
    model: "baseline",
    graphType: "actual_vs_predicted",
    title: "Baseline — Actual vs Predicted",
    image: "/figures/new_forecast_actual_vs_predicted_scatter_baseline.png",
    interpretation:
      "This scatter plot shows the relationship between actual demand values and predicted values for the baseline model. In an ideal model, the points are expected to be concentrated around the reference line. However, in the baseline model, the points are more scattered, and the alignment becomes weaker especially at higher demand levels. This indicates that the baseline approach provides a general reference but is not strong enough as a main forecasting model for a variable station-level hourly demand problem.",
    usedInMainDashboard: true,
  },
  {
    model: "baseline",
    graphType: "residual_distribution",
    title: "Baseline — Residual Distribution",
    image: "/figures/new_forecast_residual_histogram_baseline.png",
    interpretation:
      "This figure shows the distribution of prediction errors for the baseline model. A better model would produce errors that are concentrated close to zero within a narrow range. In the baseline model, the residual values are more widely spread, which shows that the model misses actual demand noticeably in some time intervals. This result supports the idea that the baseline model should mainly be used for comparison purposes and that more advanced models are necessary.",
    usedInMainDashboard: false,
  },
  {
    model: "baseline",
    graphType: "low_demand_station",
    demandGroup: "low",
    title: "Baseline — Low-Demand Station (CHI02152)",
    image:
      "/figures/new_forecast_station_timeseries_baseline_low_station_CHI02152.png",
    interpretation:
      "This figure shows the forecasting behavior of the baseline model over time for a selected low-demand station. Since actual values in low-demand stations usually change within a small range, the model predictions may remain flatter and more limited. Although the baseline approach can capture some general levels, it cannot strongly follow local fluctuations and small sudden changes. This indicates that even in low-demand stations, simple historical approaches have limited explanatory power.",
    usedInMainDashboard: false,
  },
  {
    model: "baseline",
    graphType: "medium_demand_station",
    demandGroup: "medium",
    title: "Baseline — Medium-Demand Station (CHI02114)",
    image:
      "/figures/new_forecast_station_timeseries_baseline_medium_station_CHI02114.png",
    interpretation:
      "This figure compares actual and predicted demand for the baseline model at a medium-demand station. Medium-demand stations are more informative for model evaluation because they contain more activity than low-demand stations. Although the baseline model appears to follow some general movements, its prediction line does not capture local changes in a sufficiently balanced way. Therefore, the baseline model should still be considered mainly as a simple reference point for medium-demand stations.",
    usedInMainDashboard: false,
  },
  {
    model: "baseline",
    graphType: "high_demand_station",
    demandGroup: "high",
    title: "Baseline — High-Demand Station (CHI01747)",
    image:
      "/figures/new_forecast_station_timeseries_baseline_high_station_CHI01747.png",
    interpretation:
      "This figure shows the prediction performance of the baseline model for a high-demand station. High-demand stations are operationally critical because prediction errors at these stations may directly lead to bike shortages or station overflow problems. The graph shows that baseline predictions are more unstable and sometimes produce exaggerated fluctuations compared with the actual demand structure. This result indicates that the baseline model is not suitable as the main forecasting solution for busy station scenarios.",
    usedInMainDashboard: true,
  },

  // -------------------------- Linear Regression --------------------------
  {
    model: "linear_regression",
    graphType: "actual_vs_predicted",
    title: "Linear Regression — Actual vs Predicted",
    image:
      "/figures/new_forecast_actual_vs_predicted_scatter_linearregression.png",
    interpretation:
      "This figure shows the relationship between actual and predicted demand values for the Linear Regression model. Linear Regression provides a more structured prediction pattern than the baseline model, but the points are not fully concentrated around the reference line. This indicates that station-level hourly demand cannot be explained completely with a linear structure. The model captures the general tendency to some extent, but it remains limited for demand values that include high variability or sudden increases.",
    usedInMainDashboard: false,
  },
  {
    model: "linear_regression",
    graphType: "residual_distribution",
    title: "Linear Regression — Residual Distribution",
    image: "/figures/new_forecast_residual_histogram_linearregression.png",
    interpretation:
      "This residual distribution shows how the prediction errors of the Linear Regression model are spread. Although the concentration of errors around zero is a positive sign, the width of the distribution indicates that the model cannot produce stable predictions in every case. Linear Regression is useful because it is simple and interpretable; however, for a problem such as bike-sharing demand, which depends on both time and station-level variation, the linear relationship assumption may not be sufficient.",
    usedInMainDashboard: false,
  },
  {
    model: "linear_regression",
    graphType: "low_demand_station",
    demandGroup: "low",
    title: "Linear Regression — Low-Demand Station (CHI00554)",
    image:
      "/figures/new_forecast_station_timeseries_linearregression_low_station_CHI00554.png",
    interpretation:
      "This figure shows Linear Regression predictions for a low-demand station. In low-demand stations, demand values usually change within a limited range, so the model may produce more stable predictions. However, this stable behavior can also mean that the model does not follow small local fluctuations sufficiently. Therefore, although Linear Regression can capture the general demand level in low-demand stations, it remains limited in explaining detailed station-hour changes.",
    usedInMainDashboard: false,
  },
  {
    model: "linear_regression",
    graphType: "medium_demand_station",
    demandGroup: "medium",
    title: "Linear Regression — Medium-Demand Station (CHI01050)",
    image:
      "/figures/new_forecast_station_timeseries_linearregression_medium_station_CHI01050.png",
    interpretation:
      "This figure shows the behavior of the Linear Regression model over time for a medium-demand station. The model can capture the general demand level to some extent, but it cannot adapt equally well to sharper increases and decreases in the actual series. This shows that Linear Regression is valuable as a simple and interpretable comparison model, but it is not as flexible as Random Forest in capturing more complex station-level demand behavior.",
    usedInMainDashboard: false,
  },
  {
    model: "linear_regression",
    graphType: "high_demand_station",
    demandGroup: "high",
    title: "Linear Regression — High-Demand Station (CHI01747)",
    image:
      "/figures/new_forecast_station_timeseries_linearregression_high_station_CHI01747.png",
    interpretation:
      "This figure shows the predictions of the Linear Regression model for a high-demand station. In high-demand stations, demand is more variable, which makes the limitations of the linear model more visible. Although Linear Regression can estimate some general levels, it cannot sufficiently capture sudden demand increases and local fluctuations. This result shows that a purely linear model may not be sufficient as the main solution for station-level forecasting.",
    usedInMainDashboard: false,
  },

  // ----------------------------- Random Forest -----------------------------
  {
    model: "random_forest",
    graphType: "actual_vs_predicted",
    title: "Random Forest — Actual vs Predicted",
    image:
      "/figures/new_forecast_actual_vs_predicted_scatter_random_forest_busy_top20.png",
    interpretation:
      "This scatter plot shows the relationship between actual and predicted demand values for the Random Forest model in the busiest top 20 stations scenario. The more balanced distribution of points and their closer position to the reference line indicate that the model captures the general demand structure better than Linear Regression and the baseline approach. However, some deviations still remain at higher demand values. This result shows that Random Forest is the strongest model in the current setup, although it does not completely solve sudden and high demand peaks.",
    usedInMainDashboard: true,
  },
  {
    model: "random_forest",
    graphType: "residual_distribution",
    title: "Random Forest — Residual Distribution",
    image:
      "/figures/new_forecast_residual_histogram_random_forest_busy_top20.png",
    interpretation:
      "This figure shows the distribution of prediction errors for the Random Forest model in the busiest top 20 stations scenario. The residual values are more concentrated around zero and within a narrower range, which indicates that the model produces more controlled prediction errors. This supports the conclusion that Random Forest represents the station-hour demand structure in a more balanced way. However, the distribution is not perfectly centered around zero, which shows that the model can still make errors, especially during sudden demand changes.",
    usedInMainDashboard: false,
  },
  {
    model: "random_forest",
    graphType: "low_demand_station",
    demandGroup: "low",
    title: "Random Forest — Low-Demand Station (CHI00498)",
    image:
      "/figures/new_forecast_station_timeseries_random_forest_busy_top20_low_station_CHI00498.png",
    interpretation:
      "This figure shows Random Forest predictions for a low-demand station in the busiest top 20 stations scenario. The model follows the general level of demand in a controlled way within the low-demand range. However, in low-demand stations, even small numerical changes can become visually noticeable, making every small fluctuation difficult to capture precisely. Despite this, Random Forest provides a more balanced station-level forecasting behavior compared with the baseline and Linear Regression models.",
    usedInMainDashboard: false,
  },
  {
    model: "random_forest",
    graphType: "medium_demand_station",
    demandGroup: "medium",
    title: "Random Forest — Medium-Demand Station (CHI01742)",
    image:
      "/figures/new_forecast_station_timeseries_random_forest_busy_top20_medium_station_CHI01742.png",
    interpretation:
      "This figure shows how well the Random Forest model follows the actual demand series for a medium-demand station. Medium-demand stations are important for model evaluation because they contain enough activity while usually being less extreme than high-demand stations. The graph shows that Random Forest follows the general demand pattern and keeps its predictions more balanced. This indicates that the model produces usable forecasting behavior for medium-intensity operational scenarios.",
    usedInMainDashboard: false,
  },
  {
    model: "random_forest",
    graphType: "high_demand_station",
    demandGroup: "high",
    title: "Random Forest — High-Demand Station (CHI01747)",
    image:
      "/figures/new_forecast_station_timeseries_random_forest_busy_top20_high_station_CHI01747.png",
    interpretation:
      "This figure shows the forecasting performance of the Random Forest model for a high-demand station. High-demand stations are the most critical points in bike-sharing operations because incorrect demand estimation at these stations may directly affect user satisfaction and bike redistribution decisions. Random Forest follows the general direction of the actual demand structure better than the baseline and Linear Regression models. However, some sudden peak values are still not fully captured. This indicates that the model provides an operationally useful forecasting framework, but it could still be improved with further feature engineering or additional models.",
    usedInMainDashboard: true,
  },
];

// --------------------- Helpers ---------------------
export function getFigure(
  model: ModelKey,
  graphType: GraphType,
): FigureEntry | undefined {
  return figures.find((f) => f.model === model && f.graphType === graphType);
}

export function getAppendixFigures(): FigureEntry[] {
  return figures.filter((f) => !f.usedInMainDashboard);
}

export function getStationFigure(
  model: ModelKey,
  demandGroup: DemandGroup,
): FigureEntry | undefined {
  const targetGraph: GraphType =
    demandGroup === "low"
      ? "low_demand_station"
      : demandGroup === "medium"
        ? "medium_demand_station"
        : "high_demand_station";
  return figures.find(
    (f) => f.model === model && f.graphType === targetGraph,
  );
}

export const MODEL_COMPARISON_FIGURES = {
  rmse: {
    title: "RMSE Model Comparison",
    image: "/figures/new_forecast_model_comparison_rmse.png",
    interpretation:
      "This figure compares the RMSE values of the tested models. RMSE is important because it penalizes larger prediction errors more strongly, which helps evaluate how robust a model is against sudden or large deviations. According to the results, Random Forest produced a lower RMSE value than Linear Regression. This indicates that Random Forest captured stronger variations in the station-hour demand structure more effectively. Since high-demand periods are operationally critical in bike-sharing systems, reducing larger errors increases the practical value of the model.",
  },
  mae: {
    title: "MAE Model Comparison",
    image: "/figures/new_forecast_model_comparison_mae.png",
    interpretation:
      "This figure compares the MAE values of the tested models. MAE directly shows how far the predictions are from the actual values on average. The lower MAE value of Random Forest indicates that its predictions were closer to the actual station-hour demand values compared with Linear Regression. This result supports the idea that Random Forest provides a more balanced performance not only in terms of large errors, but also in terms of average prediction accuracy.",
  },
};
