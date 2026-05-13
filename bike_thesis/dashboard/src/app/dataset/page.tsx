import { ArrowRight, Database, Layers3, Sparkles, LineChart } from "lucide-react";

import { PageHeader } from "@/components/page-header";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

const PIPELINE_STEPS = [
  {
    title: "Raw Trip Records",
    description:
      "Each row is a single Divvy ride with start time, station, user type and duration.",
    icon: Database,
  },
  {
    title: "Station-Hour Aggregation",
    description:
      "Rides are grouped by start station and hourly buckets to count ride starts.",
    icon: Layers3,
  },
  {
    title: "Forecasting Dataset",
    description:
      "Final station-hour panel; the target is the number of ride starts in the hour.",
    icon: Sparkles,
  },
  {
    title: "Model Evaluation",
    description:
      "Baseline, Linear Regression, and Random Forest compared with RMSE & MAE.",
    icon: LineChart,
  },
];

export default function DatasetPage() {
  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Dataset Transformation"
        title="From trip-level rides to a station-hour forecasting dataset"
        description="The original Divvy trip records were reshaped into a panel suitable for hourly demand forecasting."
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Processing pipeline</CardTitle>
          <CardDescription>
            High-level flow used to prepare the modeling dataset.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ol className="grid gap-4 md:grid-cols-4">
            {PIPELINE_STEPS.map((step, index) => {
              const Icon = step.icon;
              const isLast = index === PIPELINE_STEPS.length - 1;
              return (
                <li
                  key={step.title}
                  className={cn(
                    "relative flex flex-col gap-3 rounded-lg border border-border bg-secondary/30 p-4",
                  )}
                >
                  <div className="flex items-center gap-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-md bg-accent/10 text-accent">
                      <Icon className="h-4 w-4" />
                    </div>
                    <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Step {index + 1}
                    </span>
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-foreground">
                      {step.title}
                    </h3>
                    <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                      {step.description}
                    </p>
                  </div>
                  {!isLast ? (
                    <ArrowRight className="absolute -right-3 top-1/2 hidden h-5 w-5 -translate-y-1/2 text-muted-foreground md:block" />
                  ) : null}
                </li>
              );
            })}
          </ol>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Explanation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm leading-relaxed text-muted-foreground">
          <p>
            The original Divvy trip records were organized at ride level, where
            each row represented a single ride. For forecasting, the data was
            aggregated by station and hour.
          </p>
          <p>
            In the final representation, each row corresponds to a specific
            station-hour pair, and the target variable is the number of ride
            starts in that interval. This structure exposes the temporal
            patterns the models try to capture (daily and weekly seasonality,
            per-station baselines, etc.).
          </p>
          <p>
            The same transformed dataset is used as the input to all three
            models compared in this dashboard, so the performance differences
            reflect the modeling approach rather than the data preparation.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Schema at a glance</CardTitle>
          <CardDescription>
            Illustrative columns produced by the aggregation step.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-hidden rounded-lg border border-border">
            <table className="w-full border-collapse text-sm">
              <thead className="bg-secondary/60 text-xs uppercase tracking-wider text-muted-foreground">
                <tr>
                  <th className="px-4 py-2 text-left font-medium">Column</th>
                  <th className="px-4 py-2 text-left font-medium">Type</th>
                  <th className="px-4 py-2 text-left font-medium">
                    Description
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {[
                  {
                    column: "station_id",
                    type: "string",
                    desc: "Divvy start-station identifier (e.g. CHI01747).",
                  },
                  {
                    column: "datetime_hour",
                    type: "timestamp",
                    desc: "Truncated start hour of the interval.",
                  },
                  {
                    column: "hour_of_day",
                    type: "int",
                    desc: "0–23 hour component, used as a temporal feature.",
                  },
                  {
                    column: "day_of_week",
                    type: "int",
                    desc: "Weekday index, captures weekly seasonality.",
                  },
                  {
                    column: "ride_starts",
                    type: "int",
                    desc: "Target variable — ride count in this station-hour.",
                  },
                ].map((row) => (
                  <tr key={row.column} className="bg-card">
                    <td className="px-4 py-2 font-mono text-xs text-foreground">
                      {row.column}
                    </td>
                    <td className="px-4 py-2 text-xs text-muted-foreground">
                      {row.type}
                    </td>
                    <td className="px-4 py-2 text-xs text-muted-foreground">
                      {row.desc}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
