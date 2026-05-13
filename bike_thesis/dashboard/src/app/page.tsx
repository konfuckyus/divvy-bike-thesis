import Link from "next/link";
import {
  ArrowRight,
  Database,
  Target,
  Trophy,
  BarChart3,
  Compass,
  MapPin,
} from "lucide-react";

import { PageHeader } from "@/components/page-header";
import { SummaryCard } from "@/components/summary-card";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const QUICK_LINKS = [
  {
    href: "/comparison",
    title: "Model Comparison",
    description:
      "Compare Linear Regression and Random Forest using RMSE and MAE.",
    icon: BarChart3,
  },
  {
    href: "/explorer",
    title: "Model Explorer",
    description:
      "Inspect actual vs predicted, residuals, and per-station forecasts for each model.",
    icon: Compass,
  },
  {
    href: "/stations",
    title: "Station Forecast Explorer",
    description:
      "Drill into low-, medium-, and high-demand stations across all three models.",
    icon: MapPin,
  },
];

export default function OverviewPage() {
  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Dashboard Overview"
        title="Divvy Bike-Sharing Demand Forecasting Dashboard"
        description="Station-level hourly demand forecasting using the 2025 Divvy Bike Trip dataset."
      />

      <section className="grid gap-4 md:grid-cols-3">
        <SummaryCard
          label="Dataset"
          value="2025 Divvy Bike Trip Dataset"
          description="Trip-level records aggregated into a station-hour demand panel."
          
        />
        <SummaryCard
          label="Target"
          value="Hourly ride starts per station"
          description="Each row in the modeling dataset is a station-hour pair."
          
        />
        <SummaryCard
          label="Best Model"
          value="Random Forest Regressor"
          description="Best performance, especially on the busiest top-20 stations."
          
          accent
        />
      </section>

      <section>
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Demand Forecasting</CardTitle>
            <CardDescription>
              Predictive analytics for station-level hourly bike-sharing demand.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm leading-relaxed text-muted-foreground">
              This dashboard analyzes hourly demand patterns by transforming raw
              trip records into a station-specific dataset. We benchmarked
              different forecasting approaches to identify the most reliable
              model for operational planning.
            </p>
            <p className="text-sm leading-relaxed text-muted-foreground">
              Among the evaluated models(including a simple baseline and Linear
              Regression)the{" "}
              <strong className="font-semibold text-foreground">
                Random Forest Regressor
              </strong>{" "}
              delivered the strongest performance, particularly when capturing
              the complexities of the top-20 busiest stations.
            </p>
          </CardContent>
        </Card>
      </section>

      <section className="space-y-4">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-semibold tracking-tight">
            Continue exploring
          </h2>
          <p className="text-sm text-muted-foreground">
            Jump directly to the sections most relevant to your review.
          </p>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {QUICK_LINKS.map((item) => {
            const Icon = item.icon;
            return (
              <Card
                key={item.href}
                className="flex h-full flex-col justify-between transition-shadow hover:shadow-md"
              >
                <CardHeader>
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 text-accent">
                    <Icon className="h-5 w-5" />
                  </div>
                  <CardTitle className="mt-3 text-base">{item.title}</CardTitle>
                  <CardDescription>{item.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button asChild variant="ghost" className="px-0 text-accent">
                    <Link href={item.href}>
                      Open
                      <ArrowRight className="h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}
