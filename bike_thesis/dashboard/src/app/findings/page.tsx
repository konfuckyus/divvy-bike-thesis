import { CheckCircle2, TriangleAlert } from "lucide-react";

import { PageHeader } from "@/components/page-header";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const FINDINGS = [
  {
    title: "Random Forest produced the strongest forecasting behavior",
    body: "Random Forest performed better than the simpler comparison models in the current forecasting setup. The model was more successful at following the general structure of station-level hourly demand and produced more balanced predictions, especially in the busiest station scenario. This suggests that bike-sharing demand contains nonlinear patterns that are difficult to capture with a purely linear model.",
  },
  {
    title: "Baseline was useful only as a reference model",
    body: "The baseline approach was important because it provided a simple benchmark for comparison. However, the visual results showed that it was not reliable enough as a main forecasting model, especially for high-demand stations. In busy station examples, the baseline predictions became unstable and sometimes exaggerated the demand pattern.",
  },
  {
    title: "Linear Regression was interpretable but limited",
    body: "Linear Regression provided a simple and understandable comparison model. It helped show whether the station-hour demand structure could be represented with a basic linear relationship. However, the results indicated that the demand pattern was more complex than a linear model could fully explain. This made Linear Regression useful for comparison but weaker than Random Forest.",
  },
  {
    title: "Station-level forecasting is operationally meaningful",
    body: "The station-hour structure made the forecasting task more relevant for bike-sharing operations. Instead of predicting only total system demand, the project focused on estimating demand at specific stations and hours. This is more useful for understanding local shortage risk, busy stations, and potential redistribution needs.",
  },
  {
    title: "Sudden peaks remain difficult to predict",
    body: "Even the strongest model did not perfectly capture all sudden demand peaks. This shows that station-level bike-sharing demand can be irregular and difficult to forecast precisely. The current model is more useful for estimating general short-term demand behavior than for perfectly predicting every local spike.",
  },
];

const LIMITATIONS = [
  {
    title: "Only the 2025 Divvy trip data was used",
    body: "The study is limited to the 2025 Divvy Bike Trip dataset. This makes the project more focused and manageable, but it also means that the models learn patterns from a single year only. Longer historical data could help the model better understand year-to-year changes, unusual seasonal differences, and longer-term demand trends.",
  },
  {
    title: "The dataset contains trip records, not real-time bike inventory",
    body: "The Divvy trip dataset shows completed rides, including when and where rides started and ended. However, it does not provide real-time information about how many bikes were available at each station at every hour. Because of this, the project can estimate ride-start demand, but it cannot directly determine whether a station was actually empty, full, or close to a shortage at a specific time.",
  },
  {
    title: "Station imbalance can only be inferred indirectly",
    body: "Since the available data is based on ride activity rather than live station capacity, station imbalance cannot be measured perfectly. High ride-start demand may suggest stronger usage pressure at a station, but it does not automatically prove that the station experienced a bike shortage. Similarly, end-station activity may suggest accumulation, but exact fullness cannot be confirmed without station inventory or dock availability data.",
  },
  {
    title: "Some stations have sparse or irregular demand",
    body: "Not all stations have the same level of activity. Some stations are used frequently and show clearer demand patterns, while others have low or irregular usage. This makes station-level forecasting harder, especially for low-demand stations where small changes can strongly affect the visual pattern and model performance.",
  },
  {
    title: "The current model predicts demand, not complete operational decisions",
    body: "The forecasting models estimate the expected number of ride starts for station-hour combinations. However, operational decisions such as when to rebalance bikes, how many bikes to move, and which vehicle route to use require additional information such as station capacity, current inventory, dock availability, and redistribution resources. These are outside the current dataset.",
  },
];

export default function FindingsPage() {
  return (
    <div className="space-y-12">
      <PageHeader
        eyebrow="Findings & Limitations"
        title="Summary of results and current scope"
        description="A read-out of what the forecasting study showed and the dataset and modeling constraints that shape these results."
      />

      <section className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-semibold tracking-tight">Key findings</h2>
          <p className="text-sm text-muted-foreground">
            Conclusions drawn from the comparison between the baseline, Linear
            Regression, and Random Forest models on the station-hour demand
            dataset.
          </p>
        </div>
        <div className="grid gap-5 md:grid-cols-2">
          {FINDINGS.map((finding, index) => (
            <Card key={finding.title} className="flex h-full flex-col">
              <CardHeader className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent/10 text-accent">
                    <CheckCircle2 className="h-4 w-4" />
                  </div>
                  <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Key Finding {index + 1}
                  </span>
                </div>
                <CardTitle className="text-base leading-snug">
                  {finding.title}
                </CardTitle>
              </CardHeader>
              <CardContent className="flex-1">
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {finding.body}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <section className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-semibold tracking-tight">Limitations</h2>
          <p className="text-sm text-muted-foreground">
            Constraints that come from the dataset and the chosen modeling
            scope. These define the boundary of what the current study can and
            cannot answer.
          </p>
        </div>
        <div className="grid gap-5 md:grid-cols-2">
          {LIMITATIONS.map((limit, index) => (
            <Card key={limit.title} className="flex h-full flex-col">
              <CardHeader className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-amber-500/10 text-amber-700">
                    <TriangleAlert className="h-4 w-4" />
                  </div>
                  <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Limitation {index + 1}
                  </span>
                </div>
                <CardTitle className="text-base leading-snug">
                  {limit.title}
                </CardTitle>
              </CardHeader>
              <CardContent className="flex-1">
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {limit.body}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
