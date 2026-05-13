"use client";

import { useMemo, useState } from "react";

import { PageHeader } from "@/components/page-header";
import { FigureImage } from "@/components/figure-image";
import { InterpretationBox } from "@/components/interpretation-box";
import { ModelSelector } from "@/components/model-selector";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  DEMAND_GROUPS,
  DemandGroup,
  GRAPH_TYPES,
  GraphType,
  MODELS,
  ModelKey,
  getAppendixFigures,
} from "@/data/figuresData";
import { cn } from "@/lib/utils";

type ModelFilter = "all" | ModelKey;
type GraphFilter = "all" | "actual_vs_predicted" | "residual_distribution" | "station_forecast";
type DemandFilter = "all" | DemandGroup;

const STATION_GRAPHS: GraphType[] = [
  "low_demand_station",
  "medium_demand_station",
  "high_demand_station",
];

function Badge({
  children,
  variant = "default",
}: {
  children: React.ReactNode;
  variant?: "default" | "accent" | "muted";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-medium",
        variant === "default" && "border-border bg-secondary text-foreground",
        variant === "accent" && "border-accent/30 bg-accent/10 text-accent",
        variant === "muted" &&
          "border-border bg-muted text-muted-foreground",
      )}
    >
      {children}
    </span>
  );
}

function modelLabel(key: ModelKey) {
  return MODELS.find((m) => m.key === key)?.label ?? key;
}

function graphLabel(key: GraphType) {
  return GRAPH_TYPES.find((g) => g.key === key)?.label ?? key;
}

function demandLabel(key: DemandGroup) {
  return DEMAND_GROUPS.find((g) => g.key === key)?.label ?? key;
}

export default function AppendixPage() {
  const [modelFilter, setModelFilter] = useState<ModelFilter>("all");
  const [graphFilter, setGraphFilter] = useState<GraphFilter>("all");
  const [demandFilter, setDemandFilter] = useState<DemandFilter>("all");

  const allAppendix = useMemo(() => getAppendixFigures(), []);

  const filtered = useMemo(() => {
    return allAppendix.filter((f) => {
      if (modelFilter !== "all" && f.model !== modelFilter) return false;
      if (graphFilter !== "all") {
        if (graphFilter === "station_forecast") {
          if (!STATION_GRAPHS.includes(f.graphType)) return false;
        } else if (f.graphType !== graphFilter) {
          return false;
        }
      }
      if (demandFilter !== "all") {
        if (f.demandGroup !== demandFilter) return false;
      }
      return true;
    });
  }, [allAppendix, modelFilter, graphFilter, demandFilter]);

  const modelOptions: { value: ModelFilter; label: string }[] = [
    { value: "all", label: "All models" },
    ...MODELS.map((m) => ({ value: m.key as ModelFilter, label: m.label })),
  ];
  const graphOptions: { value: GraphFilter; label: string }[] = [
    { value: "all", label: "All graph types" },
    { value: "actual_vs_predicted", label: "Actual vs Predicted" },
    { value: "residual_distribution", label: "Residual Distribution" },
    { value: "station_forecast", label: "Station Forecast" },
  ];
  const demandOptions: { value: DemandFilter; label: string }[] = [
    { value: "all", label: "All demand groups" },
    ...DEMAND_GROUPS.map((g) => ({
      value: g.key as DemandFilter,
      label: g.label,
    })),
  ];

  const hasFilters =
    modelFilter !== "all" || graphFilter !== "all" || demandFilter !== "all";

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Forecasting Appendix"
        title="Supporting forecasting figures"
        description="Additional diagnostic charts and per-station forecasts that did not make it onto the main dashboard pages. Use the filters to narrow the view by model, graph type, or demand group."
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Filters</CardTitle>
          <CardDescription>
            Showing {filtered.length} of {allAppendix.length} appendix figures.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4 md:flex-row md:items-end md:flex-wrap">
          <ModelSelector
            label="Model"
            value={modelFilter}
            options={modelOptions}
            onChange={setModelFilter}
          />
          <ModelSelector
            label="Graph type"
            value={graphFilter}
            options={graphOptions}
            onChange={setGraphFilter}
          />
          <ModelSelector
            label="Demand group"
            value={demandFilter}
            options={demandOptions}
            onChange={setDemandFilter}
          />
          {hasFilters ? (
            <button
              type="button"
              onClick={() => {
                setModelFilter("all");
                setGraphFilter("all");
                setDemandFilter("all");
              }}
              className="ml-auto text-xs font-medium text-accent hover:underline"
            >
              Reset filters
            </button>
          ) : null}
        </CardContent>
      </Card>

      {filtered.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-12 text-center">
            <p className="text-sm font-medium text-foreground">
              No figures match the selected filters.
            </p>
            <p className="text-xs text-muted-foreground">
              Try clearing one or more filters above.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          {filtered.map((f) => (
            <Card key={f.image} className="flex h-full flex-col">
              <CardHeader>
                <div className="mb-2 flex flex-wrap gap-2">
                  <Badge variant="accent">{modelLabel(f.model)}</Badge>
                  <Badge>{graphLabel(f.graphType)}</Badge>
                  {f.demandGroup ? (
                    <Badge variant="muted">
                      {demandLabel(f.demandGroup)}
                    </Badge>
                  ) : null}
                </div>
                <CardTitle className="text-base">{f.title}</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-1 flex-col gap-4">
                <FigureImage src={f.image} alt={f.title} />
                <InterpretationBox text={f.interpretation} />
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
