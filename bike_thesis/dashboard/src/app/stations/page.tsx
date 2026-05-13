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
  MODELS,
  ModelKey,
  getStationFigure,
} from "@/data/figuresData";
import { cn } from "@/lib/utils";

const GROUP_ACCENTS: Record<DemandGroup, string> = {
  low: "bg-emerald-500/10 text-emerald-700",
  medium: "bg-amber-500/10 text-amber-700",
  high: "bg-rose-500/10 text-rose-700",
};

export default function StationsPage() {
  const [model, setModel] = useState<ModelKey>("random_forest");
  const [group, setGroup] = useState<DemandGroup>("high");

  const modelOptions = useMemo(
    () => MODELS.map((m) => ({ value: m.key, label: m.label })),
    [],
  );
  const groupOptions = useMemo(
    () => DEMAND_GROUPS.map((g) => ({ value: g.key, label: g.label })),
    [],
  );

  const figure = useMemo(
    () => getStationFigure(model, group),
    [model, group],
  );

  const activeModel = MODELS.find((m) => m.key === model);
  const activeGroup = DEMAND_GROUPS.find((g) => g.key === group);

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Station Forecast Explorer"
        title="Per-station hourly forecasts"
        description="Pick a model and a demand group to see how well the model tracks an example station over time."
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Selection</CardTitle>
          <CardDescription>
            Each demand group shows a representative station for the chosen
            model.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4 md:flex-row md:items-end">
          <ModelSelector
            label="Model"
            value={model}
            options={modelOptions}
            onChange={setModel}
          />
          <ModelSelector
            label="Demand group"
            value={group}
            options={groupOptions}
            onChange={setGroup}
          />
          <div className="flex flex-wrap gap-2 md:ml-auto">
            {DEMAND_GROUPS.map((g) => (
              <button
                key={g.key}
                type="button"
                onClick={() => setGroup(g.key)}
                className={cn(
                  "rounded-full border border-transparent px-3 py-1 text-xs font-medium transition",
                  group === g.key
                    ? `${GROUP_ACCENTS[g.key]} border-current/20`
                    : "bg-secondary text-muted-foreground hover:text-foreground",
                )}
              >
                {g.label}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            {figure ? figure.title : "Figure not available"}
          </CardTitle>
          <CardDescription>
            {activeModel?.label} · {activeGroup?.label}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {figure ? (
            <>
              <FigureImage src={figure.image} alt={figure.title} />
              <InterpretationBox text={figure.interpretation} />
            </>
          ) : (
            <p className="text-sm text-muted-foreground">
              No station figure is available for this selection.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            How to read these forecasts
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          <p>
            Each chart overlays the predicted hourly ride starts on top of the
            observed values for a single station from the test window. The
            closer the predicted line follows the actual line, the more
            faithfully the model captured that station&apos;s demand pattern.
          </p>
          <p>
            Low-demand stations have small absolute counts and a flatter signal,
            so any wiggle is highly visible. High-demand stations show daily
            commuting peaks that are the hardest part of the forecast.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
