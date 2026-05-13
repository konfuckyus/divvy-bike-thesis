"use client";

import { useMemo, useState } from "react";
import { Info } from "lucide-react";

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
  GRAPH_TYPES,
  GraphType,
  MODELS,
  ModelKey,
  getFigure,
} from "@/data/figuresData";

export default function ExplorerPage() {
  const [model, setModel] = useState<ModelKey>("random_forest");
  const [graphType, setGraphType] = useState<GraphType>("actual_vs_predicted");

  const modelOptions = useMemo(
    () => MODELS.map((m) => ({ value: m.key, label: m.label })),
    [],
  );
  const graphOptions = useMemo(
    () => GRAPH_TYPES.map((g) => ({ value: g.key, label: g.label })),
    [],
  );

  const figure = useMemo(
    () => getFigure(model, graphType),
    [model, graphType],
  );

  const activeModel = MODELS.find((m) => m.key === model);

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Model Explorer"
        title="Inspect each model in detail"
        description="Pick a model and a diagnostic to view its prediction behavior."
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Selection</CardTitle>
          <CardDescription>
            Combine a model with a graph type to update the panel below.
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
            label="Graph type"
            value={graphType}
            options={graphOptions}
            onChange={setGraphType}
          />
          {activeModel ? (
            <div className="flex flex-1 items-start gap-2 rounded-md border border-dashed border-border bg-secondary/40 p-3 text-xs text-muted-foreground">
              <Info className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
              <span>
                <span className="font-medium text-foreground">
                  {activeModel.label}.
                </span>{" "}
                {activeModel.description}
              </span>
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            {figure ? figure.title : "Figure not available"}
          </CardTitle>
          <CardDescription>
            {activeModel?.label} ·{" "}
            {GRAPH_TYPES.find((g) => g.key === graphType)?.label}
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
              No figure is configured for this combination yet.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
