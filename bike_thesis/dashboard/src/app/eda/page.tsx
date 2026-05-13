"use client";

import { useMemo, useState } from "react";

import { PageHeader } from "@/components/page-header";
import { FigureImage } from "@/components/figure-image";
import { InterpretationBox } from "@/components/interpretation-box";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  EDA_CATEGORIES,
  EdaCategory,
  edaFigures,
} from "@/data/edaData";
import { cn } from "@/lib/utils";

type CategoryFilter = "all" | EdaCategory;

const EDA_MISSING_MESSAGE =
  "Figure file not found. Please add the image to /public/figures/eda/.";

const CATEGORY_OPTIONS: { value: CategoryFilter; label: string }[] = [
  { value: "all", label: "All" },
  ...EDA_CATEGORIES.map((c) => ({
    value: c.key as CategoryFilter,
    label: c.label,
  })),
];

function categoryLabel(key: EdaCategory) {
  return EDA_CATEGORIES.find((c) => c.key === key)?.label ?? key;
}

export default function EdaPage() {
  const [category, setCategory] = useState<CategoryFilter>("all");

  const filtered = useMemo(() => {
    if (category === "all") return edaFigures;
    return edaFigures.filter((f) => f.category === category);
  }, [category]);

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="EDA Visualizations"
        title="Exploratory Data Analysis"
        description="This page presents the exploratory data analysis outputs generated before the forecasting stage. These visualizations help explain the temporal, user-based, and station-based structure of the 2025 Divvy Bike Trip dataset."
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Category</CardTitle>
          <CardDescription>
            Filter the EDA figures by category. Showing {filtered.length} of{" "}
            {edaFigures.length}.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {CATEGORY_OPTIONS.map((opt) => {
              const isActive = category === opt.value;
              return (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => setCategory(opt.value)}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs font-medium transition",
                    isActive
                      ? "border-accent bg-accent/10 text-accent"
                      : "border-border bg-secondary text-muted-foreground hover:bg-secondary/70 hover:text-foreground",
                  )}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {filtered.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-12 text-center">
            <p className="text-sm font-medium text-foreground">
              No figures available for this category.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          {filtered.map((f) => (
            <Card key={f.image} className="flex h-full flex-col">
              <CardHeader>
                <div className="mb-2 flex flex-wrap gap-2">
                  <span className="inline-flex items-center rounded-full border border-accent/30 bg-accent/10 px-2.5 py-0.5 text-[11px] font-medium text-accent">
                    {categoryLabel(f.category)}
                  </span>
                </div>
                <CardTitle className="text-base">{f.title}</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-1 flex-col gap-4">
                <FigureImage
                  src={f.image}
                  alt={f.title}
                  fallbackMessage={EDA_MISSING_MESSAGE}
                />
                <InterpretationBox text={f.interpretation} />
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
