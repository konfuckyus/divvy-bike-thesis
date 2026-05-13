import { LucideIcon } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface SummaryCardProps {
  label: string;
  value: string;
  description?: string;
  icon?: LucideIcon;
  accent?: boolean;
  className?: string;
}

export function SummaryCard({
  label,
  value,
  description,
  icon: Icon,
  accent,
  className,
}: SummaryCardProps) {
  return (
    <Card
      className={cn(
        "h-full transition-shadow hover:shadow-md",
        accent && "border-accent/30 ring-1 ring-accent/15",
        className,
      )}
    >
      <CardContent className="flex flex-col gap-3 p-6">
        <div className="flex items-start justify-between gap-3">
          <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            {label}
          </p>
          {Icon ? (
            <div
              className={cn(
                "flex h-9 w-9 items-center justify-center rounded-lg",
                accent
                  ? "bg-accent/10 text-accent"
                  : "bg-secondary text-muted-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
            </div>
          ) : null}
        </div>
        <p className="text-xl font-semibold leading-tight tracking-tight text-foreground">
          {value}
        </p>
        {description ? (
          <p className="text-sm leading-relaxed text-muted-foreground">
            {description}
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}
