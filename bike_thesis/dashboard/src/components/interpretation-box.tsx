import { Lightbulb } from "lucide-react";

import { cn } from "@/lib/utils";

interface InterpretationBoxProps {
  text: string;
  label?: string;
  className?: string;
}

export function InterpretationBox({
  text,
  label = "Interpretation",
  className,
}: InterpretationBoxProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-accent/20 bg-accent/5 p-4",
        className,
      )}
    >
      <div className="flex items-center gap-2">
        <Lightbulb className="h-4 w-4 text-accent" />
        <p className="text-xs font-semibold uppercase tracking-wider text-accent">
          {label}
        </p>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-foreground/90">{text}</p>
    </div>
  );
}
