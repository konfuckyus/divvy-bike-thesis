"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface Option<TValue extends string> {
  value: TValue;
  label: string;
}

interface ModelSelectorProps<TValue extends string> {
  label: string;
  value: TValue;
  options: Option<TValue>[];
  onChange: (value: TValue) => void;
  className?: string;
  placeholder?: string;
}

export function ModelSelector<TValue extends string>({
  label,
  value,
  options,
  onChange,
  className,
  placeholder,
}: ModelSelectorProps<TValue>) {
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </label>
      <Select value={value} onValueChange={(v) => onChange(v as TValue)}>
        <SelectTrigger className="w-full sm:w-64">
          <SelectValue placeholder={placeholder ?? "Select..."} />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
