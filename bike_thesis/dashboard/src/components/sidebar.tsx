"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Database,
  BarChart3,
  Compass,
  MapPin,
  ClipboardList,
  Bike,
  ScatterChart,
  BookOpen,
} from "lucide-react";

import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/dataset", label: "Dataset Transformation", icon: Database },
  { href: "/eda", label: "EDA Visualizations", icon: ScatterChart },
  { href: "/comparison", label: "Model Comparison", icon: BarChart3 },
  { href: "/explorer", label: "Model Explorer", icon: Compass },
  { href: "/stations", label: "Station Forecast", icon: MapPin },
  { href: "/findings", label: "Findings & Limitations", icon: ClipboardList },
  { href: "/appendix", label: "Forecasting Appendix", icon: BookOpen },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sticky top-0 hidden h-screen w-72 shrink-0 flex-col border-r border-border bg-card lg:flex">
      <div className="flex items-center gap-3 border-b border-border px-6 py-5">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent text-accent-foreground">
          <Bike className="h-5 w-5" />
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-semibold leading-tight">
            Divvy Forecast
          </span>
          <span className="text-xs text-muted-foreground">
            Thesis Dashboard
          </span>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-4">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-secondary hover:text-foreground",
              )}
            >
              <Icon
                className={cn(
                  "h-4 w-4 shrink-0",
                  isActive
                    ? "text-primary-foreground"
                    : "text-muted-foreground group-hover:text-foreground",
                )}
              />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-border px-6 py-4 text-xs text-muted-foreground">
        <p className="font-medium text-foreground">Graduation Thesis</p>
        <p className="mt-1 leading-relaxed">
          Station-level hourly demand forecasting on the 2025 Divvy Bike Trip
          dataset.
        </p>
      </div>
    </aside>
  );
}
