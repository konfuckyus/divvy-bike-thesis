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
  ScatterChart,
  BookOpen,
} from "lucide-react";

import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/dataset", label: "Dataset", icon: Database },
  { href: "/eda", label: "EDA", icon: ScatterChart },
  { href: "/comparison", label: "Comparison", icon: BarChart3 },
  { href: "/explorer", label: "Explorer", icon: Compass },
  { href: "/stations", label: "Stations", icon: MapPin },
  { href: "/findings", label: "Findings", icon: ClipboardList },
  { href: "/appendix", label: "Appendix", icon: BookOpen },
];

export function MobileNav() {
  const pathname = usePathname();
  return (
    <nav className="sticky top-0 z-40 flex w-full items-center gap-1 overflow-x-auto border-b border-border bg-card px-4 py-3 shadow-sm lg:hidden">
      {NAV_ITEMS.map((item) => {
        const Icon = item.icon;
        const isActive =
          item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium",
              isActive
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-secondary hover:text-foreground",
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
