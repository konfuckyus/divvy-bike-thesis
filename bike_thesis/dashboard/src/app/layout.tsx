import type { Metadata } from "next";
import { Inter } from "next/font/google";

import "./globals.css";
import { Sidebar } from "@/components/sidebar";
import { MobileNav } from "@/components/mobile-nav";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "Divvy Bike-Sharing Demand Forecasting Dashboard",
  description:
    "Station-level hourly demand forecasting using the 2025 Divvy Bike Trip dataset.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans`}>
        <div className="flex min-h-screen w-full bg-background">
          <Sidebar />
          <div className="flex min-w-0 flex-1 flex-col">
            <MobileNav />
            <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 md:px-8 md:py-10">
              {children}
            </main>
            <footer className="border-t border-border bg-card px-4 py-4 text-center text-xs text-muted-foreground md:px-8">
              Graduation Thesis · Divvy 2025 · Station-level Hourly Demand
              Forecasting
            </footer>
          </div>
        </div>
      </body>
    </html>
  );
}
