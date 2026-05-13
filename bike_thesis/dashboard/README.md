# Divvy Bike-Sharing Demand Forecasting Dashboard

A Next.js dashboard built for a graduation thesis on station-level hourly
bike-sharing demand forecasting using the 2025 Divvy Bike Trip dataset.

The dashboard is **read-only** — it visualizes pre-generated PNG figures and
ships text interpretations alongside each chart. There is no backend or live
prediction.

## Tech stack

- [Next.js 14](https://nextjs.org/) with the App Router
- TypeScript
- Tailwind CSS
- shadcn/ui–style primitives (Card, Button, Select, Tabs)
- Lucide icons

## Project structure

```
dashboard/
├── public/
│   └── figures/                  # All pre-generated PNG charts
├── src/
│   ├── app/
│   │   ├── layout.tsx            # App shell + sidebar
│   │   ├── page.tsx              # Dashboard Overview
│   │   ├── dataset/page.tsx      # Dataset Transformation
│   │   ├── comparison/page.tsx   # Model Comparison
│   │   ├── explorer/page.tsx     # Model Explorer
│   │   ├── stations/page.tsx     # Station Forecast Explorer
│   │   └── findings/page.tsx     # Findings & Limitations
│   ├── components/
│   │   ├── sidebar.tsx
│   │   ├── mobile-nav.tsx
│   │   ├── page-header.tsx
│   │   ├── summary-card.tsx
│   │   ├── figure-card.tsx
│   │   ├── figure-image.tsx      # Renders an image with a friendly fallback
│   │   ├── interpretation-box.tsx
│   │   ├── model-selector.tsx
│   │   └── ui/                   # shadcn-style primitives
│   ├── data/
│   │   └── figuresData.ts        # All figure metadata + helpers
│   └── lib/
│       └── utils.ts
└── tailwind.config.ts
```

## Pages

| Route          | Purpose                                                       |
| -------------- | ------------------------------------------------------------- |
| `/`            | Project overview + summary cards                              |
| `/dataset`     | Trip-records → station-hour forecasting dataset pipeline      |
| `/comparison`  | RMSE & MAE comparison: Linear Regression vs Random Forest     |
| `/explorer`    | Pick a model + graph type to view diagnostics & per-station forecasts |
| `/stations`    | Pick a model + demand group (low / medium / high)             |
| `/findings`    | Findings cards + limitations + future work                    |

## Adding or updating figures

1. Drop the PNG into `public/figures/`.
2. Edit `src/data/figuresData.ts` and either add a new entry to the `figures`
   array or update one of the existing entries (`model`, `graphType`,
   `demandGroup`, `title`, `image`, `interpretation`).
3. If the image is missing at runtime, the UI shows a friendly fallback card
   rather than breaking the page.

## Local development

```bash
cd dashboard
npm install
npm run dev
```

The dev server runs on http://localhost:3000.

## Build

```bash
npm run build
npm run start
```
