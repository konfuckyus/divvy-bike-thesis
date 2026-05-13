import { PageHeader } from "@/components/page-header";
import { FigureCard } from "@/components/figure-card";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { MODEL_COMPARISON_FIGURES } from "@/data/figuresData";

export default function ComparisonPage() {
  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="Model Comparison"
        title="Linear Regression vs Random Forest"
        description="Aggregate error metrics across all station-hour predictions in the test window."
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Why these metrics?</CardTitle>
          <CardDescription>
            RMSE and MAE summarize prediction quality from complementary angles.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 text-sm leading-relaxed text-muted-foreground md:grid-cols-2">
          <p>
            <span className="font-semibold text-foreground">RMSE</span> (Root
            Mean Squared Error) penalizes large prediction errors more strongly, 
            so lower RMSE indicates better performance. This is especially important for 
            high-demand stations, where severe forecasting mistakes can affect bike availability 
            and redistribution decisions.
          </p>
          <p>
            <span className="font-semibold text-foreground">MAE</span> (Mean
            Absolute Error) MAE shows how many rides the model is off by on average. 
            Lower MAE means the predictions are closer to the actual station-hour 
            demand values.
          </p>
        </CardContent>
      </Card>

      <section className="grid gap-6 lg:grid-cols-2">
        <FigureCard
          title={MODEL_COMPARISON_FIGURES.rmse.title}
          image={MODEL_COMPARISON_FIGURES.rmse.image}
          interpretation={MODEL_COMPARISON_FIGURES.rmse.interpretation}
          description="Lower RMSE means better performance, especially because large prediction errors are penalized more strongly."
        />
        <FigureCard
          title={MODEL_COMPARISON_FIGURES.mae.title}
          image={MODEL_COMPARISON_FIGURES.mae.image}
          interpretation={MODEL_COMPARISON_FIGURES.mae.interpretation}
          description="Lower MAE indicates that the model's predictions are closer to the actual station-hour demand values on average."
        />
      </section>
    </div>
  );
}
