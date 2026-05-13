import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { FigureImage } from "@/components/figure-image";
import { InterpretationBox } from "@/components/interpretation-box";
import { cn } from "@/lib/utils";

interface FigureCardProps {
  title: string;
  image: string;
  interpretation: string;
  description?: string;
  className?: string;
}

export function FigureCard({
  title,
  image,
  interpretation,
  description,
  className,
}: FigureCardProps) {
  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
        {description ? <CardDescription>{description}</CardDescription> : null}
      </CardHeader>
      <CardContent className="space-y-4">
        <FigureImage src={image} alt={title} />
        <InterpretationBox text={interpretation} />
      </CardContent>
    </Card>
  );
}
