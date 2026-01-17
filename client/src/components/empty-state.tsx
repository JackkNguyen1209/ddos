import { FileQuestion, Brain, BarChart3 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

interface EmptyStateProps {
  type: "dataset" | "analysis" | "results";
  title: string;
  description: string;
}

export function EmptyState({ type, title, description }: EmptyStateProps) {
  const icons = {
    dataset: <FileQuestion className="h-12 w-12 text-muted-foreground" />,
    analysis: <Brain className="h-12 w-12 text-muted-foreground" />,
    results: <BarChart3 className="h-12 w-12 text-muted-foreground" />,
  };

  return (
    <Card className="border-dashed">
      <CardContent className="flex flex-col items-center justify-center py-12 text-center">
        <div className="rounded-full bg-muted p-4 mb-4">
          {icons[type]}
        </div>
        <h3 className="text-lg font-medium mb-2">{title}</h3>
        <p className="text-sm text-muted-foreground max-w-sm">{description}</p>
      </CardContent>
    </Card>
  );
}
