import { Database, AlertTriangle, CheckCircle2, Trash2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import type { Dataset, DataRow } from "@shared/schema";

interface DataPreviewProps {
  dataset: Dataset;
  previewData: DataRow[];
}

export function DataPreview({ dataset, previewData }: DataPreviewProps) {
  const { dataQuality } = dataset;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5 text-primary" />
              {dataset.name}
            </CardTitle>
            <CardDescription>
              {dataset.cleanedRowCount.toLocaleString()} dòng sau khi lọc (từ {dataset.originalRowCount.toLocaleString()} dòng gốc)
            </CardDescription>
          </div>
          <Badge variant={dataset.isProcessed ? "default" : "secondary"}>
            {dataset.isProcessed ? (
              <>
                <CheckCircle2 className="mr-1 h-3 w-3" />
                Đã xử lý
              </>
            ) : (
              "Chưa xử lý"
            )}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <QualityMetric
            label="Giá trị thiếu"
            value={dataQuality.missingValues}
            type="warning"
            icon={<AlertTriangle className="h-4 w-4" />}
          />
          <QualityMetric
            label="Dòng trùng lặp"
            value={dataQuality.duplicates}
            type="warning"
            icon={<Trash2 className="h-4 w-4" />}
          />
          <QualityMetric
            label="Outliers"
            value={dataQuality.outliers}
            type="info"
            icon={<AlertTriangle className="h-4 w-4" />}
          />
          <div className="rounded-lg bg-accent/50 p-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
              <CheckCircle2 className="h-4 w-4 text-chart-2" />
              <span>Độ sạch dữ liệu</span>
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-bold text-chart-2">
                  {dataQuality.cleanedPercentage.toFixed(1)}%
                </span>
              </div>
              <Progress value={dataQuality.cleanedPercentage} className="h-2" />
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <h4 className="text-sm font-medium">Xem trước dữ liệu (10 dòng đầu)</h4>
          <div className="rounded-lg border">
            <ScrollArea className="w-full">
              <Table>
                <TableHeader>
                  <TableRow>
                    {dataset.columns.map((col) => (
                      <TableHead key={col} className="whitespace-nowrap">
                        {col}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {previewData.map((row, idx) => (
                    <TableRow key={idx} data-testid={`row-data-${idx}`}>
                      {dataset.columns.map((col) => (
                        <TableCell key={col} className="whitespace-nowrap">
                          {row[col] !== null && row[col] !== undefined
                            ? String(row[col])
                            : <span className="text-muted-foreground">null</span>}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <ScrollBar orientation="horizontal" />
            </ScrollArea>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <span className="text-sm text-muted-foreground">Các cột:</span>
          {dataset.columns.map((col) => (
            <Badge key={col} variant="secondary" className="text-xs">
              {col}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

interface QualityMetricProps {
  label: string;
  value: number;
  type: "warning" | "info";
  icon: React.ReactNode;
}

function QualityMetric({ label, value, type, icon }: QualityMetricProps) {
  return (
    <div className="rounded-lg bg-accent/50 p-4">
      <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
        <span className={type === "warning" ? "text-chart-3" : "text-chart-4"}>
          {icon}
        </span>
        <span>{label}</span>
      </div>
      <p className="text-2xl font-bold">{value.toLocaleString()}</p>
    </div>
  );
}
