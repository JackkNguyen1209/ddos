import { BarChart3, Trophy, Clock, Target, Shield, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { ML_MODELS, type AnalysisResult } from "@shared/schema";

interface AnalysisResultsProps {
  results: AnalysisResult[];
}

export function AnalysisResults({ results }: AnalysisResultsProps) {
  if (results.length === 0) {
    return null;
  }

  const sortedByAccuracy = [...results].sort((a, b) => b.accuracy - a.accuracy);
  const bestModel = sortedByAccuracy[0];

  const getModelName = (type: string) => {
    return ML_MODELS.find((m) => m.type === type)?.name || type;
  };

  const comparisonData = results.map((r) => ({
    name: getModelName(r.modelType),
    accuracy: (r.accuracy * 100).toFixed(1),
    precision: (r.precision * 100).toFixed(1),
    recall: (r.recall * 100).toFixed(1),
    f1Score: (r.f1Score * 100).toFixed(1),
  }));

  const radarData = [
    { metric: "Accuracy", ...Object.fromEntries(results.map((r) => [getModelName(r.modelType), r.accuracy * 100])) },
    { metric: "Precision", ...Object.fromEntries(results.map((r) => [getModelName(r.modelType), r.precision * 100])) },
    { metric: "Recall", ...Object.fromEntries(results.map((r) => [getModelName(r.modelType), r.recall * 100])) },
    { metric: "F1 Score", ...Object.fromEntries(results.map((r) => [getModelName(r.modelType), r.f1Score * 100])) },
  ];

  const colors = ["hsl(200, 90%, 45%)", "hsl(160, 60%, 45%)", "hsl(45, 93%, 47%)", "hsl(280, 65%, 60%)", "hsl(340, 75%, 55%)"];

  return (
    <div className="space-y-6">
      <Card className="border-primary/50 bg-primary/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Trophy className="h-5 w-5 text-chart-3" />
            Mô Hình Tốt Nhất
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col lg:flex-row gap-6">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-4">
                <div className="rounded-full bg-chart-3/20 p-3">
                  <Trophy className="h-6 w-6 text-chart-3" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">{getModelName(bestModel.modelType)}</h3>
                  <p className="text-sm text-muted-foreground">
                    Độ chính xác cao nhất trong các mô hình đã phân tích
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <MetricCard
                  label="Accuracy"
                  value={bestModel.accuracy}
                  icon={<Target className="h-4 w-4" />}
                  color="chart-1"
                />
                <MetricCard
                  label="Precision"
                  value={bestModel.precision}
                  icon={<Shield className="h-4 w-4" />}
                  color="chart-2"
                />
                <MetricCard
                  label="Recall"
                  value={bestModel.recall}
                  icon={<BarChart3 className="h-4 w-4" />}
                  color="chart-4"
                />
                <MetricCard
                  label="F1 Score"
                  value={bestModel.f1Score}
                  icon={<Target className="h-4 w-4" />}
                  color="chart-5"
                />
              </div>
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Kết quả phát hiện</span>
                <Badge variant="outline">
                  <Clock className="mr-1 h-3 w-3" />
                  {bestModel.trainingTime.toFixed(2)}s
                </Badge>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg bg-destructive/10 p-4 text-center">
                  <AlertTriangle className="h-6 w-6 text-destructive mx-auto mb-2" />
                  <p className="text-2xl font-bold text-destructive">{bestModel.ddosDetected.toLocaleString()}</p>
                  <p className="text-xs text-muted-foreground">DDoS Detected</p>
                </div>
                <div className="rounded-lg bg-chart-2/10 p-4 text-center">
                  <Shield className="h-6 w-6 text-chart-2 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-chart-2">{bestModel.normalTraffic.toLocaleString()}</p>
                  <p className="text-xs text-muted-foreground">Normal Traffic</p>
                </div>
              </div>
              <ConfusionMatrixMini matrix={bestModel.confusionMatrix} />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            So Sánh Các Mô Hình
          </CardTitle>
          <CardDescription>
            Phân tích chi tiết và so sánh hiệu suất của từng thuật toán
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="comparison" className="w-full">
            <TabsList className="grid w-full grid-cols-3 mb-6">
              <TabsTrigger value="comparison" data-testid="tab-comparison">So sánh</TabsTrigger>
              <TabsTrigger value="radar" data-testid="tab-radar">Radar Chart</TabsTrigger>
              <TabsTrigger value="details" data-testid="tab-details">Chi tiết</TabsTrigger>
            </TabsList>

            <TabsContent value="comparison" className="space-y-4">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparisonData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                    <XAxis dataKey="name" className="text-xs" tick={{ fill: 'hsl(var(--muted-foreground))' }} />
                    <YAxis domain={[0, 100]} className="text-xs" tick={{ fill: 'hsl(var(--muted-foreground))' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '0.5rem',
                      }}
                    />
                    <Legend />
                    <Bar dataKey="accuracy" name="Accuracy %" fill="hsl(200, 90%, 45%)" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="precision" name="Precision %" fill="hsl(160, 60%, 45%)" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="recall" name="Recall %" fill="hsl(45, 93%, 47%)" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="f1Score" name="F1 Score %" fill="hsl(280, 65%, 60%)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="radar" className="space-y-4">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid className="stroke-border" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={{ fill: 'hsl(var(--muted-foreground))' }} />
                    {results.map((r, idx) => (
                      <Radar
                        key={r.modelType}
                        name={getModelName(r.modelType)}
                        dataKey={getModelName(r.modelType)}
                        stroke={colors[idx % colors.length]}
                        fill={colors[idx % colors.length]}
                        fillOpacity={0.2}
                      />
                    ))}
                    <Legend />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '0.5rem',
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="details" className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {sortedByAccuracy.map((result, idx) => (
                  <ModelDetailCard
                    key={result.id}
                    result={result}
                    rank={idx + 1}
                    getModelName={getModelName}
                  />
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: number;
  icon: React.ReactNode;
  color: string;
}

function MetricCard({ label, value, icon, color }: MetricCardProps) {
  return (
    <div className="rounded-lg bg-card p-3 border">
      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
        <span className={`text-${color}`}>{icon}</span>
        <span>{label}</span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-xl font-bold">{(value * 100).toFixed(1)}</span>
        <span className="text-xs text-muted-foreground">%</span>
      </div>
      <Progress value={value * 100} className="h-1 mt-2" />
    </div>
  );
}

interface ConfusionMatrixMiniProps {
  matrix: {
    truePositive: number;
    trueNegative: number;
    falsePositive: number;
    falseNegative: number;
  };
}

function ConfusionMatrixMini({ matrix }: ConfusionMatrixMiniProps) {
  return (
    <div className="mt-4">
      <p className="text-xs text-muted-foreground mb-2">Confusion Matrix</p>
      <div className="grid grid-cols-2 gap-1 text-xs">
        <div className="bg-chart-2/20 p-2 rounded text-center">
          <p className="font-bold text-chart-2">{matrix.truePositive}</p>
          <p className="text-muted-foreground">TP</p>
        </div>
        <div className="bg-destructive/20 p-2 rounded text-center">
          <p className="font-bold text-destructive">{matrix.falsePositive}</p>
          <p className="text-muted-foreground">FP</p>
        </div>
        <div className="bg-destructive/20 p-2 rounded text-center">
          <p className="font-bold text-destructive">{matrix.falseNegative}</p>
          <p className="text-muted-foreground">FN</p>
        </div>
        <div className="bg-chart-2/20 p-2 rounded text-center">
          <p className="font-bold text-chart-2">{matrix.trueNegative}</p>
          <p className="text-muted-foreground">TN</p>
        </div>
      </div>
    </div>
  );
}

interface ModelDetailCardProps {
  result: AnalysisResult;
  rank: number;
  getModelName: (type: string) => string;
}

function ModelDetailCard({ result, rank, getModelName }: ModelDetailCardProps) {
  const pieData = [
    { name: "DDoS", value: result.ddosDetected, color: "hsl(0, 72%, 51%)" },
    { name: "Normal", value: result.normalTraffic, color: "hsl(160, 60%, 45%)" },
  ];

  return (
    <Card className={rank === 1 ? "border-chart-3" : ""} data-testid={`card-result-${result.modelType}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-base">{getModelName(result.modelType)}</CardTitle>
          {rank === 1 && (
            <Badge className="bg-chart-3 text-chart-3-foreground">
              <Trophy className="mr-1 h-3 w-3" />
              #1
            </Badge>
          )}
          {rank === 2 && <Badge variant="secondary">#2</Badge>}
          {rank === 3 && <Badge variant="outline">#3</Badge>}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <p className="text-muted-foreground text-xs">Accuracy</p>
            <p className="font-semibold">{(result.accuracy * 100).toFixed(2)}%</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Precision</p>
            <p className="font-semibold">{(result.precision * 100).toFixed(2)}%</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Recall</p>
            <p className="font-semibold">{(result.recall * 100).toFixed(2)}%</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">F1 Score</p>
            <p className="font-semibold">{(result.f1Score * 100).toFixed(2)}%</p>
          </div>
        </div>

        <div className="h-32">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={25}
                outerRadius={45}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {result.trainingTime.toFixed(2)}s
          </span>
          <span>{new Date(result.analyzedAt).toLocaleTimeString()}</span>
        </div>
      </CardContent>
    </Card>
  );
}
