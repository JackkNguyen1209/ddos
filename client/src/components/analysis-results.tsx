import { BarChart3, Trophy, Clock, Target, Shield, AlertTriangle, Info, Zap, BookOpen, CheckCircle2, XCircle, Lightbulb, Calculator, ListOrdered, Settings } from "lucide-react";
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
import { ML_MODELS, ALGORITHM_DETAILS, ATTACK_TYPE_INFO, type AnalysisResult, type AttackTypeResult } from "@shared/schema";

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
            <TabsList className="grid w-full grid-cols-6 mb-6">
              <TabsTrigger value="comparison" data-testid="tab-comparison">So sánh</TabsTrigger>
              <TabsTrigger value="attack-types" data-testid="tab-attack-types">Loại tấn công</TabsTrigger>
              <TabsTrigger value="radar" data-testid="tab-radar">Radar</TabsTrigger>
              <TabsTrigger value="algorithms" data-testid="tab-algorithms">Thuật toán</TabsTrigger>
              <TabsTrigger value="explanation" data-testid="tab-explanation">Giải thích</TabsTrigger>
              <TabsTrigger value="details" data-testid="tab-details">Chi tiết</TabsTrigger>
            </TabsList>

            <TabsContent value="attack-types" className="space-y-4">
              <AttackTypesAnalysis results={results} getModelName={getModelName} />
            </TabsContent>

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

            <TabsContent value="algorithms" className="space-y-4">
              <AlgorithmExplanations results={results} getModelName={getModelName} />
            </TabsContent>

            <TabsContent value="explanation" className="space-y-6">
              <DDoSExplanation result={bestModel} getModelName={getModelName} />
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

interface DDoSExplanationProps {
  result: AnalysisResult;
  getModelName: (type: string) => string;
}

function DDoSExplanation({ result, getModelName }: DDoSExplanationProps) {
  const featureImportance = result.featureImportance || [];
  const ddosReasons = result.ddosReasons || [];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Info className="h-5 w-5 text-primary" />
            Tại Sao Được Phân Loại Là DDoS?
          </CardTitle>
          <CardDescription>
            Giải thích chi tiết các đặc trưng khiến traffic được mô hình {getModelName(result.modelType)} phân loại là tấn công DDoS (dựa trên kết quả dự đoán của mô hình)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {ddosReasons.length > 0 ? (
            <div className="space-y-3">
              {ddosReasons.map((reason, idx) => (
                <div key={idx} className="rounded-lg border p-4 space-y-2">
                  <div className="flex items-center justify-between gap-2 flex-wrap">
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-chart-3" />
                      <span className="font-semibold">{reason.feature}</span>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      Mức đóng góp: {(reason.contribution * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{reason.description}</p>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="text-destructive">
                      Giá trị DDoS: {reason.value.toFixed(2)}
                    </span>
                    <span className="text-muted-foreground">
                      Ngưỡng: {reason.threshold.toFixed(2)}
                    </span>
                  </div>
                  <Progress value={reason.contribution * 100} className="h-1.5" />
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4 space-y-2">
              <AlertTriangle className="h-8 w-8 text-muted-foreground mx-auto" />
              <p className="text-muted-foreground">
                Không có dữ liệu giải thích. Mô hình không phát hiện đủ mẫu DDoS để phân tích hoặc dataset không có cột nhãn (label).
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <BarChart3 className="h-5 w-5 text-primary" />
            Mức Độ Quan Trọng Của Đặc Trưng
          </CardTitle>
          <CardDescription>
            Các đặc trưng ảnh hưởng nhiều nhất đến việc phát hiện DDoS
          </CardDescription>
        </CardHeader>
        <CardContent>
          {featureImportance.length > 0 ? (
            <div className="space-y-3">
              {featureImportance.map((fi, idx) => (
                <div key={idx} className="space-y-1.5">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">{fi.feature}</span>
                    <span className="text-muted-foreground">
                      {(fi.importance * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={fi.importance * 100} className="h-2" />
                  <p className="text-xs text-muted-foreground">{fi.description}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground text-center py-4">
              Không có dữ liệu về mức độ quan trọng của đặc trưng.
            </p>
          )}
        </CardContent>
      </Card>

      <Card className="bg-accent/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <AlertTriangle className="h-5 w-5 text-chart-3" />
            Tổng Kết Phân Tích
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <p>
              Mô hình <strong>{getModelName(result.modelType)}</strong> đã phát hiện{" "}
              <span className="text-destructive font-bold">{result.ddosDetected.toLocaleString()}</span> mẫu traffic là tấn công DDoS
              và <span className="text-chart-2 font-bold">{result.normalTraffic.toLocaleString()}</span> mẫu là traffic bình thường.
            </p>
            <p className="mt-2">
              Với độ chính xác <strong>{(result.accuracy * 100).toFixed(1)}%</strong>, mô hình đã xác định các đặc trưng 
              quan trọng nhất để phân biệt giữa traffic DDoS và traffic bình thường.
            </p>
            {ddosReasons.length > 0 && (
              <p className="mt-2">
                Đặc trưng <strong>{ddosReasons[0]?.feature}</strong> có mức đóng góp cao nhất 
                ({(ddosReasons[0]?.contribution * 100).toFixed(1)}%) trong việc phát hiện tấn công. 
                {ddosReasons[0]?.description}
              </p>
            )}
            {result.ddosDetected === 0 && (
              <div className="mt-4 p-4 rounded-lg bg-chart-3/10 border border-chart-3/30">
                <p className="font-semibold text-chart-3 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Lưu ý quan trọng:
                </p>
                <p className="mt-2">
                  Mô hình không phát hiện DDoS nào. Điều này có thể do:
                </p>
                <ul className="list-disc list-inside mt-1 space-y-1 text-muted-foreground">
                  <li>Dataset không có cột nhãn (label) - mô hình không có dữ liệu huấn luyện để phân biệt DDoS</li>
                  <li>Tất cả traffic trong dataset là bình thường</li>
                  <li>Dataset đã được lọc và chỉ chứa traffic bình thường</li>
                </ul>
                <p className="mt-2">
                  Để có kết quả chính xác, hãy sử dụng dataset có cột <strong>label</strong> với giá trị 0 (normal) và 1 (DDoS).
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

interface AlgorithmExplanationsProps {
  results: AnalysisResult[];
  getModelName: (type: string) => string;
}

function AlgorithmExplanations({ results, getModelName }: AlgorithmExplanationsProps) {
  return (
    <div className="space-y-4">
      <Card className="bg-primary/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <BookOpen className="h-5 w-5 text-primary" />
            Cách Phân Tích Hoạt Động
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <p>
              Hệ thống sử dụng phương pháp <strong>Supervised Machine Learning</strong> (Học máy có giám sát) để phát hiện DDoS:
            </p>
            <ol className="list-decimal list-inside space-y-2 mt-3">
              <li><strong>Tiền xử lý dữ liệu:</strong> Làm sạch dataset bằng cách loại bỏ giá trị thiếu, dòng trùng lặp và outliers</li>
              <li><strong>Trích xuất đặc trưng:</strong> Chọn các cột số làm features (src_port, dst_port, bytes, packets, duration, v.v.)</li>
              <li><strong>Chuẩn hóa:</strong> Áp dụng Min-Max normalization để đưa tất cả features về khoảng [0, 1]</li>
              <li><strong>Chia tập dữ liệu:</strong> 80% dùng để huấn luyện (training), 20% dùng để kiểm tra (testing)</li>
              <li><strong>Huấn luyện:</strong> Mô hình học từ dữ liệu huấn luyện để phân biệt traffic DDoS và bình thường</li>
              <li><strong>Dự đoán:</strong> Áp dụng mô hình đã huấn luyện lên toàn bộ dataset để phân loại</li>
              <li><strong>Đánh giá:</strong> Tính các metrics (Accuracy, Precision, Recall, F1) trên tập testing</li>
            </ol>
          </div>
        </CardContent>
      </Card>

      {results.map((result) => {
        const details = ALGORITHM_DETAILS[result.modelType];
        if (!details) return null;
        
        return (
          <Card key={result.id}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5 text-primary" />
                {getModelName(result.modelType)}
              </CardTitle>
              <CardDescription>
                Accuracy: {(result.accuracy * 100).toFixed(1)}% | 
                F1 Score: {(result.f1Score * 100).toFixed(1)}% |
                Thời gian: {result.trainingTime.toFixed(2)}s
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold flex items-center gap-2 mb-2">
                  <Lightbulb className="h-4 w-4 text-chart-3" />
                  Cách hoạt động:
                </h4>
                <p className="text-sm text-muted-foreground">{details.howItWorks}</p>
              </div>
              
              <div className="rounded-lg bg-primary/10 p-4 border border-primary/20">
                <h4 className="font-semibold flex items-center gap-2 mb-2 text-primary">
                  <Calculator className="h-4 w-4" />
                  Công thức toán học:
                </h4>
                <code className="block text-sm bg-background p-2 rounded font-mono mb-2">
                  {details.formula}
                </code>
                <p className="text-sm text-muted-foreground">{details.formulaExplanation}</p>
              </div>
              
              <div>
                <h4 className="font-semibold flex items-center gap-2 mb-2">
                  <ListOrdered className="h-4 w-4 text-chart-1" />
                  Các bước thực hiện:
                </h4>
                <ol className="text-sm space-y-1 list-none">
                  {details.steps.map((step, i) => (
                    <li key={i} className="flex items-start gap-2 text-muted-foreground">
                      <span className="text-chart-1 font-medium min-w-[20px]">{i + 1}.</span>
                      <span>{step.replace(/^\d+\.\s*/, '')}</span>
                    </li>
                  ))}
                </ol>
              </div>
              
              <div className="rounded-lg bg-accent/30 p-3">
                <h4 className="font-semibold flex items-center gap-2 mb-2 text-sm">
                  <Settings className="h-4 w-4 text-muted-foreground" />
                  Tham số sử dụng:
                </h4>
                <div className="flex flex-wrap gap-2">
                  {details.parameters.map((param, i) => (
                    <span key={i} className="text-xs bg-background px-2 py-1 rounded border">
                      {param}
                    </span>
                  ))}
                </div>
              </div>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold flex items-center gap-2 mb-2 text-chart-2">
                    <CheckCircle2 className="h-4 w-4" />
                    Điểm mạnh:
                  </h4>
                  <ul className="text-sm space-y-1">
                    {details.strengths.map((s, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="text-chart-2">+</span>
                        <span className="text-muted-foreground">{s}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold flex items-center gap-2 mb-2 text-destructive">
                    <XCircle className="h-4 w-4" />
                    Điểm yếu:
                  </h4>
                  <ul className="text-sm space-y-1">
                    {details.weaknesses.map((w, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="text-destructive">-</span>
                        <span className="text-muted-foreground">{w}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              
              <div className="rounded-lg bg-accent/50 p-3">
                <h4 className="font-semibold flex items-center gap-2 text-sm">
                  <Target className="h-4 w-4 text-primary" />
                  Phù hợp nhất cho:
                </h4>
                <p className="text-sm text-muted-foreground mt-1">{details.bestFor}</p>
              </div>
              
              <ResultInterpretation result={result} getModelName={getModelName} />
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

interface ResultInterpretationProps {
  result: AnalysisResult;
  getModelName: (type: string) => string;
}

function ResultInterpretation({ result, getModelName }: ResultInterpretationProps) {
  const total = result.ddosDetected + result.normalTraffic;
  const ddosPercent = total > 0 ? (result.ddosDetected / total * 100).toFixed(1) : "0";
  const cm = result.confusionMatrix;
  
  const hasNoLabels = cm.truePositive === 0 && cm.falsePositive === 0 && cm.falseNegative === 0;
  const isPerfectNormal = result.accuracy === 1 && result.ddosDetected === 0;
  
  return (
    <div className="border-t pt-4">
      <h4 className="font-semibold flex items-center gap-2 mb-3">
        <Info className="h-4 w-4 text-primary" />
        Diễn giải kết quả:
      </h4>
      
      {isPerfectNormal && hasNoLabels ? (
        <div className="space-y-2 text-sm">
          <div className="p-3 rounded-lg bg-chart-3/10 border border-chart-3/30">
            <p className="font-medium text-chart-3">Cảnh báo: Kết quả có thể không chính xác</p>
            <p className="text-muted-foreground mt-1">
              Accuracy 100% và 0 DDoS detected thường xảy ra khi dataset <strong>không có cột label</strong>. 
              Trong trường hợp này, mô hình mặc định coi tất cả traffic là bình thường vì không có dữ liệu 
              huấn luyện để phân biệt DDoS.
            </p>
          </div>
          <p className="text-muted-foreground">
            <strong>Giải pháp:</strong> Sử dụng dataset có cột <code className="bg-accent px-1 rounded">label</code>, 
            <code className="bg-accent px-1 rounded">class</code>, hoặc <code className="bg-accent px-1 rounded">attack</code> 
            với giá trị 0 (normal) và 1 (DDoS) để mô hình có thể học và phân loại.
          </p>
        </div>
      ) : (
        <div className="space-y-2 text-sm text-muted-foreground">
          <p>
            <strong>{getModelName(result.modelType)}</strong> đã phân tích {total.toLocaleString()} mẫu traffic và 
            phát hiện <strong className="text-destructive">{ddosPercent}%</strong> là tấn công DDoS.
          </p>
          
          {result.precision > 0 && (
            <p>
              <strong>Precision {(result.precision * 100).toFixed(1)}%:</strong> Trong các mẫu được dự đoán là DDoS, 
              {(result.precision * 100).toFixed(0)}% thực sự là DDoS (không báo động giả).
            </p>
          )}
          
          {result.recall > 0 && (
            <p>
              <strong>Recall {(result.recall * 100).toFixed(1)}%:</strong> Mô hình phát hiện được 
              {(result.recall * 100).toFixed(0)}% tổng số cuộc tấn công DDoS trong dataset.
            </p>
          )}
          
          <p>
            <strong>Confusion Matrix:</strong> TP={cm.truePositive} (đúng DDoS), 
            TN={cm.trueNegative} (đúng Normal), 
            FP={cm.falsePositive} (báo động giả), 
            FN={cm.falseNegative} (bỏ sót tấn công).
          </p>
        </div>
      )}
    </div>
  );
}

interface AttackTypesAnalysisProps {
  results: AnalysisResult[];
  getModelName: (type: string) => string;
}

function AttackTypesAnalysis({ results, getModelName }: AttackTypesAnalysisProps) {
  const allAttackTypes: AttackTypeResult[] = [];
  
  for (const result of results) {
    if (result.attackTypes) {
      for (const at of result.attackTypes) {
        const existing = allAttackTypes.find(a => a.type === at.type);
        if (existing) {
          existing.count = Math.max(existing.count, at.count);
          existing.confidence = Math.max(existing.confidence, at.confidence);
          at.indicators.forEach(ind => {
            if (!existing.indicators.includes(ind)) {
              existing.indicators.push(ind);
            }
          });
        } else {
          allAttackTypes.push({ ...at });
        }
      }
    }
  }
  
  const sortedAttacks = allAttackTypes.sort((a, b) => b.count - a.count);
  const totalAttacks = sortedAttacks.reduce((sum, at) => sum + at.count, 0);
  
  const pieColors = [
    "hsl(0, 84%, 60%)",
    "hsl(24, 95%, 53%)",
    "hsl(45, 93%, 47%)",
    "hsl(142, 71%, 45%)",
    "hsl(199, 89%, 48%)",
    "hsl(262, 83%, 58%)",
    "hsl(328, 85%, 55%)",
    "hsl(47, 96%, 53%)",
  ];
  
  const pieData = sortedAttacks.map((at, idx) => ({
    name: ATTACK_TYPE_INFO[at.type]?.nameVi || at.type,
    value: at.count,
    color: pieColors[idx % pieColors.length],
  }));
  
  const lucidResult = results.find(r => r.modelType === "lucid_cnn");

  if (sortedAttacks.length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-lg font-medium">Không phát hiện loại tấn công</p>
          <p className="text-muted-foreground mt-2">
            Không có mẫu DDoS nào được phát hiện hoặc dataset không chứa thông tin cổng/giao thức cần thiết để phân loại.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {lucidResult?.lucidAnalysis && (
        <Card className="border-primary/50 bg-primary/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Zap className="h-5 w-5 text-primary" />
              LUCID-inspired Neural Network
            </CardTitle>
            <CardDescription>
              Mạng neural với convolution filters - lấy cảm hứng từ nghiên cứu LUCID (IEEE TNSM 2020)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              <div className="text-center p-3 rounded-lg bg-accent/50">
                <p className="text-2xl font-bold text-primary">{lucidResult.lucidAnalysis.cnnLayers}</p>
                <p className="text-xs text-muted-foreground">Conv Layers</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-accent/50">
                <p className="text-2xl font-bold text-primary">{lucidResult.lucidAnalysis.kernelSize}x{lucidResult.lucidAnalysis.flowFeatures}</p>
                <p className="text-xs text-muted-foreground">Kernel Size</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-accent/50">
                <p className="text-2xl font-bold text-primary">{lucidResult.lucidAnalysis.timeWindow}s</p>
                <p className="text-xs text-muted-foreground">Time Window</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-accent/50">
                <p className="text-2xl font-bold text-primary">{lucidResult.lucidAnalysis.flowFeatures}</p>
                <p className="text-xs text-muted-foreground">Flow Features</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-accent/50">
                <p className="text-2xl font-bold text-chart-3">{(lucidResult.lucidAnalysis.anomalyScore * 100).toFixed(1)}%</p>
                <p className="text-xs text-muted-foreground">Anomaly Score</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-accent/50">
                <p className="text-2xl font-bold text-chart-2">{(lucidResult.lucidAnalysis.confidence * 100).toFixed(1)}%</p>
                <p className="text-xs text-muted-foreground">Confidence</p>
              </div>
            </div>
            <div className="mt-4 p-4 rounded-lg bg-background">
              <h4 className="font-semibold mb-2">Công thức Neural Network:</h4>
              <code className="text-sm text-muted-foreground">
                output = sigmoid(W × maxpool(ReLU(conv(X, K))) + b)
              </code>
              <p className="text-xs text-muted-foreground mt-2">
                Trong đó: X = feature matrix, K = {lucidResult.lucidAnalysis.kernelSize}×{lucidResult.lucidAnalysis.flowFeatures} convolution filters, 
                maxpool = global max pooling, W = FC weights, b = bias
              </p>
              <p className="text-xs text-chart-3 mt-2">
                Lưu ý: Đây là mô hình nhẹ lấy cảm hứng từ LUCID, phù hợp cho môi trường web. 
                Để có độ chính xác cao nhất, hãy sử dụng LUCID gốc từ GitHub với TensorFlow/Python.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <BarChart3 className="h-5 w-5 text-primary" />
              Phân Bố Loại Tấn Công
            </CardTitle>
            <CardDescription>
              Tổng cộng {totalAttacks.toLocaleString()} mẫu DDoS được phân loại
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {pieData.map((entry, idx) => (
                      <Cell key={`cell-${idx}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value: number) => [value.toLocaleString(), "Số lượng"]}
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--background))', 
                      border: '1px solid hsl(var(--border))' 
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              Chi Tiết Loại Tấn Công
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {sortedAttacks.map((attack, idx) => {
                const info = ATTACK_TYPE_INFO[attack.type];
                return (
                  <div key={attack.type} className="p-3 rounded-lg border">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: pieColors[idx % pieColors.length] }}
                        />
                        <span className="font-medium">{info?.nameVi || attack.type}</span>
                      </div>
                      <Badge variant={attack.confidence > 0.8 ? "destructive" : "secondary"}>
                        {(attack.confidence * 100).toFixed(0)}% tin cậy
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between text-sm text-muted-foreground">
                      <span>{attack.count.toLocaleString()} mẫu ({attack.percentage.toFixed(1)}%)</span>
                    </div>
                    {attack.indicators.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {attack.indicators.slice(0, 3).map((ind, i) => (
                          <Badge key={i} variant="outline" className="text-xs">
                            {ind}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <BookOpen className="h-5 w-5 text-primary" />
            Công Thức Phát Hiện Theo Loại
          </CardTitle>
          <CardDescription>
            Các quy tắc và ngưỡng được sử dụng để phân loại loại tấn công DDoS
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            {sortedAttacks.map((attack) => {
              const info = ATTACK_TYPE_INFO[attack.type];
              if (!info) return null;
              
              return (
                <div key={attack.type} className="p-4 rounded-lg border bg-accent/20">
                  <h4 className="font-semibold flex items-center gap-2 mb-2">
                    <Target className="h-4 w-4 text-primary" />
                    {info.nameVi} ({info.name})
                  </h4>
                  <p className="text-sm text-muted-foreground mb-3">{info.description}</p>
                  
                  <div className="mb-3">
                    <span className="text-xs font-medium text-primary">Công thức phát hiện:</span>
                    <code className="block mt-1 p-2 rounded bg-background text-xs overflow-x-auto">
                      {info.formula}
                    </code>
                  </div>
                  
                  <div>
                    <span className="text-xs font-medium text-primary">Dấu hiệu nhận biết:</span>
                    <ul className="mt-1 space-y-1">
                      {info.indicators.map((ind, i) => (
                        <li key={i} className="text-xs text-muted-foreground flex items-center gap-1">
                          <CheckCircle2 className="h-3 w-3 text-chart-2" />
                          {ind}
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  {info.ports && info.ports.length > 0 && (
                    <div className="mt-2">
                      <span className="text-xs font-medium text-primary">Cổng liên quan: </span>
                      <span className="text-xs text-muted-foreground">
                        {info.ports.join(", ")}
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
