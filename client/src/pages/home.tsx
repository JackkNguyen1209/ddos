import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Shield, Database, Brain, BarChart3, Loader2, AlertTriangle, CheckCircle, Eye, Tag, Sparkles } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { FileUpload } from "@/components/file-upload";
import { DataPreview } from "@/components/data-preview";
import { ModelSelector } from "@/components/model-selector";
import { AnalysisResults } from "@/components/analysis-results";
import { EmptyState } from "@/components/empty-state";
import { LearningStats } from "@/components/learning-stats";
import { FeatureReport } from "@/components/feature-report";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Dataset, DataRow, AnalysisResult, MLModelType, FeatureValidation } from "@shared/schema";

interface SchemaDetection {
  schemaType: 'cicflowmeter' | 'event_log' | 'unknown';
  schemaConfidence: number;
  recommendedModels: string[];
  columnMappings: Record<string, string>;
  normalizedColumns: string[];
}

interface FeatureReportData {
  foundFeatures: string[];
  missingFeatures: string[];
  foundPercentage: number;
  nanCount: number;
  nanPercentage: number;
  infCount: number;
  infPercentage: number;
  isReliable: boolean;
  warnings: string[];
}

interface LabelStats {
  [label: string]: {
    count: number;
    percentage: number;
    category: string;
  };
}

interface DatasetResponse {
  dataset: Dataset;
  previewData: DataRow[];
  schemaDetection?: SchemaDetection;
  featureReport?: FeatureReportData;
  labelStats?: LabelStats;
}

function ModeIndicator({ dataset }: { dataset: Dataset }) {
  const mode = dataset.mode || "supervised";
  const fv = dataset.featureValidation as FeatureValidation | undefined;
  
  const confidenceColors = {
    high: "bg-green-500",
    medium: "bg-yellow-500",
    low: "bg-red-500",
  };
  
  return (
    <Card className="mb-4" data-testid="mode-indicator">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          {mode === "supervised" ? (
            <>
              <Tag className="h-4 w-4 text-green-600" />
              <span>Mode: SUPERVISED</span>
              <Badge className="bg-green-600 text-white">Có nhãn</Badge>
            </>
          ) : (
            <>
              <Eye className="h-4 w-4 text-blue-600" />
              <span>Mode: UNLABELED INFERENCE</span>
              <Badge className="bg-blue-600 text-white">Không có nhãn</Badge>
            </>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            {mode === "supervised" ? (
              <p className="text-sm text-muted-foreground">
                <CheckCircle className="inline h-4 w-4 text-green-600 mr-1" />
                Phát hiện cột label: <strong>{dataset.labelColumn}</strong>. Sẽ train và đánh giá Accuracy/Precision/Recall.
              </p>
            ) : (
              <p className="text-sm text-muted-foreground">
                <AlertTriangle className="inline h-4 w-4 text-yellow-600 mr-1" />
                Không có cột label. Sẽ chạy inference với Anomaly Score thay vì Accuracy.
              </p>
            )}
          </div>
          
          {fv && (
            <div className="flex flex-wrap gap-2 items-center">
              <span className="text-xs text-muted-foreground">Độ tin cậy:</span>
              <Badge className={`${confidenceColors[fv.confidenceLevel]} text-white text-xs`}>
                {fv.confidenceLevel === "high" ? "Cao" : fv.confidenceLevel === "medium" ? "Trung bình" : "Thấp"}
              </Badge>
              
              {fv.availableOptional.length > 0 && (
                <span className="text-xs text-muted-foreground">
                  Features: {fv.availableOptional.join(", ")}
                </span>
              )}
              
              {fv.missingRequired.length > 0 && (
                <span className="text-xs text-orange-600">
                  Thiếu: {fv.missingRequired.join(", ")}
                </span>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default function Home() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [selectedModels, setSelectedModels] = useState<MLModelType[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);

  const { data: datasetData, isLoading: isLoadingDataset } = useQuery<DatasetResponse>({
    queryKey: ["/api/dataset"],
  });

  const { data: results = [], isLoading: isLoadingResults } = useQuery<AnalysisResult[]>({
    queryKey: ["/api/results"],
    enabled: !!datasetData?.dataset,
  });

  const uploadMutation = useMutation({
    mutationFn: async ({ file, name }: { file: File; name: string }) => {
      let data: string;
      const isExcel = file.name.endsWith(".xlsx") || file.name.endsWith(".xls");
      
      if (isExcel) {
        const arrayBuffer = await file.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);
        let binary = "";
        for (let i = 0; i < bytes.length; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        data = btoa(binary);
      } else {
        data = await file.text();
      }
      
      setUploadProgress(30);
      const response = await apiRequest("POST", "/api/upload", { name, data });
      setUploadProgress(100);
      return response.json();
    },
    onSuccess: (data: DatasetResponse & { warning?: string }) => {
      queryClient.invalidateQueries({ queryKey: ["/api/dataset"] });
      queryClient.invalidateQueries({ queryKey: ["/api/results"] });
      
      if (data.warning) {
        toast({
          title: "Upload thành công (có cảnh báo)",
          description: data.warning,
          variant: "destructive",
        });
      } else {
        toast({
          title: "Upload thành công",
          description: "Dataset đã được tải lên và xử lý",
        });
      }
      setUploadProgress(0);
    },
    onError: (error: Error) => {
      toast({
        title: "Lỗi upload",
        description: error.message,
        variant: "destructive",
      });
      setUploadProgress(0);
    },
  });

  const analyzeMutation = useMutation({
    mutationFn: async (modelTypes: MLModelType[]) => {
      if (!datasetData?.dataset) throw new Error("No dataset");
      const response = await apiRequest("POST", "/api/analyze", {
        datasetId: datasetData.dataset.id,
        modelTypes,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/results"] });
      toast({
        title: "Phân tích hoàn tất",
        description: "Kết quả đã sẵn sàng để xem",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Lỗi phân tích",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleFileSelect = (file: File, name: string) => {
    setUploadProgress(10);
    uploadMutation.mutate({ file, name });
  };

  const handleAnalyze = () => {
    if (selectedModels.length === 0) {
      toast({
        title: "Chưa chọn mô hình",
        description: "Vui lòng chọn ít nhất một mô hình ML",
        variant: "destructive",
      });
      return;
    }
    analyzeMutation.mutate(selectedModels);
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-primary p-2">
              <Shield className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-bold">DDoS Detection ML</h1>
              <p className="text-sm text-muted-foreground">
                Phát hiện tấn công DDoS bằng Machine Learning
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="space-y-8">
          <section>
            <div className="flex items-center gap-2 mb-4">
              <Database className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Bước 1: Upload Dataset</h2>
            </div>
            <FileUpload
              onFileSelect={handleFileSelect}
              isUploading={uploadMutation.isPending}
              uploadProgress={uploadProgress}
            />
          </section>

          <section>
            <div className="flex items-center gap-2 mb-4">
              <Database className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Bước 2: Xem Trước Dữ Liệu</h2>
            </div>
            {isLoadingDataset ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
            ) : datasetData?.dataset ? (
              <div className="space-y-4">
                <ModeIndicator dataset={datasetData.dataset} />
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="md:col-span-2">
                    <DataPreview
                      dataset={datasetData.dataset}
                      previewData={datasetData.previewData}
                    />
                  </div>
                  <div>
                    <FeatureReport
                      schemaDetection={datasetData.schemaDetection}
                      featureReport={datasetData.featureReport}
                      labelStats={datasetData.labelStats}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <EmptyState
                type="dataset"
                title="Chưa có dataset"
                description="Upload file CSV để bắt đầu phân tích. Hệ thống sẽ tự động kiểm tra và lọc dữ liệu."
              />
            )}
          </section>

          <section>
            <div className="flex items-center gap-2 mb-4">
              <Brain className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Bước 3: Chọn Mô Hình ML</h2>
            </div>
            <ModelSelector
              selectedModels={selectedModels}
              onModelsChange={setSelectedModels}
              onAnalyze={handleAnalyze}
              isAnalyzing={analyzeMutation.isPending}
              disabled={!datasetData?.dataset}
            />
          </section>

          <section>
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Bước 4: Kết Quả Phân Tích</h2>
            </div>
            {isLoadingResults ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
            ) : results.length > 0 ? (
              <AnalysisResults results={results} />
            ) : (
              <EmptyState
                type="results"
                title="Chưa có kết quả"
                description="Chọn các mô hình ML và nhấn 'Bắt đầu phân tích' để xem kết quả phát hiện DDoS."
              />
            )}
          </section>

          <section>
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Bước 5: Hệ Thống Tự Học</h2>
            </div>
            <LearningStats />
          </section>
        </div>
      </main>

      <footer className="border-t bg-card/50 mt-12">
        <div className="container mx-auto px-4 py-6">
          <p className="text-center text-sm text-muted-foreground">
            DDoS Detection ML - Powered by Machine Learning Algorithms
          </p>
        </div>
      </footer>
    </div>
  );
}
