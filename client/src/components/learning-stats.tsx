import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  Brain, 
  Database, 
  TrendingUp, 
  Trash2, 
  RefreshCw,
  AlertTriangle,
  Shield,
  Target,
  Clock,
  Zap
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface LearningStats {
  totalSamples: number;
  ddosSamples: number;
  normalSamples: number;
  learnedPatterns: number;
  sessionsCount: number;
  lastLearningDate: string | null;
  modelImprovements: {
    modelType: string;
    improvement: number;
    currentAccuracy: number;
  }[];
}

interface LearnedPattern {
  id: number;
  patternName: string;
  attackType: string;
  confidence: number;
  sampleCount: number;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

const MODEL_NAMES: Record<string, string> = {
  decision_tree: "Decision Tree",
  random_forest: "Random Forest",
  knn: "K-NN",
  naive_bayes: "Naive Bayes",
  logistic_regression: "Logistic Regression",
  lucid_cnn: "LUCID CNN",
};

export function LearningStats() {
  const { toast } = useToast();
  
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useQuery<LearningStats>({
    queryKey: ["/api/learning/stats"],
  });
  
  const { data: patterns, isLoading: patternsLoading, refetch: refetchPatterns } = useQuery<LearnedPattern[]>({
    queryKey: ["/api/learning/patterns"],
  });
  
  const learnMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/learning/learn");
      return res.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Học thành công",
        description: data.message,
      });
      refetchStats();
      refetchPatterns();
    },
    onError: (error: Error) => {
      toast({
        title: "Lỗi",
        description: error.message,
        variant: "destructive",
      });
    },
  });
  
  const clearMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("DELETE", "/api/learning/clear");
      return res.json();
    },
    onSuccess: () => {
      toast({
        title: "Đã xóa dữ liệu học",
        description: "Tất cả dữ liệu học đã được xóa. Hệ thống sẽ bắt đầu học lại từ đầu.",
      });
      refetchStats();
      refetchPatterns();
    },
    onError: (error: Error) => {
      toast({
        title: "Lỗi",
        description: error.message,
        variant: "destructive",
      });
    },
  });
  
  if (statsLoading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <RefreshCw className="h-5 w-5 animate-spin mr-2" />
            <span>Đang tải thống kê học...</span>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  const hasData = stats && stats.totalSamples > 0;
  
  return (
    <div className="space-y-4">
      <Card className="bg-gradient-to-r from-primary/10 to-primary/5 border-primary/20">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            Hệ Thống Tự Học
          </CardTitle>
          <CardDescription>
            Hệ thống ghi nhớ dữ liệu từ các lần upload và tự cải thiện khả năng phát hiện DDoS
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <Button 
              onClick={() => learnMutation.mutate()}
              disabled={learnMutation.isPending}
              data-testid="button-learn"
            >
              {learnMutation.isPending ? (
                <RefreshCw className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Brain className="h-4 w-4 mr-2" />
              )}
              Học từ dữ liệu hiện tại
            </Button>
            
            {hasData && (
              <Button 
                variant="destructive" 
                onClick={() => clearMutation.mutate()}
                disabled={clearMutation.isPending}
                data-testid="button-clear-learning"
              >
                {clearMutation.isPending ? (
                  <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Trash2 className="h-4 w-4 mr-2" />
                )}
                Xóa dữ liệu học
              </Button>
            )}
          </div>
          
          {!hasData && (
            <div className="text-center py-4 text-muted-foreground">
              <Database className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>Chưa có dữ liệu học. Upload dataset và nhấn "Học từ dữ liệu hiện tại" để bắt đầu.</p>
            </div>
          )}
        </CardContent>
      </Card>
      
      {hasData && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4 text-center">
                <Database className="h-6 w-6 mx-auto mb-2 text-primary" />
                <div className="text-2xl font-bold">{stats?.totalSamples.toLocaleString()}</div>
                <div className="text-xs text-muted-foreground">Tổng mẫu đã học</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4 text-center">
                <AlertTriangle className="h-6 w-6 mx-auto mb-2 text-destructive" />
                <div className="text-2xl font-bold text-destructive">{stats?.ddosSamples.toLocaleString()}</div>
                <div className="text-xs text-muted-foreground">Mẫu DDoS</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4 text-center">
                <Shield className="h-6 w-6 mx-auto mb-2 text-green-500" />
                <div className="text-2xl font-bold text-green-500">{stats?.normalSamples.toLocaleString()}</div>
                <div className="text-xs text-muted-foreground">Mẫu Normal</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4 text-center">
                <Target className="h-6 w-6 mx-auto mb-2 text-chart-3" />
                <div className="text-2xl font-bold">{stats?.learnedPatterns}</div>
                <div className="text-xs text-muted-foreground">Patterns đã học</div>
              </CardContent>
            </Card>
          </div>
          
          {stats && stats.totalSamples > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <TrendingUp className="h-4 w-4" />
                  Phân bố dữ liệu học
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-destructive">DDoS</span>
                    <span>{((stats.ddosSamples / stats.totalSamples) * 100).toFixed(1)}%</span>
                    <span className="text-green-500">Normal</span>
                  </div>
                  <div className="flex h-3 rounded-full overflow-hidden">
                    <div 
                      className="bg-destructive" 
                      style={{ width: `${(stats.ddosSamples / stats.totalSamples) * 100}%` }}
                    />
                    <div 
                      className="bg-green-500" 
                      style={{ width: `${(stats.normalSamples / stats.totalSamples) * 100}%` }}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
          
          {stats?.modelImprovements && stats.modelImprovements.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  Tiến độ cải thiện mô hình
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {stats.modelImprovements.map((mi) => (
                    <div key={mi.modelType} className="flex items-center justify-between">
                      <span className="text-sm font-medium">{MODEL_NAMES[mi.modelType] || mi.modelType}</span>
                      <div className="flex items-center gap-2">
                        <Badge variant={mi.improvement > 0 ? "default" : mi.improvement < 0 ? "destructive" : "secondary"}>
                          {mi.improvement > 0 ? "+" : ""}{mi.improvement.toFixed(1)}%
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          ({(mi.currentAccuracy * 100).toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
          
          {patterns && patterns.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  Patterns DDoS đã học ({patterns.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {patterns.map((pattern) => (
                    <div key={pattern.id} className="flex items-center justify-between p-2 rounded-lg bg-muted/50">
                      <div>
                        <span className="font-medium">{pattern.patternName}</span>
                        <span className="text-xs text-muted-foreground ml-2">
                          ({pattern.sampleCount} mẫu)
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Progress value={pattern.confidence * 100} className="w-20 h-2" />
                        <Badge variant="outline">
                          {(pattern.confidence * 100).toFixed(0)}%
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
          
          {stats?.lastLearningDate && (
            <div className="text-xs text-muted-foreground flex items-center gap-1 justify-end">
              <Clock className="h-3 w-3" />
              Học lần cuối: {new Date(stats.lastLearningDate).toLocaleString("vi-VN")}
            </div>
          )}
        </>
      )}
    </div>
  );
}
