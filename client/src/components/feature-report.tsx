import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AlertTriangle, CheckCircle, FileSearch, Layers, Tag, AlertCircle } from "lucide-react";

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

interface FeatureReportProps {
  schemaDetection?: SchemaDetection;
  featureReport?: FeatureReportData;
  labelStats?: LabelStats;
}

const schemaTypeNames: Record<string, string> = {
  'cicflowmeter': 'CICFlowMeter',
  'event_log': 'Event/Log',
  'unknown': 'Không xác định'
};

const modelNames: Record<string, string> = {
  'random_forest': 'Random Forest',
  'decision_tree': 'Decision Tree',
  'knn': 'KNN',
  'naive_bayes': 'Naive Bayes',
  'logistic_regression': 'Logistic Regression',
  'lucid': 'LUCID Neural Network',
  'isolation_forest': 'Isolation Forest',
  'lof': 'LOF'
};

export function FeatureReport({ schemaDetection, featureReport, labelStats }: FeatureReportProps) {
  if (!schemaDetection && !featureReport) {
    return null;
  }

  return (
    <div className="space-y-4" data-testid="feature-report-container">
      {schemaDetection && (
        <Card data-testid="card-schema-detection">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <FileSearch className="h-4 w-4" />
              Schema Detection
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Loại Schema:</span>
              <Badge 
                variant={schemaDetection.schemaType === 'cicflowmeter' ? 'default' : 'secondary'}
                data-testid="badge-schema-type"
              >
                {schemaTypeNames[schemaDetection.schemaType]}
              </Badge>
            </div>
            
            <div className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Độ tin cậy:</span>
                <span className="font-medium" data-testid="text-schema-confidence">
                  {schemaDetection.schemaConfidence.toFixed(0)}%
                </span>
              </div>
              <Progress value={schemaDetection.schemaConfidence} className="h-2" data-testid="progress-schema-confidence" />
            </div>

            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Model đề xuất:</span>
              <div className="flex flex-wrap gap-1 mt-1" data-testid="container-recommended-models">
                {schemaDetection.recommendedModels.slice(0, 4).map((model) => (
                  <Badge key={model} variant="outline" className="text-xs" data-testid={`badge-model-${model}`}>
                    {modelNames[model] || model}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {featureReport && (
        <Card data-testid="card-feature-report">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Feature Report
              {featureReport.isReliable ? (
                <CheckCircle className="h-4 w-4 text-green-500 ml-auto" data-testid="icon-reliable" />
              ) : (
                <AlertTriangle className="h-4 w-4 text-yellow-500 ml-auto" data-testid="icon-unreliable" />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Features tìm thấy:</span>
                <span className="font-medium" data-testid="text-features-found">
                  {featureReport.foundFeatures.length} / {featureReport.foundFeatures.length + featureReport.missingFeatures.length}
                </span>
              </div>
              <Progress 
                value={featureReport.foundPercentage} 
                className={`h-2 ${featureReport.foundPercentage < 60 ? '[&>div]:bg-yellow-500' : ''}`}
                data-testid="progress-features"
              />
              {featureReport.foundPercentage < 60 && (
                <p className="text-xs text-yellow-600 flex items-center gap-1" data-testid="text-unreliable-warning">
                  <AlertCircle className="h-3 w-3" />
                  Dưới 60% - Kết quả không tin cậy
                </p>
              )}
            </div>

            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="flex items-center justify-between p-2 bg-muted/50 rounded">
                <span className="text-muted-foreground">NaN/Null:</span>
                <span 
                  className={featureReport.nanPercentage > 10 ? 'text-yellow-600 font-medium' : ''}
                  data-testid="text-nan-percentage"
                >
                  {featureReport.nanPercentage.toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between p-2 bg-muted/50 rounded">
                <span className="text-muted-foreground">Infinity:</span>
                <span 
                  className={featureReport.infPercentage > 1 ? 'text-yellow-600 font-medium' : ''}
                  data-testid="text-inf-percentage"
                >
                  {featureReport.infPercentage.toFixed(1)}%
                </span>
              </div>
            </div>

            {featureReport.warnings.length > 0 && (
              <div className="space-y-1" data-testid="container-warnings">
                {featureReport.warnings.map((warning, i) => (
                  <p key={i} className="text-xs text-yellow-600 flex items-start gap-1" data-testid={`text-warning-${i}`}>
                    <AlertTriangle className="h-3 w-3 mt-0.5 shrink-0" />
                    {warning}
                  </p>
                ))}
              </div>
            )}

            {featureReport.missingFeatures.length > 0 && (
              <details className="text-sm" data-testid="details-missing-features">
                <summary className="cursor-pointer text-muted-foreground hover:text-foreground" data-testid="summary-missing-features">
                  Xem {featureReport.missingFeatures.length} features thiếu
                </summary>
                <div className="mt-2 flex flex-wrap gap-1">
                  {featureReport.missingFeatures.slice(0, 10).map((f) => (
                    <Badge key={f} variant="outline" className="text-xs text-muted-foreground" data-testid={`badge-missing-${f}`}>
                      {f}
                    </Badge>
                  ))}
                  {featureReport.missingFeatures.length > 10 && (
                    <Badge variant="outline" className="text-xs" data-testid="badge-more-missing">
                      +{featureReport.missingFeatures.length - 10} more
                    </Badge>
                  )}
                </div>
              </details>
            )}
          </CardContent>
        </Card>
      )}

      {labelStats && Object.keys(labelStats).length > 0 && (
        <Card data-testid="card-label-distribution">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Tag className="h-4 w-4" />
              Label Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-48 overflow-y-auto" data-testid="container-label-stats">
              {Object.entries(labelStats)
                .sort((a, b) => b[1].count - a[1].count)
                .slice(0, 10)
                .map(([label, stats], index) => (
                  <div key={label} className="space-y-1" data-testid={`label-item-${index}`}>
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <Badge 
                          variant={stats.category === 'normal' ? 'secondary' : 'destructive'}
                          className="text-xs"
                          data-testid={`badge-label-category-${index}`}
                        >
                          {stats.category === 'normal' ? 'Normal' : 'Attack'}
                        </Badge>
                        <span className="text-muted-foreground truncate max-w-32" title={label} data-testid={`text-label-name-${index}`}>
                          {label.length > 20 ? label.slice(0, 20) + '...' : label}
                        </span>
                      </div>
                      <span className="text-xs" data-testid={`text-label-count-${index}`}>
                        {stats.count.toLocaleString()} ({stats.percentage.toFixed(1)}%)
                      </span>
                    </div>
                    <Progress value={stats.percentage} className="h-1" data-testid={`progress-label-${index}`} />
                  </div>
                ))}
              {Object.keys(labelStats).length > 10 && (
                <p className="text-xs text-muted-foreground text-center" data-testid="text-more-labels">
                  +{Object.keys(labelStats).length - 10} labels khác
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
