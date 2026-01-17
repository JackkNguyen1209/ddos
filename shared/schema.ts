import { z } from "zod";

// Dataset schema
export const datasetSchema = z.object({
  id: z.string(),
  name: z.string(),
  originalRowCount: z.number(),
  cleanedRowCount: z.number(),
  columns: z.array(z.string()),
  uploadedAt: z.string(),
  isProcessed: z.boolean(),
  dataQuality: z.object({
    missingValues: z.number(),
    duplicates: z.number(),
    outliers: z.number(),
    cleanedPercentage: z.number(),
  }),
});

export type Dataset = z.infer<typeof datasetSchema>;

// ML Model types
export const mlModelTypes = [
  "decision_tree",
  "random_forest", 
  "knn",
  "naive_bayes",
  "logistic_regression",
] as const;

export type MLModelType = typeof mlModelTypes[number];

export const mlModelSchema = z.object({
  type: z.enum(mlModelTypes),
  name: z.string(),
  description: z.string(),
});

export type MLModel = z.infer<typeof mlModelSchema>;

// Analysis result schema
export const analysisResultSchema = z.object({
  id: z.string(),
  datasetId: z.string(),
  modelType: z.enum(mlModelTypes),
  accuracy: z.number(),
  precision: z.number(),
  recall: z.number(),
  f1Score: z.number(),
  confusionMatrix: z.object({
    truePositive: z.number(),
    trueNegative: z.number(),
    falsePositive: z.number(),
    falseNegative: z.number(),
  }),
  trainingTime: z.number(),
  ddosDetected: z.number(),
  normalTraffic: z.number(),
  analyzedAt: z.string(),
});

export type AnalysisResult = z.infer<typeof analysisResultSchema>;

// Data row for preview
export const dataRowSchema = z.record(z.string(), z.union([z.string(), z.number(), z.null()]));
export type DataRow = z.infer<typeof dataRowSchema>;

// Upload request
export const uploadDatasetSchema = z.object({
  name: z.string().min(1, "Dataset name is required"),
  data: z.string(), // CSV content
});

export type UploadDataset = z.infer<typeof uploadDatasetSchema>;

// Analysis request
export const analyzeRequestSchema = z.object({
  datasetId: z.string(),
  modelTypes: z.array(z.enum(mlModelTypes)).min(1, "Select at least one model"),
});

export type AnalyzeRequest = z.infer<typeof analyzeRequestSchema>;

// Model info for display
export const ML_MODELS: MLModel[] = [
  {
    type: "decision_tree",
    name: "Decision Tree",
    description: "Cây quyết định - phân loại dựa trên các quy tắc if-then",
  },
  {
    type: "random_forest",
    name: "Random Forest",
    description: "Rừng ngẫu nhiên - kết hợp nhiều cây quyết định",
  },
  {
    type: "knn",
    name: "K-Nearest Neighbors",
    description: "KNN - phân loại dựa trên k điểm lân cận gần nhất",
  },
  {
    type: "naive_bayes",
    name: "Naive Bayes",
    description: "Bayes ngây thơ - phân loại dựa trên xác suất",
  },
  {
    type: "logistic_regression",
    name: "Logistic Regression",
    description: "Hồi quy logistic - phân loại nhị phân",
  },
];
