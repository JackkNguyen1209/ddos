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

// DDoS detection reasons
export const ddosReasonSchema = z.object({
  feature: z.string(),
  value: z.number(),
  threshold: z.number(),
  contribution: z.number(),
  description: z.string(),
});

export type DDoSReason = z.infer<typeof ddosReasonSchema>;

// Feature importance for explanations
export const featureImportanceSchema = z.object({
  feature: z.string(),
  importance: z.number(),
  description: z.string(),
});

export type FeatureImportance = z.infer<typeof featureImportanceSchema>;

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
  featureImportance: z.array(featureImportanceSchema).optional(),
  ddosReasons: z.array(ddosReasonSchema).optional(),
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

// Detailed algorithm explanations
export const ALGORITHM_DETAILS: Record<string, {
  howItWorks: string;
  strengths: string[];
  weaknesses: string[];
  bestFor: string;
}> = {
  decision_tree: {
    howItWorks: "Cây quyết định xây dựng một cấu trúc cây bằng cách chia dữ liệu theo các ngưỡng của từng đặc trưng. Tại mỗi node, thuật toán chọn đặc trưng và ngưỡng tối ưu để phân chia dữ liệu thành 2 nhóm sao cho mỗi nhóm càng thuần nhất càng tốt (sử dụng chỉ số Gini hoặc Information Gain). Quá trình này lặp lại cho đến khi đạt độ sâu tối đa hoặc dữ liệu đủ thuần nhất.",
    strengths: ["Dễ hiểu và giải thích", "Xử lý được cả dữ liệu số và phân loại", "Không cần chuẩn hóa dữ liệu"],
    weaknesses: ["Dễ bị overfitting với dữ liệu nhiễu", "Nhạy cảm với thay đổi nhỏ trong dữ liệu", "Có thể tạo cây quá phức tạp"],
    bestFor: "Phát hiện các pattern rõ ràng trong traffic như SYN flood với ngưỡng cố định",
  },
  random_forest: {
    howItWorks: "Random Forest tạo nhiều cây quyết định độc lập (trong trường hợp này là 15 cây), mỗi cây được huấn luyện trên một tập con ngẫu nhiên của dữ liệu (bootstrap sampling). Khi dự đoán, mỗi cây đưa ra một phiếu bầu, và kết quả cuối cùng là lớp có nhiều phiếu nhất (majority voting). Điều này giúp giảm overfitting và tăng độ ổn định.",
    strengths: ["Giảm overfitting so với cây đơn lẻ", "Ổn định với dữ liệu nhiễu", "Có thể đánh giá tầm quan trọng của đặc trưng"],
    weaknesses: ["Chậm hơn cây đơn lẻ", "Khó giải thích hơn", "Tốn nhiều bộ nhớ"],
    bestFor: "Phát hiện DDoS với nhiều loại tấn công khác nhau nhờ tính đa dạng của các cây",
  },
  knn: {
    howItWorks: "K-Nearest Neighbors (K=5) phân loại một mẫu dựa trên 5 điểm dữ liệu gần nhất với nó trong không gian đặc trưng. Khoảng cách được tính bằng Euclidean distance trên các đặc trưng đã chuẩn hóa. Mẫu mới được gán nhãn của lớp chiếm đa số trong 5 láng giềng gần nhất.",
    strengths: ["Đơn giản, không cần huấn luyện", "Linh hoạt với mọi hình dạng dữ liệu", "Dễ hiểu trực quan"],
    weaknesses: ["Chậm với dataset lớn", "Nhạy cảm với đặc trưng không liên quan", "Cần chuẩn hóa dữ liệu"],
    bestFor: "Phát hiện anomaly khi traffic DDoS tạo thành cluster khác biệt với traffic bình thường",
  },
  naive_bayes: {
    howItWorks: "Naive Bayes sử dụng định lý Bayes với giả định các đặc trưng độc lập với nhau (naive assumption). Thuật toán tính xác suất P(DDoS|features) và P(Normal|features) dựa trên phân phối Gaussian của mỗi đặc trưng trong từng lớp, sau đó chọn lớp có xác suất cao hơn.",
    strengths: ["Rất nhanh cả huấn luyện và dự đoán", "Hoạt động tốt với dữ liệu ít", "Ít bị overfitting"],
    weaknesses: ["Giả định độc lập thường không đúng", "Nhạy cảm với đặc trưng tương quan", "Có thể đưa ra xác suất không chính xác"],
    bestFor: "Phân loại nhanh traffic realtime khi cần tốc độ cao",
  },
  logistic_regression: {
    howItWorks: "Logistic Regression tìm một đường phân cách tuyến tính trong không gian đặc trưng bằng cách học các trọng số (weights) cho mỗi đặc trưng. Sử dụng hàm sigmoid để chuyển đổi tổ hợp tuyến tính thành xác suất (0-1). Huấn luyện bằng gradient descent với 200 iterations và learning rate 0.1.",
    strengths: ["Đưa ra xác suất dự đoán", "Nhanh và hiệu quả", "Dễ giải thích qua trọng số"],
    weaknesses: ["Chỉ tìm được ranh giới tuyến tính", "Không bắt được quan hệ phi tuyến", "Cần chuẩn hóa dữ liệu"],
    bestFor: "Phân loại khi traffic DDoS có đặc trưng khác biệt rõ ràng với traffic bình thường",
  },
};
