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

// DDoS Attack types
export const ddosAttackTypes = [
  "port_scan",
  "syn_flood",
  "udp_flood",
  "icmp_flood",
  "http_flood",
  "dns_amplification",
  "ntp_amplification",
  "ldap_reflection",
  "rdp_attack",
  "ssdp_amplification",
  "memcached_amplification",
  "unknown",
] as const;

export type DDoSAttackType = typeof ddosAttackTypes[number];

// Attack type info for display
export interface AttackTypeInfo {
  type: DDoSAttackType;
  name: string;
  nameVi: string;
  description: string;
  indicators: string[];
  formula: string;
  ports?: number[];
}

export const ATTACK_TYPE_INFO: Record<DDoSAttackType, AttackTypeInfo> = {
  port_scan: {
    type: "port_scan",
    name: "Port Scan",
    nameVi: "Quét cổng",
    description: "Kẻ tấn công quét nhiều cổng để tìm dịch vụ đang mở",
    indicators: ["Nhiều cổng đích khác nhau từ cùng IP nguồn", "Kết nối ngắn", "Thất bại nhiều"],
    formula: "unique_dst_ports_per_src_ip >= 10",
    ports: [],
  },
  syn_flood: {
    type: "syn_flood",
    name: "SYN Flood",
    nameVi: "Tấn công SYN Flood",
    description: "Gửi nhiều gói SYN mà không hoàn thành TCP handshake",
    indicators: ["Cờ SYN cao", "Không có ACK", "Kết nối half-open nhiều"],
    formula: "syn_count / total_packets > 0.8 AND ack_count / syn_count < 0.1",
    ports: [80, 443, 22, 21],
  },
  udp_flood: {
    type: "udp_flood",
    name: "UDP Flood",
    nameVi: "Tấn công UDP Flood",
    description: "Gửi lượng lớn gói UDP để làm quá tải băng thông",
    indicators: ["Protocol UDP", "Packet rate cao", "Kích thước nhỏ đồng đều"],
    formula: "protocol == UDP AND packets_per_sec > 10000 AND avg_packet_size < 100",
    ports: [53, 123, 161],
  },
  icmp_flood: {
    type: "icmp_flood",
    name: "ICMP Flood",
    nameVi: "Tấn công ICMP/Ping Flood",
    description: "Gửi nhiều gói ICMP (ping) để làm tê liệt mạng",
    indicators: ["Protocol ICMP", "Echo request cao", "Nhiều nguồn"],
    formula: "protocol == ICMP AND icmp_type == 8 AND packet_rate > 5000",
    ports: [],
  },
  http_flood: {
    type: "http_flood",
    name: "HTTP Flood",
    nameVi: "Tấn công HTTP Flood",
    description: "Gửi nhiều HTTP request để làm quá tải web server",
    indicators: ["Cổng 80/443", "Request rate cao", "Nhiều connection"],
    formula: "dst_port IN (80, 443) AND request_rate > 1000/s AND unique_src_ip > 100",
    ports: [80, 443, 8080],
  },
  dns_amplification: {
    type: "dns_amplification",
    name: "DNS Amplification",
    nameVi: "Tấn công khuếch đại DNS",
    description: "Sử dụng DNS server để khuếch đại traffic đến nạn nhân",
    indicators: ["Cổng 53", "Response lớn hơn request", "Spoofed IP"],
    formula: "dst_port == 53 AND response_size / request_size > 10 AND ip_spoofing_detected",
    ports: [53],
  },
  ntp_amplification: {
    type: "ntp_amplification",
    name: "NTP Amplification",
    nameVi: "Tấn công khuếch đại NTP",
    description: "Sử dụng NTP server để khuếch đại traffic",
    indicators: ["Cổng 123", "Monlist command", "Response lớn"],
    formula: "dst_port == 123 AND response_size / request_size > 50",
    ports: [123],
  },
  ldap_reflection: {
    type: "ldap_reflection",
    name: "LDAP Reflection",
    nameVi: "Tấn công phản xạ LDAP",
    description: "Lợi dụng LDAP server để phản xạ và khuếch đại traffic",
    indicators: ["Cổng 389/636", "Connectionless LDAP", "Response lớn"],
    formula: "dst_port IN (389, 636) AND protocol == UDP AND amplification_factor > 50",
    ports: [389, 636, 3268],
  },
  rdp_attack: {
    type: "rdp_attack",
    name: "RDP Attack",
    nameVi: "Tấn công Remote Desktop",
    description: "Brute-force hoặc exploit lỗ hổng RDP",
    indicators: ["Cổng 3389", "Login attempt nhiều", "Nhiều IP nguồn"],
    formula: "dst_port == 3389 AND failed_auth > 100 AND unique_src_ip > 10",
    ports: [3389],
  },
  ssdp_amplification: {
    type: "ssdp_amplification",
    name: "SSDP Amplification",
    nameVi: "Tấn công khuếch đại SSDP",
    description: "Sử dụng SSDP/UPnP để khuếch đại traffic",
    indicators: ["Cổng 1900", "M-SEARCH request", "Response lớn"],
    formula: "dst_port == 1900 AND protocol == UDP AND amplification > 30",
    ports: [1900],
  },
  memcached_amplification: {
    type: "memcached_amplification",
    name: "Memcached Amplification",
    nameVi: "Tấn công khuếch đại Memcached",
    description: "Lợi dụng memcached server để khuếch đại DDoS",
    indicators: ["Cổng 11211", "UDP protocol", "Khuếch đại cực lớn"],
    formula: "dst_port == 11211 AND protocol == UDP AND amplification > 50000",
    ports: [11211],
  },
  unknown: {
    type: "unknown",
    name: "Unknown Attack",
    nameVi: "Tấn công không xác định",
    description: "Loại tấn công chưa được phân loại rõ ràng",
    indicators: ["Pattern bất thường", "Không khớp signature đã biết"],
    formula: "anomaly_score > threshold AND !matches_known_pattern",
    ports: [],
  },
};

// ML Model types
export const mlModelTypes = [
  "decision_tree",
  "random_forest", 
  "knn",
  "naive_bayes",
  "logistic_regression",
  "lucid_cnn",
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

// Attack type detection result
export const attackTypeResultSchema = z.object({
  type: z.enum(ddosAttackTypes),
  count: z.number(),
  percentage: z.number(),
  confidence: z.number(),
  indicators: z.array(z.string()),
});

export type AttackTypeResult = z.infer<typeof attackTypeResultSchema>;

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
  attackTypes: z.array(attackTypeResultSchema).optional(),
  lucidAnalysis: z.object({
    cnnLayers: z.number(),
    kernelSize: z.number(),
    timeWindow: z.number(),
    flowFeatures: z.number(),
    anomalyScore: z.number(),
    confidence: z.number(),
  }).optional(),
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
  {
    type: "lucid_cnn",
    name: "LUCID Neural Network",
    description: "Mạng neural nhẹ lấy cảm hứng từ LUCID - phát hiện DDoS với convolution filters",
  },
];

// Detailed algorithm explanations with formulas
export const ALGORITHM_DETAILS: Record<string, {
  howItWorks: string;
  formula: string;
  formulaExplanation: string;
  steps: string[];
  strengths: string[];
  weaknesses: string[];
  bestFor: string;
  parameters: string[];
}> = {
  decision_tree: {
    howItWorks: "Cây quyết định xây dựng một cấu trúc cây bằng cách chia dữ liệu theo các ngưỡng của từng đặc trưng. Tại mỗi node, thuật toán chọn đặc trưng và ngưỡng tối ưu để phân chia dữ liệu thành 2 nhóm sao cho mỗi nhóm càng thuần nhất càng tốt.",
    formula: "Gini(D) = 1 - Σ(pᵢ)²",
    formulaExplanation: "Gini Index đo độ không thuần nhất của tập dữ liệu D. pᵢ là tỷ lệ mẫu thuộc lớp i. Gini = 0 nghĩa là tập dữ liệu hoàn toàn thuần nhất (chỉ có 1 lớp). Gini = 0.5 là tối đa (2 lớp bằng nhau).",
    steps: [
      "1. Tính Gini Index cho mỗi đặc trưng và ngưỡng có thể",
      "2. Chọn đặc trưng và ngưỡng có Gini thấp nhất (thuần nhất nhất)",
      "3. Chia dữ liệu thành 2 nhánh: <= ngưỡng và > ngưỡng",
      "4. Lặp lại cho mỗi nhánh đến khi đạt điều kiện dừng",
      "5. Gán nhãn lá = lớp chiếm đa số trong nhánh"
    ],
    strengths: ["Dễ hiểu và giải thích", "Xử lý được cả dữ liệu số và phân loại", "Không cần chuẩn hóa dữ liệu"],
    weaknesses: ["Dễ bị overfitting với dữ liệu nhiễu", "Nhạy cảm với thay đổi nhỏ trong dữ liệu", "Có thể tạo cây quá phức tạp"],
    bestFor: "Phát hiện các pattern rõ ràng trong traffic như SYN flood với ngưỡng cố định",
    parameters: ["Độ sâu tối đa: 10", "Số mẫu tối thiểu để chia: 2"]
  },
  random_forest: {
    howItWorks: "Random Forest tạo nhiều cây quyết định độc lập, mỗi cây được huấn luyện trên một tập con ngẫu nhiên của dữ liệu (bootstrap sampling). Khi dự đoán, mỗi cây đưa ra một phiếu bầu, và kết quả cuối cùng là lớp có nhiều phiếu nhất.",
    formula: "ŷ = mode(h₁(x), h₂(x), ..., hₙ(x))",
    formulaExplanation: "Kết quả dự đoán ŷ là mode (giá trị xuất hiện nhiều nhất) của các dự đoán từ n cây quyết định h₁, h₂, ..., hₙ. Đây là majority voting - lớp nào được nhiều cây chọn nhất sẽ là kết quả.",
    steps: [
      "1. Tạo n tập dữ liệu bootstrap (lấy mẫu có hoàn lại)",
      "2. Huấn luyện 1 cây quyết định trên mỗi tập bootstrap",
      "3. Mỗi cây chỉ xét một tập con đặc trưng ngẫu nhiên tại mỗi node",
      "4. Khi dự đoán: cho x qua tất cả n cây",
      "5. Kết quả = lớp được nhiều cây chọn nhất (majority vote)"
    ],
    strengths: ["Giảm overfitting so với cây đơn lẻ", "Ổn định với dữ liệu nhiễu", "Có thể đánh giá tầm quan trọng của đặc trưng"],
    weaknesses: ["Chậm hơn cây đơn lẻ", "Khó giải thích hơn", "Tốn nhiều bộ nhớ"],
    bestFor: "Phát hiện DDoS với nhiều loại tấn công khác nhau nhờ tính đa dạng của các cây",
    parameters: ["Số cây: 15", "Độ sâu mỗi cây: 8", "Bootstrap sampling: có"]
  },
  knn: {
    howItWorks: "K-Nearest Neighbors phân loại một mẫu dựa trên K điểm dữ liệu gần nhất với nó trong không gian đặc trưng. Khoảng cách được tính bằng Euclidean distance. Mẫu mới được gán nhãn của lớp chiếm đa số trong K láng giềng.",
    formula: "d(x,y) = √[Σ(xᵢ - yᵢ)²]",
    formulaExplanation: "Euclidean distance d(x,y) là căn bậc hai của tổng bình phương chênh lệch giữa các đặc trưng của điểm x và y. Điểm nào có d nhỏ nhất là láng giềng gần nhất.",
    steps: [
      "1. Chuẩn hóa tất cả đặc trưng về khoảng [0,1]",
      "2. Với mẫu mới x, tính khoảng cách đến tất cả mẫu huấn luyện",
      "3. Chọn K mẫu có khoảng cách nhỏ nhất (K láng giềng)",
      "4. Đếm số lượng mỗi lớp trong K láng giềng",
      "5. Gán x vào lớp chiếm đa số"
    ],
    strengths: ["Đơn giản, không cần huấn luyện phức tạp", "Linh hoạt với mọi hình dạng dữ liệu", "Dễ hiểu trực quan"],
    weaknesses: ["Chậm với dataset lớn (phải tính khoảng cách đến mọi điểm)", "Nhạy cảm với đặc trưng không liên quan", "Cần chuẩn hóa dữ liệu"],
    bestFor: "Phát hiện anomaly khi traffic DDoS tạo thành cluster khác biệt với traffic bình thường",
    parameters: ["K (số láng giềng): 5", "Metric: Euclidean distance", "Chuẩn hóa: Min-Max scaling"]
  },
  naive_bayes: {
    howItWorks: "Naive Bayes sử dụng định lý Bayes với giả định các đặc trưng độc lập với nhau (naive assumption). Thuật toán tính xác suất thuộc mỗi lớp dựa trên phân phối Gaussian, sau đó chọn lớp có xác suất cao nhất.",
    formula: "P(C|X) = P(X|C) × P(C) / P(X)",
    formulaExplanation: "Xác suất hậu nghiệm P(C|X) - xác suất mẫu thuộc lớp C khi biết đặc trưng X - tính bằng: likelihood P(X|C) nhân prior P(C) chia evidence P(X). Với giả định naive: P(X|C) = Π P(xᵢ|C).",
    steps: [
      "1. Tính P(C) - xác suất tiên nghiệm của mỗi lớp (tỷ lệ mẫu)",
      "2. Với mỗi đặc trưng, tính mean và std của mỗi lớp",
      "3. P(xᵢ|C) = Gaussian(xᵢ; μ, σ²) với μ, σ² của lớp C",
      "4. P(X|C) = tích các P(xᵢ|C) (giả định độc lập)",
      "5. Chọn lớp C có P(C|X) cao nhất"
    ],
    strengths: ["Rất nhanh cả huấn luyện và dự đoán", "Hoạt động tốt với dữ liệu ít", "Ít bị overfitting"],
    weaknesses: ["Giả định độc lập thường không đúng thực tế", "Nhạy cảm với đặc trưng tương quan cao", "Xác suất có thể không chính xác tuyệt đối"],
    bestFor: "Phân loại nhanh traffic realtime khi cần tốc độ cao",
    parameters: ["Phân phối: Gaussian", "Smoothing: epsilon = 1e-9"]
  },
  logistic_regression: {
    howItWorks: "Logistic Regression tìm đường phân cách tuyến tính trong không gian đặc trưng bằng cách học các trọng số (weights). Sử dụng hàm sigmoid để chuyển tổ hợp tuyến tính thành xác suất (0-1). Huấn luyện bằng gradient descent.",
    formula: "P(y=1|X) = σ(w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ) = 1/(1 + e^(-z))",
    formulaExplanation: "Xác suất thuộc lớp DDoS = sigmoid của tổ hợp tuyến tính các đặc trưng. wᵢ là trọng số (weight) của đặc trưng xᵢ. z = Σwᵢxᵢ. Sigmoid σ(z) nén giá trị z vào khoảng (0,1).",
    steps: [
      "1. Khởi tạo weights ngẫu nhiên nhỏ",
      "2. Với mỗi mẫu: z = w·x, ŷ = sigmoid(z)",
      "3. Tính loss: L = -(y·log(ŷ) + (1-y)·log(1-ŷ))",
      "4. Cập nhật weights: w = w - α·∇L (gradient descent)",
      "5. Lặp lại đến khi hội tụ hoặc đạt max iterations"
    ],
    strengths: ["Đưa ra xác suất dự đoán trực tiếp", "Nhanh và hiệu quả bộ nhớ", "Dễ giải thích qua trọng số - weight lớn = đặc trưng quan trọng"],
    weaknesses: ["Chỉ tìm được ranh giới tuyến tính", "Không bắt được quan hệ phi tuyến phức tạp", "Cần chuẩn hóa dữ liệu để hội tụ nhanh"],
    bestFor: "Phân loại khi traffic DDoS có đặc trưng khác biệt rõ ràng với traffic bình thường",
    parameters: ["Learning rate: 0.1", "Max iterations: 200", "Regularization: L2"]
  },
  lucid_cnn: {
    howItWorks: "LUCID Neural Network lấy cảm hứng từ nghiên cứu LUCID (IEEE TNSM 2020). Reshape features thành ma trận 2D, áp dụng convolution filters để trích xuất patterns, max pooling lấy đặc trưng quan trọng nhất, fully-connected layer để phân loại.",
    formula: "output = σ(W · maxpool(ReLU(X ⊛ K)) + b)",
    formulaExplanation: "X ⊛ K = convolution của input X với kernel K. ReLU(z) = max(0,z) là hàm kích hoạt. Maxpool lấy giá trị lớn nhất từ mỗi filter. W·pooled + b là lớp fully-connected. σ = sigmoid cho xác suất đầu ra.",
    steps: [
      "1. Reshape vector đặc trưng thành ma trận 2D (mô phỏng time-window)",
      "2. Áp dụng 32 convolution filters (kernel 3×N) lên ma trận",
      "3. ReLU activation: chỉ giữ giá trị dương",
      "4. Global max pooling: lấy giá trị lớn nhất từ mỗi filter",
      "5. Fully-connected layer: z = W·pooled + b",
      "6. Sigmoid: output = 1/(1+e^(-z))",
      "7. Backpropagation cập nhật cả kernel weights và FC weights"
    ],
    strengths: ["Tự động học patterns từ dữ liệu", "Xử lý được nhiều đặc trưng đồng thời", "Nhẹ và nhanh cho môi trường web"],
    weaknesses: ["Đơn giản hơn CNN thực sự (vertical convolution only)", "Cần dữ liệu có nhãn để huấn luyện", "Kết quả phụ thuộc random initialization"],
    bestFor: "Phát hiện DDoS khi cần kết hợp nhiều đặc trưng và học patterns tự động",
    parameters: ["Số filters: 32", "Kernel size: 3×N", "Learning rate: 0.01", "Epochs: 50"]
  },
};
