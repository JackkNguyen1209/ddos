import { z } from "zod";
import { pgTable, serial, text, real, integer, timestamp, jsonb, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";

// ============== DATABASE TABLES FOR SELF-LEARNING ==============

// Training samples - accumulated from uploads
export const trainingSamples = pgTable("training_samples", {
  id: serial("id").primaryKey(),
  features: jsonb("features").notNull(), // Array of feature values
  label: integer("label").notNull(), // 0 = Normal, 1 = DDoS
  featureNames: jsonb("feature_names").notNull(), // Column names
  attackType: text("attack_type"), // Optional: type of attack detected
  source: text("source").notNull(), // Which upload this came from
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertTrainingSampleSchema = createInsertSchema(trainingSamples).omit({ id: true, createdAt: true });
export type InsertTrainingSample = z.infer<typeof insertTrainingSampleSchema>;
export type TrainingSample = typeof trainingSamples.$inferSelect;

// Model performance tracking
export const modelPerformance = pgTable("model_performance", {
  id: serial("id").primaryKey(),
  modelType: text("model_type").notNull(),
  accuracy: real("accuracy").notNull(),
  precision: real("precision_score").notNull(),
  recall: real("recall").notNull(),
  f1Score: real("f1_score").notNull(),
  trainingSamplesCount: integer("training_samples_count").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertModelPerformanceSchema = createInsertSchema(modelPerformance).omit({ id: true, createdAt: true });
export type InsertModelPerformance = z.infer<typeof insertModelPerformanceSchema>;
export type ModelPerformance = typeof modelPerformance.$inferSelect;

// Learning sessions - track each learning cycle
export const learningSessions = pgTable("learning_sessions", {
  id: serial("id").primaryKey(),
  sessionName: text("session_name").notNull(),
  samplesAdded: integer("samples_added").notNull(),
  totalSamplesAfter: integer("total_samples_after").notNull(),
  ddosSamplesAdded: integer("ddos_samples_added").notNull(),
  normalSamplesAdded: integer("normal_samples_added").notNull(),
  improvementPercentage: real("improvement_percentage"), // How much the model improved
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertLearningSessionSchema = createInsertSchema(learningSessions).omit({ id: true, createdAt: true });
export type InsertLearningSession = z.infer<typeof insertLearningSessionSchema>;
export type LearningSession = typeof learningSessions.$inferSelect;

// Learned patterns - store detected DDoS patterns
export const learnedPatterns = pgTable("learned_patterns", {
  id: serial("id").primaryKey(),
  patternName: text("pattern_name").notNull(),
  attackType: text("attack_type").notNull(),
  featureThresholds: jsonb("feature_thresholds").notNull(), // e.g., {"packets": 10000, "bytes": 1000000}
  confidence: real("confidence").notNull(),
  sampleCount: integer("sample_count").notNull(), // How many samples this pattern was learned from
  isActive: boolean("is_active").default(true).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const insertLearnedPatternSchema = createInsertSchema(learnedPatterns).omit({ id: true, createdAt: true, updatedAt: true });
export type InsertLearnedPattern = z.infer<typeof insertLearnedPatternSchema>;
export type LearnedPattern = typeof learnedPatterns.$inferSelect;

// User feedback/corrections on analysis results
export const userFeedback = pgTable("user_feedback", {
  id: serial("id").primaryKey(),
  rowIndex: integer("row_index").notNull(),
  originalLabel: text("original_label"),
  correctedLabel: text("corrected_label").notNull(),
  isAttack: boolean("is_attack").notNull(),
  category: text("category").notNull(),
  severity: text("severity").notNull(),
  userNotes: text("user_notes"),
  features: jsonb("features"),
  datasetName: text("dataset_name"),
  isApplied: boolean("is_applied").default(false).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertUserFeedbackSchema = createInsertSchema(userFeedback).omit({ id: true, createdAt: true });
export type InsertUserFeedback = z.infer<typeof insertUserFeedbackSchema>;
export type UserFeedback = typeof userFeedback.$inferSelect;

// Custom tags for data rows
export const userTags = pgTable("user_tags", {
  id: serial("id").primaryKey(),
  tagName: text("tag_name").notNull(),
  tagColor: text("tag_color").notNull(),
  description: text("description"),
  isAttackTag: boolean("is_attack_tag").default(false).notNull(),
  usageCount: integer("usage_count").default(0).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertUserTagSchema = createInsertSchema(userTags).omit({ id: true, createdAt: true });
export type InsertUserTag = z.infer<typeof insertUserTagSchema>;
export type UserTag = typeof userTags.$inferSelect;

// Audit logs for tracking all actions
export const auditLogs = pgTable("audit_logs", {
  id: serial("id").primaryKey(),
  action: text("action").notNull(), // upload, analyze, delete, feedback, export
  entityType: text("entity_type").notNull(), // dataset, analysis, label, feedback
  entityId: text("entity_id"),
  details: jsonb("details"), // Additional action details
  ipAddress: text("ip_address"),
  userAgent: text("user_agent"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertAuditLogSchema = createInsertSchema(auditLogs).omit({ id: true, createdAt: true });
export type InsertAuditLog = z.infer<typeof insertAuditLogSchema>;
export type AuditLog = typeof auditLogs.$inferSelect;

// ============== DATASETS TABLE (persistent storage) ==============
export const datasets = pgTable("datasets", {
  id: text("id").primaryKey(), // UUID
  name: text("name").notNull(),
  originalRowCount: integer("original_row_count").notNull(),
  cleanedRowCount: integer("cleaned_row_count").notNull(),
  columns: jsonb("columns").notNull(), // string[]
  mode: text("mode").notNull(), // supervised | unlabeled
  labelColumn: text("label_column"),
  featureValidation: jsonb("feature_validation"),
  dataQuality: jsonb("data_quality").notNull(),
  schemaType: text("schema_type"), // cicflowmeter | event_log | unknown
  status: text("status").default("ready").notNull(), // uploading | processing | ready | error
  filePath: text("file_path"), // Path to stored file
  previewData: jsonb("preview_data"), // First 10 rows for preview
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const insertDatasetSchema = createInsertSchema(datasets).omit({ createdAt: true, updatedAt: true });
export type InsertDataset = z.infer<typeof insertDatasetSchema>;
export type DatasetRecord = typeof datasets.$inferSelect;

// ============== ANALYSIS RESULTS TABLE ==============
export const analysisResults = pgTable("analysis_results", {
  id: serial("id").primaryKey(),
  datasetId: text("dataset_id").notNull(),
  modelType: text("model_type").notNull(),
  accuracy: real("accuracy"),
  precision: real("precision_score"),
  recall: real("recall"),
  f1Score: real("f1_score"),
  trainingTime: real("training_time").notNull(),
  ddosDetected: integer("ddos_detected").notNull(),
  normalTraffic: integer("normal_traffic").notNull(),
  mode: text("mode").notNull(), // supervised | unlabeled
  attackTypes: jsonb("attack_types"),
  confusionMatrix: jsonb("confusion_matrix"),
  unlabeledReport: jsonb("unlabeled_report"),
  advancedMetrics: jsonb("advanced_metrics"),
  featureImportance: jsonb("feature_importance"),
  hyperparameters: jsonb("hyperparameters"),
  status: text("status").default("completed").notNull(), // pending | running | completed | error
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertAnalysisResultSchema = createInsertSchema(analysisResults).omit({ id: true, createdAt: true });
export type InsertAnalysisResult = z.infer<typeof insertAnalysisResultSchema>;
export type AnalysisResultRecord = typeof analysisResults.$inferSelect;

// ============== CONFUSION MATRIX ==============
export interface ConfusionMatrix {
  truePositives: number;
  trueNegatives: number;
  falsePositives: number;
  falseNegatives: number;
  matrix: number[][];
  labels: string[];
}

// Detection Mode - Supervised (có nhãn) vs Unlabeled (không có nhãn)
export const detectionModes = ["supervised", "unlabeled"] as const;
export type DetectionMode = typeof detectionModes[number];

// Feature Contract - Định nghĩa các feature cần thiết
export const FEATURE_CONTRACT = {
  // Required features - cần ít nhất 1 từ mỗi nhóm để phân tích có ý nghĩa
  required: {
    timing: ["duration", "dur", "time", "timestamp", "start_time", "stime", "ltime"],
    volume: ["bytes", "sbytes", "dbytes", "totlen_fwd_pkts", "totlen_bwd_pkts", "tot_len", "total_bytes", "bps", "pps"],
    packets: ["packets", "spkts", "dpkts", "tot_fwd_pkts", "tot_bwd_pkts", "total_packets", "pkts"],
  },
  // Optional features - bổ sung thêm thông tin
  optional: {
    network: ["src_ip", "dst_ip", "srcip", "dstip", "saddr", "daddr", "src_port", "dst_port", "sport", "dport"],
    protocol: ["protocol", "proto", "service", "state", "flags", "tcp_flags"],
    statistics: ["mean", "std", "var", "min", "max", "iat", "psh", "urg", "fin", "syn", "rst", "ack"],
    labels: ["label", "class", "attack", "attack_cat", "category", "target", "is_attack", "ddos"],
  }
} as const;

// Feature validation result
export interface FeatureValidation {
  hasTimingFeatures: boolean;
  hasVolumeFeatures: boolean;
  hasPacketFeatures: boolean;
  hasNetworkFeatures: boolean;
  hasProtocolFeatures: boolean;
  hasLabelColumn: boolean;
  detectedLabelColumn: string | null;
  missingRequired: string[];
  availableOptional: string[];
  confidenceLevel: "high" | "medium" | "low";
  confidenceReason: string;
}

// Data validation result - phát hiện loại file
export interface DataValidationResult {
  isSchemaFile: boolean;           // File mô tả schema/dictionary
  isValidDataset: boolean;         // File dữ liệu hợp lệ
  mode: DetectionMode;             // Supervised hoặc Unlabeled
  featureValidation: FeatureValidation;
  warnings: string[];
  errors: string[];
}

// Anomaly Detection Result - cho Unlabeled mode
export interface AnomalyResult {
  isolationForestScore: number;    // Anomaly score từ Isolation Forest
  lofScore: number;                // Local Outlier Factor score
  combinedAnomalyScore: number;    // Score kết hợp
  isAnomaly: boolean;              // Kết luận có phải anomaly không
}

// Unlabeled Inference Report
export interface UnlabeledReport {
  // Confidence Report
  scoreDistribution: {
    min: number;
    max: number;
    mean: number;
    std: number;
    percentiles: { p25: number; p50: number; p75: number; p90: number; p95: number };
  };
  alertRate: number;               // Tỷ lệ cảnh báo
  
  // Data Quality Report
  dataQuality: {
    missingRate: number;
    invalidValues: number;
    totalRows: number;
    validRows: number;
  };
  
  // Drift Detection (so với baseline)
  driftReport?: {
    featureShifts: { feature: string; shift: number; severity: "low" | "medium" | "high" }[];
    overallDrift: number;
    isDistributionShifted: boolean;
  };
}

// Dataset schema
export const datasetSchema = z.object({
  id: z.string(),
  name: z.string(),
  originalRowCount: z.number(),
  cleanedRowCount: z.number(),
  columns: z.array(z.string()),
  uploadedAt: z.string(),
  isProcessed: z.boolean(),
  mode: z.enum(detectionModes).optional(),  // Thêm mode
  labelColumn: z.string().optional(),       // Cột label được phát hiện
  featureValidation: z.any().optional(),    // Feature validation result
  dataQuality: z.object({
    missingValues: z.number(),
    duplicates: z.number(),
    outliers: z.number(),
    cleanedPercentage: z.number(),
  }),
});

export type Dataset = z.infer<typeof datasetSchema>;

// Attack Categories - Phân loại nguồn tấn công
export const attackCategories = [
  "reconnaissance",      // Trinh sát - quét mạng
  "bruteforce",          // Tấn công brute force mật khẩu
  "remote_access",       // Tấn công truy cập từ xa
  "volumetric",          // Tấn công làm ngập (flood)
  "amplification",       // Tấn công khuếch đại
  "application_layer",   // Tấn công lớp ứng dụng
  "protocol_exploit",    // Khai thác lỗ hổng giao thức
] as const;

export type AttackCategory = typeof attackCategories[number];

// Attack Category Info
export interface AttackCategoryInfo {
  category: AttackCategory;
  name: string;
  nameVi: string;
  description: string;
  color: string;
  icon: string;
}

export const ATTACK_CATEGORY_INFO: Record<AttackCategory, AttackCategoryInfo> = {
  reconnaissance: {
    category: "reconnaissance",
    name: "Reconnaissance",
    nameVi: "Trinh sát mạng",
    description: "Quét và thu thập thông tin về hệ thống mục tiêu trước khi tấn công chính",
    color: "bg-yellow-500",
    icon: "Search"
  },
  bruteforce: {
    category: "bruteforce",
    name: "Brute Force",
    nameVi: "Tấn công Brute Force",
    description: "Thử đăng nhập bằng cách dò mật khẩu liên tục với nhiều tổ hợp khác nhau",
    color: "bg-orange-500",
    icon: "Key"
  },
  remote_access: {
    category: "remote_access",
    name: "Remote Access",
    nameVi: "Tấn công truy cập từ xa",
    description: "Cố gắng truy cập trái phép vào hệ thống qua các dịch vụ điều khiển từ xa",
    color: "bg-red-600",
    icon: "Monitor"
  },
  volumetric: {
    category: "volumetric",
    name: "Volumetric Attack",
    nameVi: "Tấn công làm ngập",
    description: "Gửi lượng lớn traffic để làm quá tải băng thông và tài nguyên mạng",
    color: "bg-purple-500",
    icon: "Waves"
  },
  amplification: {
    category: "amplification",
    name: "Amplification Attack",
    nameVi: "Tấn công khuếch đại",
    description: "Lợi dụng server trung gian để khuếch đại traffic gấp nhiều lần đến nạn nhân",
    color: "bg-red-500",
    icon: "Volume2"
  },
  application_layer: {
    category: "application_layer",
    name: "Application Layer",
    nameVi: "Tấn công lớp ứng dụng",
    description: "Tấn công nhắm vào các dịch vụ ứng dụng như HTTP, HTTPS để làm sập server",
    color: "bg-blue-500",
    icon: "Globe"
  },
  protocol_exploit: {
    category: "protocol_exploit",
    name: "Protocol Exploit",
    nameVi: "Khai thác giao thức",
    description: "Lợi dụng điểm yếu trong thiết kế của các giao thức mạng (TCP, ICMP)",
    color: "bg-indigo-500",
    icon: "Shield"
  }
};

// DDoS Attack types
export const ddosAttackTypes = [
  "port_scan",
  "service_scan",
  "ssh_bruteforce",
  "ftp_bruteforce",
  "telnet_bruteforce",
  "syn_flood",
  "udp_flood",
  "icmp_flood",
  "http_flood",
  "slowloris",
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
  category: AttackCategory;
  description: string;
  method: string;
  indicators: string[];
  formula: string;
  ports?: number[];
  severity: "low" | "medium" | "high" | "critical";
}

export const ATTACK_TYPE_INFO: Record<DDoSAttackType, AttackTypeInfo> = {
  port_scan: {
    type: "port_scan",
    name: "Port Scan",
    nameVi: "Quét cổng",
    category: "reconnaissance",
    description: "Kẻ tấn công quét nhiều cổng để tìm dịch vụ đang mở",
    method: "Gửi gói TCP SYN hoặc UDP đến nhiều cổng khác nhau để xác định cổng nào mở",
    indicators: ["Nhiều cổng đích khác nhau từ cùng IP nguồn", "Kết nối ngắn", "Thất bại nhiều"],
    formula: "unique_dst_ports_per_src_ip >= 10",
    ports: [],
    severity: "medium",
  },
  service_scan: {
    type: "service_scan",
    name: "Service Scan",
    nameVi: "Quét dịch vụ",
    category: "reconnaissance",
    description: "Thu thập thông tin về phiên bản và loại dịch vụ đang chạy",
    method: "Gửi các probe đặc biệt để xác định phiên bản phần mềm (banner grabbing)",
    indicators: ["Nhiều request đến cổng dịch vụ phổ biến", "Pattern nhận diện service", "Từ cùng IP nguồn"],
    formula: "connection_to_common_ports >= 5 AND banner_request_detected",
    ports: [21, 22, 23, 25, 80, 110, 143, 443, 3306, 5432],
    severity: "low",
  },
  ssh_bruteforce: {
    type: "ssh_bruteforce",
    name: "SSH Brute Force",
    nameVi: "Dò mật khẩu SSH",
    category: "bruteforce",
    description: "Thử đăng nhập SSH liên tục với nhiều username/password khác nhau",
    method: "Tự động thử hàng ngàn tổ hợp username:password để truy cập SSH",
    indicators: ["Cổng 22", "Nhiều login attempt thất bại", "Từ một hoặc nhiều IP"],
    formula: "dst_port == 22 AND failed_connections > 50 AND connection_rate > 10/min",
    ports: [22],
    severity: "high",
  },
  ftp_bruteforce: {
    type: "ftp_bruteforce",
    name: "FTP Brute Force",
    nameVi: "Dò mật khẩu FTP",
    category: "bruteforce",
    description: "Thử đăng nhập FTP liên tục để chiếm quyền truy cập file",
    method: "Tự động thử nhiều tổ hợp credentials để truy cập FTP server",
    indicators: ["Cổng 21", "530 Login incorrect nhiều", "Cùng IP hoặc IP range"],
    formula: "dst_port == 21 AND failed_auth > 30",
    ports: [21],
    severity: "high",
  },
  telnet_bruteforce: {
    type: "telnet_bruteforce",
    name: "Telnet Brute Force",
    nameVi: "Dò mật khẩu Telnet",
    category: "bruteforce",
    description: "Thử đăng nhập Telnet để điều khiển thiết bị từ xa",
    method: "Thường nhắm vào thiết bị IoT, router có Telnet mở với mật khẩu yếu",
    indicators: ["Cổng 23", "Login attempt lặp lại", "Pattern botnet như Mirai"],
    formula: "dst_port == 23 AND failed_connections > 20",
    ports: [23],
    severity: "critical",
  },
  syn_flood: {
    type: "syn_flood",
    name: "SYN Flood",
    nameVi: "Tấn công SYN Flood",
    category: "protocol_exploit",
    description: "Gửi nhiều gói SYN mà không hoàn thành TCP handshake",
    method: "Khai thác TCP 3-way handshake: gửi SYN nhưng không ACK, gây half-open connections",
    indicators: ["Cờ SYN cao", "Không có ACK", "Kết nối half-open nhiều"],
    formula: "syn_count / total_packets > 0.8 AND ack_count / syn_count < 0.1",
    ports: [80, 443, 22, 21],
    severity: "high",
  },
  udp_flood: {
    type: "udp_flood",
    name: "UDP Flood",
    nameVi: "Tấn công UDP Flood",
    category: "volumetric",
    description: "Gửi lượng lớn gói UDP để làm quá tải băng thông",
    method: "Gửi liên tục gói UDP đến các cổng ngẫu nhiên, target phải xử lý và trả ICMP unreachable",
    indicators: ["Protocol UDP", "Packet rate cao", "Kích thước nhỏ đồng đều"],
    formula: "protocol == UDP AND packets_per_sec > 10000 AND avg_packet_size < 100",
    ports: [53, 123, 161],
    severity: "high",
  },
  icmp_flood: {
    type: "icmp_flood",
    name: "ICMP Flood",
    nameVi: "Tấn công ICMP/Ping Flood",
    category: "volumetric",
    description: "Gửi nhiều gói ICMP (ping) để làm tê liệt mạng",
    method: "Gửi lượng lớn ICMP Echo Request, buộc target phải trả lời Echo Reply",
    indicators: ["Protocol ICMP", "Echo request cao", "Nhiều nguồn"],
    formula: "protocol == ICMP AND icmp_type == 8 AND packet_rate > 5000",
    ports: [],
    severity: "medium",
  },
  http_flood: {
    type: "http_flood",
    name: "HTTP Flood",
    nameVi: "Tấn công HTTP Flood",
    category: "application_layer",
    description: "Gửi nhiều HTTP request để làm quá tải web server",
    method: "Gửi request GET/POST hợp lệ với tần suất cao để exhaust server resources",
    indicators: ["Cổng 80/443", "Request rate cao", "Nhiều connection"],
    formula: "dst_port IN (80, 443) AND request_rate > 1000/s AND unique_src_ip > 100",
    ports: [80, 443, 8080],
    severity: "high",
  },
  slowloris: {
    type: "slowloris",
    name: "Slowloris",
    nameVi: "Tấn công Slowloris",
    category: "application_layer",
    description: "Giữ nhiều kết nối mở bằng cách gửi request chậm, không hoàn thành",
    method: "Mở nhiều connection HTTP và gửi headers từ từ để giữ connection mở vô thời hạn",
    indicators: ["Cổng 80/443", "Connection duration rất dài", "Incomplete headers"],
    formula: "dst_port IN (80, 443) AND avg_connection_duration > 30s AND bytes_per_sec < 10",
    ports: [80, 443],
    severity: "medium",
  },
  dns_amplification: {
    type: "dns_amplification",
    name: "DNS Amplification",
    nameVi: "Tấn công khuếch đại DNS",
    category: "amplification",
    description: "Sử dụng DNS server để khuếch đại traffic đến nạn nhân",
    method: "Gửi DNS query nhỏ với spoofed source IP, DNS server trả response lớn đến victim",
    indicators: ["Cổng 53", "Response lớn hơn request", "Spoofed IP"],
    formula: "dst_port == 53 AND response_size / request_size > 10 AND ip_spoofing_detected",
    ports: [53],
    severity: "critical",
  },
  ntp_amplification: {
    type: "ntp_amplification",
    name: "NTP Amplification",
    nameVi: "Tấn công khuếch đại NTP",
    category: "amplification",
    description: "Sử dụng NTP server để khuếch đại traffic",
    method: "Gửi monlist command (liệt kê 600 clients gần nhất) với spoofed IP, amplification 556x",
    indicators: ["Cổng 123", "Monlist command", "Response lớn"],
    formula: "dst_port == 123 AND response_size / request_size > 50",
    ports: [123],
    severity: "critical",
  },
  ldap_reflection: {
    type: "ldap_reflection",
    name: "LDAP Reflection",
    nameVi: "Tấn công phản xạ LDAP",
    category: "amplification",
    description: "Lợi dụng LDAP server để phản xạ và khuếch đại traffic",
    method: "Gửi CLDAP query với spoofed IP, nhận response khuếch đại 46-55 lần",
    indicators: ["Cổng 389/636", "Connectionless LDAP", "Response lớn"],
    formula: "dst_port IN (389, 636) AND protocol == UDP AND amplification_factor > 50",
    ports: [389, 636, 3268],
    severity: "critical",
  },
  rdp_attack: {
    type: "rdp_attack",
    name: "RDP Attack",
    nameVi: "Tấn công Remote Desktop",
    category: "remote_access",
    description: "Brute-force hoặc exploit lỗ hổng RDP để truy cập máy Windows từ xa",
    method: "Dò mật khẩu RDP hoặc khai thác CVE như BlueKeep để chiếm quyền điều khiển",
    indicators: ["Cổng 3389", "Login attempt nhiều", "Nhiều IP nguồn"],
    formula: "dst_port == 3389 AND failed_auth > 100 AND unique_src_ip > 10",
    ports: [3389],
    severity: "critical",
  },
  ssdp_amplification: {
    type: "ssdp_amplification",
    name: "SSDP Amplification",
    nameVi: "Tấn công khuếch đại SSDP",
    category: "amplification",
    description: "Sử dụng SSDP/UPnP để khuếch đại traffic",
    method: "Gửi M-SEARCH request đến thiết bị UPnP với spoofed IP, amplification 30x",
    indicators: ["Cổng 1900", "M-SEARCH request", "Response lớn"],
    formula: "dst_port == 1900 AND protocol == UDP AND amplification > 30",
    ports: [1900],
    severity: "high",
  },
  memcached_amplification: {
    type: "memcached_amplification",
    name: "Memcached Amplification",
    nameVi: "Tấn công khuếch đại Memcached",
    category: "amplification",
    description: "Lợi dụng memcached server để khuếch đại DDoS cực lớn",
    method: "Gửi 'stats' command đến memcached mở, có thể khuếch đại lên đến 51,000x",
    indicators: ["Cổng 11211", "UDP protocol", "Khuếch đại cực lớn"],
    formula: "dst_port == 11211 AND protocol == UDP AND amplification > 50000",
    ports: [11211],
    severity: "critical",
  },
  unknown: {
    type: "unknown",
    name: "Unknown Attack",
    nameVi: "Tấn công không xác định",
    category: "volumetric",
    description: "Loại tấn công chưa được phân loại rõ ràng",
    method: "Pattern bất thường không khớp với các loại tấn công đã biết",
    indicators: ["Pattern bất thường", "Không khớp signature đã biết"],
    formula: "anomaly_score > threshold AND !matches_known_pattern",
    ports: [],
    severity: "medium",
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

// Cross-Validation Result Schema
export const crossValidationResultSchema = z.object({
  foldResults: z.array(z.object({
    fold: z.number(),
    accuracy: z.number(),
    precision: z.number(),
    recall: z.number(),
    f1Score: z.number(),
  })),
  meanAccuracy: z.number(),
  stdAccuracy: z.number(),
  meanPrecision: z.number(),
  stdPrecision: z.number(),
  meanRecall: z.number(),
  stdRecall: z.number(),
  meanF1: z.number(),
  stdF1: z.number(),
  kFolds: z.number(),
});

export type CrossValidationResult = z.infer<typeof crossValidationResultSchema>;

// Grid Search Result Schema
export const gridSearchResultSchema = z.object({
  bestParams: z.record(z.any()),
  bestScore: z.number(),
  allResults: z.array(z.object({
    params: z.record(z.any()),
    score: z.number(),
  })),
  searchTime: z.number(),
  totalCombinations: z.number(),
});

export type GridSearchResult = z.infer<typeof gridSearchResultSchema>;

// Train/Val/Test Split Info Schema
export const splitInfoSchema = z.object({
  trainSize: z.number(),
  valSize: z.number(),
  testSize: z.number(),
  trainRatio: z.number(),
  valRatio: z.number(),
  testRatio: z.number(),
});

export type SplitInfo = z.infer<typeof splitInfoSchema>;

// Enhanced Metrics with Train/Val/Test
export const enhancedMetricsSchema = z.object({
  trainMetrics: z.object({
    accuracy: z.number(),
    precision: z.number(),
    recall: z.number(),
    f1Score: z.number(),
  }).optional(),
  valMetrics: z.object({
    accuracy: z.number(),
    precision: z.number(),
    recall: z.number(),
    f1Score: z.number(),
  }).optional(),
  testMetrics: z.object({
    accuracy: z.number(),
    precision: z.number(),
    recall: z.number(),
    f1Score: z.number(),
  }).optional(),
});

export type EnhancedMetrics = z.infer<typeof enhancedMetricsSchema>;

// Analysis result schema
export const analysisResultSchema = z.object({
  id: z.string(),
  datasetId: z.string(),
  modelType: z.enum(mlModelTypes),
  mode: z.enum(detectionModes).optional(),  // Supervised hoặc Unlabeled
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
  // ML Best Practices fields
  crossValidation: crossValidationResultSchema.optional(),
  gridSearch: gridSearchResultSchema.optional(),
  splitInfo: splitInfoSchema.optional(),
  enhancedMetrics: enhancedMetricsSchema.optional(),
  bestHyperparams: z.record(z.any()).optional(),
  // Warnings (e.g., label fallback, insufficient data)
  warnings: z.array(z.string()).optional(),
  // Unlabeled mode fields
  unlabeledReport: z.object({
    scoreDistribution: z.object({
      min: z.number(),
      max: z.number(),
      mean: z.number(),
      std: z.number(),
      percentiles: z.object({
        p25: z.number(),
        p50: z.number(),
        p75: z.number(),
        p90: z.number(),
        p95: z.number(),
      }),
    }),
    alertRate: z.number(),
    dataQuality: z.object({
      missingRate: z.number(),
      invalidValues: z.number(),
      totalRows: z.number(),
      validRows: z.number(),
    }),
    anomalyScores: z.object({
      isolationForest: z.number(),
      lof: z.number(),
      combined: z.number(),
    }).optional(),
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
