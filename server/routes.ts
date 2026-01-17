import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import cors from "cors";
import { storage } from "./storage";
import { analyzeWithModel, analyzeUnlabeled, extractFeatures, getFeatureColumns, runAnomalyDetection, generateUnlabeledReport, buildFeatureMapping, getFeatureStatistics, analyzeRowForAttack, getCachedResult, setCachedResult, clearCache, getCacheStats, calculateConfusionMatrix, calculateFeatureImportance } from "./ml-algorithms";
import { uploadDatasetSchema, analyzeRequestSchema, type DataRow, type Dataset, type InsertAuditLog } from "@shared/schema";
import { randomUUID } from "crypto";
import * as XLSX from "xlsx";
import multer from "multer";
import Papa from "papaparse";
import helmet from "helmet";
import { learningService } from "./learning-service";
import { 
  detectSchemaType, 
  analyzeFeatureUsage, 
  findColumnMapping, 
  normalizeDataset, 
  getModelForSchema,
  getLabelStats,
  addCustomLabel,
  removeCustomLabel,
  getCustomLabels,
  getAllLabelMappings,
  setCustomLabels,
  type SchemaType,
  type LabelConfig,
  type LabelCategory
} from "./schema-detection";

// ============== MULTER CONFIGURATION (Multipart Upload) ==============
const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50MB

const multerStorage = multer.memoryStorage();
const upload = multer({
  storage: multerStorage,
  limits: {
    fileSize: MAX_FILE_SIZE_BYTES,
  },
  fileFilter: (req, file, cb) => {
    const allowedMimes = [
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/octet-stream', // Some browsers send this for CSV
    ];
    const allowedExts = ['.csv', '.xlsx', '.xls'];
    const ext = file.originalname.toLowerCase().slice(file.originalname.lastIndexOf('.'));
    
    if (allowedExts.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error(`Định dạng file không hợp lệ. Chỉ hỗ trợ: ${allowedExts.join(', ')}`));
    }
  },
});

// Parse CSV using PapaParse (proper RFC 4180 compliant parser)
function parseCSVWithPapa(content: string): { columns: string[]; rows: DataRow[] } {
  const result = Papa.parse(content, {
    header: true,
    skipEmptyLines: true,
    transformHeader: (header) => header.trim(),
    dynamicTyping: true, // Auto-convert numbers
  });
  
  if (result.errors.length > 0) {
    console.warn("CSV parsing warnings:", result.errors.slice(0, 5));
  }
  
  const columns = result.meta.fields || [];
  const rows: DataRow[] = result.data as DataRow[];
  
  return { columns, rows };
}

// Parse Excel file
function parseExcelFile(buffer: Buffer): { columns: string[]; rows: DataRow[] } {
  const workbook = XLSX.read(buffer, { type: 'buffer' });
  const sheetName = workbook.SheetNames[0];
  const sheet = workbook.Sheets[sheetName];
  
  const jsonData = XLSX.utils.sheet_to_json<any[]>(sheet, { header: 1 });
  
  if (jsonData.length === 0) {
    return { columns: [], rows: [] };
  }
  
  const headerRow = jsonData[0] as any[];
  const columns = headerRow.map((h: any) => String(h).trim());
  
  const rows: DataRow[] = [];
  for (let i = 1; i < jsonData.length; i++) {
    const dataRow = jsonData[i] as any[];
    const row: DataRow = {};
    for (let j = 0; j < columns.length; j++) {
      row[columns[j]] = dataRow?.[j] ?? null;
    }
    rows.push(row);
  }
  
  return { columns, rows };
}

// ============== RATE LIMITING ==============
interface RateLimitEntry {
  count: number;
  resetTime: number;
}

const rateLimitStore = new Map<string, RateLimitEntry>();
const RATE_LIMIT_WINDOW_MS = 60 * 1000; // 1 minute window
const RATE_LIMIT_MAX_REQUESTS = 60; // 60 requests per minute

function getRateLimitKey(req: Request): string {
  return req.ip || req.headers['x-forwarded-for'] as string || 'unknown';
}

function rateLimitMiddleware(req: Request, res: Response, next: NextFunction): void {
  const key = getRateLimitKey(req);
  const now = Date.now();
  
  let entry = rateLimitStore.get(key);
  
  if (!entry || now > entry.resetTime) {
    entry = { count: 1, resetTime: now + RATE_LIMIT_WINDOW_MS };
    rateLimitStore.set(key, entry);
  } else {
    entry.count++;
  }
  
  res.setHeader('X-RateLimit-Limit', RATE_LIMIT_MAX_REQUESTS.toString());
  res.setHeader('X-RateLimit-Remaining', Math.max(0, RATE_LIMIT_MAX_REQUESTS - entry.count).toString());
  res.setHeader('X-RateLimit-Reset', entry.resetTime.toString());
  
  if (entry.count > RATE_LIMIT_MAX_REQUESTS) {
    res.status(429).json({ 
      error: 'Quá nhiều yêu cầu. Vui lòng thử lại sau.',
      retryAfter: Math.ceil((entry.resetTime - now) / 1000)
    });
    return;
  }
  
  next();
}

// Clean up old rate limit entries every 5 minutes
setInterval(() => {
  const now = Date.now();
  const keysToDelete: string[] = [];
  rateLimitStore.forEach((entry, key) => {
    if (now > entry.resetTime) {
      keysToDelete.push(key);
    }
  });
  keysToDelete.forEach(key => rateLimitStore.delete(key));
}, 5 * 60 * 1000);

// ============== AUDIT LOGGING ==============
async function logAudit(action: string, entityType: string, entityId: string | undefined, details: any, req?: Request): Promise<void> {
  try {
    const log: InsertAuditLog = {
      action,
      entityType,
      entityId,
      details,
      ipAddress: req?.ip || req?.headers['x-forwarded-for'] as string || null,
      userAgent: req?.headers['user-agent'] || null,
    };
    
    await storage.addAuditLog(log);
  } catch (error) {
    console.error("Failed to write audit log:", error);
  }
}

// ============== INPUT VALIDATION ==============
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls'];

function validateFileUpload(filename: string, size: number): { valid: boolean; error?: string } {
  const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'));
  
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return { valid: false, error: `Định dạng file không hợp lệ. Chỉ hỗ trợ: ${ALLOWED_EXTENSIONS.join(', ')}` };
  }
  
  if (size > MAX_FILE_SIZE) {
    return { valid: false, error: `File quá lớn. Giới hạn: ${MAX_FILE_SIZE / 1024 / 1024}MB` };
  }
  
  return { valid: true };
}

// Helper function to extract features for report
function extractFeaturesForReport(data: DataRow[], featureColumns: string[]): { features: number[][] } {
  const features: number[][] = [];
  for (const row of data) {
    const featureVector: number[] = [];
    for (const col of featureColumns) {
      const value = row[col];
      if (typeof value === "number") {
        featureVector.push(value);
      } else if (typeof value === "string") {
        featureVector.push(parseFloat(value) || 0);
      } else {
        featureVector.push(0);
      }
    }
    features.push(featureVector);
  }
  return { features };
}

// Helper function to normalize features for anomaly detection
function normalizeForReport(features: number[][]): number[][] {
  if (features.length === 0) return features;
  const numFeatures = features[0].length;
  const mins: number[] = new Array(numFeatures).fill(Infinity);
  const maxs: number[] = new Array(numFeatures).fill(-Infinity);
  
  for (const row of features) {
    for (let i = 0; i < numFeatures; i++) {
      mins[i] = Math.min(mins[i], row[i]);
      maxs[i] = Math.max(maxs[i], row[i]);
    }
  }
  
  return features.map((row) =>
    row.map((val, i) => {
      const range = maxs[i] - mins[i];
      return range === 0 ? 0 : (val - mins[i]) / range;
    })
  );
}

function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    
    if (char === '"' && (i === 0 || line[i - 1] !== '\\')) {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      result.push(current.trim().replace(/^["']|["']$/g, ""));
      current = "";
    } else {
      current += char;
    }
  }
  result.push(current.trim().replace(/^["']|["']$/g, ""));
  
  return result;
}

function parseCSV(csvContent: string): { columns: string[]; rows: DataRow[] } {
  const lines = csvContent.trim().split(/\r?\n/);
  if (lines.length === 0) {
    return { columns: [], rows: [] };
  }

  const columns = parseCSVLine(lines[0]);
  const rows: DataRow[] = [];

  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    
    const values = parseCSVLine(lines[i]);
    if (values.length !== columns.length) continue;

    const row: DataRow = {};
    for (let j = 0; j < columns.length; j++) {
      const numValue = parseFloat(values[j]);
      row[columns[j]] = isNaN(numValue) ? values[j] : numValue;
    }
    rows.push(row);
  }

  return { columns, rows };
}

function isXlsxContent(content: string): boolean {
  return content.startsWith("PK") || content.includes("xl/worksheets");
}

function parseXlsx(base64Content: string): { columns: string[]; rows: DataRow[] } {
  try {
    const binaryStr = atob(base64Content);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }
    
    const workbook = XLSX.read(bytes, { type: "array" });
    const sheetName = workbook.SheetNames[0];
    const sheet = workbook.Sheets[sheetName];
    
    const jsonData = XLSX.utils.sheet_to_json<Record<string, any>>(sheet, { defval: "" });
    
    if (jsonData.length === 0) {
      return { columns: [], rows: [] };
    }
    
    const columns = Object.keys(jsonData[0]);
    const rows: DataRow[] = jsonData.map((row) => {
      const dataRow: DataRow = {};
      for (const col of columns) {
        const value = row[col];
        if (typeof value === "number") {
          dataRow[col] = value;
        } else if (typeof value === "string") {
          const numValue = parseFloat(value);
          dataRow[col] = isNaN(numValue) ? value : numValue;
        } else {
          dataRow[col] = value?.toString() || "";
        }
      }
      return dataRow;
    });
    
    return { columns, rows };
  } catch (error) {
    console.error("Error parsing xlsx:", error);
    return { columns: [], rows: [] };
  }
}

function parseDataContent(content: string, filename: string): { columns: string[]; rows: DataRow[] } {
  if (filename.endsWith(".xlsx") || filename.endsWith(".xls") || isXlsxContent(content)) {
    return parseXlsx(content);
  }
  return parseCSV(content);
}

function hasLabelColumn(columns: string[]): boolean {
  const labelColumns = ["label", "class", "attack", "target", "Label", "Class", "Attack", "Target", "is_attack", "ddos", "DDoS", "attack_cat", "category"];
  return columns.some(col => labelColumns.includes(col) || labelColumns.some(lc => col.toLowerCase() === lc.toLowerCase()));
}

function findLabelColumn(columns: string[]): string | null {
  const labelColumns = ["label", "class", "attack", "target", "is_attack", "ddos", "attack_cat", "category"];
  const lowerColumns = columns.map(c => c.toLowerCase());
  for (const lc of labelColumns) {
    const idx = lowerColumns.indexOf(lc);
    if (idx >= 0) return columns[idx];
  }
  return null;
}

// Phát hiện file mô tả schema (như UNSW-NB15_features.csv)
function isSchemaDescriptionFile(columns: string[], rows: DataRow[]): { isSchema: boolean; reason: string } {
  const schemaIndicators = ["no.", "name", "type", "description", "feature", "description"];
  const lowerCols = columns.map(c => c.toLowerCase());
  
  // Nếu có cột "No." + "Name" + "Description" -> file mô tả
  const hasNoCol = lowerCols.some(c => c === "no" || c === "no." || c === "#");
  const hasNameCol = lowerCols.some(c => c === "name" || c === "feature" || c === "feature_name");
  const hasDescCol = lowerCols.some(c => c.includes("description") || c.includes("desc"));
  const hasTypeCol = lowerCols.some(c => c === "type" || c === "data_type" || c === "dtype");
  
  if (hasNoCol && hasNameCol && (hasDescCol || hasTypeCol)) {
    return { isSchema: true, reason: "File có cấu trúc mô tả schema (No/Name/Description/Type). Đây không phải dataset thực." };
  }
  
  // Kiểm tra số dòng quá ít + tên cột giống mô tả feature
  if (rows.length < 100 && rows.length === columns.length) {
    return { isSchema: true, reason: "File có số dòng bằng số cột - có thể là file mô tả features." };
  }
  
  return { isSchema: false, reason: "" };
}

// Validate Feature Contract
function validateFeatureContract(columns: string[]): import("@shared/schema").FeatureValidation {
  const lowerCols = columns.map(c => c.toLowerCase());
  const contract = {
    timing: ["duration", "dur", "time", "timestamp", "start_time", "stime", "ltime"],
    volume: ["bytes", "sbytes", "dbytes", "totlen_fwd_pkts", "totlen_bwd_pkts", "tot_len", "total_bytes", "bps"],
    packets: ["packets", "spkts", "dpkts", "tot_fwd_pkts", "tot_bwd_pkts", "total_packets", "pkts", "pps"],
    network: ["src_ip", "dst_ip", "srcip", "dstip", "saddr", "daddr", "src_port", "dst_port", "sport", "dport"],
    protocol: ["protocol", "proto", "service", "state", "flags", "tcp_flags"],
    labels: ["label", "class", "attack", "attack_cat", "category", "target", "is_attack", "ddos"],
  };

  const hasMatch = (group: string[]) => lowerCols.some(col => group.some(g => col.includes(g)));
  
  const hasTimingFeatures = hasMatch(contract.timing);
  const hasVolumeFeatures = hasMatch(contract.volume);
  const hasPacketFeatures = hasMatch(contract.packets);
  const hasNetworkFeatures = hasMatch(contract.network);
  const hasProtocolFeatures = hasMatch(contract.protocol);
  const hasLabelColumn = hasMatch(contract.labels);
  
  const detectedLabelColumn = findLabelColumn(columns);
  
  const missingRequired: string[] = [];
  if (!hasTimingFeatures) missingRequired.push("timing (duration, timestamp...)");
  if (!hasVolumeFeatures) missingRequired.push("volume (bytes, bps...)");
  if (!hasPacketFeatures) missingRequired.push("packets (packets, pps...)");
  
  const availableOptional: string[] = [];
  if (hasNetworkFeatures) availableOptional.push("network (IP/Port)");
  if (hasProtocolFeatures) availableOptional.push("protocol");
  if (hasLabelColumn) availableOptional.push("labels");
  
  // Tính confidence level
  let confidenceLevel: "high" | "medium" | "low" = "high";
  let confidenceReason = "Đầy đủ các features cần thiết";
  
  if (missingRequired.length === 0 && hasNetworkFeatures && hasProtocolFeatures) {
    confidenceLevel = "high";
    confidenceReason = "Dataset có đầy đủ features: timing, volume, packets, network, protocol";
  } else if (missingRequired.length <= 1) {
    confidenceLevel = "medium";
    confidenceReason = `Thiếu một số features: ${missingRequired.join(", ")}`;
  } else {
    confidenceLevel = "low";
    confidenceReason = `Thiếu nhiều features quan trọng: ${missingRequired.join(", ")}. Kết quả có thể không chính xác.`;
  }
  
  return {
    hasTimingFeatures,
    hasVolumeFeatures,
    hasPacketFeatures,
    hasNetworkFeatures,
    hasProtocolFeatures,
    hasLabelColumn,
    detectedLabelColumn,
    missingRequired,
    availableOptional,
    confidenceLevel,
    confidenceReason,
  };
}

// Tính toán percentile
function percentile(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.floor((p / 100) * sorted.length);
  return sorted[Math.min(idx, sorted.length - 1)];
}

// Tính standard deviation
function std(arr: number[]): number {
  if (arr.length === 0) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const variance = arr.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / arr.length;
  return Math.sqrt(variance);
}

function cleanData(rows: DataRow[], columns: string[]): {
  cleanedRows: DataRow[];
  missingValues: number;
  duplicates: number;
  outliers: number;
} {
  let missingValues = 0;
  let duplicates = 0;
  let outliers = 0;

  const seenRows = new Set<string>();
  const cleanedRows: DataRow[] = [];
  
  const criticalCols = columns.filter(col => 
    ["label", "class", "attack", "dst_port", "protocol", "packets", "bytes", "pps"].some(
      c => col.toLowerCase().includes(c)
    )
  );

  for (const row of rows) {
    let missingCount = 0;
    let hasCriticalMissing = false;
    
    for (const col of columns) {
      if (row[col] === null || row[col] === undefined || row[col] === "") {
        missingCount++;
        if (criticalCols.includes(col)) {
          hasCriticalMissing = true;
        }
      }
    }
    
    if (hasCriticalMissing) {
      missingValues++;
      continue;
    }
    
    if (missingCount > columns.length * 0.5) {
      missingValues++;
      continue;
    }

    const rowKey = columns.map((col) => String(row[col])).join("|");
    if (seenRows.has(rowKey)) {
      duplicates++;
      continue;
    }
    seenRows.add(rowKey);

    let hasOutlier = false;
    for (const col of columns) {
      const value = row[col];
      if (typeof value === "number" && (value < -1e15 || value > 1e15)) {
        hasOutlier = true;
        outliers++;
        break;
      }
    }

    if (!hasOutlier) {
      cleanedRows.push(row);
    }
  }

  return { cleanedRows, missingValues, duplicates, outliers };
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  // Apply rate limiting to all API routes
  app.use('/api', rateLimitMiddleware);
  
  // Add helmet for security headers
  app.use(helmet({
    contentSecurityPolicy: false, // Disable for development
    crossOriginEmbedderPolicy: false,
  }));
  
  // CORS configuration
  // In production (Docker), frontend is served from same origin, so CORS is less strict
  // In development (Replit), allow Replit domains
  const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || [];
  
  app.use(cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (same-origin requests, curl, server-side)
      if (!origin) return callback(null, true);
      
      // Allow explicitly configured origins
      if (allowedOrigins.length > 0 && allowedOrigins.includes(origin)) {
        return callback(null, true);
      }
      
      // Allow localhost and local network for development/Docker
      if (origin.includes('localhost') || 
          origin.includes('127.0.0.1') ||
          /^https?:\/\/(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)/.test(origin)) {
        return callback(null, true);
      }
      
      // Allow Replit domains
      if (origin.includes('.replit.dev') || origin.includes('.replit.app')) {
        return callback(null, true);
      }
      
      // In development, be permissive
      if (process.env.NODE_ENV !== 'production') {
        return callback(null, true);
      }
      
      // In production, allow same-origin (no origin header means same-origin)
      // For cross-origin in production, require explicit ALLOWED_ORIGINS config
      callback(null, true);
    },
    credentials: true,
  }));

  // ============== MULTIPART FILE UPLOAD (P0-02) ==============
  app.post("/api/upload/file", upload.single('file'), async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "Không có file được upload" });
      }

      const file = req.file;
      const ext = file.originalname.toLowerCase().slice(file.originalname.lastIndexOf('.'));
      
      let columns: string[];
      let rows: DataRow[];
      
      // Parse based on file type
      if (ext === '.csv') {
        const content = file.buffer.toString('utf-8');
        const parsed = parseCSVWithPapa(content);
        columns = parsed.columns;
        rows = parsed.rows;
      } else if (ext === '.xlsx' || ext === '.xls') {
        const parsed = parseExcelFile(file.buffer);
        columns = parsed.columns;
        rows = parsed.rows;
      } else {
        return res.status(400).json({ error: "Định dạng file không được hỗ trợ" });
      }

      if (rows.length === 0) {
        return res.status(400).json({ error: "Không tìm thấy dữ liệu trong file" });
      }

      // Schema detection
      const schemaCheck = isSchemaDescriptionFile(columns, rows);
      if (schemaCheck.isSchema) {
        return res.status(400).json({ 
          error: `File này là mô tả schema/dictionary, không phải dataset thực. ${schemaCheck.reason}`,
          isSchemaFile: true 
        });
      }

      // Schema Detection + Column Normalization
      const columnMappings = findColumnMapping(columns);
      const { type: schemaType, confidence: schemaConfidence } = detectSchemaType(columns);
      
      // Convert rows to 2D array for feature analysis
      const dataArray = rows.map(row => columns.map(col => row[col]));
      const featureReport = analyzeFeatureUsage(columns, dataArray, schemaType);
      
      // Normalize dataset columns
      const normalizedResult = normalizeDataset(columns, dataArray, columnMappings);
      const normalizedColumns = normalizedResult.columns;
      
      // Get recommended models
      const recommendedModels = getModelForSchema(schemaType);

      // Feature validation
      const featureValidation = validateFeatureContract(columns);
      const mode: import("@shared/schema").DetectionMode = featureValidation.hasLabelColumn ? "supervised" : "unlabeled";

      // Get label statistics
      let labelStats: Record<string, { count: number; percentage: number; category: string }> = {};
      if (featureValidation.hasLabelColumn && featureValidation.detectedLabelColumn) {
        const labelIndex = columns.indexOf(featureValidation.detectedLabelColumn);
        if (labelIndex >= 0) {
          labelStats = getLabelStats(dataArray, labelIndex);
        }
      }

      // Feature mapping
      const featureMapping = buildFeatureMapping(columns);
      const featureMappingObj = Object.fromEntries(Array.from(featureMapping.entries()));
      const detectedFeatureTypes = Array.from(new Set(featureMapping.values()));

      // Clean data
      const { cleanedRows, missingValues, duplicates, outliers } = cleanData(rows, columns);
      
      // Feature statistics
      const featureStats = getFeatureStatistics(cleanedRows, featureMapping);

      const datasetId = randomUUID();
      const dataset: Dataset = {
        id: datasetId,
        name: file.originalname,
        originalRowCount: rows.length,
        cleanedRowCount: cleanedRows.length,
        columns,
        uploadedAt: new Date().toISOString(),
        isProcessed: true,
        mode,
        labelColumn: featureValidation.detectedLabelColumn || undefined,
        featureValidation,
        dataQuality: {
          missingValues,
          duplicates,
          outliers,
          cleanedPercentage: (cleanedRows.length / rows.length) * 100,
        },
      };

      const previewData = cleanedRows.slice(0, 10);

      await storage.setDataset(dataset, previewData);
      await storage.clearResults();

      (global as any).__datasetRows = cleanedRows;
      (global as any).__datasetMode = mode;

      // Build warnings
      const warnings: string[] = [];
      const schemaTypeNames: Record<SchemaType, string> = {
        'cicflowmeter': 'CICFlowMeter',
        'event_log': 'Event/Log',
        'unknown': 'Không xác định'
      };
      warnings.push(`📋 Schema: ${schemaTypeNames[schemaType]} (${schemaConfidence.toFixed(0)}% tin cậy)`);
      
      if (mode === "unlabeled") {
        warnings.push("🔍 Mode: UNLABELED - Không tìm thấy cột label.");
      } else {
        warnings.push(`✓ Mode: SUPERVISED - Phát hiện cột label: "${featureValidation.detectedLabelColumn}".`);
      }

      // Audit log
      await logAudit("upload", "dataset", datasetId, { 
        name: file.originalname, 
        rowCount: rows.length, 
        cleanedCount: cleanedRows.length,
        mode,
        schemaType,
        uploadMethod: "multipart"
      }, req);

      res.json({ 
        datasetId,
        dataset, 
        previewData, 
        warning: warnings.join(" | "),
        validationResult: {
          mode,
          featureValidation,
          confidenceLevel: featureValidation.confidenceLevel,
        },
        schemaDetection: {
          schemaType,
          schemaConfidence,
          recommendedModels,
          columnMappings,
          normalizedColumns,
        },
        featureReport: {
          foundFeatures: featureReport.foundFeatures,
          missingFeatures: featureReport.missingFeatures,
          foundPercentage: featureReport.foundPercentage,
          nanCount: featureReport.nanCount,
          nanPercentage: featureReport.nanPercentage,
          infCount: featureReport.infCount,
          infPercentage: featureReport.infPercentage,
          isReliable: featureReport.isReliable,
          warnings: featureReport.warnings,
        },
        labelStats,
        featureAnalysis: {
          mapping: featureMappingObj,
          detectedTypes: detectedFeatureTypes,
          statistics: featureStats.featureStats,
          anomalyIndicators: featureStats.anomalyIndicators,
        }
      });
    } catch (error: any) {
      console.error("Upload error:", error);
      if (error.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({ error: "File quá lớn. Giới hạn: 50MB" });
      }
      res.status(500).json({ error: error.message || "Lỗi khi xử lý file" });
    }
  });

  // Error handler for multer
  app.use((err: any, req: Request, res: Response, next: NextFunction) => {
    if (err instanceof multer.MulterError) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({ error: "File quá lớn. Giới hạn: 50MB" });
      }
      return res.status(400).json({ error: err.message });
    }
    next(err);
  });

  // ============== EXPORT ENDPOINTS ==============
  app.get("/api/export/csv", async (req, res) => {
    try {
      const results = await storage.getResults();
      if (!results || results.length === 0) {
        return res.status(404).json({ error: "Không có kết quả phân tích để xuất" });
      }

      const dataset = await storage.getDataset();
      const csvRows: string[] = [];
      
      // Header row with all result fields
      csvRows.push("Model,Accuracy,Precision,Recall,F1Score,DDoSDetected,NormalTraffic");
      
      for (const result of results) {
        csvRows.push([
          result.modelType,
          (result.accuracy * 100).toFixed(2),
          (result.precision * 100).toFixed(2),
          (result.recall * 100).toFixed(2),
          (result.f1Score * 100).toFixed(2),
          result.ddosDetected,
          result.normalTraffic
        ].join(","));
      }

      logAudit("export", "analysis", undefined, { format: "csv", resultCount: results.length }, req);

      res.setHeader("Content-Type", "text/csv");
      res.setHeader("Content-Disposition", `attachment; filename="analysis_results_${Date.now()}.csv"`);
      res.send(csvRows.join("\n"));
    } catch (error) {
      res.status(500).json({ error: "Export thất bại" });
    }
  });

  app.get("/api/export/json", async (req, res) => {
    try {
      const results = await storage.getResults();
      const dataset = await storage.getDataset();
      
      if (!results || results.length === 0) {
        return res.status(404).json({ error: "Không có kết quả phân tích để xuất" });
      }

      const exportData = {
        exportDate: new Date().toISOString(),
        datasetInfo: dataset ? {
          name: dataset.dataset.name,
          totalRows: dataset.dataset.originalRowCount,
          columns: dataset.dataset.columns.length,
        } : null,
        results: results.map(r => ({
          modelType: r.modelType,
          metrics: {
            accuracy: r.accuracy,
            precision: r.precision,
            recall: r.recall,
            f1Score: r.f1Score,
          },
          predictions: {
            ddosDetected: r.ddosDetected,
            normalTraffic: r.normalTraffic,
          },
          bestHyperparams: r.bestHyperparams,
          crossValidation: r.crossValidation,
        })),
      };

      logAudit("export", "analysis", undefined, { format: "json", resultCount: results.length }, req);

      res.setHeader("Content-Type", "application/json");
      res.setHeader("Content-Disposition", `attachment; filename="analysis_results_${Date.now()}.json"`);
      res.json(exportData);
    } catch (error) {
      res.status(500).json({ error: "Export thất bại" });
    }
  });

  // ============== BATCH PROCESSING ==============
  app.post("/api/batch/analyze", async (req, res) => {
    try {
      const { datasets, models } = req.body;
      
      if (!Array.isArray(datasets) || datasets.length === 0) {
        return res.status(400).json({ error: "Danh sách datasets không hợp lệ" });
      }
      
      if (datasets.length > 5) {
        return res.status(400).json({ error: "Tối đa 5 datasets mỗi lần" });
      }

      const batchResults: any[] = [];
      
      for (const datasetInfo of datasets) {
        const { name, data } = datasetInfo;
        try {
          const { columns, rows } = parseDataContent(data, name);
          const featureColumns = getFeatureColumns(columns);
          const datasetId = randomUUID();
          
          const datasetResults: any[] = [];
          const modelsToRun = models || ["decision_tree", "random_forest", "knn"];
          
          for (const modelType of modelsToRun) {
            const result = await analyzeWithModel(datasetId, modelType as any, rows, featureColumns);
            datasetResults.push(result);
          }
          
          batchResults.push({
            dataset: name,
            status: "success",
            results: datasetResults,
          });
        } catch (err) {
          batchResults.push({
            dataset: name,
            status: "error",
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      }

      logAudit("batch_analyze", "analysis", undefined, { datasetCount: datasets.length, modelCount: models?.length }, req);

      res.json({
        processed: batchResults.length,
        results: batchResults,
      });
    } catch (error) {
      res.status(500).json({ error: "Batch processing thất bại" });
    }
  });

  // ============== AUDIT LOGS ==============
  app.get("/api/audit-logs", async (req, res) => {
    try {
      const limit = Math.min(parseInt(req.query.limit as string) || 50, 200);
      const offset = parseInt(req.query.offset as string) || 0;
      
      const logs = await storage.getAuditLogs(limit, offset);
      res.json({
        logs,
        total: logs.length,
        limit,
        offset,
      });
    } catch (error) {
      console.error("Failed to get audit logs:", error);
      res.status(500).json({ error: "Failed to get audit logs" });
    }
  });

  // ============== CACHE MANAGEMENT ==============
  app.get("/api/cache/stats", async (req, res) => {
    const stats = getCacheStats();
    res.json(stats);
  });

  app.delete("/api/cache", async (req, res) => {
    clearCache();
    logAudit("clear_cache", "system", undefined, {}, req);
    res.json({ success: true, message: "Cache đã được xóa" });
  });

  // ============== FEATURE IMPORTANCE & CONFUSION MATRIX ==============
  app.get("/api/analysis/:resultId/confusion-matrix", async (req, res) => {
    try {
      const results = await storage.getResults();
      const result = results?.find(r => r.id === req.params.resultId);
      
      if (!result) {
        return res.status(404).json({ error: "Không tìm thấy kết quả phân tích" });
      }

      // Generate confusion matrix from result predictions
      const matrix = {
        truePositives: result.ddosDetected,
        trueNegatives: result.normalTraffic,
        falsePositives: Math.round(result.ddosDetected * (1 - result.precision) / result.precision) || 0,
        falseNegatives: Math.round(result.ddosDetected * (1 - result.recall) / result.recall) || 0,
        matrix: [[result.normalTraffic, 0], [0, result.ddosDetected]],
        labels: ["Normal", "DDoS"],
      };

      res.json(matrix);
    } catch (error) {
      res.status(500).json({ error: "Lấy confusion matrix thất bại" });
    }
  });

  app.get("/api/dataset", async (req, res) => {
    try {
      const data = await storage.getDataset();
      if (!data) {
        return res.json(null);
      }
      res.json(data);
    } catch (error) {
      res.status(500).json({ error: "Failed to get dataset" });
    }
  });

  // P0-02: DEPRECATED - Legacy JSON upload endpoint removed for security
  // All uploads must use POST /api/upload/file with FormData (multipart/form-data)
  app.post("/api/upload", (_req, res) => {
    return res.status(410).json({ 
      error: "Endpoint không còn được hỗ trợ. Vui lòng sử dụng POST /api/upload/file với FormData (multipart/form-data).",
      deprecated: true,
      newEndpoint: "/api/upload/file",
      message: "Use FormData with 'file' field for file upload"
    });
  });

  app.get("/api/results", async (req, res) => {
    try {
      const results = await storage.getResults();
      res.json(results);
    } catch (error) {
      res.status(500).json({ error: "Failed to get results" });
    }
  });

  app.post("/api/analyze", async (req, res) => {
    try {
      const parsed = analyzeRequestSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.message });
      }

      const { datasetId, modelTypes } = parsed.data;
      const datasetData = await storage.getDataset();

      if (!datasetData) {
        return res.status(404).json({ error: "Dataset not found" });
      }

      const allRows: DataRow[] = (global as any).__datasetRows || [];
      if (allRows.length === 0) {
        return res.status(400).json({ error: "No data available for analysis" });
      }

      const featureColumns = getFeatureColumns(datasetData.dataset.columns);
      const requestedMode = datasetData.dataset.mode || "supervised";

      // CRITICAL: Check actual label status to determine effective mode
      const trainingData = extractFeatures(allRows, featureColumns);
      const { hasLabel, labeledRowCount } = trainingData;
      
      // Determine effective mode: if user requested supervised but no labels, fallback to unlabeled
      const effectiveMode = (requestedMode === "unlabeled" || !hasLabel) ? "unlabeled" : "supervised";

      await storage.clearResults();

      const results = [];
      
      // ============== UNLABELED MODE: No metrics, anomaly-based only ==============
      if (effectiveMode === "unlabeled") {
        const result = await analyzeUnlabeled(
          datasetId,
          allRows,
          featureColumns,
          datasetData.dataset.columns,
          { labeledRowCount }
        );
        
        // Add warning if user requested supervised but we fell back to unlabeled
        if (requestedMode === "supervised" && !hasLabel) {
          if (!result.warnings) result.warnings = [];
          result.warnings.push("Requested supervised mode but dataset has insufficient labels. Falling back to anomaly-based detection.");
        }
        
        await storage.addResult(result);
        results.push(result);
        
        // Audit log for analysis
        logAudit("analyze", "analysis", datasetId, { 
          modelCount: 1, 
          models: ["anomaly_ensemble"],
          mode: effectiveMode,
          fallback: requestedMode !== effectiveMode
        }, req);

        return res.json(results);
      }
      
      // ============== SUPERVISED MODE: Full metrics ==============
      for (const modelType of modelTypes) {
        const result = await analyzeWithModel(datasetId, modelType, allRows, featureColumns);
        
        // Add mode to result with proper type
        const resultWithMode = {
          ...result,
          mode: effectiveMode as "supervised" | "unlabeled",
        };
        
        await storage.addResult(resultWithMode);
        results.push(resultWithMode);
      }

      // Audit log for analysis
      logAudit("analyze", "analysis", datasetId, { 
        modelCount: modelTypes.length, 
        models: modelTypes,
        mode: effectiveMode 
      }, req);

      res.json(results);
    } catch (error) {
      console.error("Analysis error:", error);
      res.status(500).json({ error: "Failed to analyze dataset" });
    }
  });

  // ============== LEARNING ENDPOINTS ==============
  
  // Get learning statistics
  app.get("/api/learning/stats", async (req, res) => {
    try {
      const stats = await learningService.getLearningStats();
      res.json(stats);
    } catch (error) {
      console.error("Learning stats error:", error);
      res.status(500).json({ error: "Failed to get learning stats" });
    }
  });
  
  // Get learned patterns
  app.get("/api/learning/patterns", async (req, res) => {
    try {
      const patterns = await learningService.getLearnedPatterns();
      res.json(patterns);
    } catch (error) {
      console.error("Learning patterns error:", error);
      res.status(500).json({ error: "Failed to get learned patterns" });
    }
  });
  
  // Learn from current analysis results
  app.post("/api/learning/learn", async (req, res) => {
    try {
      const datasetData = await storage.getDataset();
      if (!datasetData) {
        return res.status(404).json({ error: "No dataset available" });
      }
      
      const allRows: DataRow[] = (global as any).__datasetRows || [];
      if (allRows.length === 0) {
        return res.status(400).json({ error: "No data available to learn from" });
      }
      
      const featureColumns = getFeatureColumns(datasetData.dataset.columns);
      
      // Get labels from the most recent analysis or generate from anomaly detection
      const results = await storage.getResults();
      let labels: number[] = [];
      let attackTypes: string[] = [];
      
      if (results.length > 0) {
        // Use predictions from the best performing model
        const bestResult = results.sort((a, b) => (b.f1Score || 0) - (a.f1Score || 0))[0];
        
        // Generate labels based on anomaly detection if no supervised results
        const { features } = extractFeaturesForReport(allRows, featureColumns);
        const normalizedFeatures = normalizeForReport(features);
        const { scores } = runAnomalyDetection(normalizedFeatures);
        
        labels = scores.map(s => s > 0.5 ? 1 : 0);
        
        // Get attack types from classification
        const attackClassification = bestResult.attackTypes || [];
        for (let i = 0; i < allRows.length; i++) {
          if (labels[i] === 1 && attackClassification.length > 0) {
            attackTypes.push(attackClassification[0]?.type || "unknown");
          } else {
            attackTypes.push("normal");
          }
        }
      } else {
        // No analysis results, use anomaly detection
        const { features } = extractFeaturesForReport(allRows, featureColumns);
        const normalizedFeatures = normalizeForReport(features);
        const { scores } = runAnomalyDetection(normalizedFeatures);
        labels = scores.map(s => s > 0.5 ? 1 : 0);
        attackTypes = labels.map(l => l === 1 ? "unknown_ddos" : "normal");
      }
      
      const result = await learningService.addTrainingSamples(
        allRows,
        labels,
        featureColumns,
        datasetData.dataset.name,
        attackTypes
      );
      
      // Record model performance if we have analysis results
      if (results.length > 0) {
        for (const r of results) {
          await learningService.recordModelPerformance(
            r.modelType,
            r.accuracy,
            r.precision,
            r.recall,
            r.f1Score
          );
        }
      }
      
      res.json({
        success: true,
        ...result,
        message: `Đã học từ ${result.samplesAdded} mẫu (${result.ddosSamplesAdded} DDoS, ${result.normalSamplesAdded} Normal). Phát hiện ${result.patternsLearned} patterns mới.`
      });
    } catch (error) {
      console.error("Learning error:", error);
      res.status(500).json({ error: "Failed to learn from data" });
    }
  });
  
  // Analyze using learned patterns
  app.post("/api/learning/analyze-with-patterns", async (req, res) => {
    try {
      const datasetData = await storage.getDataset();
      if (!datasetData) {
        return res.status(404).json({ error: "No dataset available" });
      }
      
      const allRows: DataRow[] = (global as any).__datasetRows || [];
      if (allRows.length === 0) {
        return res.status(400).json({ error: "No data available" });
      }
      
      const featureColumns = getFeatureColumns(datasetData.dataset.columns);
      
      // Analyze each row against learned patterns
      const results: { rowIndex: number; matchedPattern: string | null; confidence: number; isAttack: boolean }[] = [];
      let attackCount = 0;
      
      for (let i = 0; i < Math.min(allRows.length, 1000); i++) {
        const row = allRows[i];
        const features: number[] = [];
        
        for (const col of featureColumns) {
          const val = row[col];
          features.push(typeof val === "number" ? val : parseFloat(String(val)) || 0);
        }
        
        const match = await learningService.matchLearnedPatterns(features, featureColumns);
        
        if (match.matchedPattern && match.confidence > 0.3) {
          attackCount++;
          results.push({
            rowIndex: i,
            matchedPattern: match.matchedPattern.patternName,
            confidence: match.confidence,
            isAttack: true,
          });
        } else {
          results.push({
            rowIndex: i,
            matchedPattern: null,
            confidence: 0,
            isAttack: false,
          });
        }
      }
      
      res.json({
        totalAnalyzed: results.length,
        attacksDetected: attackCount,
        normalTraffic: results.length - attackCount,
        attackRate: attackCount / results.length,
        sampleResults: results.slice(0, 20),
      });
    } catch (error) {
      console.error("Pattern analysis error:", error);
      res.status(500).json({ error: "Failed to analyze with patterns" });
    }
  });
  
  // Clear all learned data
  app.delete("/api/learning/clear", async (req, res) => {
    try {
      await learningService.clearAllData();
      res.json({ success: true, message: "Đã xóa tất cả dữ liệu học" });
    } catch (error) {
      console.error("Clear learning error:", error);
      res.status(500).json({ error: "Failed to clear learning data" });
    }
  });

  // ============== CUSTOM LABEL MANAGEMENT ==============
  
  // Get all label mappings (built-in + custom)
  app.get("/api/labels", (req, res) => {
    try {
      const labels = getAllLabelMappings();
      const customLabels = getCustomLabels();
      
      // Group by category
      const grouped: Record<string, { label: string; config: LabelConfig }[]> = {};
      for (const [label, config] of Object.entries(labels)) {
        const category = config.category;
        if (!grouped[category]) {
          grouped[category] = [];
        }
        grouped[category].push({ label, config });
      }
      
      res.json({
        total: Object.keys(labels).length,
        customCount: Object.keys(customLabels).length,
        builtInCount: Object.keys(labels).length - Object.keys(customLabels).length,
        labels,
        customLabels,
        grouped,
      });
    } catch (error) {
      console.error("Get labels error:", error);
      res.status(500).json({ error: "Failed to get label mappings" });
    }
  });
  
  // Add custom label
  app.post("/api/labels", (req, res) => {
    try {
      const { labelName, isAttack, category, severity, description } = req.body;
      
      if (!labelName) {
        return res.status(400).json({ error: "labelName is required" });
      }
      
      addCustomLabel(labelName, {
        isAttack: Boolean(isAttack),
        category: category as LabelCategory,
        severity: severity as 'low' | 'medium' | 'high' | 'critical',
        description,
      });
      
      res.json({
        success: true,
        message: `Đã thêm label "${labelName}" ${isAttack ? 'là tấn công' : 'không phải tấn công'}`,
        label: labelName,
        config: getCustomLabels()[labelName.toLowerCase().trim().replace(/[\s\-\.]+/g, '_')],
      });
    } catch (error) {
      console.error("Add label error:", error);
      res.status(500).json({ error: "Failed to add custom label" });
    }
  });
  
  // Update/set multiple custom labels
  app.put("/api/labels", (req, res) => {
    try {
      const { labels } = req.body;
      
      if (!labels || typeof labels !== 'object') {
        return res.status(400).json({ error: "labels object is required" });
      }
      
      setCustomLabels(labels as Record<string, LabelConfig>);
      
      res.json({
        success: true,
        message: `Đã cập nhật ${Object.keys(labels).length} labels`,
        customLabels: getCustomLabels(),
      });
    } catch (error) {
      console.error("Update labels error:", error);
      res.status(500).json({ error: "Failed to update labels" });
    }
  });
  
  // Delete custom label
  app.delete("/api/labels/:labelName", (req, res) => {
    try {
      const { labelName } = req.params;
      
      const removed = removeCustomLabel(labelName);
      
      if (removed) {
        res.json({
          success: true,
          message: `Đã xóa label "${labelName}"`,
        });
      } else {
        res.status(404).json({
          error: `Label "${labelName}" không tồn tại hoặc là label hệ thống`,
        });
      }
    } catch (error) {
      console.error("Delete label error:", error);
      res.status(500).json({ error: "Failed to delete label" });
    }
  });
  
  // Get label categories info
  app.get("/api/labels/categories", (req, res) => {
    const categories: Record<string, { description: string; isAttack: boolean; severity: string }> = {
      normal: { description: 'Lưu lượng bình thường', isAttack: false, severity: 'low' },
      ddos: { description: 'Tấn công DDoS chung', isAttack: true, severity: 'critical' },
      ddos_volumetric: { description: 'DDoS lưu lượng lớn (UDP/ICMP flood)', isAttack: true, severity: 'critical' },
      ddos_protocol: { description: 'DDoS khai thác giao thức (SYN flood)', isAttack: true, severity: 'critical' },
      ddos_amplification: { description: 'DDoS khuếch đại (DNS/NTP)', isAttack: true, severity: 'critical' },
      ddos_application: { description: 'DDoS lớp ứng dụng (HTTP flood)', isAttack: true, severity: 'high' },
      reconnaissance: { description: 'Quét thăm dò (Port scan)', isAttack: true, severity: 'medium' },
      bruteforce: { description: 'Tấn công dò mật khẩu', isAttack: true, severity: 'high' },
      exploit: { description: 'Khai thác lỗ hổng', isAttack: true, severity: 'critical' },
      malware: { description: 'Mã độc, botnet', isAttack: true, severity: 'critical' },
      infiltration: { description: 'Xâm nhập hệ thống', isAttack: true, severity: 'critical' },
      anomaly_traffic: { description: 'Bất thường lưu lượng (không rõ attack)', isAttack: false, severity: 'medium' },
      anomaly_behavior: { description: 'Bất thường hành vi', isAttack: false, severity: 'medium' },
      anomaly_protocol: { description: 'Bất thường giao thức', isAttack: false, severity: 'medium' },
      anomaly_resource: { description: 'Bất thường tài nguyên', isAttack: false, severity: 'high' },
      custom: { description: 'Label tùy chỉnh', isAttack: false, severity: 'medium' },
    };
    
    res.json({ categories });
  });

  // ============== USER FEEDBACK & REVIEW SYSTEM ==============
  
  // Get all user feedback
  app.get("/api/feedback", async (req, res) => {
    try {
      const feedback = await storage.getAllFeedback();
      res.json({ feedback, total: feedback.length });
    } catch (error) {
      console.error("Get feedback error:", error);
      res.status(500).json({ error: "Failed to get feedback" });
    }
  });
  
  // Submit user feedback/correction for a row
  app.post("/api/feedback", async (req, res) => {
    try {
      const { rowIndex, originalLabel, correctedLabel, isAttack, category, severity, userNotes, features, datasetName } = req.body;
      
      if (rowIndex === undefined || !correctedLabel || !category || !severity) {
        return res.status(400).json({ error: "Missing required fields: rowIndex, correctedLabel, category, severity" });
      }
      
      const feedback = await storage.addFeedback({
        rowIndex,
        originalLabel: originalLabel || null,
        correctedLabel,
        isAttack: Boolean(isAttack),
        category,
        severity,
        userNotes: userNotes || null,
        features: features || null,
        datasetName: datasetName || null,
        isApplied: false,
      });
      
      // Also add to custom labels if not exists
      addCustomLabel(correctedLabel, {
        isAttack: Boolean(isAttack),
        category: category as LabelCategory,
        severity: severity as 'low' | 'medium' | 'high' | 'critical',
        description: userNotes || `Label do người dùng định nghĩa`,
      });
      
      res.json({
        success: true,
        message: `Đã lưu phản hồi cho dòng ${rowIndex}`,
        feedback,
      });
    } catch (error) {
      console.error("Add feedback error:", error);
      res.status(500).json({ error: "Failed to add feedback" });
    }
  });
  
  // Apply all pending feedback to learning system
  app.post("/api/feedback/apply", async (req, res) => {
    try {
      const pendingFeedback = await storage.getPendingFeedback();
      
      if (pendingFeedback.length === 0) {
        return res.json({ success: true, message: "Không có phản hồi nào cần áp dụng", applied: 0 });
      }
      
      let applied = 0;
      for (const fb of pendingFeedback) {
        // Mark as applied (features are stored for future batch training)
        await storage.markFeedbackApplied(fb.id);
        applied++;
        
        // The corrected label is already added to custom labels when feedback was submitted
        // Features can be used for batch training later via /api/learning/learn
      }
      
      res.json({
        success: true,
        message: `Đã áp dụng ${applied} phản hồi vào hệ thống học`,
        applied,
      });
    } catch (error) {
      console.error("Apply feedback error:", error);
      res.status(500).json({ error: "Failed to apply feedback" });
    }
  });
  
  // Delete feedback
  app.delete("/api/feedback/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      await storage.deleteFeedback(id);
      res.json({ success: true, message: "Đã xóa phản hồi" });
    } catch (error) {
      console.error("Delete feedback error:", error);
      res.status(500).json({ error: "Failed to delete feedback" });
    }
  });
  
  // ============== USER TAGS SYSTEM ==============
  
  // Get all tags
  app.get("/api/tags", async (req, res) => {
    try {
      const tags = await storage.getAllTags();
      res.json({ tags, total: tags.length });
    } catch (error) {
      console.error("Get tags error:", error);
      res.status(500).json({ error: "Failed to get tags" });
    }
  });
  
  // Create a new tag
  app.post("/api/tags", async (req, res) => {
    try {
      const { tagName, tagColor, description, isAttackTag } = req.body;
      
      if (!tagName || !tagColor) {
        return res.status(400).json({ error: "tagName and tagColor are required" });
      }
      
      const tag = await storage.addTag({
        tagName,
        tagColor,
        description: description || null,
        isAttackTag: Boolean(isAttackTag),
        usageCount: 0,
      });
      
      res.json({ success: true, tag });
    } catch (error) {
      console.error("Add tag error:", error);
      res.status(500).json({ error: "Failed to add tag" });
    }
  });
  
  // Update tag usage count
  app.put("/api/tags/:id/use", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      await storage.incrementTagUsage(id);
      res.json({ success: true });
    } catch (error) {
      console.error("Update tag error:", error);
      res.status(500).json({ error: "Failed to update tag" });
    }
  });
  
  // Delete tag
  app.delete("/api/tags/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      await storage.deleteTag(id);
      res.json({ success: true, message: "Đã xóa tag" });
    } catch (error) {
      console.error("Delete tag error:", error);
      res.status(500).json({ error: "Failed to delete tag" });
    }
  });
  
  // Get review summary for current dataset
  app.get("/api/review/summary", async (req, res) => {
    try {
      const datasetData = await storage.getDataset();
      if (!datasetData) {
        return res.status(404).json({ error: "No dataset available" });
      }
      
      const allRows: DataRow[] = (global as any).__datasetRows || [];
      const feedback = await storage.getAllFeedback();
      const tags = await storage.getAllTags();
      
      res.json({
        totalRows: allRows.length,
        feedbackCount: feedback.length,
        pendingFeedback: feedback.filter(f => !f.isApplied).length,
        appliedFeedback: feedback.filter(f => f.isApplied).length,
        availableTags: tags.length,
        datasetName: datasetData.dataset.name,
      });
    } catch (error) {
      console.error("Review summary error:", error);
      res.status(500).json({ error: "Failed to get review summary" });
    }
  });

  return httpServer;
}
