import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { analyzeWithModel, getFeatureColumns, runAnomalyDetection, generateUnlabeledReport, buildFeatureMapping, getFeatureStatistics, analyzeRowForAttack } from "./ml-algorithms";
import { uploadDatasetSchema, analyzeRequestSchema, type DataRow, type Dataset } from "@shared/schema";
import { randomUUID } from "crypto";
import * as XLSX from "xlsx";
import { learningService } from "./learning-service";
import { 
  detectSchemaType, 
  analyzeFeatureUsage, 
  findColumnMapping, 
  normalizeDataset, 
  getModelForSchema,
  getLabelStats,
  type SchemaType
} from "./schema-detection";

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

  app.post("/api/upload", async (req, res) => {
    try {
      const parsed = uploadDatasetSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: parsed.error.message });
      }

      const { name, data } = parsed.data;
      const { columns, rows } = parseDataContent(data, name);

      if (rows.length === 0) {
        return res.status(400).json({ error: "Không tìm thấy dữ liệu trong file. Hãy đảm bảo file có định dạng CSV hoặc Excel (.xlsx) hợp lệ." });
      }

      // Epic 1.1: Phát hiện file mô tả schema
      const schemaCheck = isSchemaDescriptionFile(columns, rows);
      if (schemaCheck.isSchema) {
        return res.status(400).json({ 
          error: `File này là mô tả schema/dictionary, không phải dataset thực. ${schemaCheck.reason} Hãy upload file dữ liệu records thực.`,
          isSchemaFile: true 
        });
      }

      // Schema Detection + Column Normalization
      const columnMappings = findColumnMapping(columns);
      const { type: schemaType, confidence: schemaConfidence } = detectSchemaType(columns);
      
      // Convert rows to 2D array for feature analysis
      const dataArray = rows.map(row => columns.map(col => row[col]));
      const featureReport = analyzeFeatureUsage(columns, dataArray, schemaType);
      
      // Normalize dataset columns and labels
      const normalizedResult = normalizeDataset(columns, dataArray, columnMappings);
      const normalizedColumns = normalizedResult.columns;
      
      // Get recommended models based on schema type
      const recommendedModels = getModelForSchema(schemaType);

      // Epic 1.2: Validate Feature Contract và chọn Mode
      const featureValidation = validateFeatureContract(columns);
      const mode: import("@shared/schema").DetectionMode = featureValidation.hasLabelColumn ? "supervised" : "unlabeled";

      // Get label statistics if has label column
      let labelStats: Record<string, { count: number; percentage: number; category: string }> = {};
      if (featureValidation.hasLabelColumn && featureValidation.detectedLabelColumn) {
        const labelIndex = columns.indexOf(featureValidation.detectedLabelColumn);
        if (labelIndex >= 0) {
          labelStats = getLabelStats(dataArray, labelIndex);
        }
      }

      // Smart Feature Mapping - tự động nhận diện các cột feature
      const featureMapping = buildFeatureMapping(columns);
      const featureMappingObj = Object.fromEntries(Array.from(featureMapping.entries()));
      const detectedFeatureTypes = Array.from(new Set(featureMapping.values()));

      const { cleanedRows, missingValues, duplicates, outliers } = cleanData(rows, columns);
      
      // Get feature statistics for the dataset
      const featureStats = getFeatureStatistics(cleanedRows, featureMapping);

      const dataset: Dataset = {
        id: randomUUID(),
        name,
        originalRowCount: rows.length,
        cleanedRowCount: cleanedRows.length,
        columns,
        uploadedAt: new Date().toISOString(),
        isProcessed: true,
        mode,  // Thêm mode vào dataset
        labelColumn: featureValidation.detectedLabelColumn || undefined,
        featureValidation,  // Thêm feature validation
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

      // Tạo warnings dựa trên validation
      const warnings: string[] = [];
      
      // Schema type warning
      const schemaTypeNames: Record<SchemaType, string> = {
        'cicflowmeter': 'CICFlowMeter',
        'event_log': 'Event/Log',
        'unknown': 'Không xác định'
      };
      warnings.push(`📋 Schema: ${schemaTypeNames[schemaType]} (${schemaConfidence.toFixed(0)}% tin cậy)`);
      
      if (mode === "unlabeled") {
        warnings.push("🔍 Mode: UNLABELED - Không tìm thấy cột label. Sẽ chạy inference và hiển thị score/cảnh báo thay vì accuracy.");
      } else {
        warnings.push(`✓ Mode: SUPERVISED - Phát hiện cột label: "${featureValidation.detectedLabelColumn}". Sẽ train và đánh giá chuẩn.`);
      }
      
      // Feature report warnings
      if (!featureReport.isReliable) {
        warnings.push(`⚠️ Kết quả có thể không đáng tin cậy`);
      }
      
      for (const warn of featureReport.warnings) {
        warnings.push(`⚠️ ${warn}`);
      }
      
      if (featureValidation.missingRequired.length > 0 && featureReport.warnings.length === 0) {
        warnings.push(`⚠️ Thiếu features quan trọng: ${featureValidation.missingRequired.join(", ")}. ${featureValidation.confidenceReason}`);
      }
      
      if (featureValidation.confidenceLevel === "low") {
        warnings.push("⚠️ Độ tin cậy THẤP: Dataset thiếu nhiều features cần thiết cho phát hiện DDoS chính xác.");
      }

      // Add feature detection info
      if (detectedFeatureTypes.length > 0) {
        warnings.push(`📊 Phát hiện ${detectedFeatureTypes.length} loại feature: ${detectedFeatureTypes.slice(0, 5).join(", ")}${detectedFeatureTypes.length > 5 ? "..." : ""}`);
      }
      
      // Add anomaly indicators from feature statistics
      if (featureStats.anomalyIndicators.length > 0) {
        warnings.push(`⚠️ Dấu hiệu bất thường: ${featureStats.anomalyIndicators.slice(0, 3).join(", ")}`);
      }

      res.json({ 
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
    } catch (error) {
      console.error("Upload error:", error);
      res.status(500).json({ error: "Failed to process dataset" });
    }
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
      const mode = datasetData.dataset.mode || "supervised";

      await storage.clearResults();

      const results = [];
      for (const modelType of modelTypes) {
        const result = await analyzeWithModel(datasetId, modelType, allRows, featureColumns);
        
        // Add mode to result
        const resultWithMode = {
          ...result,
          mode,
        };
        
        // Add unlabeled report if unlabeled mode
        if (mode === "unlabeled") {
          const { features } = extractFeaturesForReport(allRows, featureColumns);
          const normalizedFeatures = normalizeForReport(features);
          const { scores, alertRate } = runAnomalyDetection(normalizedFeatures);
          
          resultWithMode.unlabeledReport = generateUnlabeledReport(
            allRows,
            datasetData.dataset.columns,
            features,
            scores
          );
        }
        
        await storage.addResult(resultWithMode);
        results.push(resultWithMode);
      }

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

  return httpServer;
}
