import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { analyzeWithModel, getFeatureColumns } from "./ml-algorithms";
import { uploadDatasetSchema, analyzeRequestSchema, type DataRow, type Dataset } from "@shared/schema";
import { randomUUID } from "crypto";

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

function hasLabelColumn(columns: string[]): boolean {
  const labelColumns = ["label", "class", "attack", "target", "Label", "Class", "Attack", "Target", "is_attack", "ddos", "DDoS"];
  return columns.some(col => labelColumns.includes(col) || labelColumns.some(lc => col.toLowerCase() === lc.toLowerCase()));
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

  for (const row of rows) {
    let hasMissing = false;
    for (const col of columns) {
      if (row[col] === null || row[col] === undefined || row[col] === "") {
        hasMissing = true;
        missingValues++;
        break;
      }
    }

    if (hasMissing) continue;

    const rowKey = columns.map((col) => String(row[col])).join("|");
    if (seenRows.has(rowKey)) {
      duplicates++;
      continue;
    }
    seenRows.add(rowKey);

    let hasOutlier = false;
    for (const col of columns) {
      const value = row[col];
      if (typeof value === "number" && (value < -1e10 || value > 1e10)) {
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
      const { columns, rows } = parseCSV(data);

      if (rows.length === 0) {
        return res.status(400).json({ error: "No data found in CSV" });
      }

      const { cleanedRows, missingValues, duplicates, outliers } = cleanData(rows, columns);

      const dataset: Dataset = {
        id: randomUUID(),
        name,
        originalRowCount: rows.length,
        cleanedRowCount: cleanedRows.length,
        columns,
        uploadedAt: new Date().toISOString(),
        isProcessed: true,
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

      const hasLabel = hasLabelColumn(columns);
      const warning = hasLabel 
        ? undefined 
        : "Cảnh báo: Không tìm thấy cột label (như 'label', 'class', 'attack'). Kết quả phân tích có thể không chính xác.";

      res.json({ dataset, previewData, warning });
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

      await storage.clearResults();

      const results = [];
      for (const modelType of modelTypes) {
        const result = await analyzeWithModel(datasetId, modelType, allRows, featureColumns);
        await storage.addResult(result);
        results.push(result);
      }

      res.json(results);
    } catch (error) {
      console.error("Analysis error:", error);
      res.status(500).json({ error: "Failed to analyze dataset" });
    }
  });

  return httpServer;
}
