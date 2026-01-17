import { randomUUID } from "crypto";
import { eq, desc, and } from "drizzle-orm";
import { db } from "./db";
import { 
  datasets, 
  analysisResults, 
  userFeedback, 
  userTags,
  auditLogs,
  type InsertDataset,
  type DatasetRecord,
  type InsertAnalysisResult,
  type AnalysisResultRecord,
  type InsertUserFeedback, 
  type UserFeedback, 
  type InsertUserTag, 
  type UserTag,
  type InsertAuditLog,
  type AuditLog,
  type Dataset, 
  type DataRow, 
  type AnalysisResult
} from "@shared/schema";

export interface IStorage {
  // Dataset methods
  createDataset(data: InsertDataset, previewData: DataRow[]): Promise<DatasetRecord>;
  getDatasetById(id: string): Promise<DatasetRecord | null>;
  getAllDatasets(): Promise<DatasetRecord[]>;
  updateDatasetStatus(id: string, status: string, errorMessage?: string): Promise<void>;
  deleteDataset(id: string): Promise<void>;
  
  // Analysis result methods
  createAnalysisResult(data: InsertAnalysisResult): Promise<AnalysisResultRecord>;
  getResultsByDatasetId(datasetId: string): Promise<AnalysisResultRecord[]>;
  getResultById(id: number): Promise<AnalysisResultRecord | null>;
  deleteResultsByDatasetId(datasetId: string): Promise<void>;
  
  // Legacy interface (for backwards compatibility during migration)
  setDataset(dataset: Dataset, previewData: DataRow[]): Promise<string>;
  getDataset(): Promise<{ dataset: Dataset; previewData: DataRow[] } | null>;
  addResult(result: AnalysisResult): Promise<void>;
  getResults(): Promise<AnalysisResult[]>;
  clearResults(): Promise<void>;
  
  // Feedback methods
  addFeedback(feedback: InsertUserFeedback): Promise<UserFeedback>;
  getAllFeedback(): Promise<UserFeedback[]>;
  getPendingFeedback(): Promise<UserFeedback[]>;
  markFeedbackApplied(id: number): Promise<void>;
  deleteFeedback(id: number): Promise<void>;
  
  // Tag methods
  addTag(tag: InsertUserTag): Promise<UserTag>;
  getAllTags(): Promise<UserTag[]>;
  incrementTagUsage(id: number): Promise<void>;
  deleteTag(id: number): Promise<void>;
  
  // Audit log methods
  addAuditLog(log: InsertAuditLog): Promise<AuditLog>;
  getAuditLogs(limit?: number, offset?: number): Promise<AuditLog[]>;
}

export class DatabaseStorage implements IStorage {
  private currentDatasetId: string | null = null;

  // ============== DATASET METHODS ==============
  
  async createDataset(data: InsertDataset, previewData: DataRow[]): Promise<DatasetRecord> {
    const [result] = await db.insert(datasets).values({
      ...data,
      previewData: previewData as any,
    }).returning();
    this.currentDatasetId = result.id;
    return result;
  }

  async getDatasetById(id: string): Promise<DatasetRecord | null> {
    const [result] = await db.select().from(datasets).where(eq(datasets.id, id));
    return result || null;
  }

  async getAllDatasets(): Promise<DatasetRecord[]> {
    return await db.select().from(datasets).orderBy(desc(datasets.createdAt));
  }

  async updateDatasetStatus(id: string, status: string, errorMessage?: string): Promise<void> {
    await db.update(datasets)
      .set({ status, updatedAt: new Date() })
      .where(eq(datasets.id, id));
  }

  async deleteDataset(id: string): Promise<void> {
    await db.delete(analysisResults).where(eq(analysisResults.datasetId, id));
    await db.delete(datasets).where(eq(datasets.id, id));
  }

  // ============== ANALYSIS RESULT METHODS ==============
  
  async createAnalysisResult(data: InsertAnalysisResult): Promise<AnalysisResultRecord> {
    const [result] = await db.insert(analysisResults).values(data).returning();
    return result;
  }

  async getResultsByDatasetId(datasetId: string): Promise<AnalysisResultRecord[]> {
    return await db.select().from(analysisResults)
      .where(eq(analysisResults.datasetId, datasetId))
      .orderBy(desc(analysisResults.createdAt));
  }

  async getResultById(id: number): Promise<AnalysisResultRecord | null> {
    const [result] = await db.select().from(analysisResults).where(eq(analysisResults.id, id));
    return result || null;
  }

  async deleteResultsByDatasetId(datasetId: string): Promise<void> {
    await db.delete(analysisResults).where(eq(analysisResults.datasetId, datasetId));
  }

  // ============== LEGACY INTERFACE (backwards compatibility) ==============
  
  async setDataset(dataset: Dataset, previewData: DataRow[]): Promise<string> {
    const existingDataset = await this.getDatasetById(dataset.id);
    
    if (existingDataset) {
      await db.update(datasets)
        .set({
          name: dataset.name,
          originalRowCount: dataset.originalRowCount,
          cleanedRowCount: dataset.cleanedRowCount,
          columns: dataset.columns as any,
          mode: dataset.mode || "supervised",
          labelColumn: dataset.labelColumn,
          featureValidation: dataset.featureValidation as any,
          dataQuality: dataset.dataQuality as any,
          previewData: previewData as any,
          updatedAt: new Date(),
        })
        .where(eq(datasets.id, dataset.id));
      this.currentDatasetId = dataset.id;
      return dataset.id;
    }
    
    const insertData: InsertDataset = {
      id: dataset.id,
      name: dataset.name,
      originalRowCount: dataset.originalRowCount,
      cleanedRowCount: dataset.cleanedRowCount,
      columns: dataset.columns as any,
      mode: dataset.mode || "supervised",
      labelColumn: dataset.labelColumn,
      featureValidation: dataset.featureValidation as any,
      dataQuality: dataset.dataQuality as any,
      status: "ready",
    };
    
    await this.createDataset(insertData, previewData);
    return dataset.id;
  }

  async getDataset(): Promise<{ dataset: Dataset; previewData: DataRow[] } | null> {
    if (!this.currentDatasetId) {
      const allDatasets = await this.getAllDatasets();
      if (allDatasets.length === 0) return null;
      this.currentDatasetId = allDatasets[0].id;
    }
    
    const record = await this.getDatasetById(this.currentDatasetId);
    if (!record) return null;
    
    const dataset: Dataset = {
      id: record.id,
      name: record.name,
      originalRowCount: record.originalRowCount,
      cleanedRowCount: record.cleanedRowCount,
      columns: record.columns as string[],
      uploadedAt: record.createdAt.toISOString(),
      isProcessed: record.status === "ready",
      mode: record.mode as any,
      labelColumn: record.labelColumn || undefined,
      featureValidation: record.featureValidation as any,
      dataQuality: record.dataQuality as any,
    };
    
    return { 
      dataset, 
      previewData: (record.previewData as DataRow[]) || [] 
    };
  }

  async addResult(result: AnalysisResult): Promise<void> {
    if (!this.currentDatasetId) {
      throw new Error("No dataset selected");
    }
    
    const existingResults = await this.getResultsByDatasetId(this.currentDatasetId);
    const existing = existingResults.find(r => r.modelType === result.modelType);
    
    if (existing) {
      await db.delete(analysisResults).where(eq(analysisResults.id, existing.id));
    }
    
    await this.createAnalysisResult({
      datasetId: this.currentDatasetId,
      modelType: result.modelType,
      accuracy: result.accuracy,
      precision: result.precision,
      recall: result.recall,
      f1Score: result.f1Score,
      trainingTime: result.trainingTime,
      ddosDetected: result.ddosDetected,
      normalTraffic: result.normalTraffic,
      mode: result.mode || "supervised",
      attackTypes: result.attackTypes as any,
      confusionMatrix: result.confusionMatrix as any,
      unlabeledReport: result.unlabeledReport as any,
      advancedMetrics: result.enhancedMetrics as any,
      featureImportance: result.featureImportance as any,
      warnings: result.warnings as any,
      status: "completed",
    });
  }

  async getResults(): Promise<AnalysisResult[]> {
    if (!this.currentDatasetId) return [];
    
    const records = await this.getResultsByDatasetId(this.currentDatasetId);
    return records.map(r => ({
      id: String(r.id),
      datasetId: r.datasetId,
      modelType: r.modelType as any,
      accuracy: r.accuracy || 0,
      precision: r.precision || 0,
      recall: r.recall || 0,
      f1Score: r.f1Score || 0,
      trainingTime: r.trainingTime,
      ddosDetected: r.ddosDetected,
      normalTraffic: r.normalTraffic,
      analyzedAt: r.createdAt.toISOString(),
      mode: r.mode as any,
      attackTypes: (r.attackTypes as any) || [],
      confusionMatrix: r.confusionMatrix as any || { truePositive: 0, trueNegative: 0, falsePositive: 0, falseNegative: 0 },
      unlabeledReport: r.unlabeledReport as any,
      enhancedMetrics: r.advancedMetrics as any,
      featureImportance: r.featureImportance as any,
      warnings: (r.warnings as string[]) || undefined,
    }));
  }

  async clearResults(): Promise<void> {
    if (!this.currentDatasetId) return;
    await this.deleteResultsByDatasetId(this.currentDatasetId);
  }

  // ============== FEEDBACK METHODS ==============
  
  async addFeedback(feedback: InsertUserFeedback): Promise<UserFeedback> {
    const [result] = await db.insert(userFeedback).values(feedback).returning();
    return result;
  }

  async getAllFeedback(): Promise<UserFeedback[]> {
    return await db.select().from(userFeedback).orderBy(desc(userFeedback.createdAt));
  }

  async getPendingFeedback(): Promise<UserFeedback[]> {
    return await db.select().from(userFeedback)
      .where(eq(userFeedback.isApplied, false))
      .orderBy(desc(userFeedback.createdAt));
  }

  async markFeedbackApplied(id: number): Promise<void> {
    await db.update(userFeedback)
      .set({ isApplied: true })
      .where(eq(userFeedback.id, id));
  }

  async deleteFeedback(id: number): Promise<void> {
    await db.delete(userFeedback).where(eq(userFeedback.id, id));
  }

  // ============== TAG METHODS ==============
  
  async addTag(tag: InsertUserTag): Promise<UserTag> {
    const [result] = await db.insert(userTags).values(tag).returning();
    return result;
  }

  async getAllTags(): Promise<UserTag[]> {
    return await db.select().from(userTags).orderBy(desc(userTags.createdAt));
  }

  async incrementTagUsage(id: number): Promise<void> {
    const [tag] = await db.select().from(userTags).where(eq(userTags.id, id));
    if (tag) {
      await db.update(userTags)
        .set({ usageCount: tag.usageCount + 1 })
        .where(eq(userTags.id, id));
    }
  }

  async deleteTag(id: number): Promise<void> {
    await db.delete(userTags).where(eq(userTags.id, id));
  }

  // ============== AUDIT LOG METHODS ==============
  
  async addAuditLog(log: InsertAuditLog): Promise<AuditLog> {
    const [result] = await db.insert(auditLogs).values(log).returning();
    return result;
  }

  async getAuditLogs(limit: number = 50, offset: number = 0): Promise<AuditLog[]> {
    return await db.select().from(auditLogs)
      .orderBy(desc(auditLogs.createdAt))
      .limit(limit)
      .offset(offset);
  }
}

export const storage = new DatabaseStorage();
