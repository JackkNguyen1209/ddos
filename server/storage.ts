import { randomUUID } from "crypto";
import type { Dataset, DataRow, AnalysisResult, MLModelType, InsertUserFeedback, UserFeedback, InsertUserTag, UserTag } from "@shared/schema";

export interface IStorage {
  setDataset(dataset: Dataset, previewData: DataRow[]): Promise<void>;
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
}

export class MemStorage implements IStorage {
  private dataset: Dataset | null = null;
  private previewData: DataRow[] = [];
  private results: AnalysisResult[] = [];
  private feedback: UserFeedback[] = [];
  private tags: UserTag[] = [];
  private feedbackIdCounter = 1;
  private tagIdCounter = 1;

  async setDataset(dataset: Dataset, previewData: DataRow[]): Promise<void> {
    this.dataset = dataset;
    this.previewData = previewData;
    this.results = [];
  }

  async getDataset(): Promise<{ dataset: Dataset; previewData: DataRow[] } | null> {
    if (!this.dataset) return null;
    return { dataset: this.dataset, previewData: this.previewData };
  }

  async addResult(result: AnalysisResult): Promise<void> {
    const existingIndex = this.results.findIndex(
      (r) => r.modelType === result.modelType
    );
    if (existingIndex >= 0) {
      this.results[existingIndex] = result;
    } else {
      this.results.push(result);
    }
  }

  async getResults(): Promise<AnalysisResult[]> {
    return [...this.results];
  }

  async clearResults(): Promise<void> {
    this.results = [];
  }

  // Feedback methods
  async addFeedback(feedback: InsertUserFeedback): Promise<UserFeedback> {
    const newFeedback: UserFeedback = {
      id: this.feedbackIdCounter++,
      rowIndex: feedback.rowIndex,
      originalLabel: feedback.originalLabel ?? null,
      correctedLabel: feedback.correctedLabel,
      isAttack: feedback.isAttack,
      category: feedback.category,
      severity: feedback.severity,
      userNotes: feedback.userNotes ?? null,
      features: feedback.features ?? null,
      datasetName: feedback.datasetName ?? null,
      isApplied: feedback.isApplied ?? false,
      createdAt: new Date(),
    };
    this.feedback.push(newFeedback);
    return newFeedback;
  }

  async getAllFeedback(): Promise<UserFeedback[]> {
    return [...this.feedback];
  }

  async getPendingFeedback(): Promise<UserFeedback[]> {
    return this.feedback.filter(f => !f.isApplied);
  }

  async markFeedbackApplied(id: number): Promise<void> {
    const fb = this.feedback.find(f => f.id === id);
    if (fb) {
      fb.isApplied = true;
    }
  }

  async deleteFeedback(id: number): Promise<void> {
    this.feedback = this.feedback.filter(f => f.id !== id);
  }

  // Tag methods
  async addTag(tag: InsertUserTag): Promise<UserTag> {
    const newTag: UserTag = {
      id: this.tagIdCounter++,
      tagName: tag.tagName,
      tagColor: tag.tagColor,
      description: tag.description ?? null,
      isAttackTag: tag.isAttackTag ?? false,
      usageCount: tag.usageCount ?? 0,
      createdAt: new Date(),
    };
    this.tags.push(newTag);
    return newTag;
  }

  async getAllTags(): Promise<UserTag[]> {
    return [...this.tags];
  }

  async incrementTagUsage(id: number): Promise<void> {
    const tag = this.tags.find(t => t.id === id);
    if (tag) {
      tag.usageCount++;
    }
  }

  async deleteTag(id: number): Promise<void> {
    this.tags = this.tags.filter(t => t.id !== id);
  }
}

export const storage = new MemStorage();
