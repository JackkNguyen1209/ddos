import { randomUUID } from "crypto";
import type { Dataset, DataRow, AnalysisResult, MLModelType } from "@shared/schema";

export interface IStorage {
  setDataset(dataset: Dataset, previewData: DataRow[]): Promise<void>;
  getDataset(): Promise<{ dataset: Dataset; previewData: DataRow[] } | null>;
  addResult(result: AnalysisResult): Promise<void>;
  getResults(): Promise<AnalysisResult[]>;
  clearResults(): Promise<void>;
}

export class MemStorage implements IStorage {
  private dataset: Dataset | null = null;
  private previewData: DataRow[] = [];
  private results: AnalysisResult[] = [];

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
}

export const storage = new MemStorage();
