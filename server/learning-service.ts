import { db } from "./db";
import { 
  trainingSamples, 
  modelPerformance, 
  learningSessions, 
  learnedPatterns,
  InsertTrainingSample,
  InsertModelPerformance,
  InsertLearningSession,
  InsertLearnedPattern,
  TrainingSample,
  LearnedPattern,
  type DataRow
} from "@shared/schema";
import { eq, desc, sql, and, gte } from "drizzle-orm";

export interface LearningStats {
  totalSamples: number;
  ddosSamples: number;
  normalSamples: number;
  learnedPatterns: number;
  sessionsCount: number;
  lastLearningDate: string | null;
  modelImprovements: {
    modelType: string;
    improvement: number;
    currentAccuracy: number;
  }[];
}

export interface LearnFromDataResult {
  samplesAdded: number;
  ddosSamplesAdded: number;
  normalSamplesAdded: number;
  patternsLearned: number;
  sessionId: number;
}

export class LearningService {
  async addTrainingSamples(
    rows: DataRow[],
    labels: number[],
    featureColumns: string[],
    source: string,
    attackTypes?: string[]
  ): Promise<LearnFromDataResult> {
    let ddosCount = 0;
    let normalCount = 0;
    
    const samples: InsertTrainingSample[] = [];
    
    for (let i = 0; i < rows.length; i++) {
      const features: number[] = [];
      for (const col of featureColumns) {
        const val = rows[i][col];
        features.push(typeof val === "number" ? val : parseFloat(String(val)) || 0);
      }
      
      const label = labels[i];
      if (label === 1) ddosCount++;
      else normalCount++;
      
      samples.push({
        features,
        label,
        featureNames: featureColumns,
        attackType: attackTypes?.[i] || null,
        source,
      });
    }
    
    if (samples.length > 0) {
      await db.insert(trainingSamples).values(samples);
    }
    
    const totalSamples = await this.getTotalSamples();
    
    const patternsLearned = await this.updateLearnedPatterns(samples.filter(s => s.label === 1));
    
    const [session] = await db.insert(learningSessions).values({
      sessionName: source,
      samplesAdded: samples.length,
      totalSamplesAfter: totalSamples,
      ddosSamplesAdded: ddosCount,
      normalSamplesAdded: normalCount,
      improvementPercentage: null,
    }).returning();
    
    return {
      samplesAdded: samples.length,
      ddosSamplesAdded: ddosCount,
      normalSamplesAdded: normalCount,
      patternsLearned,
      sessionId: session.id,
    };
  }
  
  async getAccumulatedSamples(limit?: number): Promise<TrainingSample[]> {
    if (limit) {
      return db.select().from(trainingSamples).orderBy(desc(trainingSamples.createdAt)).limit(limit);
    }
    return db.select().from(trainingSamples);
  }
  
  async getTotalSamples(): Promise<number> {
    const result = await db.select({ count: sql<number>`count(*)` }).from(trainingSamples);
    return Number(result[0]?.count || 0);
  }
  
  async getSampleCounts(): Promise<{ total: number; ddos: number; normal: number }> {
    const result = await db.select({
      total: sql<number>`count(*)`,
      ddos: sql<number>`count(*) filter (where label = 1)`,
      normal: sql<number>`count(*) filter (where label = 0)`,
    }).from(trainingSamples);
    
    return {
      total: Number(result[0]?.total || 0),
      ddos: Number(result[0]?.ddos || 0),
      normal: Number(result[0]?.normal || 0),
    };
  }
  
  async recordModelPerformance(
    modelType: string,
    accuracy: number,
    precision: number,
    recall: number,
    f1Score: number
  ): Promise<void> {
    const counts = await this.getSampleCounts();
    
    await db.insert(modelPerformance).values({
      modelType,
      accuracy,
      precision,
      recall,
      f1Score,
      trainingSamplesCount: counts.total,
    });
  }
  
  async getModelHistory(modelType: string): Promise<typeof modelPerformance.$inferSelect[]> {
    return db.select()
      .from(modelPerformance)
      .where(eq(modelPerformance.modelType, modelType))
      .orderBy(desc(modelPerformance.createdAt))
      .limit(10);
  }
  
  async updateLearnedPatterns(ddosSamples: InsertTrainingSample[]): Promise<number> {
    if (ddosSamples.length === 0) return 0;
    
    const attackTypeGroups = new Map<string, InsertTrainingSample[]>();
    
    for (const sample of ddosSamples) {
      const attackType = sample.attackType || "unknown";
      if (!attackTypeGroups.has(attackType)) {
        attackTypeGroups.set(attackType, []);
      }
      attackTypeGroups.get(attackType)!.push(sample);
    }
    
    let patternsUpdated = 0;
    
    for (const [attackType, samples] of Array.from(attackTypeGroups.entries())) {
      if (samples.length < 3) continue;
      
      const featureNames = samples[0].featureNames as string[];
      const thresholds: Record<string, { min: number; max: number; mean: number }> = {};
      
      for (let i = 0; i < featureNames.length; i++) {
        const values = samples.map((s: InsertTrainingSample) => (s.features as number[])[i]);
        const min = Math.min(...values);
        const max = Math.max(...values);
        const mean = values.reduce((a: number, b: number) => a + b, 0) / values.length;
        
        thresholds[featureNames[i]] = { min, max, mean };
      }
      
      const existingPattern = await db.select()
        .from(learnedPatterns)
        .where(eq(learnedPatterns.attackType, attackType))
        .limit(1);
      
      if (existingPattern.length > 0) {
        await db.update(learnedPatterns)
          .set({
            featureThresholds: thresholds,
            confidence: Math.min(0.5 + (samples.length * 0.01), 0.99),
            sampleCount: sql`${learnedPatterns.sampleCount} + ${samples.length}`,
            updatedAt: new Date(),
          })
          .where(eq(learnedPatterns.id, existingPattern[0].id));
      } else {
        await db.insert(learnedPatterns).values({
          patternName: `${attackType} Pattern`,
          attackType,
          featureThresholds: thresholds,
          confidence: Math.min(0.5 + (samples.length * 0.01), 0.95),
          sampleCount: samples.length,
          isActive: true,
        });
      }
      
      patternsUpdated++;
    }
    
    return patternsUpdated;
  }
  
  async getLearnedPatterns(): Promise<LearnedPattern[]> {
    return db.select()
      .from(learnedPatterns)
      .where(eq(learnedPatterns.isActive, true))
      .orderBy(desc(learnedPatterns.confidence));
  }
  
  async getLearningStats(): Promise<LearningStats> {
    const counts = await this.getSampleCounts();
    
    const patternsCount = await db.select({ count: sql<number>`count(*)` })
      .from(learnedPatterns)
      .where(eq(learnedPatterns.isActive, true));
    
    const sessions = await db.select()
      .from(learningSessions)
      .orderBy(desc(learningSessions.createdAt))
      .limit(1);
    
    const sessionsCountResult = await db.select({ count: sql<number>`count(*)` })
      .from(learningSessions);
    
    const latestPerformance = await db.select()
      .from(modelPerformance)
      .orderBy(desc(modelPerformance.createdAt))
      .limit(6);
    
    const modelImprovements = await this.calculateModelImprovements();
    
    return {
      totalSamples: counts.total,
      ddosSamples: counts.ddos,
      normalSamples: counts.normal,
      learnedPatterns: Number(patternsCount[0]?.count || 0),
      sessionsCount: Number(sessionsCountResult[0]?.count || 0),
      lastLearningDate: sessions[0]?.createdAt?.toISOString() || null,
      modelImprovements,
    };
  }
  
  async calculateModelImprovements(): Promise<{ modelType: string; improvement: number; currentAccuracy: number }[]> {
    const modelTypes = ["decision_tree", "random_forest", "knn", "naive_bayes", "logistic_regression", "lucid_cnn"];
    const improvements: { modelType: string; improvement: number; currentAccuracy: number }[] = [];
    
    for (const modelType of modelTypes) {
      const history = await db.select()
        .from(modelPerformance)
        .where(eq(modelPerformance.modelType, modelType))
        .orderBy(desc(modelPerformance.createdAt))
        .limit(2);
      
      if (history.length >= 2) {
        const current = history[0].accuracy;
        const previous = history[1].accuracy;
        improvements.push({
          modelType,
          improvement: ((current - previous) / previous) * 100,
          currentAccuracy: current,
        });
      } else if (history.length === 1) {
        improvements.push({
          modelType,
          improvement: 0,
          currentAccuracy: history[0].accuracy,
        });
      }
    }
    
    return improvements;
  }
  
  async matchLearnedPatterns(features: number[], featureNames: string[]): Promise<{
    matchedPattern: LearnedPattern | null;
    confidence: number;
    matchDetails: { feature: string; value: number; inRange: boolean }[];
  }> {
    const patterns = await this.getLearnedPatterns();
    
    let bestMatch: LearnedPattern | null = null;
    let bestConfidence = 0;
    let bestMatchDetails: { feature: string; value: number; inRange: boolean }[] = [];
    
    for (const pattern of patterns) {
      const thresholds = pattern.featureThresholds as Record<string, { min: number; max: number; mean: number }>;
      const matchDetails: { feature: string; value: number; inRange: boolean }[] = [];
      let matchCount = 0;
      let totalChecks = 0;
      
      for (let i = 0; i < featureNames.length; i++) {
        const featureName = featureNames[i];
        const value = features[i];
        
        if (thresholds[featureName]) {
          totalChecks++;
          const { min, max } = thresholds[featureName];
          const margin = (max - min) * 0.2;
          const inRange = value >= min - margin && value <= max + margin;
          
          if (inRange) matchCount++;
          matchDetails.push({ feature: featureName, value, inRange });
        }
      }
      
      if (totalChecks > 0) {
        const matchRatio = matchCount / totalChecks;
        const confidence = matchRatio * pattern.confidence;
        
        if (confidence > bestConfidence) {
          bestConfidence = confidence;
          bestMatch = pattern;
          bestMatchDetails = matchDetails;
        }
      }
    }
    
    return {
      matchedPattern: bestMatch,
      confidence: bestConfidence,
      matchDetails: bestMatchDetails,
    };
  }
  
  async clearAllData(): Promise<void> {
    await db.delete(trainingSamples);
    await db.delete(modelPerformance);
    await db.delete(learningSessions);
    await db.delete(learnedPatterns);
  }
}

export const learningService = new LearningService();
