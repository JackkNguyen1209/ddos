import { describe, it, expect } from "vitest";
import {
  setGlobalSeed,
  getGlobalSeed,
  makeSplitIndices,
  makeKFolds,
  kFoldCrossValidation,
  runAnomalyDetection,
  extractFeatures,
  filterLabeledData,
} from "./ml-algorithms";

describe("ML Pipeline Determinism", () => {
  describe("global seed management", () => {
    it("should set and retrieve global seed", () => {
      setGlobalSeed("my-test-seed");
      expect(getGlobalSeed()).toBe("my-test-seed");
    });
  });

  describe("makeSplitIndices", () => {
    it("should produce identical splits with same seed", () => {
      const split1 = makeSplitIndices(100, "split-seed-123");
      const split2 = makeSplitIndices(100, "split-seed-123");
      
      expect(split1.trainIdx).toEqual(split2.trainIdx);
      expect(split1.valIdx).toEqual(split2.valIdx);
      expect(split1.testIdx).toEqual(split2.testIdx);
    });

    it("should produce different splits with different seeds", () => {
      const split1 = makeSplitIndices(100, "seed-a");
      const split2 = makeSplitIndices(100, "seed-b");
      
      expect(split1.trainIdx).not.toEqual(split2.trainIdx);
    });

    it("should have correct proportions (60/20/20)", () => {
      const split = makeSplitIndices(100, "proportion-test");
      
      expect(split.trainIdx.length).toBe(60);
      expect(split.valIdx.length).toBe(20);
      expect(split.testIdx.length).toBe(20);
    });

    it("should partition all indices without overlap", () => {
      const n = 100;
      const split = makeSplitIndices(n, "partition-test");
      const allIndices = [...split.trainIdx, ...split.valIdx, ...split.testIdx].sort((a, b) => a - b);
      const expected = Array.from({ length: n }, (_, i) => i);
      
      expect(allIndices).toEqual(expected);
    });
  });

  describe("makeKFolds", () => {
    it("should produce identical folds with same seed", () => {
      const folds1 = makeKFolds(100, 5, "fold-seed");
      const folds2 = makeKFolds(100, 5, "fold-seed");
      
      expect(folds1).toEqual(folds2);
    });

    it("should produce different folds with different seeds", () => {
      const folds1 = makeKFolds(100, 5, "fold-a");
      const folds2 = makeKFolds(100, 5, "fold-b");
      
      expect(folds1).not.toEqual(folds2);
    });

    it("should create correct number of folds", () => {
      const folds = makeKFolds(100, 5, "k-test");
      expect(folds.length).toBe(5);
    });

    it("should partition all indices", () => {
      const n = 100;
      const folds = makeKFolds(n, 5, "partition-test");
      const allIndices = folds.flat().sort((a, b) => a - b);
      const expected = Array.from({ length: n }, (_, i) => i);
      
      expect(allIndices).toEqual(expected);
    });
  });

  describe("kFoldCrossValidation", () => {
    it("should produce identical results with same seed", () => {
      const features = Array.from({ length: 50 }, (_, i) => 
        [i % 5, (i * 2) % 7, (i * 3) % 11, (i + 1) % 3, i % 2]
      );
      const labels = features.map((_, i) => i % 2);
      
      const result1 = kFoldCrossValidation(
        features, labels, "decision_tree", 3, undefined, "cv-seed"
      );
      const result2 = kFoldCrossValidation(
        features, labels, "decision_tree", 3, undefined, "cv-seed"
      );
      
      expect(result1.foldResults).toEqual(result2.foldResults);
      expect(result1.meanAccuracy).toBe(result2.meanAccuracy);
    });

    it("should produce different folds with different seeds", () => {
      const features = Array.from({ length: 50 }, (_, i) => 
        [i % 5, (i * 2) % 7, (i * 3) % 11, (i + 1) % 3, i % 2]
      );
      const labels = features.map((_, i) => i % 2);
      
      const folds1 = makeKFolds(50, 3, "cv-seed-a");
      const folds2 = makeKFolds(50, 3, "cv-seed-b");
      
      expect(folds1).not.toEqual(folds2);
    });
  });

  describe("runAnomalyDetection", () => {
    it("should produce identical results with same seed", () => {
      const features = Array.from({ length: 30 }, (_, i) => 
        [i % 5, (i * 2) % 7, (i * 3) % 11]
      );
      
      const result1 = runAnomalyDetection(features, 0.5, "anomaly-seed");
      const result2 = runAnomalyDetection(features, 0.5, "anomaly-seed");
      
      expect(result1.scores).toEqual(result2.scores);
      expect(result1.isAnomalous).toEqual(result2.isAnomalous);
    });

    it("should produce different results with different seeds", () => {
      const features = Array.from({ length: 30 }, (_, i) => 
        [i % 5, (i * 2) % 7, (i * 3) % 11]
      );
      
      const result1 = runAnomalyDetection(features, 0.5, "seed-x");
      const result2 = runAnomalyDetection(features, 0.5, "seed-y");
      
      expect(result1.scores).not.toEqual(result2.scores);
    });

    it("should compute LOF score for each point (not random index)", () => {
      const features = Array.from({ length: 50 }, (_, i) => 
        [i % 5, (i * 2) % 7, (i * 3) % 11]
      );
      
      const result1 = runAnomalyDetection(features, 0.5, "lof-test");
      const result2 = runAnomalyDetection(features, 0.5, "lof-test");
      
      expect(result1.scores.length).toBe(features.length);
      expect(result1.scores).toEqual(result2.scores);
    });
  });

  describe("extractFeatures with partial labels", () => {
    it("should handle full labels correctly", () => {
      const data = [
        { feature1: 1, feature2: 2, label: 1 },
        { feature1: 3, feature2: 4, label: 0 },
        { feature1: 5, feature2: 6, label: 1 },
      ];
      
      const result = extractFeatures(data, ["feature1", "feature2"]);
      
      // 3 rows < MIN_LABELED_ROWS(50) but 100% labeled > MIN_LABELED_RATIO(10%), so hasLabel=true
      expect(result.hasLabel).toBe(true);
      expect(result.labeledRowCount).toBe(3);
      expect(result.validLabelMask).toEqual([true, true, true]);
      expect(result.labels).toEqual([1, 0, 1]);
    });

    it("should handle missing labels with null", () => {
      const data = [
        { feature1: 1, feature2: 2, label: 1 },
        { feature1: 3, feature2: 4 },
        { feature1: 5, feature2: 6, label: 0 },
      ];
      
      const result = extractFeatures(data, ["feature1", "feature2"]);
      
      expect(result.labeledRowCount).toBe(2);
      expect(result.validLabelMask).toEqual([true, false, true]);
      expect(result.labels).toEqual([1, null, 0]);
    });

    it("should not force labels to 0 when missing", () => {
      const data = [
        { feature1: 1, feature2: 2 },
        { feature1: 3, feature2: 4 },
      ];
      
      const result = extractFeatures(data, ["feature1", "feature2"]);
      
      expect(result.labeledRowCount).toBe(0);
      expect(result.hasLabel).toBe(false);
      expect(result.labels).toEqual([null, null]);
    });

    it("should detect hasLabel=true when sufficient labels", () => {
      const data = Array.from({ length: 100 }, (_, i) => ({
        feature1: i,
        feature2: i * 2,
        label: i % 2,
      }));
      
      const result = extractFeatures(data, ["feature1", "feature2"]);
      
      expect(result.hasLabel).toBe(true);
      expect(result.labeledRowCount).toBe(100);
    });
  });

  describe("filterLabeledData", () => {
    it("should filter to only labeled rows", () => {
      const trainingData = {
        features: [[1, 2], [3, 4], [5, 6]],
        labels: [1, null, 0] as (number | null)[],
        validLabelMask: [true, false, true],
        hasLabel: false,
        labeledRowCount: 2,
      };
      
      const filtered = filterLabeledData(trainingData);
      
      expect(filtered.features).toEqual([[1, 2], [5, 6]]);
      expect(filtered.labels).toEqual([1, 0]);
    });

    it("should return empty arrays when no labels", () => {
      const trainingData = {
        features: [[1, 2], [3, 4]],
        labels: [null, null] as (number | null)[],
        validLabelMask: [false, false],
        hasLabel: false,
        labeledRowCount: 0,
      };
      
      const filtered = filterLabeledData(trainingData);
      
      expect(filtered.features).toEqual([]);
      expect(filtered.labels).toEqual([]);
    });
  });
});
