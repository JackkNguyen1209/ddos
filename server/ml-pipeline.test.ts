import { describe, it, expect } from "vitest";
import {
  setGlobalSeed,
  getGlobalSeed,
  makeSplitIndices,
  makeKFolds,
  kFoldCrossValidation,
  runAnomalyDetection,
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
  });
});
