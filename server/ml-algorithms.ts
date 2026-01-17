import type { DataRow, MLModelType, AnalysisResult, FeatureImportance, DDoSReason, AttackTypeResult, DDoSAttackType } from "@shared/schema";
import { randomUUID } from "crypto";

interface TrainingData {
  features: number[][];
  labels: number[];
}

function extractFeatures(data: DataRow[], featureColumns: string[]): TrainingData {
  const features: number[][] = [];
  const labels: number[] = [];

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

    const label = row["label"] || row["Label"] || row["class"] || row["Class"] || row["attack"] || row["Attack"];
    if (label !== undefined) {
      if (typeof label === "number") {
        labels.push(label > 0 ? 1 : 0);
      } else {
        const lowerLabel = String(label).toLowerCase();
        labels.push(lowerLabel.includes("ddos") || lowerLabel.includes("attack") || lowerLabel === "1" ? 1 : 0);
      }
    } else {
      labels.push(0);
    }
  }

  return { features, labels };
}

function normalizeFeatures(features: number[][]): number[][] {
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

function splitData(
  features: number[][],
  labels: number[],
  testRatio: number = 0.2
): {
  trainFeatures: number[][];
  trainLabels: number[];
  testFeatures: number[][];
  testLabels: number[];
} {
  const indices = features.map((_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const splitPoint = Math.floor(features.length * (1 - testRatio));

  return {
    trainFeatures: indices.slice(0, splitPoint).map((i) => features[i]),
    trainLabels: indices.slice(0, splitPoint).map((i) => labels[i]),
    testFeatures: indices.slice(splitPoint).map((i) => features[i]),
    testLabels: indices.slice(splitPoint).map((i) => labels[i]),
  };
}

function calculateMetrics(predicted: number[], actual: number[]) {
  let tp = 0, tn = 0, fp = 0, fn = 0;

  for (let i = 0; i < predicted.length; i++) {
    if (predicted[i] === 1 && actual[i] === 1) tp++;
    else if (predicted[i] === 0 && actual[i] === 0) tn++;
    else if (predicted[i] === 1 && actual[i] === 0) fp++;
    else if (predicted[i] === 0 && actual[i] === 1) fn++;
  }

  const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1Score = 2 * (precision * recall) / (precision + recall) || 0;

  return {
    accuracy,
    precision,
    recall,
    f1Score,
    confusionMatrix: {
      truePositive: tp,
      trueNegative: tn,
      falsePositive: fp,
      falseNegative: fn,
    },
  };
}

class DecisionTree {
  private maxDepth: number;
  private tree: any = null;

  constructor(maxDepth: number = 10) {
    this.maxDepth = maxDepth;
  }

  private gini(labels: number[]): number {
    if (labels.length === 0) return 0;
    const counts = labels.reduce((acc, l) => {
      acc[l] = (acc[l] || 0) + 1;
      return acc;
    }, {} as Record<number, number>);
    let gini = 1;
    for (const count of Object.values(counts)) {
      gini -= Math.pow(count / labels.length, 2);
    }
    return gini;
  }

  private bestSplit(features: number[][], labels: number[]): { feature: number; threshold: number } | null {
    let bestGain = 0;
    let bestSplit: { feature: number; threshold: number } | null = null;
    const parentGini = this.gini(labels);

    for (let f = 0; f < features[0].length; f++) {
      const values = Array.from(new Set(features.map((row) => row[f]))).sort((a, b) => a - b);
      for (let i = 0; i < values.length - 1; i++) {
        const threshold = (values[i] + values[i + 1]) / 2;
        const leftIdx = features.map((row, idx) => (row[f] <= threshold ? idx : -1)).filter((i) => i >= 0);
        const rightIdx = features.map((row, idx) => (row[f] > threshold ? idx : -1)).filter((i) => i >= 0);

        if (leftIdx.length === 0 || rightIdx.length === 0) continue;

        const leftLabels = leftIdx.map((i) => labels[i]);
        const rightLabels = rightIdx.map((i) => labels[i]);

        const leftGini = this.gini(leftLabels);
        const rightGini = this.gini(rightLabels);

        const weightedGini =
          (leftLabels.length / labels.length) * leftGini +
          (rightLabels.length / labels.length) * rightGini;

        const gain = parentGini - weightedGini;

        if (gain > bestGain) {
          bestGain = gain;
          bestSplit = { feature: f, threshold };
        }
      }
    }

    return bestSplit;
  }

  private buildTree(features: number[][], labels: number[], depth: number): any {
    if (depth >= this.maxDepth || labels.length < 2) {
      const ones = labels.filter((l) => l === 1).length;
      return { leaf: true, prediction: ones >= labels.length / 2 ? 1 : 0 };
    }

    const uniqueLabels = Array.from(new Set(labels));
    if (uniqueLabels.length === 1) {
      return { leaf: true, prediction: uniqueLabels[0] };
    }

    const split = this.bestSplit(features, labels);
    if (!split) {
      const ones = labels.filter((l) => l === 1).length;
      return { leaf: true, prediction: ones >= labels.length / 2 ? 1 : 0 };
    }

    const leftIdx: number[] = [];
    const rightIdx: number[] = [];
    features.forEach((row, idx) => {
      if (row[split.feature] <= split.threshold) {
        leftIdx.push(idx);
      } else {
        rightIdx.push(idx);
      }
    });

    return {
      leaf: false,
      feature: split.feature,
      threshold: split.threshold,
      left: this.buildTree(
        leftIdx.map((i) => features[i]),
        leftIdx.map((i) => labels[i]),
        depth + 1
      ),
      right: this.buildTree(
        rightIdx.map((i) => features[i]),
        rightIdx.map((i) => labels[i]),
        depth + 1
      ),
    };
  }

  train(features: number[][], labels: number[]): void {
    this.tree = this.buildTree(features, labels, 0);
  }

  private predictOne(features: number[]): number {
    let node = this.tree;
    while (!node.leaf) {
      if (features[node.feature] <= node.threshold) {
        node = node.left;
      } else {
        node = node.right;
      }
    }
    return node.prediction;
  }

  predict(features: number[][]): number[] {
    return features.map((f) => this.predictOne(f));
  }
}

class RandomForest {
  private trees: DecisionTree[] = [];
  private numTrees: number;

  constructor(numTrees: number = 10) {
    this.numTrees = numTrees;
  }

  train(features: number[][], labels: number[]): void {
    this.trees = [];
    for (let t = 0; t < this.numTrees; t++) {
      const sampleSize = Math.floor(features.length * 0.8);
      const indices: number[] = [];
      for (let i = 0; i < sampleSize; i++) {
        indices.push(Math.floor(Math.random() * features.length));
      }
      const sampledFeatures = indices.map((i) => features[i]);
      const sampledLabels = indices.map((i) => labels[i]);

      const tree = new DecisionTree(8);
      tree.train(sampledFeatures, sampledLabels);
      this.trees.push(tree);
    }
  }

  predict(features: number[][]): number[] {
    return features.map((f) => {
      const predictions = this.trees.map((tree) => tree.predict([f])[0]);
      const ones = predictions.filter((p) => p === 1).length;
      return ones > predictions.length / 2 ? 1 : 0;
    });
  }
}

class KNN {
  private k: number;
  private trainFeatures: number[][] = [];
  private trainLabels: number[] = [];

  constructor(k: number = 5) {
    this.k = k;
  }

  train(features: number[][], labels: number[]): void {
    this.trainFeatures = features;
    this.trainLabels = labels;
  }

  private distance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += Math.pow(a[i] - b[i], 2);
    }
    return Math.sqrt(sum);
  }

  predict(features: number[][]): number[] {
    return features.map((f) => {
      const distances = this.trainFeatures.map((tf, idx) => ({
        distance: this.distance(f, tf),
        label: this.trainLabels[idx],
      }));
      distances.sort((a, b) => a.distance - b.distance);
      const kNearest = distances.slice(0, this.k);
      const ones = kNearest.filter((n) => n.label === 1).length;
      return ones > this.k / 2 ? 1 : 0;
    });
  }
}

class NaiveBayes {
  private classPriors: Record<number, number> = {};
  private featureMeans: Record<number, number[]> = {};
  private featureVars: Record<number, number[]> = {};

  train(features: number[][], labels: number[]): void {
    const classes = Array.from(new Set(labels));

    for (const c of classes) {
      const classFeatures = features.filter((_, i) => labels[i] === c);
      this.classPriors[c] = classFeatures.length / features.length;

      const numFeatures = features[0].length;
      this.featureMeans[c] = [];
      this.featureVars[c] = [];

      for (let f = 0; f < numFeatures; f++) {
        const values = classFeatures.map((row) => row[f]);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length + 1e-9;
        this.featureMeans[c].push(mean);
        this.featureVars[c].push(variance);
      }
    }
  }

  private gaussian(x: number, mean: number, variance: number): number {
    return (1 / Math.sqrt(2 * Math.PI * variance)) * Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
  }

  predict(features: number[][]): number[] {
    return features.map((f) => {
      let bestClass = 0;
      let bestProb = -Infinity;

      for (const c of Object.keys(this.classPriors).map(Number)) {
        let prob = Math.log(this.classPriors[c]);
        for (let i = 0; i < f.length; i++) {
          prob += Math.log(this.gaussian(f[i], this.featureMeans[c][i], this.featureVars[c][i]) + 1e-9);
        }
        if (prob > bestProb) {
          bestProb = prob;
          bestClass = c;
        }
      }

      return bestClass;
    });
  }
}

class LogisticRegression {
  private weights: number[] = [];
  private bias: number = 0;
  private learningRate: number;
  private iterations: number;

  constructor(learningRate: number = 0.1, iterations: number = 100) {
    this.learningRate = learningRate;
    this.iterations = iterations;
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }

  train(features: number[][], labels: number[]): void {
    const numFeatures = features[0].length;
    this.weights = new Array(numFeatures).fill(0);
    this.bias = 0;

    for (let iter = 0; iter < this.iterations; iter++) {
      const gradW = new Array(numFeatures).fill(0);
      let gradB = 0;

      for (let i = 0; i < features.length; i++) {
        const z = features[i].reduce((sum, f, j) => sum + f * this.weights[j], 0) + this.bias;
        const pred = this.sigmoid(z);
        const error = pred - labels[i];

        for (let j = 0; j < numFeatures; j++) {
          gradW[j] += error * features[i][j];
        }
        gradB += error;
      }

      for (let j = 0; j < numFeatures; j++) {
        this.weights[j] -= (this.learningRate * gradW[j]) / features.length;
      }
      this.bias -= (this.learningRate * gradB) / features.length;
    }
  }

  predict(features: number[][]): number[] {
    return features.map((f) => {
      const z = f.reduce((sum, val, j) => sum + val * this.weights[j], 0) + this.bias;
      return this.sigmoid(z) >= 0.5 ? 1 : 0;
    });
  }
}

class LUCIDCNN {
  private kernels: number[][][] = [];
  private fcWeights: number[] = [];
  private fcBias: number = 0;
  private numKernels: number;
  private kernelRows: number;
  private learningRate: number;
  private epochs: number;
  
  constructor(numKernels: number = 64, kernelRows: number = 3, learningRate: number = 0.01, epochs: number = 50) {
    this.numKernels = numKernels;
    this.kernelRows = kernelRows;
    this.learningRate = learningRate;
    this.epochs = epochs;
  }
  
  private relu(x: number): number {
    return Math.max(0, x);
  }
  
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }
  
  private initializeKernels(numFeatures: number): void {
    this.kernels = [];
    for (let k = 0; k < this.numKernels; k++) {
      const kernel: number[][] = [];
      for (let i = 0; i < this.kernelRows; i++) {
        const row: number[] = [];
        for (let j = 0; j < numFeatures; j++) {
          row.push((Math.random() - 0.5) * 0.1);
        }
        kernel.push(row);
      }
      this.kernels.push(kernel);
    }
    
    this.fcWeights = new Array(this.numKernels).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    this.fcBias = 0;
  }
  
  private conv2d(input: number[][], kernel: number[][]): number[] {
    const outputSize = input.length - kernel.length + 1;
    const output: number[] = [];
    
    for (let i = 0; i < outputSize; i++) {
      let sum = 0;
      for (let kr = 0; kr < kernel.length; kr++) {
        for (let kc = 0; kc < kernel[0].length; kc++) {
          sum += input[i + kr][kc] * kernel[kr][kc];
        }
      }
      output.push(this.relu(sum));
    }
    
    return output;
  }
  
  private maxPool(input: number[]): number {
    return Math.max(...input, 0);
  }
  
  private forward(features: number[][]): { pooled: number[]; prediction: number } {
    const pooled: number[] = [];
    
    for (const kernel of this.kernels) {
      const convOutput = this.conv2d(features, kernel);
      const maxVal = this.maxPool(convOutput);
      pooled.push(maxVal);
    }
    
    const z = pooled.reduce((sum, val, i) => sum + val * this.fcWeights[i], 0) + this.fcBias;
    const prediction = this.sigmoid(z);
    
    return { pooled, prediction };
  }
  
  private reshapeToMatrix(features: number[], numRows: number): number[][] {
    const numCols = Math.ceil(features.length / numRows);
    const matrix: number[][] = [];
    
    for (let i = 0; i < numRows; i++) {
      const row: number[] = [];
      for (let j = 0; j < numCols; j++) {
        const idx = i * numCols + j;
        row.push(idx < features.length ? features[idx] : 0);
      }
      matrix.push(row);
    }
    
    return matrix;
  }
  
  train(features: number[][], labels: number[]): void {
    const numFeatures = features[0].length;
    const matrixRows = Math.max(this.kernelRows + 2, 5);
    
    this.initializeKernels(Math.ceil(numFeatures / matrixRows));
    
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      for (let i = 0; i < features.length; i++) {
        const matrix = this.reshapeToMatrix(features[i], matrixRows);
        const { pooled, prediction } = this.forward(matrix);
        
        const error = prediction - labels[i];
        
        for (let j = 0; j < this.fcWeights.length; j++) {
          this.fcWeights[j] -= this.learningRate * error * pooled[j];
        }
        this.fcBias -= this.learningRate * error;
      }
    }
  }
  
  predict(features: number[][]): number[] {
    const matrixRows = Math.max(this.kernelRows + 2, 5);
    
    return features.map((f) => {
      const matrix = this.reshapeToMatrix(f, matrixRows);
      const { prediction } = this.forward(matrix);
      return prediction >= 0.5 ? 1 : 0;
    });
  }
  
  getAnomalyScores(features: number[][]): number[] {
    const matrixRows = Math.max(this.kernelRows + 2, 5);
    
    return features.map((f) => {
      const matrix = this.reshapeToMatrix(f, matrixRows);
      const { prediction } = this.forward(matrix);
      return prediction;
    });
  }
}

function classifyAttackTypes(
  data: DataRow[],
  predictions: number[],
  featureColumns: string[]
): AttackTypeResult[] {
  const attackCounts: Record<DDoSAttackType, { count: number; indicators: Set<string>; confidence: number }> = {
    port_scan: { count: 0, indicators: new Set(), confidence: 0 },
    syn_flood: { count: 0, indicators: new Set(), confidence: 0 },
    udp_flood: { count: 0, indicators: new Set(), confidence: 0 },
    icmp_flood: { count: 0, indicators: new Set(), confidence: 0 },
    http_flood: { count: 0, indicators: new Set(), confidence: 0 },
    dns_amplification: { count: 0, indicators: new Set(), confidence: 0 },
    ntp_amplification: { count: 0, indicators: new Set(), confidence: 0 },
    ldap_reflection: { count: 0, indicators: new Set(), confidence: 0 },
    rdp_attack: { count: 0, indicators: new Set(), confidence: 0 },
    ssdp_amplification: { count: 0, indicators: new Set(), confidence: 0 },
    memcached_amplification: { count: 0, indicators: new Set(), confidence: 0 },
    unknown: { count: 0, indicators: new Set(), confidence: 0 },
  };

  const portCol = featureColumns.find(c => c.toLowerCase().includes("dst_port") || c.toLowerCase().includes("dstport") || c.toLowerCase().includes("destination_port"));
  const srcPortCol = featureColumns.find(c => c.toLowerCase().includes("src_port") || c.toLowerCase().includes("srcport") || c.toLowerCase().includes("source_port"));
  const protocolCol = featureColumns.find(c => c.toLowerCase().includes("protocol") || c.toLowerCase() === "proto");
  const bytesCol = featureColumns.find(c => c.toLowerCase().includes("bytes") || c.toLowerCase().includes("length"));
  const packetsCol = featureColumns.find(c => c.toLowerCase().includes("packets") || c.toLowerCase().includes("pkts"));
  const durationCol = featureColumns.find(c => c.toLowerCase().includes("duration") || c.toLowerCase().includes("time"));
  const flagsCol = featureColumns.find(c => c.toLowerCase().includes("flag") || c.toLowerCase().includes("tcp_flags"));

  for (let i = 0; i < data.length; i++) {
    if (predictions[i] !== 1) continue;
    
    const row = data[i];
    let attackType: DDoSAttackType = "unknown";
    let confidence = 0.5;
    const indicators: string[] = [];

    const dstPort = portCol ? Number(row[portCol]) || 0 : 0;
    const srcPort = srcPortCol ? Number(row[srcPortCol]) || 0 : 0;
    const protocol = protocolCol ? String(row[protocolCol]).toLowerCase() : "";
    const bytes = bytesCol ? Number(row[bytesCol]) || 0 : 0;
    const packets = packetsCol ? Number(row[packetsCol]) || 0 : 0;
    const duration = durationCol ? Number(row[durationCol]) || 0 : 0;
    const flags = flagsCol ? String(row[flagsCol]).toUpperCase() : "";

    if (dstPort === 3389) {
      attackType = "rdp_attack";
      confidence = 0.85;
      indicators.push("Cổng RDP 3389");
      if (duration < 1) indicators.push("Kết nối ngắn");
    } else if (dstPort === 389 || dstPort === 636 || dstPort === 3268) {
      attackType = "ldap_reflection";
      confidence = 0.9;
      indicators.push(`Cổng LDAP ${dstPort}`);
      if (protocol.includes("udp")) indicators.push("Giao thức UDP");
    } else if (dstPort === 53) {
      attackType = "dns_amplification";
      confidence = 0.8;
      indicators.push("Cổng DNS 53");
      if (bytes > 512) indicators.push("Response size lớn");
    } else if (dstPort === 123) {
      attackType = "ntp_amplification";
      confidence = 0.85;
      indicators.push("Cổng NTP 123");
    } else if (dstPort === 1900) {
      attackType = "ssdp_amplification";
      confidence = 0.85;
      indicators.push("Cổng SSDP 1900");
    } else if (dstPort === 11211) {
      attackType = "memcached_amplification";
      confidence = 0.9;
      indicators.push("Cổng Memcached 11211");
    } else if (dstPort === 80 || dstPort === 443 || dstPort === 8080) {
      if (flags.includes("S") && !flags.includes("A")) {
        attackType = "syn_flood";
        confidence = 0.85;
        indicators.push("SYN flag without ACK");
      } else {
        attackType = "http_flood";
        confidence = 0.75;
      }
      indicators.push(`Cổng HTTP ${dstPort}`);
    } else if (protocol.includes("udp") || protocol === "17") {
      attackType = "udp_flood";
      confidence = 0.7;
      indicators.push("Giao thức UDP");
      if (packets > 100) indicators.push("Số packet cao");
    } else if (protocol.includes("icmp") || protocol === "1") {
      attackType = "icmp_flood";
      confidence = 0.8;
      indicators.push("Giao thức ICMP");
    } else if (flags.includes("S") && !flags.includes("A")) {
      attackType = "syn_flood";
      confidence = 0.8;
      indicators.push("SYN flag cao");
    }

    if (indicators.length === 0 && attackType === "unknown") {
      indicators.push("Pattern bất thường");
    }

    attackCounts[attackType].count++;
    attackCounts[attackType].confidence = Math.max(attackCounts[attackType].confidence, confidence);
    indicators.forEach(ind => attackCounts[attackType].indicators.add(ind));
  }

  const totalDDoS = predictions.filter(p => p === 1).length;
  
  const results: AttackTypeResult[] = Object.entries(attackCounts)
    .filter(([_, data]) => data.count > 0)
    .map(([type, data]) => ({
      type: type as DDoSAttackType,
      count: data.count,
      percentage: totalDDoS > 0 ? (data.count / totalDDoS) * 100 : 0,
      confidence: data.confidence,
      indicators: Array.from(data.indicators),
    }))
    .sort((a, b) => b.count - a.count);

  return results;
}

export async function analyzeWithModel(
  datasetId: string,
  modelType: MLModelType,
  data: DataRow[],
  featureColumns: string[]
): Promise<AnalysisResult> {
  const startTime = Date.now();

  const { features, labels } = extractFeatures(data, featureColumns);
  const normalizedFeatures = normalizeFeatures(features);
  const { trainFeatures, trainLabels, testFeatures, testLabels } = splitData(normalizedFeatures, labels);

  let model: any;
  let lucidModel: LUCIDCNN | null = null;
  
  switch (modelType) {
    case "decision_tree":
      model = new DecisionTree(10);
      break;
    case "random_forest":
      model = new RandomForest(15);
      break;
    case "knn":
      model = new KNN(5);
      break;
    case "naive_bayes":
      model = new NaiveBayes();
      break;
    case "logistic_regression":
      model = new LogisticRegression(0.1, 200);
      break;
    case "lucid_cnn":
      lucidModel = new LUCIDCNN(32, 3, 0.01, 30);
      model = lucidModel;
      break;
    default:
      throw new Error(`Unknown model type: ${modelType}`);
  }

  model.train(trainFeatures, trainLabels);
  const predictions = model.predict(testFeatures);
  const metrics = calculateMetrics(predictions, testLabels);

  const allPredictions = model.predict(normalizedFeatures);
  const ddosDetected = allPredictions.filter((p: number) => p === 1).length;
  const normalTraffic = allPredictions.filter((p: number) => p === 0).length;

  const trainingTime = (Date.now() - startTime) / 1000;

  const featureImportance = calculateFeatureImportance(normalizedFeatures, allPredictions, featureColumns);
  const ddosReasons = generateDDoSReasons(normalizedFeatures, allPredictions, featureColumns, featureImportance);
  
  const attackTypes = classifyAttackTypes(data, allPredictions, featureColumns);
  
  let lucidAnalysis = undefined;
  if (modelType === "lucid_cnn" && lucidModel) {
    const anomalyScores = lucidModel.getAnomalyScores(normalizedFeatures);
    const avgAnomalyScore = anomalyScores.reduce((a, b) => a + b, 0) / anomalyScores.length;
    
    lucidAnalysis = {
      cnnLayers: 1,
      kernelSize: 3,
      timeWindow: 10,
      flowFeatures: featureColumns.length,
      anomalyScore: avgAnomalyScore,
      confidence: metrics.f1Score,
    };
  }

  return {
    id: randomUUID(),
    datasetId,
    modelType,
    accuracy: metrics.accuracy,
    precision: metrics.precision,
    recall: metrics.recall,
    f1Score: metrics.f1Score,
    confusionMatrix: metrics.confusionMatrix,
    trainingTime,
    ddosDetected,
    normalTraffic,
    analyzedAt: new Date().toISOString(),
    featureImportance,
    ddosReasons,
    attackTypes,
    lucidAnalysis,
  };
}

export function getFeatureColumns(columns: string[]): string[] {
  const labelColumns = ["label", "class", "attack", "target", "Label", "Class", "Attack", "Target"];
  const idColumns = ["id", "ID", "Id", "index", "Index"];
  
  return columns.filter(
    (col) =>
      !labelColumns.includes(col) &&
      !idColumns.includes(col)
  );
}

const FEATURE_DESCRIPTIONS: Record<string, string> = {
  bytes: "Lượng dữ liệu (bytes) cao bất thường có thể chỉ ra tấn công volumetric",
  packets: "Số lượng gói tin lớn trong thời gian ngắn là dấu hiệu của DDoS flood",
  duration: "Thời lượng kết nối ngắn bất thường là đặc điểm của SYN flood",
  src_ip: "Nhiều địa chỉ IP nguồn khác nhau có thể là botnet phân tán",
  dst_ip: "Tập trung vào một IP đích là mục tiêu của cuộc tấn công",
  protocol: "Giao thức bất thường (UDP flood, ICMP flood) thường dùng trong DDoS",
  src_port: "Các cổng nguồn ngẫu nhiên là đặc điểm của IP spoofing",
  dst_port: "Tập trung vào cổng cụ thể (80, 443, 53) là mục tiêu tấn công",
  flags: "Cờ TCP bất thường (SYN flood, ACK flood) là dấu hiệu tấn công",
  flow_duration: "Luồng dữ liệu ngắn lặp lại nhiều lần là đặc điểm botnet",
  total_fwd_packets: "Số gói chuyển tiếp cao bất thường chỉ ra traffic flood",
  total_bwd_packets: "Số gói phản hồi thấp bất thường chỉ ra tấn công một chiều",
  flow_bytes_per_s: "Tốc độ bytes/giây cao là dấu hiệu tấn công bandwidth",
  flow_packets_per_s: "Tốc độ gói/giây cao là dấu hiệu flood attack",
  avg_packet_size: "Kích thước gói tin nhỏ đồng đều là đặc điểm SYN flood",
  default: "Đặc trưng mạng bất thường đóng góp vào phát hiện DDoS",
};

function getFeatureDescription(feature: string): string {
  const lowerFeature = feature.toLowerCase();
  for (const [key, desc] of Object.entries(FEATURE_DESCRIPTIONS)) {
    if (lowerFeature.includes(key)) {
      return desc;
    }
  }
  return FEATURE_DESCRIPTIONS.default;
}

function calculateFeatureImportance(
  features: number[][],
  labels: number[],
  featureColumns: string[]
): FeatureImportance[] {
  const importances: FeatureImportance[] = [];
  
  for (let i = 0; i < featureColumns.length; i++) {
    const featureValues = features.map((f) => f[i]);
    
    const ddosValues = featureValues.filter((_, idx) => labels[idx] === 1);
    const normalValues = featureValues.filter((_, idx) => labels[idx] === 0);
    
    const ddosMean = ddosValues.length > 0 ? ddosValues.reduce((a, b) => a + b, 0) / ddosValues.length : 0;
    const normalMean = normalValues.length > 0 ? normalValues.reduce((a, b) => a + b, 0) / normalValues.length : 0;
    
    const ddosVar = ddosValues.length > 0 
      ? ddosValues.reduce((sum, v) => sum + Math.pow(v - ddosMean, 2), 0) / ddosValues.length 
      : 1;
    const normalVar = normalValues.length > 0 
      ? normalValues.reduce((sum, v) => sum + Math.pow(v - normalMean, 2), 0) / normalValues.length 
      : 1;
    
    const pooledStd = Math.sqrt((ddosVar + normalVar) / 2) || 1;
    const importance = Math.abs(ddosMean - normalMean) / pooledStd;
    
    importances.push({
      feature: featureColumns[i],
      importance: Math.min(importance, 1),
      description: getFeatureDescription(featureColumns[i]),
    });
  }
  
  return importances.sort((a, b) => b.importance - a.importance).slice(0, 10);
}

function generateDDoSReasons(
  features: number[][],
  labels: number[],
  featureColumns: string[],
  featureImportance: FeatureImportance[]
): DDoSReason[] {
  const reasons: DDoSReason[] = [];
  
  const ddosIndices = labels.map((l, i) => l === 1 ? i : -1).filter((i) => i !== -1);
  const normalIndices = labels.map((l, i) => l === 0 ? i : -1).filter((i) => i !== -1);
  
  const topFeatures = featureImportance.slice(0, 5);
  
  for (const fi of topFeatures) {
    const featureIdx = featureColumns.indexOf(fi.feature);
    if (featureIdx === -1) continue;
    
    const ddosValues = ddosIndices.map((i) => features[i][featureIdx]);
    const normalValues = normalIndices.map((i) => features[i][featureIdx]);
    
    const ddosMean = ddosValues.length > 0 ? ddosValues.reduce((a, b) => a + b, 0) / ddosValues.length : 0;
    const normalMean = normalValues.length > 0 ? normalValues.reduce((a, b) => a + b, 0) / normalValues.length : 0;
    
    const threshold = (ddosMean + normalMean) / 2;
    
    reasons.push({
      feature: fi.feature,
      value: ddosMean,
      threshold: threshold,
      contribution: fi.importance,
      description: fi.description,
    });
  }
  
  return reasons;
}
