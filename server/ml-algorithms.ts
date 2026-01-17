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
  private numCols: number = 0;
  
  constructor(numKernels: number = 32, kernelRows: number = 3, learningRate: number = 0.01, epochs: number = 30) {
    this.numKernels = numKernels;
    this.kernelRows = kernelRows;
    this.learningRate = learningRate;
    this.epochs = epochs;
  }
  
  private relu(x: number): number {
    return Math.max(0, x);
  }
  
  private reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }
  
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }
  
  private initializeKernels(numFeatures: number): void {
    this.numCols = numFeatures;
    this.kernels = [];
    const scale = Math.sqrt(2.0 / (this.kernelRows * numFeatures));
    
    for (let k = 0; k < this.numKernels; k++) {
      const kernel: number[][] = [];
      for (let i = 0; i < this.kernelRows; i++) {
        const row: number[] = [];
        for (let j = 0; j < numFeatures; j++) {
          row.push((Math.random() - 0.5) * scale);
        }
        kernel.push(row);
      }
      this.kernels.push(kernel);
    }
    
    this.fcWeights = new Array(this.numKernels).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    this.fcBias = 0;
  }
  
  private conv2d(input: number[][], kernel: number[][]): { output: number[]; preActivation: number[] } {
    const outputSize = Math.max(1, input.length - kernel.length + 1);
    const output: number[] = [];
    const preActivation: number[] = [];
    
    for (let i = 0; i < outputSize; i++) {
      let sum = 0;
      for (let kr = 0; kr < kernel.length && (i + kr) < input.length; kr++) {
        for (let kc = 0; kc < kernel[0].length && kc < input[0].length; kc++) {
          sum += input[i + kr][kc] * kernel[kr][kc];
        }
      }
      preActivation.push(sum);
      output.push(this.relu(sum));
    }
    
    return { output, preActivation };
  }
  
  private maxPoolWithIndex(input: number[]): { value: number; index: number } {
    if (input.length === 0) return { value: 0, index: 0 };
    let maxVal = input[0];
    let maxIdx = 0;
    for (let i = 1; i < input.length; i++) {
      if (input[i] > maxVal) {
        maxVal = input[i];
        maxIdx = i;
      }
    }
    return { value: maxVal, index: maxIdx };
  }
  
  private forward(matrix: number[][]): { 
    pooled: number[]; 
    poolIndices: number[]; 
    convOutputs: number[][]; 
    preActivations: number[][]; 
    prediction: number;
    z: number;
  } {
    const pooled: number[] = [];
    const poolIndices: number[] = [];
    const convOutputs: number[][] = [];
    const preActivations: number[][] = [];
    
    for (const kernel of this.kernels) {
      const { output, preActivation } = this.conv2d(matrix, kernel);
      convOutputs.push(output);
      preActivations.push(preActivation);
      const { value, index } = this.maxPoolWithIndex(output);
      pooled.push(value);
      poolIndices.push(index);
    }
    
    const z = pooled.reduce((sum, val, i) => sum + val * this.fcWeights[i], 0) + this.fcBias;
    const prediction = this.sigmoid(z);
    
    return { pooled, poolIndices, convOutputs, preActivations, prediction, z };
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
    if (!features || features.length === 0 || !features[0]) {
      this.initializeKernels(10);
      return;
    }
    const numFeatures = features[0].length;
    const matrixRows = Math.max(this.kernelRows + 2, 5);
    const numCols = Math.ceil(numFeatures / matrixRows);
    
    this.initializeKernels(numCols);
    
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let epochLoss = 0;
      
      for (let i = 0; i < features.length; i++) {
        const matrix = this.reshapeToMatrix(features[i], matrixRows);
        const { pooled, poolIndices, preActivations, prediction } = this.forward(matrix);
        
        const error = prediction - labels[i];
        epochLoss += error * error;
        
        for (let k = 0; k < this.numKernels; k++) {
          const cachedFCWeight = this.fcWeights[k];
          const gradFC = error * pooled[k];
          
          const poolIdx = poolIndices[k];
          if (poolIdx < preActivations[k].length) {
            const dRelu = this.reluDerivative(preActivations[k][poolIdx]);
            const gradKernel = error * cachedFCWeight * dRelu;
            
            for (let kr = 0; kr < this.kernelRows && (poolIdx + kr) < matrix.length; kr++) {
              for (let kc = 0; kc < this.kernels[k][kr].length && kc < matrix[0].length; kc++) {
                this.kernels[k][kr][kc] -= this.learningRate * gradKernel * matrix[poolIdx + kr][kc] * 0.1;
              }
            }
          }
          
          this.fcWeights[k] -= this.learningRate * gradFC;
        }
        
        this.fcBias -= this.learningRate * error;
      }
      
      if (epoch > 5 && epochLoss / features.length < 0.01) break;
    }
  }
  
  predict(features: number[][]): number[] {
    if (!features || features.length === 0) {
      return [];
    }
    const matrixRows = Math.max(this.kernelRows + 2, 5);
    
    return features.map((f) => {
      const matrix = this.reshapeToMatrix(f, matrixRows);
      const { prediction } = this.forward(matrix);
      return prediction >= 0.5 ? 1 : 0;
    });
  }
  
  getAnomalyScores(features: number[][]): number[] {
    if (!features || features.length === 0) {
      return [];
    }
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
    service_scan: { count: 0, indicators: new Set(), confidence: 0 },
    ssh_bruteforce: { count: 0, indicators: new Set(), confidence: 0 },
    ftp_bruteforce: { count: 0, indicators: new Set(), confidence: 0 },
    telnet_bruteforce: { count: 0, indicators: new Set(), confidence: 0 },
    syn_flood: { count: 0, indicators: new Set(), confidence: 0 },
    udp_flood: { count: 0, indicators: new Set(), confidence: 0 },
    icmp_flood: { count: 0, indicators: new Set(), confidence: 0 },
    http_flood: { count: 0, indicators: new Set(), confidence: 0 },
    slowloris: { count: 0, indicators: new Set(), confidence: 0 },
    dns_amplification: { count: 0, indicators: new Set(), confidence: 0 },
    ntp_amplification: { count: 0, indicators: new Set(), confidence: 0 },
    ldap_reflection: { count: 0, indicators: new Set(), confidence: 0 },
    rdp_attack: { count: 0, indicators: new Set(), confidence: 0 },
    ssdp_amplification: { count: 0, indicators: new Set(), confidence: 0 },
    memcached_amplification: { count: 0, indicators: new Set(), confidence: 0 },
    unknown: { count: 0, indicators: new Set(), confidence: 0 },
  };

  const portCol = featureColumns.find(c => c.toLowerCase().includes("dst_port") || c.toLowerCase().includes("dstport") || c.toLowerCase().includes("destination_port") || c.toLowerCase().includes("port"));
  const srcPortCol = featureColumns.find(c => c.toLowerCase().includes("src_port") || c.toLowerCase().includes("srcport") || c.toLowerCase().includes("source_port"));
  const protocolCol = featureColumns.find(c => c.toLowerCase().includes("protocol") || c.toLowerCase() === "proto");
  const bytesCol = featureColumns.find(c => c.toLowerCase().includes("bytes") || c.toLowerCase().includes("length") || c.toLowerCase().includes("size"));
  const packetsCol = featureColumns.find(c => c.toLowerCase().includes("packets") || c.toLowerCase().includes("pkts") || c.toLowerCase().includes("count"));
  const durationCol = featureColumns.find(c => c.toLowerCase().includes("duration") || c.toLowerCase().includes("time"));
  const flagsCol = featureColumns.find(c => c.toLowerCase().includes("flag") || c.toLowerCase().includes("tcp_flags"));
  const srcIpCol = featureColumns.find(c => c.toLowerCase().includes("src_ip") || c.toLowerCase().includes("srcip") || c.toLowerCase().includes("source_ip") || c.toLowerCase().includes("src_addr"));
  const dstIpCol = featureColumns.find(c => c.toLowerCase().includes("dst_ip") || c.toLowerCase().includes("dstip") || c.toLowerCase().includes("dest_ip") || c.toLowerCase().includes("dst_addr"));

  const srcIpToDstPorts: Map<string, Set<number>> = new Map();
  const srcIpToDstIps: Map<string, Set<string>> = new Map();
  const srcIpDurations: Map<string, number[]> = new Map();
  
  for (let i = 0; i < data.length; i++) {
    if (predictions[i] !== 1) continue;
    
    const row = data[i];
    const srcIp = srcIpCol ? String(row[srcIpCol] || "unknown") : "unknown";
    const dstPort = portCol ? Number(row[portCol]) || 0 : 0;
    const dstIp = dstIpCol ? String(row[dstIpCol] || "unknown") : "unknown";
    const duration = durationCol ? Number(row[durationCol]) || 0 : 0;
    
    if (!srcIpToDstPorts.has(srcIp)) srcIpToDstPorts.set(srcIp, new Set());
    if (dstPort > 0) srcIpToDstPorts.get(srcIp)!.add(dstPort);
    
    if (!srcIpToDstIps.has(srcIp)) srcIpToDstIps.set(srcIp, new Set());
    srcIpToDstIps.get(srcIp)!.add(dstIp);
    
    if (!srcIpDurations.has(srcIp)) srcIpDurations.set(srcIp, []);
    srcIpDurations.get(srcIp)!.push(duration);
  }
  
  const portScanners = new Set<string>();
  Array.from(srcIpToDstPorts.entries()).forEach(([srcIp, ports]) => {
    if (ports.size >= 10) {
      portScanners.add(srcIp);
    }
  });
  
  const ipBasedAttackers = new Set<string>();
  Array.from(srcIpToDstIps.entries()).forEach(([srcIp, dstIps]) => {
    if (dstIps.size >= 5) {
      ipBasedAttackers.add(srcIp);
    }
  });

  for (let i = 0; i < data.length; i++) {
    if (predictions[i] !== 1) continue;
    
    const row = data[i];
    let attackType: DDoSAttackType = "unknown";
    let confidence = 0.5;
    const indicators: string[] = [];

    const dstPort = portCol ? Number(row[portCol]) || 0 : 0;
    const protocol = protocolCol ? String(row[protocolCol]).toLowerCase() : "";
    const bytes = bytesCol ? Number(row[bytesCol]) || 0 : 0;
    const packets = packetsCol ? Number(row[packetsCol]) || 0 : 0;
    const duration = durationCol ? Number(row[durationCol]) || 0 : 0;
    const flags = flagsCol ? String(row[flagsCol]).toUpperCase() : "";
    const srcIp = srcIpCol ? String(row[srcIpCol] || "unknown") : "unknown";

    if (portScanners.has(srcIp)) {
      const uniquePorts = srcIpToDstPorts.get(srcIp)?.size || 0;
      attackType = "port_scan";
      confidence = Math.min(0.95, 0.6 + (uniquePorts / 100));
      indicators.push(`${uniquePorts} cổng đích khác nhau`);
      if (duration < 1) indicators.push("Kết nối ngắn");
      indicators.push("Quét nhiều cổng từ 1 IP");
    } else if (dstPort === 22) {
      attackType = "ssh_bruteforce";
      confidence = 0.85;
      indicators.push("Cổng SSH 22");
      indicators.push("Dò mật khẩu SSH");
      if (duration < 1) indicators.push("Nhiều connection attempt");
    } else if (dstPort === 21) {
      attackType = "ftp_bruteforce";
      confidence = 0.8;
      indicators.push("Cổng FTP 21");
      indicators.push("Dò mật khẩu FTP");
    } else if (dstPort === 23) {
      attackType = "telnet_bruteforce";
      confidence = 0.9;
      indicators.push("Cổng Telnet 23");
      indicators.push("Dò mật khẩu Telnet");
      indicators.push("Nguy cơ IoT botnet cao");
    } else if (dstPort === 3389) {
      attackType = "rdp_attack";
      confidence = 0.85;
      indicators.push("Cổng RDP 3389");
      indicators.push("Tấn công Remote Desktop");
      if (duration < 1) indicators.push("Kết nối ngắn");
      if (ipBasedAttackers.has(srcIp)) indicators.push("Tấn công từ nhiều IP");
    } else if (dstPort === 389 || dstPort === 636 || dstPort === 3268) {
      attackType = "ldap_reflection";
      confidence = 0.9;
      indicators.push(`Cổng LDAP ${dstPort}`);
      if (protocol.includes("udp") || protocol === "17") indicators.push("Giao thức UDP");
      if (bytes > 1000) indicators.push("Phản hồi lớn (amplification)");
    } else if (dstPort === 53) {
      attackType = "dns_amplification";
      confidence = 0.8;
      indicators.push("Cổng DNS 53");
      if (bytes > 512) indicators.push("Response size lớn");
      if (protocol.includes("udp") || protocol === "17") indicators.push("Giao thức UDP");
    } else if (dstPort === 123) {
      attackType = "ntp_amplification";
      confidence = 0.85;
      indicators.push("Cổng NTP 123");
      if (bytes > 400) indicators.push("Khuếch đại cao");
    } else if (dstPort === 1900) {
      attackType = "ssdp_amplification";
      confidence = 0.85;
      indicators.push("Cổng SSDP 1900");
    } else if (dstPort === 11211) {
      attackType = "memcached_amplification";
      confidence = 0.9;
      indicators.push("Cổng Memcached 11211");
      indicators.push("Khuếch đại cực cao (51,000x)");
    } else if (dstPort === 80 || dstPort === 443 || dstPort === 8080) {
      if (flags.includes("S") && !flags.includes("A")) {
        attackType = "syn_flood";
        confidence = 0.85;
        indicators.push("SYN flag without ACK");
      } else {
        attackType = "http_flood";
        confidence = 0.75;
        if (packets > 100) indicators.push("Số request cao");
      }
      indicators.push(`Cổng HTTP ${dstPort}`);
    } else if (protocol.includes("udp") || protocol === "17") {
      attackType = "udp_flood";
      confidence = 0.7;
      indicators.push("Giao thức UDP");
      if (packets > 100) indicators.push("Số packet cao");
      if (bytes > 1000) indicators.push("Lượng data lớn");
    } else if (protocol.includes("icmp") || protocol === "1") {
      attackType = "icmp_flood";
      confidence = 0.8;
      indicators.push("Giao thức ICMP");
      if (packets > 50) indicators.push("Ping flood");
    } else if (flags.includes("S") && !flags.includes("A")) {
      attackType = "syn_flood";
      confidence = 0.8;
      indicators.push("SYN flag cao");
    }

    if (indicators.length === 0 && attackType === "unknown") {
      indicators.push("Pattern bất thường");
      if (packets > 100) indicators.push("Traffic volume cao");
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

// ============== SMART FEATURE DETECTION & MAPPING ==============

// Network traffic feature patterns for auto-detection
const FEATURE_PATTERNS = {
  // Packet features
  fwd_packets: ["fwd_pkts", "fwd_packets", "fwdpackets", "forward_packets", "tot_fwd_pkts", "total_fwd_packets"],
  bwd_packets: ["bwd_pkts", "bwd_packets", "bwdpackets", "backward_packets", "tot_bwd_pkts", "total_bwd_packets"],
  fwd_packet_length: ["fwd_pkt_len", "fwd_packet_len", "fwdpktlen", "fwd_seg_size", "fwd_packet_length", "fwd_pkt_length_total", "fwd_packets_length_total"],
  bwd_packet_length: ["bwd_pkt_len", "bwd_packet_len", "bwdpktlen", "bwd_seg_size", "bwd_packet_length", "bwd_pkt_length_total"],
  packet_length_total: ["pkt_len_total", "totlen_fwd_pkts", "totlen_bwd_pkts", "flow_bytes", "total_length_of_fwd_packets"],
  
  // Flow features
  flow_duration: ["flow_duration", "duration", "flow_time", "flowduration"],
  flow_bytes_per_sec: ["flow_byts_s", "flow_bytes_s", "flow_bytes_per_sec", "bytes_per_second", "flowbytespersec"],
  flow_packets_per_sec: ["flow_pkts_s", "flow_packets_s", "flow_packets_per_sec", "packets_per_second", "flowpktspersec"],
  
  // IAT (Inter-Arrival Time) features
  flow_iat_mean: ["flow_iat_mean", "flowiatmean", "iat_mean"],
  flow_iat_std: ["flow_iat_std", "flowiatstd", "iat_std"],
  flow_iat_max: ["flow_iat_max", "flowiatmax", "iat_max"],
  flow_iat_min: ["flow_iat_min", "flowiatmin", "iat_min"],
  fwd_iat_mean: ["fwd_iat_mean", "fwdiatmean"],
  bwd_iat_mean: ["bwd_iat_mean", "bwdiatmean"],
  
  // Header features
  fwd_header_length: ["fwd_header_length", "fwd_header_len", "fwdheaderlen", "fwd_seg_size_min"],
  bwd_header_length: ["bwd_header_length", "bwd_header_len", "bwdheaderlen"],
  
  // Flag features
  syn_flag: ["syn_flag", "syn_flag_cnt", "synflag", "syn_count"],
  ack_flag: ["ack_flag", "ack_flag_cnt", "ackflag", "ack_count"],
  fin_flag: ["fin_flag", "fin_flag_cnt", "finflag", "fin_count"],
  rst_flag: ["rst_flag", "rst_flag_cnt", "rstflag", "rst_count"],
  psh_flag: ["psh_flag", "psh_flag_cnt", "pshflag", "push_count"],
  urg_flag: ["urg_flag", "urg_flag_cnt", "urgflag", "urg_count"],
  
  // Network features
  src_port: ["src_port", "srcport", "source_port", "sport"],
  dst_port: ["dst_port", "dstport", "dest_port", "destination_port", "dport"],
  src_ip: ["src_ip", "srcip", "source_ip", "src_addr"],
  dst_ip: ["dst_ip", "dstip", "dest_ip", "dst_addr", "destination_ip"],
  protocol: ["protocol", "proto", "ip_protocol"],
  
  // Statistics
  pkt_size_avg: ["pkt_size_avg", "packet_size_avg", "avg_pkt_size", "packet_length_mean"],
  pkt_size_std: ["pkt_size_std", "packet_size_std", "std_pkt_size", "packet_length_std"],
  init_win_bytes_fwd: ["init_win_bytes_forward", "init_win_bytes_fwd", "init_fwd_win_byts"],
  init_win_bytes_bwd: ["init_win_bytes_backward", "init_win_bytes_bwd", "init_bwd_win_byts"],
  
  // Active/Idle features
  active_mean: ["active_mean", "activemean", "act_data_pkt_fwd"],
  idle_mean: ["idle_mean", "idlemean"],
  subflow_fwd_pkts: ["subflow_fwd_packets", "subflow_fwd_pkts"],
  subflow_bwd_pkts: ["subflow_bwd_packets", "subflow_bwd_pkts"],
};

// Map column name to feature category
export function mapColumnToFeature(columnName: string): string | null {
  const lowerCol = columnName.toLowerCase().replace(/[_\s-]/g, "");
  
  for (const [featureType, patterns] of Object.entries(FEATURE_PATTERNS)) {
    for (const pattern of patterns) {
      const lowerPattern = pattern.toLowerCase().replace(/[_\s-]/g, "");
      if (lowerCol.includes(lowerPattern) || lowerPattern.includes(lowerCol)) {
        return featureType;
      }
    }
  }
  return null;
}

// Analyze data row to determine if it's an attack based on actual feature values
export function analyzeRowForAttack(row: DataRow, featureMapping: Map<string, string>): {
  isAttack: boolean;
  attackType: DDoSAttackType;
  confidence: number;
  reasons: string[];
} {
  const reasons: string[] = [];
  let attackScore = 0;
  let attackType: DDoSAttackType = "unknown";
  
  // Get actual values from mapped features
  const getValue = (featureType: string): number => {
    const entries = Array.from(featureMapping.entries());
    for (const [col, mappedType] of entries) {
      if (mappedType === featureType) {
        const val = row[col];
        return typeof val === "number" ? val : parseFloat(String(val)) || 0;
      }
    }
    return 0;
  };
  
  const fwdPackets = getValue("fwd_packets");
  const bwdPackets = getValue("bwd_packets");
  const fwdPacketLength = getValue("fwd_packet_length");
  const bwdPacketLength = getValue("bwd_packet_length");
  const flowDuration = getValue("flow_duration");
  const flowBytesPerSec = getValue("flow_bytes_per_sec");
  const flowPacketsPerSec = getValue("flow_packets_per_sec");
  const flowIatMean = getValue("flow_iat_mean");
  const synFlag = getValue("syn_flag");
  const ackFlag = getValue("ack_flag");
  const finFlag = getValue("fin_flag");
  const rstFlag = getValue("rst_flag");
  const dstPort = getValue("dst_port");
  const srcPort = getValue("src_port");
  const protocol = getValue("protocol");
  const pktSizeAvg = getValue("pkt_size_avg");
  const initWinFwd = getValue("init_win_bytes_fwd");
  
  // High packet rate detection (potential volumetric attack)
  if (flowPacketsPerSec > 1000) {
    attackScore += 3;
    reasons.push(`Tốc độ packet rất cao: ${flowPacketsPerSec.toFixed(0)} pkts/s`);
  } else if (flowPacketsPerSec > 100) {
    attackScore += 1;
    reasons.push(`Tốc độ packet cao: ${flowPacketsPerSec.toFixed(0)} pkts/s`);
  }
  
  // High bytes per second (bandwidth exhaustion)
  if (flowBytesPerSec > 100000000) { // > 100MB/s
    attackScore += 3;
    reasons.push(`Bandwidth rất cao: ${(flowBytesPerSec / 1000000).toFixed(1)} MB/s`);
  } else if (flowBytesPerSec > 10000000) { // > 10MB/s
    attackScore += 2;
    reasons.push(`Bandwidth cao: ${(flowBytesPerSec / 1000000).toFixed(1)} MB/s`);
  }
  
  // Very short flow duration with many packets (burst attack)
  if (flowDuration < 1000 && fwdPackets > 50) {
    attackScore += 2;
    reasons.push(`Burst: ${fwdPackets} packets trong ${flowDuration}ms`);
  }
  
  // SYN Flood detection
  if (synFlag > 0 && ackFlag === 0 && finFlag === 0) {
    attackScore += 3;
    attackType = "syn_flood";
    reasons.push("SYN flag cao, không có ACK/FIN (SYN Flood)");
  }
  
  // RST flood
  if (rstFlag > 10) {
    attackScore += 2;
    reasons.push(`Nhiều RST flag: ${rstFlag}`);
  }
  
  // Asymmetric traffic (potential reflection/amplification)
  if (bwdPacketLength > fwdPacketLength * 10 && bwdPacketLength > 10000) {
    attackScore += 3;
    reasons.push(`Amplification detected: Response ${bwdPacketLength} >> Request ${fwdPacketLength}`);
    
    // Check specific amplification attack types
    if (dstPort === 53 || srcPort === 53) {
      attackType = "dns_amplification";
      reasons.push("Cổng DNS 53 - DNS Amplification");
    } else if (dstPort === 123 || srcPort === 123) {
      attackType = "ntp_amplification";
      reasons.push("Cổng NTP 123 - NTP Amplification");
    } else if (dstPort === 1900 || srcPort === 1900) {
      attackType = "ssdp_amplification";
      reasons.push("Cổng SSDP 1900 - SSDP Amplification");
    } else if (dstPort === 11211 || srcPort === 11211) {
      attackType = "memcached_amplification";
      reasons.push("Cổng Memcached 11211 - Memcached Amplification");
    }
  }
  
  // Very low IAT (fast packet transmission)
  if (flowIatMean < 10 && fwdPackets > 20) {
    attackScore += 2;
    reasons.push(`IAT rất thấp: ${flowIatMean.toFixed(2)}ms giữa các packet`);
  }
  
  // Small packet flood
  if (fwdPackets > 100 && pktSizeAvg < 100) {
    attackScore += 2;
    reasons.push(`Small packet flood: ${fwdPackets} packets, avg size ${pktSizeAvg.toFixed(0)} bytes`);
  }
  
  // Protocol-based detection
  if (protocol === 17 || String(row["protocol"]).toLowerCase() === "udp") {
    if (flowPacketsPerSec > 500) {
      attackScore += 2;
      if (attackType === "unknown") attackType = "udp_flood";
      reasons.push("UDP flood pattern detected");
    }
  } else if (protocol === 1 || String(row["protocol"]).toLowerCase() === "icmp") {
    if (fwdPackets > 50) {
      attackScore += 2;
      if (attackType === "unknown") attackType = "icmp_flood";
      reasons.push("ICMP flood pattern detected");
    }
  }
  
  // Port-based attack classification
  if (attackType === "unknown") {
    if (dstPort === 22) {
      if (fwdPackets > 20 && flowDuration < 5000) {
        attackType = "ssh_bruteforce";
        attackScore += 2;
        reasons.push("SSH bruteforce pattern: nhiều connection attempt");
      }
    } else if (dstPort === 21) {
      if (fwdPackets > 20) {
        attackType = "ftp_bruteforce";
        attackScore += 2;
        reasons.push("FTP bruteforce pattern");
      }
    } else if (dstPort === 23) {
      attackType = "telnet_bruteforce";
      attackScore += 2;
      reasons.push("Telnet attack - nguy cơ IoT botnet");
    } else if (dstPort === 3389) {
      if (fwdPackets > 10) {
        attackType = "rdp_attack";
        attackScore += 2;
        reasons.push("RDP attack pattern");
      }
    } else if (dstPort === 80 || dstPort === 443 || dstPort === 8080) {
      if (fwdPackets > 100 && flowPacketsPerSec > 50) {
        attackType = "http_flood";
        attackScore += 2;
        reasons.push(`HTTP flood: ${fwdPackets} requests`);
      }
    }
  }
  
  // Calculate confidence
  const confidence = Math.min(0.95, 0.3 + (attackScore * 0.1));
  const isAttack = attackScore >= 3;
  
  if (attackType === "unknown" && isAttack) {
    // Try to classify based on primary indicators
    if (flowBytesPerSec > 10000000) {
      attackType = "udp_flood";
    } else if (flowPacketsPerSec > 500) {
      attackType = "syn_flood";
    }
  }
  
  return { isAttack, attackType, confidence, reasons };
}

// Build feature mapping from columns
export function buildFeatureMapping(columns: string[]): Map<string, string> {
  const mapping = new Map<string, string>();
  
  for (const col of columns) {
    const featureType = mapColumnToFeature(col);
    if (featureType) {
      mapping.set(col, featureType);
    }
  }
  
  return mapping;
}

// Get feature statistics for unlabeled data analysis
export function getFeatureStatistics(data: DataRow[], featureMapping: Map<string, string>): {
  detectedFeatures: string[];
  featureStats: Record<string, { min: number; max: number; mean: number; std: number }>;
  anomalyIndicators: string[];
} {
  const detectedFeatures: string[] = Array.from(new Set(featureMapping.values()));
  const featureStats: Record<string, { min: number; max: number; mean: number; std: number }> = {};
  const anomalyIndicators: string[] = [];
  
  // Calculate statistics for each mapped feature
  const entries = Array.from(featureMapping.entries());
  for (const [col, featureType] of entries) {
    const values = data.map(row => {
      const val = row[col];
      return typeof val === "number" ? val : parseFloat(String(val)) || 0;
    }).filter(v => !isNaN(v));
    
    if (values.length > 0) {
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
      
      featureStats[featureType] = {
        min: Math.min(...values),
        max: Math.max(...values),
        mean,
        std: Math.sqrt(variance),
      };
      
      // Check for anomaly indicators based on feature statistics
      if (featureType === "flow_packets_per_sec" && mean > 500) {
        anomalyIndicators.push(`Tốc độ packet trung bình cao: ${mean.toFixed(0)} pkts/s`);
      }
      if (featureType === "flow_bytes_per_sec" && mean > 10000000) {
        anomalyIndicators.push(`Bandwidth trung bình cao: ${(mean / 1000000).toFixed(1)} MB/s`);
      }
      if (featureType === "fwd_packets" && mean > 100) {
        anomalyIndicators.push(`Nhiều forward packets: avg ${mean.toFixed(0)}`);
      }
    }
  }
  
  return { detectedFeatures, featureStats, anomalyIndicators };
}

export function getFeatureColumns(columns: string[]): string[] {
  const labelColumns = ["label", "class", "attack", "target", "Label", "Class", "Attack", "Target", "attack_cat", "category"];
  const idColumns = ["id", "ID", "Id", "index", "Index", "flow_id", "timestamp", "src_ip", "dst_ip", "src_port"];
  
  return columns.filter(
    (col) =>
      !labelColumns.includes(col) &&
      !idColumns.includes(col)
  );
}

// ============== ANOMALY DETECTION ALGORITHMS (Epic 4) ==============

// Isolation Forest - Anomaly Detection
class IsolationForest {
  private trees: IsolationTree[] = [];
  private numTrees: number;
  private sampleSize: number;
  
  constructor(numTrees: number = 100, sampleSize: number = 256) {
    this.numTrees = numTrees;
    this.sampleSize = sampleSize;
  }
  
  fit(data: number[][]): void {
    this.trees = [];
    for (let i = 0; i < this.numTrees; i++) {
      const sample = this.subsample(data, Math.min(this.sampleSize, data.length));
      const tree = new IsolationTree();
      tree.fit(sample, 0, Math.ceil(Math.log2(this.sampleSize)));
      this.trees.push(tree);
    }
  }
  
  private subsample(data: number[][], size: number): number[][] {
    const indices = new Set<number>();
    while (indices.size < size) {
      indices.add(Math.floor(Math.random() * data.length));
    }
    return Array.from(indices).map(i => data[i]);
  }
  
  predict(data: number[][]): number[] {
    return data.map(point => this.anomalyScore(point));
  }
  
  private anomalyScore(point: number[]): number {
    const avgPathLength = this.trees.reduce((sum, tree) => sum + tree.pathLength(point, 0), 0) / this.trees.length;
    const c = this.expectedPathLength(this.sampleSize);
    return Math.pow(2, -avgPathLength / c);
  }
  
  private expectedPathLength(n: number): number {
    if (n <= 1) return 0;
    if (n === 2) return 1;
    const H = Math.log(n - 1) + 0.5772156649;
    return 2 * H - (2 * (n - 1) / n);
  }
}

class IsolationTree {
  private splitFeature: number = 0;
  private splitValue: number = 0;
  private left: IsolationTree | null = null;
  private right: IsolationTree | null = null;
  private isLeaf: boolean = true;
  private size: number = 0;
  
  fit(data: number[][], depth: number, maxDepth: number): void {
    this.size = data.length;
    
    if (depth >= maxDepth || data.length <= 1) {
      this.isLeaf = true;
      return;
    }
    
    const numFeatures = data[0].length;
    this.splitFeature = Math.floor(Math.random() * numFeatures);
    
    const featureValues = data.map(row => row[this.splitFeature]);
    const minVal = Math.min(...featureValues);
    const maxVal = Math.max(...featureValues);
    
    if (minVal === maxVal) {
      this.isLeaf = true;
      return;
    }
    
    this.splitValue = minVal + Math.random() * (maxVal - minVal);
    this.isLeaf = false;
    
    const leftData = data.filter(row => row[this.splitFeature] < this.splitValue);
    const rightData = data.filter(row => row[this.splitFeature] >= this.splitValue);
    
    if (leftData.length > 0) {
      this.left = new IsolationTree();
      this.left.fit(leftData, depth + 1, maxDepth);
    }
    
    if (rightData.length > 0) {
      this.right = new IsolationTree();
      this.right.fit(rightData, depth + 1, maxDepth);
    }
  }
  
  pathLength(point: number[], currentDepth: number): number {
    if (this.isLeaf) {
      return currentDepth + this.expectedPathLength(this.size);
    }
    
    if (point[this.splitFeature] < this.splitValue) {
      return this.left ? this.left.pathLength(point, currentDepth + 1) : currentDepth + 1;
    } else {
      return this.right ? this.right.pathLength(point, currentDepth + 1) : currentDepth + 1;
    }
  }
  
  private expectedPathLength(n: number): number {
    if (n <= 1) return 0;
    if (n === 2) return 1;
    const H = Math.log(n - 1) + 0.5772156649;
    return 2 * H - (2 * (n - 1) / n);
  }
}

// Local Outlier Factor (LOF)
class LocalOutlierFactor {
  private k: number;
  private data: number[][] = [];
  
  constructor(k: number = 20) {
    this.k = k;
  }
  
  fit(data: number[][]): void {
    this.data = data;
  }
  
  predict(data: number[][]): number[] {
    return data.map(point => this.lofScore(point));
  }
  
  private euclideanDistance(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - (b[i] || 0), 2), 0));
  }
  
  private getKNeighbors(point: number[]): { distances: number[]; indices: number[] } {
    const distances = this.data.map((p, i) => ({
      distance: this.euclideanDistance(point, p),
      index: i
    }));
    
    distances.sort((a, b) => a.distance - b.distance);
    const kNearest = distances.slice(0, this.k);
    
    return {
      distances: kNearest.map(d => d.distance),
      indices: kNearest.map(d => d.index)
    };
  }
  
  private reachabilityDistance(point: number[], neighborIdx: number): number {
    const neighborNeighbors = this.getKNeighbors(this.data[neighborIdx]);
    const kDistance = neighborNeighbors.distances[this.k - 1] || neighborNeighbors.distances[neighborNeighbors.distances.length - 1] || 0.001;
    const distance = this.euclideanDistance(point, this.data[neighborIdx]);
    return Math.max(kDistance, distance);
  }
  
  private localReachabilityDensity(point: number[]): number {
    const neighbors = this.getKNeighbors(point);
    if (neighbors.indices.length === 0) return 1;
    
    const avgReachDist = neighbors.indices.reduce((sum, idx) => 
      sum + this.reachabilityDistance(point, idx), 0) / neighbors.indices.length;
    
    return avgReachDist > 0 ? 1 / avgReachDist : 1;
  }
  
  private lofScore(point: number[]): number {
    const neighbors = this.getKNeighbors(point);
    if (neighbors.indices.length === 0) return 1;
    
    const pointLrd = this.localReachabilityDensity(point);
    if (pointLrd === 0) return 1;
    
    const avgNeighborLrd = neighbors.indices.reduce((sum, idx) => 
      sum + this.localReachabilityDensity(this.data[idx]), 0) / neighbors.indices.length;
    
    return avgNeighborLrd / pointLrd;
  }
}

// Anomaly detection cho Unlabeled mode
export function runAnomalyDetection(
  features: number[][],
  threshold: number = 0.5
): { scores: number[]; isAnomalous: boolean[]; avgScore: number; alertRate: number } {
  // Isolation Forest
  const isoForest = new IsolationForest(50, Math.min(256, features.length));
  isoForest.fit(features);
  const isoScores = isoForest.predict(features);
  
  // LOF (chỉ chạy với sample nhỏ để tối ưu performance)
  const sampleSize = Math.min(500, features.length);
  const lof = new LocalOutlierFactor(Math.min(10, Math.floor(sampleSize / 10)));
  const sampleFeatures = features.slice(0, sampleSize);
  lof.fit(sampleFeatures);
  
  // Combine scores
  const scores = isoScores.map((isoScore, i) => {
    // Normalize LOF để scale tương tự Isolation Forest
    const lofIdx = i < sampleSize ? i : Math.floor(Math.random() * sampleSize);
    const lofScoreNorm = Math.min(1, Math.max(0, (lof.predict([features[lofIdx]])[0] - 1) / 2));
    return (isoScore * 0.7 + lofScoreNorm * 0.3);
  });
  
  const isAnomalous = scores.map(s => s > threshold);
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  const alertRate = isAnomalous.filter(Boolean).length / isAnomalous.length;
  
  return { scores, isAnomalous, avgScore, alertRate };
}

// Unlabeled inference report generation
export function generateUnlabeledReport(
  data: DataRow[],
  columns: string[],
  features: number[][],
  anomalyScores: number[]
): import("@shared/schema").AnalysisResult["unlabeledReport"] {
  // Score distribution
  const sortedScores = [...anomalyScores].sort((a, b) => a - b);
  const mean = anomalyScores.reduce((a, b) => a + b, 0) / anomalyScores.length;
  const variance = anomalyScores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / anomalyScores.length;
  
  // Data quality
  let missingCount = 0;
  let invalidCount = 0;
  for (const row of data) {
    for (const col of columns) {
      const val = row[col];
      if (val === null || val === undefined || val === "") {
        missingCount++;
      } else if (typeof val === "number" && (isNaN(val) || !isFinite(val))) {
        invalidCount++;
      }
    }
  }
  
  const totalCells = data.length * columns.length;
  
  return {
    scoreDistribution: {
      min: sortedScores[0] || 0,
      max: sortedScores[sortedScores.length - 1] || 0,
      mean,
      std: Math.sqrt(variance),
      percentiles: {
        p25: sortedScores[Math.floor(sortedScores.length * 0.25)] || 0,
        p50: sortedScores[Math.floor(sortedScores.length * 0.50)] || 0,
        p75: sortedScores[Math.floor(sortedScores.length * 0.75)] || 0,
        p90: sortedScores[Math.floor(sortedScores.length * 0.90)] || 0,
        p95: sortedScores[Math.floor(sortedScores.length * 0.95)] || 0,
      },
    },
    alertRate: anomalyScores.filter(s => s > 0.5).length / anomalyScores.length,
    dataQuality: {
      missingRate: totalCells > 0 ? missingCount / totalCells : 0,
      invalidValues: invalidCount,
      totalRows: data.length,
      validRows: data.filter(row => 
        columns.every(col => row[col] !== null && row[col] !== undefined)
      ).length,
    },
  };
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
