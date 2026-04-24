/**
 * Client-side model inference engine.
 * Loads JSON artifacts exported by train_model.py and replicates
 * the full transform → predict → simulate pipeline in JavaScript.
 */

let modelConfig = null;
let capperBounds = null;
let transformerConfig = null;
let scalerParams = null;
let modelWeights = null;
let dataSnapshot = null;
let featureRanges = null;

/** Load all model artifacts from static JSON files */
export async function loadModel() {
  const base = '/model_artifacts';
  const [config, capper, transformer, scaler, weights, snapshot, ranges] = await Promise.all([
    fetch(`${base}/model_config.json`).then(r => r.json()),
    fetch(`${base}/capper_bounds.json`).then(r => r.json()),
    fetch(`${base}/transformer_config.json`).then(r => r.json()),
    fetch(`${base}/scaler_params.json`).then(r => r.json()),
    fetch(`${base}/model_weights.json`).then(r => r.json()),
    fetch(`${base}/data_snapshot.json`).then(r => r.json()),
    fetch(`${base}/feature_ranges.json`).then(r => r.json()),
  ]);

  modelConfig = config;
  capperBounds = capper;
  transformerConfig = transformer;
  scalerParams = scaler;
  modelWeights = weights;
  dataSnapshot = snapshot;
  featureRanges = ranges;

  return { modelConfig, featureRanges, dataSnapshot };
}

/** Get the loaded feature ranges */
export function getFeatureRanges() {
  return featureRanges;
}

/** Get GWU current values */
export function getGWUValues() {
  return featureRanges?._gwu_current || {};
}

/** Get GWU school name */
export function getGWUSchoolName() {
  return featureRanges?._gwu_school_name || 'George Washington University';
}

/** Get GWU current rank */
export function getGWUCurrentRank() {
  return featureRanges?._gwu_current_rank || null;
}

/** Get GWU current score */
export function getGWUCurrentScore() {
  return featureRanges?._gwu_current_score || null;
}

// ============================================================
// TRANSFORM PIPELINE (mirrors Python's OutlierCapper → 
// RankingFeatureTransformer → StandardScaler)
// ============================================================

/** Apply OutlierCapper: clip values to learned percentile bounds */
function applyCapper(row) {
  const result = { ...row };
  for (const [col, bounds] of Object.entries(capperBounds)) {
    if (col in result) {
      result[col] = Math.max(bounds.lower, Math.min(bounds.upper, result[col]));
    }
  }
  return result;
}

/** Apply RankingFeatureTransformer: log, logit, inv_norm transforms */
function applyTransformer(row) {
  const result = { ...row };

  // Log transformation: log1p(x)
  for (const col of transformerConfig.log_cols) {
    if (col in result) {
      result[col] = Math.log1p(result[col]);
    }
  }

  // Logit transformation: log(p / (1-p))
  for (const col of transformerConfig.logit_cols) {
    if (col in result) {
      let p = Math.max(0.001, Math.min(0.999, result[col]));
      result[col] = Math.log(p / (1 - p));
    }
  }

  // Inverse Normal transformation (not used in 8-feature model, but included for completeness)
  for (const col of transformerConfig.inv_norm_cols) {
    if (col in result) {
      const N = transformerConfig.rank_counts[col] || 120;
      let percentile = (result[col] - 0.5) / N;
      percentile = Math.max(0.001, Math.min(0.999, percentile));
      result[col] = -1 * normPPF(percentile);
    }
  }

  return result;
}

/** Apply StandardScaler: (x - mean) / scale */
function applyScaler(row) {
  const features = scalerParams.feature_names;
  const values = features.map((feat, i) => {
    return (row[feat] - scalerParams.mean[i]) / scalerParams.scale[i];
  });
  return values;
}

/** Full prediction: capper → transformer → scaler → dot product + intercept */
function predictScore(row) {
  const capped = applyCapper(row);
  const transformed = applyTransformer(capped);
  const scaled = applyScaler(transformed);
  
  let score = modelWeights.intercept;
  for (let i = 0; i < scaled.length; i++) {
    score += scaled[i] * modelWeights.coef[i];
  }
  return score;
}

// ============================================================
// MONTE CARLO RANK SIMULATION
// (mirrors simulate_advanced_rank from rank_scenario_planning.ipynb)
// ============================================================

/**
 * Simulate rank for a school with custom metrics.
 * @param {Object} customMetrics - Feature values to use for the target school
 * @param {string} targetSchool - School name to simulate (default: GWU)
 * @param {number} nSimulations - Number of Monte Carlo iterations
 * @returns {Object} Results including medianRank, CI, score, distribution
 */
export function simulateRank(customMetrics, targetSchool = null, nSimulations = 10000) {
  if (!dataSnapshot || !modelWeights) {
    throw new Error('Model not loaded. Call loadModel() first.');
  }

  targetSchool = targetSchool || getGWUSchoolName();

  // 1. Find target school index
  const targetIdx = dataSnapshot.findIndex(s => s.School === targetSchool);
  if (targetIdx === -1) {
    throw new Error(`School "${targetSchool}" not found in dataset.`);
  }

  // 2. Build simulation data with custom metrics applied
  const simData = dataSnapshot.map((school, i) => {
    if (i === targetIdx) {
      // Apply custom metrics to target school
      const modified = { ...school };
      for (const [key, val] of Object.entries(customMetrics)) {
        modified[key] = val;
      }
      return modified;
    }
    return { ...school };
  });

  // 3. Predict scores for all schools
  const baseScores = simData.map(school => {
    const features = {};
    for (const feat of modelConfig.features) {
      features[feat] = school[feat];
    }
    return predictScore(features);
  });

  // 4. Setup volatility (tiered noise)
  const noiseScale = simData.map((school, i) => {
    if (i === targetIdx) return 0; // Target is deterministic
    const rank = school.Rank;
    if (rank <= 20) return 0.8;
    if (rank <= 50) return 1.5;
    return 2.5;
  });

  // 5. Monte Carlo loop
  const predictedRanks = [];
  const n = baseScores.length;

  for (let sim = 0; sim < nSimulations; sim++) {
    // Generate noise
    const scenarioScores = baseScores.map((score, i) => {
      return score + gaussianRandom() * noiseScale[i];
    });

    // Sort descending to get ranks (highest score = rank 1)
    const indices = Array.from({ length: n }, (_, i) => i);
    indices.sort((a, b) => scenarioScores[b] - scenarioScores[a]);

    // Find target's rank
    const rank = indices.indexOf(targetIdx) + 1;
    predictedRanks.push(rank);
  }

  // 6. Compile results
  predictedRanks.sort((a, b) => a - b);

  const medianRank = predictedRanks[Math.floor(predictedRanks.length / 2)];
  const p5 = predictedRanks[Math.floor(predictedRanks.length * 0.05)];
  const p95 = predictedRanks[Math.floor(predictedRanks.length * 0.95)];
  const scenarioScore = baseScores[targetIdx];

  // Build histogram data for chart
  const rankCounts = {};
  for (const r of predictedRanks) {
    rankCounts[r] = (rankCounts[r] || 0) + 1;
  }
  
  // Convert to probability
  const distribution = {};
  for (const [rank, count] of Object.entries(rankCounts)) {
    distribution[rank] = count / nSimulations;
  }

  return {
    medianRank,
    range90: [p5, p95],
    scenarioScore: Math.round(scenarioScore * 100) / 100,
    rankDistribution: distribution,
    rawRanks: predictedRanks,
  };
}

// ============================================================
// MATH UTILITIES
// ============================================================

/** Box-Muller transform for generating standard normal random numbers */
function gaussianRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Approximate inverse normal CDF (PPF) using rational approximation.
 * Accurate to ~1e-9 for 0 < p < 1.
 */
function normPPF(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;

  // Rational approximation by Peter Acklam
  const a = [
    -3.969683028665376e+01, 2.209460984245205e+02,
    -2.759285104469687e+02, 1.383577518672690e+02,
    -3.066479806614716e+01, 2.506628277459239e+00
  ];
  const b = [
    -5.447609879822406e+01, 1.615858368580409e+02,
    -1.556989798598866e+02, 6.680131188771972e+01,
    -1.328068155288572e+01
  ];
  const c = [
    -7.784894002430293e-03, -3.223964580411365e-01,
    -2.400758277161838e+00, -2.549732539343734e+00,
    4.374664141464968e+00, 2.938163982698783e+00
  ];
  const d = [
    7.784695709041462e-03, 3.224671290700398e-01,
    2.445134137142996e+00, 3.754408661907416e+00
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q, r;

  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  } else if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
}
