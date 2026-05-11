/**
 * Client-side regression-only inference engine for the GWSB Ranking Predictor.
 *
 * Single bootstrapped ElasticNet engine + Monte Carlo rank simulator. The
 * scoring path is: capper → log/logit → standardize → coef·x + intercept.
 */

let modelConfig = null;
let capperBounds = null;
let transformerConfig = null;
let scalerParams = null;
let modelWeights = null;
let dataSnapshot = null;
let featureRanges = null;
let gmatCurves = null;
let modelExplainability = null;

const ARTIFACTS_BASE = '/model_artifacts';

export async function loadModel() {
  const [config, capper, transformer, scaler, weights, snapshot, ranges, curves, expl] = await Promise.all([
    fetch(`${ARTIFACTS_BASE}/model_config.json`).then(r => r.json()),
    fetch(`${ARTIFACTS_BASE}/capper_bounds.json`).then(r => r.json()),
    fetch(`${ARTIFACTS_BASE}/transformer_config.json`).then(r => r.json()),
    fetch(`${ARTIFACTS_BASE}/scaler_params.json`).then(r => r.json()),
    fetch(`${ARTIFACTS_BASE}/model_weights.json`).then(r => r.json()),
    fetch(`${ARTIFACTS_BASE}/data_snapshot.json`).then(r => r.json()),
    fetch(`${ARTIFACTS_BASE}/feature_ranges.json`).then(r => r.json()),
    fetch(`${ARTIFACTS_BASE}/gmat_inference_curves.json`).then(r => r.json()).catch(() => null),
    fetch(`${ARTIFACTS_BASE}/model_explainability.json`).then(r => r.json()).catch(() => null),
  ]);
  modelConfig = config;
  capperBounds = capper;
  transformerConfig = transformer;
  scalerParams = scaler;
  modelWeights = weights;
  dataSnapshot = snapshot;
  featureRanges = ranges;
  gmatCurves = curves;
  modelExplainability = expl;
  return { modelConfig, featureRanges, dataSnapshot };
}

export function getFeatureRanges() { return featureRanges; }
export function getGWUValues() { return featureRanges?._gwu_current || {}; }
export function getGWUSchoolName() { return featureRanges?._gwu_school_name || 'George Washington University'; }
export function getGWUCurrentRank() { return featureRanges?._gwu_current_rank || null; }
export function getGWUCurrentScore() { return featureRanges?._gwu_current_score || null; }
export function getGmatInputConfig() { return featureRanges?._gmat_input_config || {}; }
export function getModelExplainability() { return modelExplainability; }
export function getDataSnapshot() { return dataSnapshot; }
export function getModelConfig() { return modelConfig; }

/** Get SBP cohort means + slider config + occupation list. */
export function getSbpConfig() {
  return {
    cohort:      featureRanges?._sbp_cohort      || {},
    occupations: featureRanges?._sbp_occupations || [],
    slider:      featureRanges?._sbp_slider      || { min: 60000, max: 260000, step: 1000, format: 'dollar' },
  };
}

/** Compute SBP ratio from per-occupation {salary, n} inputs. */
export function computeSBPRatio(perOccupation) {
  const cohort = featureRanges?._sbp_cohort || {};
  const MIN_N = 3;
  let ratio_sum = 0, n_sum = 0;
  for (const [occ, info] of Object.entries(perOccupation || {})) {
    const cm = cohort[occ]?.cohort_mean;
    if (!cm || cm <= 0) continue;
    const sal = Number(info?.salary);
    const n = Number(info?.n);
    if (!isFinite(sal) || sal <= 0 || !isFinite(n) || n < MIN_N) continue;
    ratio_sum += (sal / cm) * n;
    n_sum += n;
  }
  return n_sum > 0 ? ratio_sum / n_sum : null;
}

// ============================================================
// GMAT/GRE blended percentile (5-distribution, 40/40/20 GRE internal)
// ============================================================

const GMAT_BLEND_THRESHOLD = 0.25;

function percentileRank(score, sortedArr) {
  if (!sortedArr || sortedArr.length === 0 || score == null || Number.isNaN(score)) return null;
  let lo = 0, hi = sortedArr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sortedArr[mid] < score) lo = mid + 1; else hi = mid;
  }
  let upper = lo;
  while (upper < sortedArr.length && sortedArr[upper] === score) upper++;
  return (lo + upper + 1) / 2 / sortedArr.length;
}

/**
 * Cohort floor (lowest GMAT_Combined among reporting schools) used as a fallback
 * when the user has not provided ANY test inputs (per-methodology missing-data
 * rule). Sourced from the data snapshot at load time.
 */
function gmatCohortFloor() {
  if (!dataSnapshot) return 0;
  const vals = dataSnapshot.map(s => s.GMAT_Combined).filter(v => v != null && !Number.isNaN(v));
  if (!vals.length) return 0;
  return Math.min(...vals);
}

/**
 * Compute the 0-100 blended GMAT/GRE percentile from raw user inputs.
 * If neither GMAT nor GRE is provided -> cohort floor (matches official
 * methodology missing-data rule).
 */
export function computeBlendedGMAT({ scale, gmat_score, gre_q, gre_v, gre_aw, gre_enabled, gmat_enabled = true }) {
  if (!gmatCurves) return gmat_score ?? 0;

  let p_old = 0, p_new = 0, p_gre = 0;
  let r_old = null, r_new = null, r_gre = null;

  if (gmat_enabled && gmat_score != null) {
    if (scale === 'old') {
      r_old = percentileRank(gmat_score, gmatCurves.gmat_old);
      if (r_old !== null) p_old = 1;
    } else if (scale === 'new') {
      r_new = percentileRank(gmat_score, gmatCurves.gmat_new);
      if (r_new !== null) p_new = 1;
    }
  }
  if (gre_enabled) {
    const rq = gre_q != null ? percentileRank(gre_q, gmatCurves.gre_q) : null;
    const rv = gre_v != null ? percentileRank(gre_v, gmatCurves.gre_v) : null;
    const ra = gre_aw != null ? percentileRank(gre_aw, gmatCurves.gre_aw) : null;
    if (rq !== null && rv !== null && ra !== null) r_gre = 0.4 * rq + 0.4 * rv + 0.2 * ra;
    else if (rq !== null && rv !== null) r_gre = 0.5 * rq + 0.5 * rv;
    else if (rq !== null) r_gre = rq;
    else if (rv !== null) r_gre = rv;
    if (r_gre !== null) p_gre = 1;
  }

  const total = p_old + p_new + p_gre;
  if (total <= 0) return gmatCohortFloor();   // No inputs → cohort floor

  let blended = (p_old * (r_old || 0) + p_new * (r_new || 0) + p_gre * (r_gre || 0)) / total;
  return Math.max(0, Math.min(100, blended * 100));
}

// ============================================================
// Regression scoring path
// ============================================================

function applyCapper(row) {
  const result = { ...row };
  for (const [col, bounds] of Object.entries(capperBounds)) {
    if (col in result) {
      result[col] = Math.max(bounds.lower, Math.min(bounds.upper, result[col]));
    }
  }
  return result;
}

function applyTransformer(row) {
  const result = { ...row };
  for (const col of transformerConfig.log_cols || []) {
    if (col in result) result[col] = Math.log1p(result[col]);
  }
  for (const col of transformerConfig.logit_cols || []) {
    if (col in result) {
      const p = Math.max(0.001, Math.min(0.999, result[col]));
      result[col] = Math.log(p / (1 - p));
    }
  }
  return result;
}

function applyScaler(row) {
  const features = scalerParams.feature_names;
  return features.map((f, i) => (row[f] - scalerParams.mean[i]) / scalerParams.scale[i]);
}

function predictScore(row) {
  const capped = applyCapper(row);
  const transformed = applyTransformer(capped);
  const scaled = applyScaler(transformed);
  let s = modelWeights.intercept;
  for (let i = 0; i < scaled.length; i++) s += scaled[i] * modelWeights.coef[i];
  return s;
}

// ============================================================
// Monte Carlo rank simulation
// ============================================================

export function simulateRank(customMetrics, targetSchool = null, nSimulations = 10000) {
  if (!dataSnapshot) throw new Error('Model not loaded.');
  targetSchool = targetSchool || getGWUSchoolName();
  const targetIdx = dataSnapshot.findIndex(s => s.School === targetSchool);
  if (targetIdx === -1) throw new Error(`School "${targetSchool}" not found.`);

  const simData = dataSnapshot.map((school, i) => {
    if (i === targetIdx) {
      const modified = { ...school };
      for (const [k, v] of Object.entries(customMetrics)) {
        if (k === 'GMAT_Combined' && v !== null && typeof v === 'object') {
          modified[k] = computeBlendedGMAT(v);
        } else if (k === 'SalaryByProfession' && v !== null && typeof v === 'object') {
          const ratio = computeSBPRatio(v);
          if (ratio !== null) modified[k] = ratio;
        } else {
          modified[k] = v;
        }
      }
      return modified;
    }
    return { ...school };
  });

  const baseScores = simData.map((school, i) => {
    const trueScore = dataSnapshot[i].OverallScore;
    const baselineFeatures = {};
    for (const f of modelConfig.features) baselineFeatures[f] = dataSnapshot[i][f];
    const baselinePred = predictScore(baselineFeatures);
    const residual = (trueScore != null && !Number.isNaN(trueScore)) ? (trueScore - baselinePred) : 0;
    const simFeatures = {};
    for (const f of modelConfig.features) simFeatures[f] = school[f];
    return predictScore(simFeatures) + residual;
  });

  const noiseScale = simData.map((s, i) => {
    if (i === targetIdx) return 0;
    const r = s.Rank;
    if (r <= 20) return 0.8;
    if (r <= 50) return 1.5;
    return 2.5;
  });

  const predictedRanks = [];
  const n = baseScores.length;
  for (let sim = 0; sim < nSimulations; sim++) {
    const scenario = baseScores.map((s, i) => s + gaussianRandom() * noiseScale[i]);
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => scenario[b] - scenario[a]);
    predictedRanks.push(idx.indexOf(targetIdx) + 1);
  }
  predictedRanks.sort((a, b) => a - b);
  const median = predictedRanks[Math.floor(predictedRanks.length / 2)];
  const p5 = predictedRanks[Math.floor(predictedRanks.length * 0.05)];
  const p95 = predictedRanks[Math.floor(predictedRanks.length * 0.95)];

  const counts = {};
  for (const r of predictedRanks) counts[r] = (counts[r] || 0) + 1;
  const distribution = {};
  for (const [r, c] of Object.entries(counts)) distribution[r] = c / nSimulations;

  return {
    medianRank: median,
    range90: [p5, p95],
    scenarioScore: Math.round(baseScores[targetIdx] * 100) / 100,
    rankDistribution: distribution,
    rawRanks: predictedRanks,
  };
}

function gaussianRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
