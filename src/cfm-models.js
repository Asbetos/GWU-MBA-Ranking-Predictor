/**
 * Client-side mirror of the 8 indirect (core-feature) sklearn pipelines.
 *
 * Each Python pipeline is:
 *   SimpleImputer(strategy="median") -> StandardScaler -> {RidgeCV | ElasticNetCV}
 * with a per-target transform on y at fit time:
 *   log1p / logit / identity, inverse-applied at predict time, then clipped.
 */

const ARTIFACTS_BASE = '/cfm_artifacts';

const TARGET_NAMES = [
  'EmployedAtGrad', 'Employed3Mo', 'AvgSalaryBonus', 'MedianGPA',
  'AcceptanceRate', 'PeerScore', 'RecruiterScore', 'GMAT_Combined',
];

let summary = null;
const models = {}; // target_name -> per-model JSON

export async function loadCfmArtifacts() {
  const summaryPromise = fetch(`${ARTIFACTS_BASE}/summary.json`).then((r) => r.json());
  const modelPromises = TARGET_NAMES.map((t) =>
    fetch(`${ARTIFACTS_BASE}/model_${t}.json`).then((r) => r.json()),
  );
  const [s, ...modelJsons] = await Promise.all([summaryPromise, ...modelPromises]);
  summary = s;
  modelJsons.forEach((m) => { models[m.target_name] = m; });
  return { summary, models };
}

export function getCfmSummary() { return summary; }
export function getCoreTargetSummaries() { return summary?.core_targets || []; }
export function getLeverMetadata() { return summary?.lever_metadata || []; }
export function getGwuBaseline() { return summary?.gwu_baseline || null; }
export function getGwuPredictorValues() { return summary?.gwu_baseline?.predictor_values || {}; }
export function getActualCoreFeatures() { return summary?.gwu_baseline?.actual_core_features || {}; }
export function getConfidenceFor(target) {
  const row = (summary?.core_targets || []).find((t) => t.target_name === target);
  return row?.confidence || { label: 'Low', tone: 'low', weight: 0.25 };
}
export function getCoreTargetOrder() { return summary?.core_target_order || TARGET_NAMES; }
export function getCoreTargetLabel(target) {
  return summary?.core_target_labels?.[target] || target;
}

function inverseTargetTransform(target, raw) {
  const m = models[target];
  let v = raw;
  if (m.target_transform === 'log1p') v = Math.expm1(raw);
  else if (m.target_transform === 'logit') v = 1.0 / (1.0 + Math.exp(-raw));
  if (m.clip_lower !== null && m.clip_lower !== undefined) v = Math.max(m.clip_lower, v);
  if (m.clip_upper !== null && m.clip_upper !== undefined) v = Math.min(m.clip_upper, v);
  return v;
}

export function predictCoreFeature(target, predictorRow) {
  const m = models[target];
  if (!m) throw new Error(`Model for ${target} not loaded`);
  const { feature_columns, imputer_statistics, scaler_mean, scaler_scale, intercept, coef } = m;

  let z = intercept;
  for (let i = 0; i < feature_columns.length; i++) {
    const key = feature_columns[i];
    let raw = predictorRow?.[key];
    if (raw === null || raw === undefined || Number.isNaN(raw)) {
      raw = imputer_statistics[i];
    }
    const scaled = (raw - scaler_mean[i]) / scaler_scale[i];
    z += scaled * coef[i];
  }
  return inverseTargetTransform(target, z);
}

export function predictAllCoreFeatures(predictorRow) {
  const out = {};
  for (const t of TARGET_NAMES) {
    out[t] = predictCoreFeature(t, predictorRow);
  }
  return out;
}
