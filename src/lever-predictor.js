/**
 * Lever Predictor — adjust non-method levers; predict the 8 core features and
 * pipe them into the regression engine + Monte Carlo rank simulator.
 *
 * Levers use their original CFM-trained metadata (min / max / step / format
 * from cfm_artifacts/summary.json). Percentage levers stay as percentages;
 * no count conversion. Each slider is tinted by its primary CFM's confidence
 * (high / medium / low) so users see at a glance which inputs the model
 * reads confidently.
 */

import { simulateRank, computeBlendedGMAT } from './model.js';
import {
  getLeverMetadata,
  getGwuPredictorValues,
  getActualCoreFeatures,
  getCoreTargetOrder,
  getCoreTargetLabel,
  getConfidenceFor,
  getLeverPrimaryTargets,
  predictAllCoreFeatures,
  getGwuBaseline,
} from './cfm-models.js';

const LEVER_DEBOUNCE_MS = 200;
let leverValues = {};
let debounceTimer = null;
let previousRank = null;

// ============================================================
// Formatters
// ============================================================

function fmtFor(format, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  switch (format) {
    case 'percent': return `${(value * 100).toFixed(1)}%`;
    case 'dollar': return `$${Math.round(value).toLocaleString()}`;
    case 'binary': return value >= 0.5 ? 'Yes' : 'No';
    case 'number':
    default: return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }
}

function fmtCoreValue(target, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (['EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate'].includes(target)) {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (target === 'AvgSalaryBonus') return `$${Math.round(value).toLocaleString()}`;
  if (target === 'GMAT_Combined') return Math.round(value).toString();
  return value.toFixed(2);
}

function fmtCoreDelta(target, delta) {
  if (delta === null || delta === undefined || Number.isNaN(delta)) return '';
  const sign = delta >= 0 ? '+' : '−';
  const abs = Math.abs(delta);
  let body;
  if (['EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate'].includes(target)) body = `${(abs * 100).toFixed(1)}pp`;
  else if (target === 'AvgSalaryBonus') body = `$${Math.round(abs).toLocaleString()}`;
  else if (target === 'GMAT_Combined') body = abs.toFixed(0);
  else body = abs.toFixed(2);
  return `${sign}${body}`;
}

// ============================================================
// Slider building
// ============================================================

function buildSlider(meta, tone) {
  const id = `lever-${meta.key.replace(/[^a-z0-9]+/gi, '_')}`;
  const wrap = document.createElement('div');
  wrap.className = `slider-container lever-slider tone-${tone}`;
  wrap.dataset.leverKey = meta.key;
  wrap.innerHTML = `
    <div class="flex items-center justify-between mb-2">
      <label class="text-sm font-medium text-gray-300" for="${id}-range">${meta.label}</label>
      <span class="text-sm font-mono font-semibold text-cyan-300 bg-cyan-500/10 px-2.5 py-1 rounded-lg" data-role="value">${fmtFor(meta.format, meta.gwu_current)}</span>
    </div>
    <input type="range" id="${id}-range" min="${meta.min}" max="${meta.max}" step="${meta.step}" value="${meta.gwu_current}" class="w-full" />
    <div class="flex justify-between mt-1.5">
      <span class="text-xs text-gray-600">${fmtFor(meta.format, meta.min)}</span>
      <span class="text-xs text-gray-600">${fmtFor(meta.format, meta.max)}</span>
    </div>
  `;
  return wrap;
}

function attachHandlers(wrap, meta) {
  const range = wrap.querySelector('input[type="range"]');
  const display = wrap.querySelector('[data-role="value"]');
  range.addEventListener('input', e => {
    let v = parseFloat(e.target.value);
    if (meta.format === 'binary') v = v >= 0.5 ? 1 : 0;
    leverValues[meta.key] = v;
    display.textContent = fmtFor(meta.format, v);
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(recompute, LEVER_DEBOUNCE_MS);
  });
}

// ============================================================
// Prediction + render
// ============================================================

function buildPredictorRow() {
  const baseline = { ...getGwuPredictorValues() };
  for (const [k, v] of Object.entries(leverValues)) baseline[k] = v;
  return baseline;
}

function buildCoreFeatureRow(target, predictedValue, actualValue) {
  const conf = getConfidenceFor(target);
  const delta = (actualValue !== null && actualValue !== undefined && !Number.isNaN(actualValue))
    ? predictedValue - actualValue : null;
  const row = document.createElement('div');
  row.className = `core-feature-row core-feature-row--${conf.tone}`;
  const label = getCoreTargetLabel(target);
  const deltaStr = fmtCoreDelta(target, delta);
  const deltaCls = delta === null ? 'core-delta core-delta--null'
    : delta >= 0 ? 'core-delta core-delta--up' : 'core-delta core-delta--down';
  row.innerHTML = `
    <div class="flex items-center justify-between gap-1">
      <p class="text-[11px] font-semibold text-white truncate" title="${label}">${label}</p>
      <span class="confidence-pill confidence-${conf.tone}" style="font-size: 0.55rem; padding: 0.1rem 0.4rem;">${conf.label}</span>
    </div>
    <div class="flex items-baseline justify-between mt-1">
      <p class="text-base font-bold text-white">${fmtCoreValue(target, predictedValue)}</p>
      <p class="${deltaCls}" style="font-size: 0.7rem;">${deltaStr}</p>
    </div>
  `;
  return row;
}

function animateLeverRank(targetRank) {
  const el = document.getElementById('lever-rank-display');
  if (!el) return;
  const start = previousRank ?? targetRank;
  previousRank = targetRank;
  const duration = 500;
  const t0 = performance.now();
  function tick(now) {
    const p = Math.min((now - t0) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = `#${Math.round(start + (targetRank - start) * eased)}`;
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function renderResults(predictedCore, simResults) {
  const actuals = getActualCoreFeatures();
  const grid = document.getElementById('lever-core-features');
  if (grid) {
    grid.innerHTML = '';
    // Sort core features by confidence: high → medium → low
    const CONFIDENCE_ORDER = { high: 0, medium: 1, low: 2 };
    const targets = getCoreTargetOrder().slice();
    targets.sort((a, b) => {
      const aTone = getConfidenceFor(a)?.tone || 'low';
      const bTone = getConfidenceFor(b)?.tone || 'low';
      return (CONFIDENCE_ORDER[aTone] ?? 2) - (CONFIDENCE_ORDER[bTone] ?? 2);
    });
    for (const target of targets) {
      grid.appendChild(buildCoreFeatureRow(target, predictedCore[target], actuals[target]));
    }
  }
  if (simResults) {
    animateLeverRank(simResults.medianRank);
    const sub = document.getElementById('lever-rank-subtitle');
    if (sub) sub.textContent = `Median from 10,000 simulations`;
    const score = document.getElementById('lever-score');
    if (score) score.textContent = simResults.scenarioScore.toFixed(1);
    const ci = document.getElementById('lever-ci');
    if (ci) ci.textContent = `${simResults.range90[0]}–${simResults.range90[1]}`;
  }
}

function recompute() {
  const row = buildPredictorRow();
  const predictedCore = predictAllCoreFeatures(row);
  // CFM was trained on legacy raw-GMAT GMAT_Combined; convert to v2's blended
  // 0-100 scale. SBP isn't a CFM target; the snapshot value flows through.
  const cfmFeed = { ...predictedCore };
  if (typeof cfmFeed.GMAT_Combined === 'number') {
    cfmFeed.GMAT_Combined = computeBlendedGMAT({
      scale: 'old',
      gmat_score: cfmFeed.GMAT_Combined,
      gre_q: null, gre_v: null, gre_aw: null,
      gre_enabled: false,
    });
  }
  let sim = null;
  try {
    sim = simulateRank(cfmFeed);
  } catch (err) {
    console.error('[lever] simulateRank failed', err);
  }
  renderResults(predictedCore, sim);
}

function renderBaselineCard() {
  const baseline = getGwuBaseline();
  if (!baseline) return;
  const rankEl = document.getElementById('lever-baseline-rank');
  const scoreEl = document.getElementById('lever-baseline-score');
  if (rankEl) rankEl.textContent = baseline.actual_rank ? `#${baseline.actual_rank}` : '#—';
  if (scoreEl) scoreEl.textContent = baseline.actual_overall_score?.toFixed(1) ?? '—';
}

export function initLeverPredictor() {
  const container = document.getElementById('lever-sliders-container');
  if (!container) return;
  container.innerHTML = '';

  const metadata = getLeverMetadata();
  const primaries = getLeverPrimaryTargets();

  for (const meta of metadata) {
    leverValues[meta.key] = meta.gwu_current;
    const target = primaries[meta.key]?.target || 'unranked';
    const conf = getConfidenceFor(target);
    const tone = conf?.tone || 'low';
    const wrap = buildSlider(meta, tone);
    attachHandlers(wrap, meta);
    container.appendChild(wrap);
  }

  renderBaselineCard();

  const resetBtn = document.getElementById('lever-reset-btn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      for (const meta of metadata) {
        leverValues[meta.key] = meta.gwu_current;
        const wrap = container.querySelector(`[data-lever-key="${CSS.escape(meta.key)}"]`);
        if (!wrap) continue;
        wrap.querySelector('input[type="range"]').value = meta.gwu_current;
        wrap.querySelector('[data-role="value"]').textContent = fmtFor(meta.format, meta.gwu_current);
      }
      previousRank = null;
      recompute();
    });
  }

  recompute();
}
