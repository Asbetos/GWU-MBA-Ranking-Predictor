/**
 * 9-feature slider panel with composite GMAT/GRE control (Old/New toggle + optional GRE Q/V/AW).
 */

import { getFeatureRanges, getGWUValues, getGmatInputConfig, computeBlendedGMAT, getSbpConfig, computeSBPRatio } from './model.js';

const FEATURE_ORDER = [
  'EmployedAtGrad', 'Employed3Mo', 'AvgSalaryBonus',
  'MedianGPA', 'AcceptanceRate',
  'PeerScore', 'RecruiterScore',
  'GMAT_Combined',
  'SalaryByProfession',
];

let currentValues = {};
let gmatState = { scale: 'old', gmat_score: null, gre_q: null, gre_v: null, gre_aw: null, gre_enabled: false };
let sbpState = {};
let onChangeCallback = null;
let debounceTimer = null;

function formatValue(value, format) {
  switch (format) {
    case 'percent': return `${(value * 100).toFixed(1)}%`;
    case 'dollar': return `$${Math.round(value).toLocaleString()}`;
    case 'number':
    default: return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }
}

function createSlider(featureKey, config, initialValue) {
  const container = document.createElement('div');
  container.className = 'slider-container';
  container.id = `slider-${featureKey}`;
  const { min, max, step, label, format } = config;
  const displayVal = formatValue(initialValue, format);

  container.innerHTML = `
    <div class="flex items-center justify-between mb-2">
      <label class="text-sm font-medium text-gray-300" for="range-${featureKey}">${label}</label>
      <div class="flex flex-col items-end">
        <span class="text-sm font-mono font-semibold text-indigo-400 bg-indigo-500/10 px-2.5 py-1 rounded-lg cursor-pointer hover:bg-indigo-500/20 transition-colors" id="value-${featureKey}" title="Click to edit">${displayVal}</span>
        <input type="number" id="input-${featureKey}" class="hidden w-24 bg-navy-800 border border-indigo-500/50 text-indigo-400 rounded px-2 py-0.5 text-sm font-mono text-right focus:outline-none focus:border-indigo-500 transition-colors" min="${min}" max="${max}" step="${step}" value="${initialValue}" />
      </div>
    </div>
    <div id="error-${featureKey}" class="hidden text-red-400 text-[10px] text-right mb-1 -mt-1">Invalid range (${min}-${max})</div>
    <input type="range" id="range-${featureKey}" min="${min}" max="${max}" step="${step}" value="${initialValue}" class="w-full" />
    <div class="flex justify-between mt-1.5">
      <span class="text-xs text-gray-600">${formatValue(min, format)}</span>
      <span class="text-xs text-gray-600">${formatValue(max, format)}</span>
    </div>
  `;
  return container;
}

function createGmatComposite(gmatCfg, gwuValues) {
  const wrap = document.createElement('div');
  wrap.className = 'slider-container col-span-full';
  wrap.id = 'slider-GMAT_Combined';

  const scaleDefault = gmatCfg.gmat_scale_default || 'old';
  const oldR = gmatCfg.gmat_old_range || { min: 500, max: 800, step: 5 };
  const newR = gmatCfg.gmat_new_range || { min: 505, max: 805, step: 5 };
  const qR   = gmatCfg.gre_q_range || { min: 130, max: 170, step: 1 };
  const vR   = gmatCfg.gre_v_range || { min: 130, max: 170, step: 1 };
  const awR  = gmatCfg.gre_aw_range || { min: 0, max: 6, step: 0.5 };

  const initialGmat = scaleDefault === 'old'
    ? (gwuValues.gmat_old ?? Math.round((oldR.min + oldR.max) / 2))
    : (gwuValues.gmat_new ?? Math.round((newR.min + newR.max) / 2));
  const initQ  = gwuValues.gre_q  ?? Math.round((qR.min + qR.max) / 2);
  const initV  = gwuValues.gre_v  ?? Math.round((vR.min + vR.max) / 2);
  const initAW = gwuValues.gre_aw ?? ((awR.min + awR.max) / 2);
  const initialGreEnabled = !!gmatCfg.gre_default_enabled;

  const initialGmatEnabled = true;
  gmatState = {
    scale: scaleDefault, gmat_score: initialGmat,
    gre_q: initQ, gre_v: initV, gre_aw: initAW,
    gre_enabled: initialGreEnabled,
    gmat_enabled: initialGmatEnabled,
  };

  wrap.innerHTML = `
    <div class="flex items-center justify-between mb-2">
      <label class="text-sm font-medium text-gray-300">GMAT / GRE Score</label>
      <span class="text-sm font-mono font-semibold text-indigo-400 bg-indigo-500/10 px-2.5 py-1 rounded-lg" id="value-GMAT_Combined">—</span>
    </div>

    <div class="pb-2 border-b border-white/5">
      <label class="toggle-switch text-xs mb-2">
        <input type="checkbox" id="gmat-toggle" ${initialGmatEnabled ? 'checked' : ''} />
        <span class="toggle-switch-track"><span class="toggle-switch-knob"></span></span>
        <span class="text-gray-300 font-semibold">Include GMAT input</span>
      </label>
      <div id="gmat-controls" class="${initialGmatEnabled ? '' : 'hidden'}">
        <div class="flex gap-1 mb-2 text-xs">
          <button type="button" data-gmat-scale="old"
            class="gmat-scale-btn px-2 py-1 rounded ${scaleDefault === 'old' ? 'bg-indigo-500/30 text-indigo-200' : 'bg-white/5 text-gray-400'}">Old GMAT (200–800)</button>
          <button type="button" data-gmat-scale="new"
            class="gmat-scale-btn px-2 py-1 rounded ${scaleDefault === 'new' ? 'bg-indigo-500/30 text-indigo-200' : 'bg-white/5 text-gray-400'}">New GMAT (205–805)</button>
        </div>
        <input type="range" id="range-GMAT_Score"
          min="${scaleDefault === 'old' ? oldR.min : newR.min}"
          max="${scaleDefault === 'old' ? oldR.max : newR.max}"
          step="${scaleDefault === 'old' ? oldR.step : newR.step}"
          value="${initialGmat}" class="w-full" />
        <div class="flex justify-between mt-1.5 text-xs text-gray-600">
          <span id="gmat-min-label">${scaleDefault === 'old' ? oldR.min : newR.min}</span>
          <span class="text-indigo-300 font-mono" id="gmat-score-display">${Math.round(initialGmat)}</span>
          <span id="gmat-max-label">${scaleDefault === 'old' ? oldR.max : newR.max}</span>
        </div>
      </div>
    </div>

    <div class="mt-3 pt-2 border-t border-white/5">
      <label class="toggle-switch text-xs">
        <input type="checkbox" id="gre-toggle" ${initialGreEnabled ? 'checked' : ''} />
        <span class="toggle-switch-track"><span class="toggle-switch-knob"></span></span>
        <span class="text-gray-300 font-semibold">Include GRE input</span>
        <span class="text-[10px] text-gray-500">(40% Q + 40% V + 20% AW)</span>
      </label>
      <p class="text-[10px] text-gray-500 mt-1">If both GMAT and GRE are OFF, the cohort floor is used (matches US News missing-data rule).</p>
      <div id="gre-controls" class="mt-2 ${initialGreEnabled ? '' : 'hidden'} space-y-2">
        <div>
          <div class="flex justify-between text-[11px] text-gray-500"><span>GRE Quantitative</span><span class="text-cyan-300 font-mono" id="gre-q-display">${initQ}</span></div>
          <input type="range" id="range-GRE_Q" min="${qR.min}" max="${qR.max}" step="${qR.step}" value="${initQ}" class="w-full" />
        </div>
        <div>
          <div class="flex justify-between text-[11px] text-gray-500"><span>GRE Verbal</span><span class="text-cyan-300 font-mono" id="gre-v-display">${initV}</span></div>
          <input type="range" id="range-GRE_V" min="${vR.min}" max="${vR.max}" step="${vR.step}" value="${initV}" class="w-full" />
        </div>
        <div>
          <div class="flex justify-between text-[11px] text-gray-500"><span>GRE Analytical Writing</span><span class="text-cyan-300 font-mono" id="gre-aw-display">${initAW.toFixed(1)}</span></div>
          <input type="range" id="range-GRE_AW" min="${awR.min}" max="${awR.max}" step="${awR.step}" value="${initAW}" class="w-full" />
        </div>
      </div>
    </div>

    <p class="text-[10px] text-gray-500 mt-2">
      Blended percentile (vs. ${scaleDefault === 'old' ? '2025' : '2025'} cohort) — fed to score model.
    </p>
  `;
  return wrap;
}

function attachGmatHandlers(wrap, gmatCfg) {
  const oldR = gmatCfg.gmat_old_range || { min: 500, max: 800, step: 5 };
  const newR = gmatCfg.gmat_new_range || { min: 505, max: 805, step: 5 };

  const range = wrap.querySelector('#range-GMAT_Score');
  const display = wrap.querySelector('#gmat-score-display');
  const minLabel = wrap.querySelector('#gmat-min-label');
  const maxLabel = wrap.querySelector('#gmat-max-label');
  const buttons = wrap.querySelectorAll('.gmat-scale-btn');
  const gmatToggle = wrap.querySelector('#gmat-toggle');
  const gmatControls = wrap.querySelector('#gmat-controls');
  const greToggle = wrap.querySelector('#gre-toggle');
  const greControls = wrap.querySelector('#gre-controls');
  const valueChip = wrap.querySelector('#value-GMAT_Combined');

  const greQRange = wrap.querySelector('#range-GRE_Q');
  const greVRange = wrap.querySelector('#range-GRE_V');
  const greAWRange = wrap.querySelector('#range-GRE_AW');
  const greQDisp = wrap.querySelector('#gre-q-display');
  const greVDisp = wrap.querySelector('#gre-v-display');
  const greAWDisp = wrap.querySelector('#gre-aw-display');

  const updateBlended = () => {
    const payload = {
      scale: gmatState.scale,
      gmat_score: gmatState.gmat_enabled ? gmatState.gmat_score : null,
      gre_q: gmatState.gre_enabled ? gmatState.gre_q : null,
      gre_v: gmatState.gre_enabled ? gmatState.gre_v : null,
      gre_aw: gmatState.gre_enabled ? gmatState.gre_aw : null,
      gre_enabled: gmatState.gre_enabled,
      gmat_enabled: gmatState.gmat_enabled,
    };
    const blended = computeBlendedGMAT(payload);
    valueChip.textContent = blended.toFixed(1);
    currentValues.GMAT_Combined = payload;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => onChangeCallback && onChangeCallback({ ...currentValues }), 300);
  };

  range.addEventListener('input', e => {
    gmatState.gmat_score = parseFloat(e.target.value);
    display.textContent = Math.round(gmatState.gmat_score);
    updateBlended();
  });

  for (const btn of buttons) {
    btn.addEventListener('click', () => {
      const scale = btn.dataset.gmatScale;
      if (scale === gmatState.scale) return;
      gmatState.scale = scale;
      const r = scale === 'old' ? oldR : newR;
      range.min = r.min; range.max = r.max; range.step = r.step;
      let v = gmatState.gmat_score;
      if (v == null || v < r.min || v > r.max) v = Math.round((r.min + r.max) / 2);
      range.value = v;
      gmatState.gmat_score = v;
      display.textContent = Math.round(v);
      minLabel.textContent = r.min;
      maxLabel.textContent = r.max;
      buttons.forEach(b => {
        const active = b.dataset.gmatScale === scale;
        b.classList.toggle('bg-indigo-500/30', active);
        b.classList.toggle('text-indigo-200', active);
        b.classList.toggle('bg-white/5', !active);
        b.classList.toggle('text-gray-400', !active);
      });
      updateBlended();
    });
  }

  gmatToggle.addEventListener('change', e => {
    gmatState.gmat_enabled = e.target.checked;
    gmatControls.classList.toggle('hidden', !gmatState.gmat_enabled);
    updateBlended();
  });

  greToggle.addEventListener('change', e => {
    gmatState.gre_enabled = e.target.checked;
    greControls.classList.toggle('hidden', !gmatState.gre_enabled);
    updateBlended();
  });

  greQRange.addEventListener('input', e => {
    gmatState.gre_q = parseFloat(e.target.value);
    greQDisp.textContent = Math.round(gmatState.gre_q);
    if (gmatState.gre_enabled) updateBlended();
  });
  greVRange.addEventListener('input', e => {
    gmatState.gre_v = parseFloat(e.target.value);
    greVDisp.textContent = Math.round(gmatState.gre_v);
    if (gmatState.gre_enabled) updateBlended();
  });
  greAWRange.addEventListener('input', e => {
    gmatState.gre_aw = parseFloat(e.target.value);
    greAWDisp.textContent = gmatState.gre_aw.toFixed(1);
    if (gmatState.gre_enabled) updateBlended();
  });

  updateBlended();
}

// ============================================================
// Composite SBP control: 7 industries × {salary, n_reporting}
// ============================================================

function createSBPComposite(sbpCfg, gwuSBP) {
  const wrap = document.createElement('div');
  wrap.className = 'slider-container col-span-full';
  wrap.id = 'slider-SalaryByProfession';

  const occs = sbpCfg.occupations;
  const cohort = sbpCfg.cohort;
  const sld = sbpCfg.slider;

  // Initialize state from GWU's per-occupation values (fallback to cohort means).
  sbpState = {};
  for (const occ of occs) {
    const gw = gwuSBP?.[occ];
    sbpState[occ] = {
      salary: (gw?.salary != null) ? gw.salary : (cohort[occ]?.cohort_mean ?? 130000),
      n: (gw?.n_reporting != null) ? gw.n_reporting : 0,
    };
  }

  const rows = occs.map(occ => {
    const cm = cohort[occ]?.cohort_mean;
    const cmStr = cm ? `cohort $${Math.round(cm).toLocaleString()}` : 'no cohort data';
    const init = sbpState[occ];
    const safeKey = occ.replace(/[^a-z0-9]+/gi, '_');
    return `
      <div class="rounded-lg p-2" style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);">
        <div class="flex items-center justify-between mb-1">
          <p class="text-xs font-semibold text-white">${occ}</p>
          <span class="text-[10px] text-gray-500">${cmStr}</span>
        </div>
        <div class="sbp-row">
          <input type="number" id="sbp-sal-input-${safeKey}" data-occ="${occ}" data-role="salary-input"
                 class="sbp-salary-input"
                 min="${sld.min}" max="${sld.max}" step="${sld.step}" value="${Math.round(init.salary)}" />
          <input type="range" id="sbp-sal-${safeKey}" data-occ="${occ}" data-role="salary"
                 min="${sld.min}" max="${sld.max}" step="${sld.step}" value="${init.salary}" />
          <input type="number" id="sbp-n-${safeKey}" data-occ="${occ}" data-role="n"
                 min="0" max="200" step="1" value="${init.n}"
                 class="bg-ink-800 border border-indigo-500/30 text-indigo-300 rounded px-1 py-0.5 text-[10px] font-mono text-right" title="number of reporting graduates" style="width: 100%;" />
        </div>
        <div class="flex justify-between mt-0.5">
          <span class="text-[9px] text-gray-500">salary $</span>
          <span class="text-[9px] text-gray-500">n reporting (≥3 to count)</span>
        </div>
      </div>
    `;
  }).join('');

  wrap.innerHTML = `
    <div class="flex items-center justify-between mb-2">
      <div>
        <label class="text-sm font-medium text-gray-300">Salary by Profession</label>
        <p class="text-[10px] text-gray-500">Per-industry salary × cohort ratio (drops industries with <3 reporters).</p>
      </div>
      <span class="text-sm font-mono font-semibold text-indigo-400 bg-indigo-500/10 px-2.5 py-1 rounded-lg" id="value-SalaryByProfession">—</span>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-1.5">
      ${rows}
    </div>
  `;
  return wrap;
}

function attachSBPHandlers(wrap) {
  const chip = wrap.querySelector('#value-SalaryByProfession');
  const update = () => {
    const ratio = computeSBPRatio(sbpState);
    chip.textContent = ratio !== null ? ratio.toFixed(3) : '—';
    currentValues.SalaryByProfession = { ...sbpState };
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => onChangeCallback && onChangeCallback({ ...currentValues }), 250);
  };
  // Slider drives both: state + paired number-input
  wrap.querySelectorAll('input[data-role="salary"]').forEach(el => {
    el.addEventListener('input', e => {
      const occ = e.target.dataset.occ;
      const v = parseFloat(e.target.value);
      sbpState[occ].salary = v;
      const safeKey = occ.replace(/[^a-z0-9]+/gi, '_');
      const numInput = wrap.querySelector(`#sbp-sal-input-${safeKey}`);
      if (numInput) numInput.value = Math.round(v);
      update();
    });
  });
  // Number input on the LEFT — keystroke commits when valid
  wrap.querySelectorAll('input[data-role="salary-input"]').forEach(el => {
    const commit = (clamp = false) => {
      const occ = el.dataset.occ;
      const safeKey = occ.replace(/[^a-z0-9]+/gi, '_');
      const slider = wrap.querySelector(`#sbp-sal-${safeKey}`);
      let v = parseFloat(el.value);
      if (isNaN(v)) return;
      const min = parseFloat(slider.min);
      const max = parseFloat(slider.max);
      if (clamp) v = Math.max(min, Math.min(max, v));
      sbpState[occ].salary = v;
      slider.value = Math.max(min, Math.min(max, v));
      update();
    };
    el.addEventListener('input', () => commit(false));
    el.addEventListener('blur', () => commit(true));
    el.addEventListener('keydown', e => { if (e.key === 'Enter') el.blur(); });
  });
  wrap.querySelectorAll('input[data-role="n"]').forEach(el => {
    el.addEventListener('input', e => {
      const occ = e.target.dataset.occ;
      const v = parseInt(e.target.value, 10);
      sbpState[occ].n = isNaN(v) ? 0 : Math.max(0, v);
      update();
    });
  });
  update();
}

export function initSliders(containerId, onChange) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const ranges = getFeatureRanges();
  const gwuValues = getGWUValues();
  const gmatCfg = getGmatInputConfig();
  onChangeCallback = onChange;

  const sbpCfg = getSbpConfig();
  for (const key of FEATURE_ORDER) {
    if (key === 'GMAT_Combined' || key === 'SalaryByProfession') continue;
    currentValues[key] = gwuValues[key] ?? ranges[key]?.data_median ?? ranges[key]?.min ?? 0;
  }
  currentValues.GMAT_Combined = null;
  currentValues.SalaryByProfession = null;

  for (const key of FEATURE_ORDER) {
    if (key === 'GMAT_Combined') {
      const composite = createGmatComposite(gmatCfg, gwuValues);
      container.appendChild(composite);
      attachGmatHandlers(composite, gmatCfg);
      continue;
    }
    if (key === 'SalaryByProfession') {
      const composite = createSBPComposite(sbpCfg, gwuValues.sbp_per_occupation || {});
      container.appendChild(composite);
      attachSBPHandlers(composite);
      continue;
    }
    const config = ranges[key];
    if (!config) continue;
    const slider = createSlider(key, config, currentValues[key]);
    container.appendChild(slider);

    const rangeInput = slider.querySelector(`#range-${key}`);
    const numInput = slider.querySelector(`#input-${key}`);
    const display = slider.querySelector(`#value-${key}`);
    const errorMsg = slider.querySelector(`#error-${key}`);

    const validateInput = (val) => !isNaN(val) && val >= config.min && val <= config.max;
    const toggleError = (isInvalid) => {
      if (isInvalid) {
        errorMsg.classList.remove('hidden');
        numInput.classList.remove('border-indigo-500/50', 'text-indigo-400', 'focus:border-indigo-500');
        numInput.classList.add('border-red-500', 'text-red-400', 'focus:border-red-500');
      } else {
        errorMsg.classList.add('hidden');
        numInput.classList.add('border-indigo-500/50', 'text-indigo-400', 'focus:border-indigo-500');
        numInput.classList.remove('border-red-500', 'text-red-400', 'focus:border-red-500');
      }
    };
    const updateFromValue = (val) => {
      val = Math.max(config.min, Math.min(config.max, val));
      currentValues[key] = val;
      rangeInput.value = val;
      numInput.value = val;
      display.textContent = formatValue(val, config.format);
      toggleError(false);
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => onChangeCallback && onChangeCallback({ ...currentValues }), 300);
    };
    rangeInput.addEventListener('input', (e) => updateFromValue(parseFloat(e.target.value)));
    numInput.addEventListener('input', (e) => toggleError(!validateInput(parseFloat(e.target.value))));
    display.addEventListener('click', () => {
      display.classList.add('hidden');
      numInput.classList.remove('hidden');
      numInput.focus();
    });
    const commitInput = () => {
      const val = parseFloat(numInput.value);
      if (validateInput(val)) updateFromValue(val);
      else {
        numInput.value = currentValues[key];
        toggleError(false);
      }
      numInput.classList.add('hidden');
      display.classList.remove('hidden');
    };
    numInput.addEventListener('blur', commitInput);
    numInput.addEventListener('keydown', e => { if (e.key === 'Enter') numInput.blur(); });
  }

  return { ...currentValues };
}

export function resetSliders() {
  const ranges = getFeatureRanges();
  const gwuValues = getGWUValues();
  const gmatCfg = getGmatInputConfig();

  const sbpCfg = getSbpConfig();
  for (const key of FEATURE_ORDER) {
    if (key === 'GMAT_Combined' || key === 'SalaryByProfession') continue;
    const config = ranges[key];
    if (!config) continue;
    const value = gwuValues[key] ?? config.data_median ?? config.min;
    currentValues[key] = value;
    const r = document.getElementById(`range-${key}`);
    if (r) r.value = value;
    const n = document.getElementById(`input-${key}`);
    if (n) n.value = value;
    const d = document.getElementById(`value-${key}`);
    if (d) d.textContent = formatValue(value, config.format);
  }

  // Reset SBP composite
  const sbpWrap = document.getElementById('slider-SalaryByProfession');
  if (sbpWrap) {
    const gwuSBP = gwuValues.sbp_per_occupation || {};
    sbpState = {};
    for (const occ of (sbpCfg.occupations || [])) {
      const gw = gwuSBP[occ];
      sbpState[occ] = {
        salary: gw?.salary ?? sbpCfg.cohort[occ]?.cohort_mean ?? 130000,
        n: gw?.n_reporting ?? 0,
      };
      const safeKey = occ.replace(/[^a-z0-9]+/gi, '_');
      const sal = sbpWrap.querySelector(`#sbp-sal-${safeKey}`);
      const salNum = sbpWrap.querySelector(`#sbp-sal-input-${safeKey}`);
      const nIn = sbpWrap.querySelector(`#sbp-n-${safeKey}`);
      if (sal) sal.value = sbpState[occ].salary;
      if (salNum) salNum.value = Math.round(sbpState[occ].salary);
      if (nIn) nIn.value = sbpState[occ].n;
    }
    const ratio = computeSBPRatio(sbpState);
    sbpWrap.querySelector('#value-SalaryByProfession').textContent = ratio !== null ? ratio.toFixed(3) : '—';
    currentValues.SalaryByProfession = { ...sbpState };
  }

  const wrap = document.getElementById('slider-GMAT_Combined');
  if (wrap) {
    const scaleDefault = gmatCfg.gmat_scale_default || 'old';
    const oldR = gmatCfg.gmat_old_range || { min: 500, max: 800, step: 5 };
    const newR = gmatCfg.gmat_new_range || { min: 505, max: 805, step: 5 };
    const r = scaleDefault === 'old' ? oldR : newR;
    const initGmat = scaleDefault === 'old'
      ? (gwuValues.gmat_old ?? Math.round((oldR.min + oldR.max) / 2))
      : (gwuValues.gmat_new ?? Math.round((newR.min + newR.max) / 2));
    const qR = gmatCfg.gre_q_range, vR = gmatCfg.gre_v_range, awR = gmatCfg.gre_aw_range;
    const initQ = gwuValues.gre_q ?? Math.round((qR.min + qR.max) / 2);
    const initV = gwuValues.gre_v ?? Math.round((vR.min + vR.max) / 2);
    const initAW = gwuValues.gre_aw ?? ((awR.min + awR.max) / 2);
    const initialGreEnabled = !!gmatCfg.gre_default_enabled;

    gmatState = { scale: scaleDefault, gmat_score: initGmat,
                  gre_q: initQ, gre_v: initV, gre_aw: initAW,
                  gre_enabled: initialGreEnabled, gmat_enabled: true };

    const range = wrap.querySelector('#range-GMAT_Score');
    range.min = r.min; range.max = r.max; range.step = r.step; range.value = initGmat;
    wrap.querySelector('#gmat-score-display').textContent = Math.round(initGmat);
    wrap.querySelector('#gmat-min-label').textContent = r.min;
    wrap.querySelector('#gmat-max-label').textContent = r.max;
    wrap.querySelectorAll('.gmat-scale-btn').forEach(b => {
      const active = b.dataset.gmatScale === scaleDefault;
      b.classList.toggle('bg-indigo-500/30', active);
      b.classList.toggle('text-indigo-200', active);
      b.classList.toggle('bg-white/5', !active);
      b.classList.toggle('text-gray-400', !active);
    });
    wrap.querySelector('#gre-toggle').checked = initialGreEnabled;
    wrap.querySelector('#gre-controls').classList.toggle('hidden', !initialGreEnabled);
    wrap.querySelector('#range-GRE_Q').value = initQ;
    wrap.querySelector('#range-GRE_V').value = initV;
    wrap.querySelector('#range-GRE_AW').value = initAW;
    wrap.querySelector('#gre-q-display').textContent = Math.round(initQ);
    wrap.querySelector('#gre-v-display').textContent = Math.round(initV);
    wrap.querySelector('#gre-aw-display').textContent = initAW.toFixed(1);

    // Reset GMAT toggle to ON (default)
    const gmatTog = wrap.querySelector('#gmat-toggle');
    if (gmatTog) gmatTog.checked = true;
    const gmatCtrl = wrap.querySelector('#gmat-controls');
    if (gmatCtrl) gmatCtrl.classList.remove('hidden');

    const payload = {
      scale: scaleDefault, gmat_score: initGmat,
      gre_q: initialGreEnabled ? initQ : null,
      gre_v: initialGreEnabled ? initV : null,
      gre_aw: initialGreEnabled ? initAW : null,
      gre_enabled: initialGreEnabled,
      gmat_enabled: true,
    };
    const blended = computeBlendedGMAT(payload);
    wrap.querySelector('#value-GMAT_Combined').textContent = blended.toFixed(1);
    currentValues.GMAT_Combined = payload;
  }

  if (onChangeCallback) onChangeCallback({ ...currentValues });
}

export function getCurrentValues() { return { ...currentValues }; }
