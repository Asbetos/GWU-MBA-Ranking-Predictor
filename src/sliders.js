/**
 * Slider component — creates and manages the 8 interactive range sliders.
 */

import { getFeatureRanges, getGWUValues, getGmatInputConfig, computeBlendedGMAT } from './model.js';

const FEATURE_ORDER = [
  'EmployedAtGrad', 'Employed3Mo', 'AvgSalaryBonus', 'MedianGPA',
  'AcceptanceRate', 'PeerScore', 'RecruiterScore', 'GMAT_Combined'
];

let currentValues = {};
// Composite GMAT control state (separate from the model-facing GMAT_Combined value).
let gmatState = { scale: 'old', gmat_score: null, gre_total: null, gre_enabled: false };
let onChangeCallback = null;
let debounceTimer = null;

/** Format a value for display based on its format type */
function formatValue(value, format) {
  switch (format) {
    case 'percent':
      return `${(value * 100).toFixed(1)}%`;
    case 'dollar':
      return `$${Math.round(value).toLocaleString()}`;
    case 'number':
    default:
      return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }
}

/** Create a single slider element */
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
    <input
      type="range"
      id="range-${featureKey}"
      min="${min}"
      max="${max}"
      step="${step}"
      value="${initialValue}"
      class="w-full"
    />
    <div class="flex justify-between mt-1.5">
      <span class="text-xs text-gray-600">${formatValue(min, format)}</span>
      <span class="text-xs text-gray-600">${formatValue(max, format)}</span>
    </div>
  `;

  return container;
}

/** Build the composite GMAT/GRE control (scale toggle + score slider + optional GRE) */
function createGmatComposite(gmatCfg, gwuValues) {
  const wrap = document.createElement('div');
  wrap.className = 'slider-container md:col-span-2';
  wrap.id = 'slider-GMAT_Combined';

  const scaleDefault = gmatCfg.gmat_scale_default || 'old';
  const oldRange = gmatCfg.gmat_old_range || { min: 500, max: 800, step: 5 };
  const newRange = gmatCfg.gmat_new_range || { min: 505, max: 805, step: 5 };
  const greRange = gmatCfg.gre_range || { min: 280, max: 340, step: 1 };

  const initialScale = scaleDefault;
  const initialGmat = initialScale === 'old'
    ? (gwuValues.gmat_old ?? Math.round((oldRange.min + oldRange.max) / 2))
    : (gwuValues.gmat_new ?? Math.round((newRange.min + newRange.max) / 2));
  const initialGre = gwuValues.gre_total ?? Math.round((greRange.min + greRange.max) / 2);
  const initialGreEnabled = !!gmatCfg.gre_default_enabled;

  gmatState = {
    scale: initialScale,
    gmat_score: initialGmat,
    gre_total: initialGreEnabled ? initialGre : null,
    gre_enabled: initialGreEnabled,
  };

  wrap.innerHTML = `
    <div class="flex items-center justify-between mb-2">
      <label class="text-sm font-medium text-gray-300">GMAT / GRE Score</label>
      <span class="text-sm font-mono font-semibold text-indigo-400 bg-indigo-500/10 px-2.5 py-1 rounded-lg" id="value-GMAT_Combined">—</span>
    </div>
    <div class="flex gap-1 mb-2 text-xs">
      <button type="button" data-gmat-scale="old"
        class="gmat-scale-btn px-2 py-1 rounded ${initialScale === 'old' ? 'bg-indigo-500/30 text-indigo-200' : 'bg-white/5 text-gray-400'}">Old GMAT (200–800)</button>
      <button type="button" data-gmat-scale="new"
        class="gmat-scale-btn px-2 py-1 rounded ${initialScale === 'new' ? 'bg-indigo-500/30 text-indigo-200' : 'bg-white/5 text-gray-400'}">New GMAT (205–805)</button>
    </div>
    <input type="range"
      id="range-GMAT_Score"
      min="${initialScale === 'old' ? oldRange.min : newRange.min}"
      max="${initialScale === 'old' ? oldRange.max : newRange.max}"
      step="${initialScale === 'old' ? oldRange.step : newRange.step}"
      value="${initialGmat}" class="w-full" />
    <div class="flex justify-between mt-1.5 text-xs text-gray-600">
      <span id="gmat-min-label">${initialScale === 'old' ? oldRange.min : newRange.min}</span>
      <span class="text-indigo-300 font-mono" id="gmat-score-display">${initialGmat}</span>
      <span id="gmat-max-label">${initialScale === 'old' ? oldRange.max : newRange.max}</span>
    </div>

    <div class="mt-4 pt-3 border-t border-white/5">
      <label class="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
        <input type="checkbox" id="gre-toggle" ${initialGreEnabled ? 'checked' : ''} class="accent-indigo-500" />
        Include GRE (total verbal+quant)
      </label>
      <div id="gre-controls" class="mt-2 ${initialGreEnabled ? '' : 'hidden'}">
        <input type="range" id="range-GRE_Total"
          min="${greRange.min}" max="${greRange.max}" step="${greRange.step}"
          value="${initialGre}" class="w-full" />
        <div class="flex justify-between mt-1.5 text-xs text-gray-600">
          <span>${greRange.min}</span>
          <span class="text-cyan-300 font-mono" id="gre-score-display">${initialGre}</span>
          <span>${greRange.max}</span>
        </div>
      </div>
    </div>

    <p class="text-[10px] text-gray-500 mt-2">
      Blended percentile (vs. ${scaleDefault === 'old' ? '2024–2025' : ''} cohort) — fed to score model.
    </p>
  `;

  return wrap;
}

/** Wire up the composite GMAT control */
function attachGmatHandlers(wrap, gmatCfg) {
  const oldRange = gmatCfg.gmat_old_range || { min: 500, max: 800, step: 5 };
  const newRange = gmatCfg.gmat_new_range || { min: 505, max: 805, step: 5 };

  const range = wrap.querySelector('#range-GMAT_Score');
  const display = wrap.querySelector('#gmat-score-display');
  const minLabel = wrap.querySelector('#gmat-min-label');
  const maxLabel = wrap.querySelector('#gmat-max-label');
  const buttons = wrap.querySelectorAll('.gmat-scale-btn');
  const greToggle = wrap.querySelector('#gre-toggle');
  const greControls = wrap.querySelector('#gre-controls');
  const greRange = wrap.querySelector('#range-GRE_Total');
  const greDisplay = wrap.querySelector('#gre-score-display');
  const valueChip = wrap.querySelector('#value-GMAT_Combined');

  const updateBlendedDisplay = () => {
    const blended = computeBlendedGMAT({
      scale: gmatState.scale,
      gmat_score: gmatState.gmat_score,
      gre_total: gmatState.gre_enabled ? gmatState.gre_total : null,
    });
    valueChip.textContent = blended.toFixed(1);
    currentValues.GMAT_Combined = {
      scale: gmatState.scale,
      gmat_score: gmatState.gmat_score,
      gre_total: gmatState.gre_enabled ? gmatState.gre_total : null,
    };
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (onChangeCallback) onChangeCallback({ ...currentValues });
    }, 300);
  };

  range.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    gmatState.gmat_score = v;
    display.textContent = Math.round(v);
    updateBlendedDisplay();
  });

  for (const btn of buttons) {
    btn.addEventListener('click', () => {
      const scale = btn.dataset.gmatScale;
      if (scale === gmatState.scale) return;
      gmatState.scale = scale;
      const r = scale === 'old' ? oldRange : newRange;
      range.min = r.min;
      range.max = r.max;
      range.step = r.step;
      // Snap into range, default to midpoint if old value is outside
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
      updateBlendedDisplay();
    });
  }

  greToggle.addEventListener('change', (e) => {
    gmatState.gre_enabled = e.target.checked;
    greControls.classList.toggle('hidden', !gmatState.gre_enabled);
    updateBlendedDisplay();
  });

  greRange.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    gmatState.gre_total = v;
    greDisplay.textContent = Math.round(v);
    if (gmatState.gre_enabled) updateBlendedDisplay();
  });

  // Initial paint
  updateBlendedDisplay();
}

/** Initialize all sliders */
export function initSliders(containerId, onChange) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const ranges = getFeatureRanges();
  const gwuValues = getGWUValues();
  const gmatCfg = getGmatInputConfig();
  onChangeCallback = onChange;

  // Initialize current values from GWU defaults
  for (const key of FEATURE_ORDER) {
    if (key === 'GMAT_Combined') continue; // handled by composite control
    currentValues[key] = gwuValues[key] ?? ranges[key]?.data_median ?? ranges[key]?.min ?? 0;
  }
  // Placeholder: will be set by attachGmatHandlers's initial paint
  currentValues.GMAT_Combined = null;

  // Create slider elements
  for (const key of FEATURE_ORDER) {
    if (key === 'GMAT_Combined') {
      const composite = createGmatComposite(gmatCfg, gwuValues);
      container.appendChild(composite);
      attachGmatHandlers(composite, gmatCfg);
      continue;
    }

    const config = ranges[key];
    if (!config) continue;

    const slider = createSlider(key, config, currentValues[key]);
    container.appendChild(slider);

    // Attach event listeners
    const rangeInput = slider.querySelector(`#range-${key}`);
    const numInput = slider.querySelector(`#input-${key}`);
    const display = slider.querySelector(`#value-${key}`);
    const errorMsg = slider.querySelector(`#error-${key}`);

    const validateInput = (val) => {
      return !isNaN(val) && val >= config.min && val <= config.max;
    };

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
      // Validate clamp
      val = Math.max(config.min, Math.min(config.max, val));
      
      currentValues[key] = val;
      rangeInput.value = val;
      numInput.value = val;
      display.textContent = formatValue(val, config.format);

      toggleError(false);

      // Debounced callback
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        if (onChangeCallback) {
          onChangeCallback({ ...currentValues });
        }
      }, 300);
    };

    rangeInput.addEventListener('input', (e) => {
      updateFromValue(parseFloat(e.target.value));
    });

    numInput.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      toggleError(!validateInput(val));
    });

    display.addEventListener('click', () => {
      display.classList.add('hidden');
      numInput.classList.remove('hidden');
      numInput.focus();
    });

    const commitInput = () => {
      const val = parseFloat(numInput.value);
      if (validateInput(val)) {
        updateFromValue(val);
      } else {
        // revert
        numInput.value = currentValues[key];
        toggleError(false);
      }
      numInput.classList.add('hidden');
      display.classList.remove('hidden');
    };

    numInput.addEventListener('blur', commitInput);
    numInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        numInput.blur();
      }
    });
  }

  return { ...currentValues };
}

/** Reset all sliders to GWU current values */
export function resetSliders() {
  const ranges = getFeatureRanges();
  const gwuValues = getGWUValues();
  const gmatCfg = getGmatInputConfig();

  for (const key of FEATURE_ORDER) {
    if (key === 'GMAT_Combined') continue;
    const config = ranges[key];
    if (!config) continue;

    const value = gwuValues[key] ?? config.data_median ?? config.min;
    currentValues[key] = value;

    const rangeInput = document.getElementById(`range-${key}`);
    if (rangeInput) rangeInput.value = value;
    const numInput = document.getElementById(`input-${key}`);
    if (numInput) numInput.value = value;
    const display = document.getElementById(`value-${key}`);
    if (display) display.textContent = formatValue(value, config.format);
  }

  // Reset composite GMAT control
  const wrap = document.getElementById('slider-GMAT_Combined');
  if (wrap) {
    const scaleDefault = gmatCfg.gmat_scale_default || 'old';
    const oldRange = gmatCfg.gmat_old_range || { min: 500, max: 800, step: 5 };
    const newRange = gmatCfg.gmat_new_range || { min: 505, max: 805, step: 5 };
    const r = scaleDefault === 'old' ? oldRange : newRange;
    const initialGmat = scaleDefault === 'old'
      ? (gwuValues.gmat_old ?? Math.round((oldRange.min + oldRange.max) / 2))
      : (gwuValues.gmat_new ?? Math.round((newRange.min + newRange.max) / 2));
    const greRange = gmatCfg.gre_range || { min: 280, max: 340, step: 1 };
    const initialGre = gwuValues.gre_total ?? Math.round((greRange.min + greRange.max) / 2);
    const initialGreEnabled = !!gmatCfg.gre_default_enabled;

    gmatState = { scale: scaleDefault, gmat_score: initialGmat, gre_total: initialGre, gre_enabled: initialGreEnabled };

    const range = wrap.querySelector('#range-GMAT_Score');
    range.min = r.min; range.max = r.max; range.step = r.step; range.value = initialGmat;
    wrap.querySelector('#gmat-score-display').textContent = Math.round(initialGmat);
    wrap.querySelector('#gmat-min-label').textContent = r.min;
    wrap.querySelector('#gmat-max-label').textContent = r.max;
    wrap.querySelectorAll('.gmat-scale-btn').forEach(b => {
      const active = b.dataset.gmatScale === scaleDefault;
      b.classList.toggle('bg-indigo-500/30', active);
      b.classList.toggle('text-indigo-200', active);
      b.classList.toggle('bg-white/5', !active);
      b.classList.toggle('text-gray-400', !active);
    });
    const greToggle = wrap.querySelector('#gre-toggle');
    greToggle.checked = initialGreEnabled;
    wrap.querySelector('#gre-controls').classList.toggle('hidden', !initialGreEnabled);
    wrap.querySelector('#range-GRE_Total').value = initialGre;
    wrap.querySelector('#gre-score-display').textContent = Math.round(initialGre);

    const blended = computeBlendedGMAT({
      scale: scaleDefault, gmat_score: initialGmat, gre_total: initialGreEnabled ? initialGre : null,
    });
    wrap.querySelector('#value-GMAT_Combined').textContent = blended.toFixed(1);
    currentValues.GMAT_Combined = {
      scale: scaleDefault, gmat_score: initialGmat, gre_total: initialGreEnabled ? initialGre : null,
    };
  }

  if (onChangeCallback) onChangeCallback({ ...currentValues });
}

/** Get current slider values */
export function getCurrentValues() {
  return { ...currentValues };
}
