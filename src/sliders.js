/**
 * Slider component — creates and manages the 8 interactive range sliders.
 */

import { getFeatureRanges, getGWUValues } from './model.js';

const FEATURE_ORDER = [
  'EmployedAtGrad', 'Employed3Mo', 'AvgSalaryBonus', 'MedianGPA',
  'AcceptanceRate', 'PeerScore', 'RecruiterScore', 'GMAT_Combined'
];

let currentValues = {};
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

/** Initialize all sliders */
export function initSliders(containerId, onChange) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const ranges = getFeatureRanges();
  const gwuValues = getGWUValues();
  onChangeCallback = onChange;

  // Initialize current values from GWU defaults
  for (const key of FEATURE_ORDER) {
    currentValues[key] = gwuValues[key] ?? ranges[key]?.data_median ?? ranges[key]?.min ?? 0;
  }

  // Create slider elements
  for (const key of FEATURE_ORDER) {
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

  for (const key of FEATURE_ORDER) {
    const config = ranges[key];
    if (!config) continue;

    const value = gwuValues[key] ?? config.data_median ?? config.min;
    currentValues[key] = value;

    const rangeInput = document.getElementById(`range-${key}`);
    if (rangeInput) {
      rangeInput.value = value;
    }

    const numInput = document.getElementById(`input-${key}`);
    if (numInput) {
      numInput.value = value;
    }

    const display = document.getElementById(`value-${key}`);
    if (display) {
      display.textContent = formatValue(value, config.format);
    }
  }

  if (onChangeCallback) {
    onChangeCallback({ ...currentValues });
  }
}

/** Get current slider values */
export function getCurrentValues() {
  return { ...currentValues };
}
