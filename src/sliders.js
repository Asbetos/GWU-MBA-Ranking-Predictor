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
    <div class="flex items-center justify-between mb-3">
      <label class="text-sm font-medium text-gray-300" for="range-${featureKey}">${label}</label>
      <span class="text-sm font-mono font-semibold text-indigo-400 bg-indigo-500/10 px-2.5 py-1 rounded-lg" id="value-${featureKey}">${displayVal}</span>
    </div>
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

    // Attach event listener
    const input = slider.querySelector(`#range-${key}`);
    input.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      currentValues[key] = val;

      // Update display value
      const display = document.getElementById(`value-${key}`);
      if (display) {
        display.textContent = formatValue(val, config.format);
      }

      // Debounced callback
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        if (onChangeCallback) {
          onChangeCallback({ ...currentValues });
        }
      }, 300);
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

    const input = document.getElementById(`range-${key}`);
    if (input) {
      input.value = value;
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
