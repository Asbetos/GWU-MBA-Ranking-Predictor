/**
 * GWU MBA Ranking Predictor — Main Application
 * Bootstraps the model, initializes sliders, and wires up the prediction loop.
 */

import { loadModel, simulateRank, getGWUSchoolName, getGWUCurrentRank, getGWUCurrentScore } from './model.js';
import { initSliders, resetSliders } from './sliders.js';
import { updateResults, showCurrentInfo } from './results.js';

async function init() {
  try {
    // 1. Load model artifacts
    console.log('[init] Loading model artifacts...');
    await loadModel();
    console.log('[init] Model loaded successfully.');

    // 2. Show current GWU info
    const schoolName = getGWUSchoolName();
    const currentRank = getGWUCurrentRank();
    const currentScore = getGWUCurrentScore();
    showCurrentInfo(schoolName, currentRank, currentScore);

    // 3. Initialize sliders with change callback
    const initialValues = initSliders('sliders-container', handleSliderChange);

    // 4. Run initial simulation
    if (initialValues) {
      handleSliderChange(initialValues);
    }

    // 5. Wire reset button
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
      resetBtn.addEventListener('click', resetSliders);
    }

    console.log('[init] App ready.');
  } catch (err) {
    console.error('[init] Failed to initialize:', err);
    showError(err.message);
  }
}

function handleSliderChange(values) {
  try {
    const results = simulateRank(values);
    updateResults(results);
  } catch (err) {
    console.error('[predict] Simulation error:', err);
  }
}

function showError(message) {
  const rankEl = document.getElementById('rank-display');
  if (rankEl) {
    rankEl.textContent = '⚠';
    rankEl.style.fontSize = '4rem';
  }
  const subtitle = document.getElementById('rank-subtitle');
  if (subtitle) {
    subtitle.textContent = `Error: ${message}`;
    subtitle.style.color = '#ef4444';
  }
}

// Boot the app
init();
