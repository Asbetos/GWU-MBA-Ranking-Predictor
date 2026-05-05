/**
 * GWU MBA Ranking Predictor — Main Application
 * Bootstraps all three tabs and wires them up.
 */

import { loadModel, simulateRank, getGWUSchoolName, getGWUCurrentRank, getGWUCurrentScore } from './model.js';
import { initSliders, resetSliders } from './sliders.js';
import { updateResults, showCurrentInfo } from './results.js';
import { loadCfmArtifacts } from './cfm-models.js';
import { renderExplainability } from './explainability.js';
import { initLeverPredictor } from './lever-predictor.js';
import { initTabs } from './tabs.js';
import { initScoreModelTab } from './score-model.js';

async function init() {
  try {
    console.log('[init] Loading model artifacts...');
    await Promise.all([loadModel(), loadCfmArtifacts()]);
    console.log('[init] Model + CFM artifacts loaded.');

    // Tab nav first so panes show/hide correctly.
    // Register lazy-init handlers BEFORE initTabs so the initial activation
    // event fires through them.
    initScoreModelTab();
    initTabs();

    // ----- Tab 1 (Direct Predictor) -----
    const schoolName = getGWUSchoolName();
    showCurrentInfo(schoolName, getGWUCurrentRank(), getGWUCurrentScore());

    const initialValues = initSliders('sliders-container', handleSliderChange);
    if (initialValues) handleSliderChange(initialValues);

    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) resetBtn.addEventListener('click', resetSliders);

    // ----- Tab 2 (Performance) -----
    renderExplainability();

    // ----- Tab 3 (Lever Predictor) -----
    initLeverPredictor();

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

init();
