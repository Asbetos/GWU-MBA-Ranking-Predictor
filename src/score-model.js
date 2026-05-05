/**
 * Tab 4 - Score Model Insights.
 * Renders performance metrics, coefficient table, and per-feature contribution
 * chart for the main bootstrapped ElasticNet OverallScore model.
 */

import Chart from 'chart.js/auto';
import { getModelExplainability } from './model.js';

const FEATURE_LABELS = {
  EmployedAtGrad: 'Employed at Graduation',
  Employed3Mo: 'Employed 3 Months After',
  AvgSalaryBonus: 'Avg Salary + Bonus',
  MedianGPA: 'Median GPA',
  AcceptanceRate: 'Acceptance Rate',
  PeerScore: 'Peer Assessment',
  RecruiterScore: 'Recruiter Assessment',
  GMAT_Combined: 'GMAT/GRE Blended Percentile',
};

let contribChart = null;
let contribMode = 'avg'; // 'avg' or 'gwu'
let inited = false;

function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}

function renderPerformance(data) {
  const grid = document.getElementById('score-model-perf');
  if (!grid) return;
  const p = data.performance || {};
  const metrics = [
    { label: 'MAE', value: fmt(p.mae, 2), help: 'Mean Absolute Error (score points)' },
    { label: 'RMSE', value: fmt(p.rmse, 2), help: 'Root Mean Squared Error' },
    { label: 'R²', value: fmt(p.r2, 3), help: 'Variance explained' },
    { label: 'Spearman ρ', value: fmt(p.spearman, 3), help: 'Rank correlation' },
  ];
  grid.innerHTML = metrics.map(m => `
    <div class="glass-panel p-4 text-center">
      <p class="text-[11px] uppercase tracking-widest text-gray-500 mb-1">${m.label}</p>
      <p class="text-3xl font-bold text-white">${m.value}</p>
      <p class="text-[10px] text-gray-500 mt-1">${m.help}</p>
    </div>
  `).join('');
}

function ciBadge(lo, hi, sig) {
  const cls = sig ? 'text-emerald-300' : 'text-gray-500';
  return `<span class="${cls} font-mono text-xs">[${fmt(lo, 2)}, ${fmt(hi, 2)}]</span>`;
}

function renderCoefTable(data) {
  const wrap = document.getElementById('score-model-coef-table');
  if (!wrap) return;
  const coefs = (data.coefficients || []).slice().sort((a, b) =>
    Math.abs(b.mean_weight) - Math.abs(a.mean_weight)
  );
  const intercept = data.intercept;
  const rows = coefs.map(c => `
    <tr class="${c.is_significant ? '' : 'opacity-60'}">
      <td class="py-1.5 px-2 text-sm text-white">${FEATURE_LABELS[c.feature] || c.feature}</td>
      <td class="py-1.5 px-2 text-right font-mono text-sm ${c.mean_weight >= 0 ? 'text-indigo-300' : 'text-pink-300'}">${fmt(c.mean_weight, 3)}</td>
      <td class="py-1.5 px-2 text-right">${ciBadge(c.lower_95_ci, c.upper_95_ci, c.is_significant)}</td>
      <td class="py-1.5 px-2 text-center">${c.is_significant
        ? '<span class="text-emerald-300 text-xs">✓ sig</span>'
        : '<span class="text-gray-500 text-xs">n.s.</span>'}</td>
    </tr>
  `).join('');

  wrap.innerHTML = `
    <table class="w-full text-sm">
      <thead>
        <tr class="text-gray-500 text-[11px] uppercase tracking-widest border-b border-white/5">
          <th class="text-left py-1.5 px-2 font-medium">Feature</th>
          <th class="text-right py-1.5 px-2 font-medium">Mean weight</th>
          <th class="text-right py-1.5 px-2 font-medium">95% CI</th>
          <th class="text-center py-1.5 px-2 font-medium">Significance</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
      <tfoot>
        <tr class="border-t border-white/5 text-gray-400 text-xs">
          <td class="py-1.5 px-2">Intercept</td>
          <td class="py-1.5 px-2 text-right font-mono">${fmt(intercept, 3)}</td>
          <td colspan="2" class="py-1.5 px-2 text-right text-[10px]">calibrated so #1 school = 100</td>
        </tr>
      </tfoot>
    </table>
  `;
}

function buildContribDataset(data, mode) {
  if (mode === 'gwu') {
    const arr = data.gwu_contribution_pct || [];
    return arr.map(r => ({ feature: r.feature, value: r.signed_pct }));
  }
  const arr = data.avg_abs_contribution_pct || [];
  return arr.map(r => ({ feature: r.feature, value: r.pct }));
}

function renderContribChart(data) {
  const canvas = document.getElementById('sm-contrib-chart');
  if (!canvas) return;

  const rows = buildContribDataset(data, contribMode)
    .slice()
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const labels = rows.map(r => FEATURE_LABELS[r.feature] || r.feature);
  const values = rows.map(r => r.value);
  const colors = values.map(v => v >= 0 ? 'rgba(99, 102, 241, 0.85)' : 'rgba(244, 114, 182, 0.85)');

  if (contribChart) contribChart.destroy();
  contribChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: colors, borderRadius: 3 }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          callbacks: {
            label: (item) => `${item.raw.toFixed(2)}%`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#64748b', font: { size: 10 }, callback: (v) => `${v}%` },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 11 } },
        },
      },
    },
  });

  const caption = document.getElementById('sm-contrib-caption');
  if (caption) {
    caption.textContent = contribMode === 'gwu'
      ? 'Signed contributions for GWU\'s most recent prediction. Positive = pushed score up; negative = pulled down. Bars sum to 100% of |contribution|.'
      : 'Average |contribution| across all schools, normalized to sum to 100%. Highlights which features drive most of the ranking variation.';
  }
}

function renderMethodology(data) {
  const m = data.methodology || {};
  const txt = document.getElementById('sm-methodology-text');
  if (txt) {
    txt.innerHTML = `
      <strong class="text-white">GMAT/GRE blend:</strong> ${m.gmat_blend || ''}<br/>
      <strong class="text-white mt-1 inline-block">Training:</strong> ${m.training || ''}
    `;
  }
  const yrEl = document.getElementById('sm-training-years');
  if (yrEl && data.performance?.training_years) {
    yrEl.textContent = data.performance.training_years.join(', ');
  }
  const nObsEl = document.getElementById('sm-n-obs');
  if (nObsEl && data.performance?.n_observations) {
    nObsEl.textContent = data.performance.n_observations;
  }
  const itersEl = document.getElementById('sm-bootstrap-iters');
  if (itersEl && data.performance?.n_bootstrap_iterations) {
    itersEl.textContent = data.performance.n_bootstrap_iterations.toLocaleString();
  }
}

function attachToggleHandlers(data) {
  const avgBtn = document.getElementById('sm-contrib-avg');
  const gwuBtn = document.getElementById('sm-contrib-gwu');
  if (!avgBtn || !gwuBtn) return;
  const setMode = (mode) => {
    contribMode = mode;
    avgBtn.classList.toggle('active', mode === 'avg');
    gwuBtn.classList.toggle('active', mode === 'gwu');
    renderContribChart(data);
  };
  avgBtn.addEventListener('click', () => setMode('avg'));
  gwuBtn.addEventListener('click', () => setMode('gwu'));
}

export function renderScoreModel() {
  if (inited) return;
  const data = getModelExplainability();
  if (!data) {
    console.warn('[score-model] explainability artifact missing');
    return;
  }
  renderPerformance(data);
  renderCoefTable(data);
  renderContribChart(data);
  renderMethodology(data);
  attachToggleHandlers(data);
  inited = true;
}

/** Lazy-init: render once when the score-model tab first becomes visible. */
export function initScoreModelTab() {
  document.addEventListener('tab:activated', (e) => {
    if (e.detail?.tabKey === 'score-model') renderScoreModel();
  });
}
