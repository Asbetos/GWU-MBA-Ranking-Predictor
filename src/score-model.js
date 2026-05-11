/**
 * Direct Model Insights — single-engine (regression) view.
 */

import Chart from 'chart.js/auto';
import { getModelExplainability } from './model.js';

const FEATURE_LABELS = {
  EmployedAtGrad:     'Employed at Graduation',
  Employed3Mo:        'Employed 3 Months After',
  AvgSalaryBonus:     'Avg Salary + Bonus',
  SalaryByProfession: 'Salary by Profession',
  MedianGPA:          'Median GPA',
  AcceptanceRate:     'Acceptance Rate',
  PeerScore:          'Peer Assessment',
  RecruiterScore:     'Recruiter Assessment',
  GMAT_Combined:      'GMAT/GRE Blended',
};

let contribChart = null;
let contribAudience = 'avg';
let inited = false;

function fmt(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(d);
}

function renderPerformance(perf) {
  const grid = document.getElementById('sm-perf-regression');
  if (!grid) return;
  const metrics = [
    {
      label: 'Avg score error (MAE)',
      value: fmt(perf.mae, 2),
      hint: 'How far off the model is on a typical school, on the published 0–100 score.',
    },
    {
      label: 'Worst-case error (RMSE)',
      value: fmt(perf.rmse, 2),
      hint: 'Like average error but more sensitive to big misses. Lower is better.',
    },
    {
      label: 'Variance explained (R²)',
      value: fmt(perf.r2, 3),
      hint: 'Share of the score variation the model captures. 1.00 = perfect; 0 = no better than guessing the mean.',
    },
    {
      label: 'Rank agreement (ρ)',
      value: fmt(perf.spearman, 3),
      hint: "Spearman's rank correlation with US News' published rank order. 1.00 = identical ordering.",
    },
  ];
  grid.innerHTML = metrics.map(m => `
    <div class="bg-white/5 rounded-lg p-3 text-center" title="${m.hint}">
      <p class="text-[11px] uppercase tracking-widest text-gray-500 mb-1">${m.label}</p>
      <p class="text-2xl font-bold text-white">${m.value}</p>
    </div>
  `).join('');
}

function renderCoefTable(data) {
  const wrap = document.getElementById('score-model-coef-table');
  if (!wrap) return;
  const coefs = (data.regression?.coefficients || []).slice().sort((a, b) =>
    Math.abs(b.mean_weight) - Math.abs(a.mean_weight)
  );
  const intercept = data.regression?.intercept;

  const rows = coefs.map(c => `
    <tr class="${c.is_significant ? '' : 'opacity-60'}">
      <td class="py-1.5 px-2 text-sm text-white">${FEATURE_LABELS[c.feature] || c.feature}</td>
      <td class="py-1.5 px-2 text-right font-mono text-sm ${c.mean_weight >= 0 ? 'text-cyan-300' : 'text-pink-300'}">${fmt(c.mean_weight, 3)}</td>
      <td class="py-1.5 px-2 text-right text-[11px] text-gray-500 font-mono">[${fmt(c.lower_95_ci, 2)}, ${fmt(c.upper_95_ci, 2)}]</td>
      <td class="py-1.5 px-2 text-center">${c.is_significant
        ? '<span class="text-emerald-300 text-xs">✓ sig</span>'
        : '<span class="text-gray-500 text-xs">n.s.</span>'}</td>
    </tr>
  `).join('');

  wrap.innerHTML = `
    <table class="w-full text-sm">
      <thead>
        <tr class="text-gray-500 text-[11px] uppercase tracking-widest border-b border-white/5">
          <th class="text-left py-1.5 px-2 font-medium" title="Ranking indicator">Indicator</th>
          <th class="text-right py-1.5 px-2 font-medium" title="Score points added (or removed) for a one-standard-deviation improvement.">Impact weight</th>
          <th class="text-right py-1.5 px-2 font-medium" title="Range of impact-weight values seen across 10,000 re-trainings on resampled data. Tight = confident.">Uncertainty range</th>
          <th class="text-center py-1.5 px-2 font-medium" title="A weight is 'confident' when its uncertainty range stays on one side of zero — i.e. we're sure about the direction.">Confidence</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
      <tfoot>
        <tr class="border-t border-white/5 text-gray-400 text-xs">
          <td class="py-1.5 px-2">Baseline (intercept)</td>
          <td class="py-1.5 px-2 text-right font-mono">${fmt(intercept, 3)}</td>
          <td colspan="2" class="py-1.5 px-2 text-right text-[10px]">calibrated so #1 school = 100</td>
        </tr>
      </tfoot>
    </table>
  `;
}

function buildContribDataset(data) {
  const r = data.regression || {};
  if (contribAudience === 'gwu') {
    return (r.gwu_contribution_pct || []).map(x => ({ feature: x.feature, value: x.signed_pct }));
  }
  return (r.avg_abs_contribution_pct || []).map(x => ({ feature: x.feature, value: x.pct }));
}

function renderContribChart(data) {
  const canvas = document.getElementById('sm-contrib-chart');
  if (!canvas) return;
  const rows = buildContribDataset(data).slice().sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const labels = rows.map(r => FEATURE_LABELS[r.feature] || r.feature);
  const values = rows.map(r => r.value);
  const colors = values.map(v => v >= 0 ? 'rgba(99, 102, 241, 0.85)' : 'rgba(244, 114, 182, 0.85)');

  if (contribChart) contribChart.destroy();
  contribChart = new Chart(canvas, {
    type: 'bar',
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 3 }] },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.95)', callbacks: { label: it => `${it.raw.toFixed(2)}%` } },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { size: 10 }, callback: v => `${v}%` } },
        y: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 11 } } },
      },
    },
  });

  const caption = document.getElementById('sm-contrib-caption');
  if (caption) {
    caption.textContent = contribAudience === 'gwu'
      ? "Bars show what's currently driving GWU's score. Indigo bars are helping, pink bars are hurting. Total absolute share = 100%."
      : 'Bars show how much each indicator typically drives a school\'s score, averaged across all 122 schools. Total share = 100%.';
  }
}

function renderMethodology(data) {
  const txt = document.getElementById('sm-methodology-text');
  if (txt) {
    txt.innerHTML = `
      <strong class="text-cyan-300">The model.</strong>
      A regularised linear regression (ElasticNet) trained to reproduce US News' published overall score from the 9 ranking
      indicators. We trained the model 10,000 times on slightly different slices of the data (a technique called
      <em>bootstrapping</em>) and averaged the results, so we get not just a single weight per indicator but also a sense of
      how stable each weight is. Weights are sign-constrained to match domain direction — every "higher is better" indicator
      gets a non-negative weight, and Acceptance Rate is constrained negative (lower = more selective = better).
      <br/><br/>
      <strong class="text-white">How GMAT and GRE feed in.</strong>
      Each school's mix of GMAT-old, GMAT-new, and GRE submitters is collapsed into a single 0–100 score:
      we rank each test against the cohort (per-year percentile), blend GRE-internal Q/V/AW at the published 40/40/20 weights,
      then average across GMAT-old, GMAT-new and GRE weighted by the school's share of submitters in each.
      If a school reports no test scores, it gets the cohort floor (lowest scoring school).
      <br/><br/>
      <strong class="text-white">Why the rank can be uncertain even when the score is precise.</strong>
      A school's published rank is a function of where its score lands relative to every other school's score.
      Other schools' scores wobble year-to-year because of measurement noise. The Monte Carlo simulation adds that noise
      to every competitor 10,000 times and reports the distribution of resulting ranks — that's the 90% confidence band you
      see on the Direct Predictor tab.
    `;
  }
  const yrEl = document.getElementById('sm-training-years');
  if (yrEl && data.regression?.performance?.training_years) {
    yrEl.textContent = data.regression.performance.training_years.join(', ');
  }
  const nObsEl = document.getElementById('sm-n-obs');
  if (nObsEl && data.regression?.performance?.n_observations) {
    nObsEl.textContent = data.regression.performance.n_observations;
  }
  const itersEl = document.getElementById('sm-bootstrap-iters');
  if (itersEl && data.regression?.performance?.n_bootstrap_iterations) {
    itersEl.textContent = data.regression.performance.n_bootstrap_iterations.toLocaleString();
  }
}

function attachToggleHandlers(data) {
  const avgBtn = document.getElementById('sm-contrib-avg');
  const gwuBtn = document.getElementById('sm-contrib-gwu');
  if (!avgBtn || !gwuBtn) return;
  const setAud = aud => {
    contribAudience = aud;
    avgBtn.classList.toggle('active', aud === 'avg');
    gwuBtn.classList.toggle('active', aud === 'gwu');
    renderContribChart(data);
  };
  avgBtn.addEventListener('click', () => setAud('avg'));
  gwuBtn.addEventListener('click', () => setAud('gwu'));
}

export function renderScoreModel() {
  if (inited) return;
  const data = getModelExplainability();
  if (!data) return;
  renderPerformance(data.regression?.performance || {});
  renderCoefTable(data);
  renderContribChart(data);
  renderMethodology(data);
  attachToggleHandlers(data);
  inited = true;
}

export function initScoreModelTab() {
  document.addEventListener('tab:activated', e => {
    if (e.detail?.tabKey === 'score-model') renderScoreModel();
  });
}
