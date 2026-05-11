/**
 * Tab 4 — Indirect Model Insights.
 * Renders one card per core-feature model with metrics, confidence pill,
 * signed-coefficient bar chart, and inline feature-contribution % bars.
 */

import Chart from 'chart.js/auto';
import { getCoreTargetSummaries, getCfmTopFeatures } from './cfm-models.js';

const FEATURE_LABEL_OVERRIDES = {
  // Make the longer raw column names a bit kinder to the eye.
  'admissions_and_enrollment.average_age_of_new_entrants': 'avg age of new entrants',
  'admissions_and_enrollment.average_work_experience_months': 'avg work experience (mo)',
  'admissions_and_enrollment.students_with_prior_work_experience_count': 'entrants w/ prior work exp.',
  'admissions_and_enrollment.total_applicants': 'total applicants',
  'application_info.application_fee': 'application fee',
  'application_info.test_optional_admissions': 'test-optional admissions',
  'gpa_data.percent_new_entrants_providing_gpa': '% submitting GPA',
  'gmat_data.percent_new_entrants_providing_gmat_old': '% submitting old GMAT',
  'gmat_data.percent_new_entrants_providing_gmat_new': '% submitting new GMAT',
  'gre_data.gre_accepted_for_admissions': 'GRE accepted',
  'gre_data.percent_new_entrants_providing_gre': '% submitting GRE',
  'toefl_ielts_data.minimum_ielts_score_required': 'min IELTS required',
  'tuition_and_fees.full_time_tuition_per_year_out_of_state': 'FT tuition (OOS)',
  'tuition_and_fees.food_housing_books_misc_full_time_mba': 'FT MBA living costs',
  'student_body_fulltime_mba.male_percent': '% male (FT MBA)',
  'student_body_fulltime_mba.female_percent': '% female (FT MBA)',
  'student_body_fulltime_mba.international_students_percent': '% international (FT MBA)',
  'school_info.graduate_enrollment': 'graduate enrollment',
  'school_info.full_time_degree_seeking_percent': '% FT degree seeking',
  'school_info.school_type': 'school type (public=0/priv=1)',
  'school_info.total_enrollment': 'total enrollment',
  'school_info.total_enrollment_all_programs': 'total enrollment (all)',
  'specialty_masters_admissions.application_fee': 'spec. masters app fee',
  'specialty_masters_admissions.percent_providing_gpa': 'spec. masters % GPA',
  'specialty_masters_admissions.average_work_experience_months': 'spec. masters work exp.',
  'specialty_masters_admissions.average_age_of_new_entrants': 'spec. masters avg age',
  'specialty_masters_admissions.students_with_prior_work_experience': 'spec. masters w/ work exp.',
  'student_body_fulltime_mba.race_ethnicity.international': '% international (race)',
  'student_body_fulltime_mba.race_ethnicity.hispanic': '% hispanic',
  'student_body_fulltime_mba.race_ethnicity.black': '% black',
  'student_body_fulltime_mba.race_ethnicity.white': '% white',
  'student_body_fulltime_mba.race_ethnicity.asian': '% asian',
  'student_body_fulltime_mba.race_ethnicity.pacific_islander': '% pacific islander',
  'student_body_fulltime_mba.race_ethnicity.unknown': '% unknown',
  'student_body_fulltime_mba.race_ethnicity.two_or_more_races': '% two or more races',
  'student_body_all_programs.countries_most_represented[0].percent': 'top country % (rank 1)',
  'student_body_all_programs.countries_most_represented[2].percent': 'top country % (rank 3)',
  'student_body_all_programs.countries_most_represented[4].percent': 'top country % (rank 5)',
  'undergraduate_majors.engineering': 'undergrad engineering %',
  'undergraduate_majors.business_and_commerce': 'undergrad business %',
  'undergraduate_majors.humanities': 'undergrad humanities %',
  'student_indebtedness_graduates.average_indebtedness_full_time_mba': 'avg FT MBA indebtedness',
  'financial_aid.research_assistantships': 'research assistantships',
  'financial_aid.fellowships': 'fellowships',
};

const CONFIDENCE_COPY = {
  high: 'Strong fit on unseen schools. Both the direction and the rough size of changes are trustworthy — actionable for planning.',
  medium: 'Decent fit. The direction is reliable, but treat exact magnitudes as approximations rather than precise estimates.',
  low: 'Weak fit. Treat as suggestive only — small lever changes here are unlikely to translate into measurable rank movement.',
};

function fmtNum(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return value.toFixed(digits);
}

function fmtMae(target, mae) {
  if (mae === null || mae === undefined) return '—';
  if (target === 'AvgSalaryBonus') return `$${Math.round(mae).toLocaleString()}`;
  if (target === 'GMAT_Combined') return mae.toFixed(1);
  if (['EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate'].includes(target)) {
    return `${(mae * 100).toFixed(1)} pp`;
  }
  return mae.toFixed(3);
}

function featureLabel(rawKey) {
  return FEATURE_LABEL_OVERRIDES[rawKey] || rawKey;
}

/** Horizontal bar chart showing signed coefficients (original view). */
function buildChart(canvas, topCoefficients) {
  const sorted = topCoefficients.slice(0, 8);
  const labels = sorted.map((c) => featureLabel(c.feature));
  const data = sorted.map((c) => c.coefficient);
  const colors = data.map((v) =>
    v >= 0 ? 'rgba(99, 102, 241, 0.85)' : 'rgba(244, 114, 182, 0.85)',
  );
  const borders = data.map((v) =>
    v >= 0 ? 'rgba(99, 102, 241, 1)' : 'rgba(244, 114, 182, 1)',
  );

  return new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 1,
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          borderColor: 'rgba(99, 102, 241, 0.3)',
          borderWidth: 1,
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          cornerRadius: 8,
          padding: 10,
          callbacks: {
            label: (item) => `coef: ${item.raw.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#64748b', font: { size: 10 } },
          border: { display: false },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 10 } },
          border: { color: 'rgba(255,255,255,0.05)' },
        },
      },
    },
  });
}

/**
 * Build inline CSS-only contribution bars for the top features driving a target.
 * Each row shows: feature name, a proportional gradient bar, direction arrow, and
 * the % contribution label. Top 6 features are shown to keep cards compact.
 */
function buildContribBars(targetName) {
  const features = getCfmTopFeatures(targetName, 6);
  if (!features.length) return '<p class="text-gray-600 text-xs">No feature data available.</p>';

  const maxPct = features[0]?.pct || 1; // first item has highest pct (already sorted)

  const rows = features.map(f => {
    const label = featureLabel(f.feature);
    const isPositive = f.coefficient >= 0;
    const barClass = isPositive ? 'contrib-bar--pos' : 'contrib-bar--neg';
    const arrow = isPositive ? '↑' : '↓';
    const arrowClass = isPositive ? 'text-indigo-300' : 'text-pink-300';
    // Scale bar width relative to the top contributor so the largest = 100%
    const barWidth = Math.max(2, (f.pct / maxPct) * 100);

    return `
      <div class="contrib-row">
        <span class="contrib-label" title="${f.feature}">${label}</span>
        <div class="contrib-bar-track">
          <div class="contrib-bar ${barClass}" style="width: ${barWidth.toFixed(1)}%"></div>
        </div>
        <span class="contrib-pct">
          <span class="${arrowClass}">${arrow}</span> ${f.pct.toFixed(1)}%
        </span>
      </div>
    `;
  }).join('');

  return `<div class="contrib-bars">${rows}</div>`;
}

function renderCard(target) {
  const conf = target.confidence || { label: 'Low', tone: 'low' };
  const card = document.createElement('div');
  card.className = `glass-panel p-5 perf-card perf-card--${conf.tone}`;
  card.innerHTML = `
    <div class="flex items-start justify-between mb-3">
      <div>
        <p class="text-[10px] uppercase tracking-widest text-gray-500">${target.target_name}</p>
        <p class="text-lg font-bold text-white">${target.label}</p>
      </div>
      <span class="confidence-pill confidence-${conf.tone}">${conf.label}</span>
    </div>

    <div class="grid grid-cols-4 gap-2 mb-3 text-center">
      <div class="metric-mini" title="Cross-validated R² — fit on schools the model hasn't seen. 1.00 = perfect, 0.00 = no better than guessing the average.">
        <p class="metric-mini-label">Cross-val fit</p>
        <p class="metric-mini-value">${fmtNum(target.grouped_cv?.r2, 2)}</p>
      </div>
      <div class="metric-mini" title="Forward-in-time test — train on 2024 data, predict 2025. Tests how well the model holds up year-over-year.">
        <p class="metric-mini-label">Next-year fit</p>
        <p class="metric-mini-value">${fmtNum(target.temporal_holdout?.r2, 2)}</p>
      </div>
      <div class="metric-mini" title="Average prediction error in this indicator's own units. Lower = better.">
        <p class="metric-mini-label">Avg error</p>
        <p class="metric-mini-value">${fmtMae(target.target_name, target.grouped_cv?.mae)}</p>
      </div>
      <div class="metric-mini" title="Number of school-year observations used to train the model.">
        <p class="metric-mini-label">Trained on</p>
        <p class="metric-mini-value">${target.training_row_count}</p>
      </div>
    </div>

    <p class="text-[11px] text-gray-500 mb-2">Strongest levers — green pushes the indicator up, pink pushes it down</p>
    <div class="perf-chart-wrap"><canvas></canvas></div>

    <div class="contrib-section mt-4">
      <p class="text-[11px] text-gray-500 mb-2">
        Where the model's signal comes from
        <span class="text-gray-600 ml-1">(% share of total influence · ↑ helps · ↓ hurts)</span>
      </p>
      ${buildContribBars(target.target_name)}
    </div>

    <p class="confidence-copy mt-3">${CONFIDENCE_COPY[conf.tone] || ''}</p>
  `;
  const canvas = card.querySelector('canvas');
  buildChart(canvas, target.top_coefficients || []);
  return card;
}

/* Confidence priority for sorting: high first, then medium, then low */
const CONFIDENCE_ORDER = { high: 0, medium: 1, low: 2 };

export function renderExplainability() {
  const grid = document.getElementById('performance-grid');
  if (!grid) return;
  grid.innerHTML = '';

  // Sort cards by confidence level: high → medium → low
  const targets = getCoreTargetSummaries().slice();
  targets.sort((a, b) => {
    const aTone = (a.confidence?.tone || 'low');
    const bTone = (b.confidence?.tone || 'low');
    return (CONFIDENCE_ORDER[aTone] ?? 2) - (CONFIDENCE_ORDER[bTone] ?? 2);
  });

  for (const target of targets) {
    const card = renderCard(target);
    grid.appendChild(card);
  }
}

