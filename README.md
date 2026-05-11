# GWU MBA Ranking Predictor v2

A faithful reproduction of the **official US News Best Business Schools 2026 methodology** running side-by-side with a 9-feature bootstrapped regression, with Monte Carlo rank simulation and a methodology-comparison tab.

> **Why v2?** v1 used a single ML regression with 8 features and a hand-rolled GMAT/GRE blend that introduced a non-canonical "<25% submission penalty" not present in the official US News methodology. v2 (a) implements the official 9-indicator z-score+fixed-weights formula, (b) adds the missing **Salary by Profession** indicator (10% of the published score), (c) corrects the GMAT/GRE blend to the published 40/40/20 GRE-internal split with submission-proportion cross-exam blending, (d) keeps the regression on board for comparison, and (e) lets the user toggle between engines on the Direct Predictor tab.

---

## Quick Start

```bash
cd webapp-v2
npm install
npm run dev
# → http://localhost:3100/
```

---

## What's new in v2

| Change | Detail |
|---|---|
| **9 indicators** (vs. 8 in v1) | Adds `SalaryByProfession` (per-occupation salary ratio) — 10% of the official score. |
| **Two scoring engines** | (a) Official US News calculator: z-score + fixed weights + min-max rescale. (b) Bootstrapped ElasticNet regression on stacked 2024+2025. |
| **Engine toggle** | On the Direct Predictor tab, switch which engine drives the rank prediction. Both share the same Monte Carlo volatility model. |
| **Corrected GMAT/GRE blend** | Per-year percentile rank for `GMAT_old`, `GMAT_new`, `GRE_Q`, `GRE_V`, `GRE_AW`. GRE-internal 40/40/20. Cross-exam submission-proportion weighted. **No spurious <25% penalty** (that was a v1 bug). |
| **GRE input** | Slider exposes `GRE Quantitative`, `GRE Verbal`, `GRE Analytical Writing` separately when GRE is enabled. Median GRE is approximated as the midpoint of each section's reported 10th-90th range (the dataset does not carry true medians). |
| **5th tab — Methodology Comparison** | Per-school side-by-side: published US News rank/score vs. our official reproduction vs. the regression. Sortable, searchable, with summary correlation and mean-error stats. |
| **Score Model Insights tab** | Now dual-engine: official weights table (with direction higher↑/lower↑) alongside regression coefficients with bootstrap 95% CIs. |
| **Same Tailwind theme** | Identical glass panels, indigo/cyan palette, Inter + JetBrains Mono fonts. |

---

## Tabs

| # | Tab | Purpose |
|---|---|---|
| 1 | **Direct Predictor** | 9 sliders. Toggle between the official engine and the regression. Monte Carlo simulates 10,000 rank outcomes per slider change. |
| 2 | **Lever Models** | Reliability of the 8 indirect "core feature" models (CFM artifacts copied from v1; SalaryByProfession is not yet a CFM target). |
| 3 | **Lever Predictor** | Adjust non-method levers; see predicted core features feed into the chosen score engine. |
| 4 | **Score Model Insights** | Official methodology weights and direction vs. regression mean coefficients with bootstrap 95% CIs. Per-feature contribution charts (avg or GWU; either engine). |
| 5 | **Methodology Comparison** | Per-school deltas: Published vs. Official vs. Regression. Sort by absolute disagreement to find where the engines diverge. |

---

## Official methodology reproduced

Source: <https://www.usnews.com/education/best-graduate-schools/articles/business-schools-methodology>

| Indicator | Weight |
|---|---|
| Employment rates at graduation (2-yr weighted avg) | 7% |
| Employment rates 3 months after graduation (2-yr weighted avg) | 13% |
| Mean starting salary + bonus (2-yr weighted avg) | 20% |
| **Salary by profession** | 10% |
| Peer assessment | 12.5% |
| Recruiter assessment (3-yr weighted avg) | 12.5% |
| Median GMAT/GRE (40/40/20 GRE blend, submission-weighted across exams) | 13% |
| Median undergraduate GPA | 10% |
| Acceptance rate (lower is better) | 2% |

Each indicator is z-scored against the cohort, multiplied by its weight (acceptance rate negated), summed, and rescaled so the top school = 100.

### Salary by profession

For each of 7 professions {Consulting, Finance/Accounting, General Management, Human Resources, Marketing/Sales, IT/MIS, Operations/Logistics}: compute `school_avg_salary / cohort_weighted_avg`. Drop "Other" and any profession with <3 reporting graduates. Each school's score is the weighted average of these ratios across professions, weighted by the school's number of reporters per profession.

### GMAT/GRE blend

1. Per-year percentile rank for each of the 5 score distributions: GMAT-old, GMAT-new, GRE Q, GRE V, GRE AW.
2. GRE-internal blend: `0.4·pct(Q) + 0.4·pct(V) + 0.2·pct(AW)` → single GRE percentile.
3. Cross-exam blend: weighted by the school's `Pct_GMAT_Old`, `Pct_GMAT_New`, `Pct_GRE` submission proportions, normalised to sum to 1.
4. Multiply by 100 → 0–100 score.

If a percentile is missing for a given exam, that exam contributes zero weight (and the others re-normalise). If all three are missing, the school gets the cohort floor (the lowest blended score among schools that did report).

---

## Re-training

```bash
cd webapp-v2/scripts
pip install -r requirements.txt   # one-time
python train_model.py
# → 10 JSON artifacts written to public/model_artifacts/
```

The script:
1. Loads `../../all_schools_flat_2024.csv` and `../../all_schools_flat_2025.csv`.
2. Per-year KNN-imputes the 9 indicators + raw test scores.
3. Computes the salary-by-profession indicator (per-year cohort means).
4. Computes the corrected GMAT/GRE blended percentile.
5. Runs the official z-score + fixed-weight calculator and reports MAE/R²/Spearman vs. the published `OverallScore`.
6. Trains the bootstrapped ElasticNet (10,000 iterations) and reports the same.
7. Exports all artifacts including `methodology_comparison.json` for the comparison tab.

---

## Artifacts

In `public/model_artifacts/`:

| File | Purpose |
|---|---|
| `model_config.json` | 9-feature list, target name, official weights, transform config |
| `official_params.json` | Cohort means/stds per indicator + min/max weighted-sum for rescaling |
| `capper_bounds.json` | Outlier caps used by the regression-track only |
| `transformer_config.json` | log/logit columns for the regression track |
| `scaler_params.json` | Regression-track StandardScaler mean/scale |
| `model_weights.json` | Regression coef + intercept |
| `data_snapshot.json` | All schools (snapshot year) with imputed features + computed `official_score` |
| `feature_ranges.json` | Slider ranges + GMAT input config + GWU current values |
| `gmat_inference_curves.json` | Sorted GMAT_old/new + GRE Q/V/AW score arrays for client-side percentile rank |
| `model_explainability.json` | Performance + coefficients + contribution percentages for both engines |
| `methodology_comparison.json` | Per-school rows: published / official / regression scores & ranks |

---

## Deploying to Vercel via a separate GitHub repo

This module is **independent** of the original `webapp/` and is designed to deploy to its own Vercel project.

### One-time setup

```bash
# From inside webapp-v2/
cd "D:/work/US news/notebooks/webapp-v2"

# Initialise git, commit
git init
git add .
git commit -m "Initial commit: GWU MBA Ranking Predictor v2 (faithful methodology + regression)"

# Create a new GitHub repo (via gh CLI, or manually on github.com)
gh repo create GWU-MBA-Ranking-Predictor-v2 --public --source=. --remote=origin --push
# OR manually:
#   git remote add origin git@github.com:<your-user>/GWU-MBA-Ranking-Predictor-v2.git
#   git push -u origin main
```

### Vercel project

1. Go to [vercel.com](https://vercel.com) → "Add New Project".
2. Import the new GitHub repo.
3. Vercel auto-detects Vite. No env vars or extra configuration needed.
4. **Build command**: `npx vite build` (already set in `vercel.json`).
5. **Output directory**: `dist` (already set).
6. Click Deploy.

Subsequent pushes to `main` auto-deploy. Treat the v2 Vercel project as a parallel surface; the original webapp continues to deploy independently from its own repo.

---

## Tech stack

| Layer | Same as v1? | Notes |
|---|---|---|
| Model training | Python 3.10+, scikit-learn, scipy, numpy, pandas, joblib | Adds the Salary-by-Profession reducer |
| Frontend bundler | Vite 8 | port 3100 (vs. v1's 3000) for side-by-side dev |
| Styling | Tailwind CSS v3 with the same custom theme | Identical glass-panel + navy/indigo/cyan palette |
| Charts | Chart.js 4 | 2 charts (rank distribution, contribution) |
| Inference | Pure JavaScript (browser-side) | Two engines: official z-score formula and regression coefficient dot-product |
| Deployment | Vercel static hosting | Independent project, independent GitHub repo |

---

## Known limitations

1. **GRE medians are approximated** as range midpoints — the published dataset does not include true medians per school. For schools with skewed GRE distributions this introduces a small bias.
2. **Cohort definition** — US News' published z-scores are computed on the 134 schools they actually rank. Our 2024+2025 stack has ~243 rows including some unranked observations; this shifts cohort means/stds slightly. Spearman rank correlation with the published ranking is excellent (~0.96), but the absolute score scale has a ~10-point linear offset.
3. **CFM models are unchanged from v1** — they predict the legacy 8-feature schema and the legacy raw-GMAT `GMAT_Combined`. Tab 3 (Lever Predictor) adapts the CFM output to v2's blended scale on the fly, but does not re-train CFMs against the new 9-indicator schema. Re-training the CFMs is out of scope for v2.
