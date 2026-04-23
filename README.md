# GWU MBA Ranking Predictor

An interactive what-if scenario planning tool for predicting George Washington University's US News MBA ranking. Move 8 slider levers to simulate how changes in key metrics affect the predicted ranking in real time.

---

## Table of Contents

- [Quick Start](#quick-start)
- [How the Model Works](#how-the-model-works)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Re-training the Model](#re-training-the-model)
- [Deploying to Vercel](#deploying-to-vercel)

---

## Quick Start

### Prerequisites

- **Node.js** v18+ ([download](https://nodejs.org/))
- **Python** 3.10+ (only needed if re-training the model)

### Step-by-Step Launch

```bash
# 1. Navigate to the webapp directory
cd notebooks/webapp

# 2. Install JavaScript dependencies
npm install

# 3. Start the development server
npm run dev

# 4. Open the app in your browser
#    → http://localhost:3000/
```

That's it. The app loads pre-trained model artifacts from `public/model_artifacts/` and runs all predictions client-side in the browser — no backend server needed.

### Other Commands

```bash
# Build for production (outputs to dist/)
npm run build

# Preview the production build locally
npm run preview
```

---

## How the Model Works

### Overview

The model predicts US News MBA ranking scores using 8 input features, trained on the 2025 US News dataset (122 schools). A Monte Carlo simulation then converts the predicted score into a rank distribution by simulating market volatility across all competing schools.

### Data Preprocessing Pipeline

The raw CSV data (`data/us_news_data_2025.csv`) goes through the following steps, replicating the logic from `[NEW]us_news_model_script_v2.ipynb`:

**1. Column Renaming**

Raw column names from the CSV are mapped to clean model names:

| Raw CSV Column | Model Name |
|---|---|
| `school_info.school_name` | `School` |
| `school_info.us_news_rank` | `Rank` |
| `school_info.us_news_overall_score` | `OverallScore` |
| `ranking_scores_two_year_averages.fulltime_employed_at_graduation_two_yr_avg` | `EmployedAtGrad` |
| `ranking_scores_two_year_averages.fulltime_employed_3_months_after_two_yr_avg` | `Employed3Mo` |
| `ranking_scores_two_year_averages.avg_starting_salary_and_bonus_two_yr_avg` | `AvgSalaryBonus` |
| `ranking_scores_two_year_averages.median_undergraduate_gpa` | `MedianGPA` |
| `ranking_scores_two_year_averages.acceptance_rate` | `AcceptanceRate` |
| `ranking_scores_two_year_averages.peer_assessment_score_out_of_5` | `PeerScore` |
| `ranking_scores_two_year_averages.recruiter_assessment_score_out_of_5` | `RecruiterScore` |
| `ranking_scores_two_year_averages.median_gmat_score_fulltime_old` | `GMAT_Old` |
| `ranking_scores_two_year_averages.median_gmat_score_fulltime_new` | `GMAT_New` |

**2. Unused Columns Dropped**

- `school_info.us_news_rank_out_of` — constant value, not predictive
- `ranking_scores_two_year_averages.salaries_by_profession_indicator_rank` — excluded from the 8-feature model

**3. GMAT Score Combination**

The dataset contains two GMAT columns (old 200-800 scale and new 205-805 Focus Edition scale). They are combined with priority to the old score:

```python
df['GMAT_Combined'] = df['GMAT_Old'].fillna(df['GMAT_New'])
```

**4. KNN Imputation**

Missing values are imputed using scikit-learn's `KNNImputer` with `n_neighbors=5` and `weights='distance'`. In the 2025 dataset, the affected columns were:

| Column | Missing Count | Missing % |
|---|---|---|
| `GMAT_Combined` | 65 | 53.3% |
| `RecruiterScore` | 3 | 2.5% |
| `MedianGPA` | 2 | 1.6% |

### Model Training Pipeline

After preprocessing, the data is fed into the `USNewsRankingSystem` class, which implements a 3-stage sklearn Pipeline followed by bootstrapped regression:

**Stage 1: OutlierCapper**

Clips each feature to its 5th and 95th percentile bounds learned from the training data. This prevents extreme outliers from distorting the regression.

**Stage 2: RankingFeatureTransformer**

Applies domain-specific transformations to normalize feature distributions:

| Transform | Applied To | Formula | Rationale |
|---|---|---|---|
| **Log** | `AvgSalaryBonus`, `GMAT_Combined` | `log(1 + x)` | Compresses right-skewed salary/score distributions |
| **Logit** | `EmployedAtGrad`, `Employed3Mo`, `AcceptanceRate` | `log(p / (1-p))` | Maps bounded [0,1] percentages to unbounded (-∞, +∞) space |
| **Inverse Normal** | *(not used in this model)* | `-Φ⁻¹((rank - 0.5) / N)` | Converts ordinal ranks to Z-scores |

**Stage 3: StandardScaler**

Centers and scales each transformed feature to zero mean and unit variance: `z = (x - μ) / σ`

**Stage 4: Bootstrapped ElasticNetCV (1,000 iterations)**

Instead of fitting a single regression, the model bootstraps 1,000 samples and fits `ElasticNetCV` on each:

1. Resample the data with replacement
2. Fit `ElasticNetCV` with `l1_ratio=[.1, .5, .7, .9, .95, .99, 1]` and 5-fold CV
3. Collect the coefficient vector from each iteration
4. Final coefficients = mean of all 1,000 bootstrap coefficient vectors

This produces robust, variance-stabilized weights and enables confidence interval estimation.

**Stage 5: Calibration**

The model is calibrated so that the #1-ranked school (Stanford) receives a predicted score of exactly 100:

```
intercept = 100 - dot(transform(Stanford_features), coefficients)
```

### Trained Model Coefficients

| Feature | Mean Weight | 95% CI | Significant? |
|---|---|---|---|
| `AvgSalaryBonus` | 9.42 | [6.91, 11.73] | ✓ |
| `PeerScore` | 5.22 | [3.58, 6.94] | ✓ |
| `RecruiterScore` | 3.58 | [2.63, 4.35] | ✓ |
| `MedianGPA` | 2.92 | [2.07, 3.86] | ✓ |
| `Employed3Mo` | 2.36 | [1.12, 3.65] | ✓ |
| `EmployedAtGrad` | 1.78 | [0.61, 2.96] | ✓ |
| `GMAT_Combined` | 0.79 | [-0.52, 2.51] | ✗ |
| `AcceptanceRate` | -1.38 | [-2.45, -0.23] | ✓ |

**Key insights:**
- Average Salary + Bonus is by far the strongest predictor (weight 9.42)
- GMAT score is the only non-significant feature (CI crosses zero)
- Acceptance Rate has a negative weight: lower acceptance → higher score (more selective = better)

### Model Performance

| Metric | Value |
|---|---|
| MAE | 4.69 |
| RMSE | 5.55 |
| R² | 0.924 |
| Spearman ρ | 0.982 |

### Monte Carlo Rank Simulation

The trained model predicts *scores*, not *ranks*. To convert a score into a rank, the app runs a Monte Carlo simulation (adapted from `rank_scenario_planning.ipynb`):

1. **Predict scores** for all 122 schools using the trained pipeline
2. **Override the target school's score** with the user's what-if scenario (deterministic)
3. **Add tiered Gaussian noise** to competitor scores (simulating year-to-year volatility):
   - Top 20 schools: σ = 0.8 (very stable)
   - Ranks 21-50: σ = 1.5 (moderate volatility)
   - Ranks 51+: σ = 2.5 (high volatility)
4. **Rank all schools** by descending score in each simulation
5. **Repeat 5,000 times** and compile the rank distribution

The result is a probability distribution of ranks, from which we extract:
- **Median rank** — the most likely outcome
- **90% confidence interval** — the range from the 5th to 95th percentile
- **Rank distribution histogram** — the full probability profile

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  TRAINING (Python, runs once)                            │
│                                                          │
│  Raw CSV → Rename → GMAT Combine → KNN Impute           │
│      ↓                                                   │
│  OutlierCapper → FeatureTransformer → StandardScaler     │
│      ↓                                                   │
│  Bootstrapped ElasticNetCV (1000x) → Calibration         │
│      ↓                                                   │
│  Export 7 JSON artifacts                                 │
└──────────────────────────┬───────────────────────────────┘
                           │
                    JSON artifacts
                           │
┌──────────────────────────▼───────────────────────────────┐
│  FRONTEND (JavaScript, runs in browser)                  │
│                                                          │
│  Load JSON artifacts → Replicate transform pipeline      │
│      ↓                                                   │
│  8 Interactive Sliders → Debounced input (300ms)         │
│      ↓                                                   │
│  Client-side Monte Carlo simulation (5000 iterations)    │
│      ↓                                                   │
│  Predicted Rank + CI + Distribution Chart                │
└──────────────────────────────────────────────────────────┘
```

The key design choice is **"Train in Python, Serve in JavaScript"**: the heavy scikit-learn training runs once offline, exports all learned parameters as JSON, and the browser replicates the lightweight transform + predict + simulate pipeline in pure JavaScript for instant real-time predictions.

---

## File Structure

```
webapp/
├── index.html                     # Main HTML page
├── package.json                   # Node.js dependencies and scripts
├── vite.config.js                 # Vite bundler configuration
├── tailwind.config.js             # Tailwind CSS theme and plugins
├── postcss.config.js              # PostCSS configuration
├── vercel.json                    # Vercel deployment settings
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
│
├── styles/
│   └── index.css                  # Tailwind directives + custom components
│
├── src/
│   ├── main.js                    # Application entry point
│   ├── model.js                   # Client-side inference engine
│   ├── sliders.js                 # Slider UI component
│   └── results.js                 # Results panel + Chart.js visualization
│
├── public/
│   └── model_artifacts/           # Pre-trained model parameters (7 JSON files)
│       ├── model_config.json      #   Feature list, transform config
│       ├── capper_bounds.json     #   OutlierCapper percentile bounds
│       ├── transformer_config.json#   Log/logit/inv_norm column mappings
│       ├── scaler_params.json     #   StandardScaler mean and scale vectors
│       ├── model_weights.json     #   Regression coefficients + intercept
│       ├── data_snapshot.json     #   All 122 schools with imputed features
│       └── feature_ranges.json   #   Slider ranges + GWU current values
│
└── scripts/
    ├── train_model.py             # Full training pipeline (Python)
    └── requirements.txt           # Python dependencies
```

---

## Re-training the Model

When new ranking data is available:

```bash
# 1. Place the new CSV at the expected path
#    → ../data/us_news_data_2025.csv (relative to webapp/)

# 2. Install Python dependencies (one-time)
cd scripts
pip install -r requirements.txt

# 3. Run the training script
python train_model.py
#    → Outputs 7 JSON files to public/model_artifacts/

# 4. The webapp will automatically use the new artifacts on next load
```

The training script takes ~40 seconds on a modern machine (bottlenecked by 1,000 ElasticNetCV bootstrap iterations with 4-core parallelism).

---

## Deploying to Vercel

### Steps

1. **Create a GitHub repository** and push the `webapp/` folder contents

2. **Import on Vercel:**
   - Go to [vercel.com](https://vercel.com) → "Add New Project"
   - Import your GitHub repository
   - Vercel auto-detects Vite — no additional configuration needed

3. **Deploy:**
   - Vercel runs `npx vite build` automatically
   - Model artifacts from `public/model_artifacts/` are included in the static build
   - The app is deployed as a fully static site (no server required)

### How It Works in Production

The production build (`dist/`) is a fully self-contained static site:
- `index.html` + bundled JS/CSS
- `model_artifacts/` served as static JSON files
- All predictions run client-side — zero server cost, instant response times

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model Training | Python 3.10+, scikit-learn, scipy, numpy, pandas, joblib |
| Frontend Framework | Vite 8 |
| Styling | Tailwind CSS v3 |
| Charts | Chart.js 4 |
| Inference | Pure JavaScript (browser-side) |
| Deployment | Vercel (static hosting) |
