"""
GWSB Ranking Predictor — Regression Training Pipeline
======================================================
Single bootstrapped ElasticNet engine over 9 indicators on the stacked
2024 + 2025 panel. The regression replaces the full US News scoring
methodology with a learned linear approximation calibrated so the rank-1
school maps to a score of 100. The Monte Carlo rank simulator wraps the
regression score with tiered Gaussian volatility.

Indicators (9):
  EmployedAtGrad, Employed3Mo, AvgSalaryBonus, SalaryByProfession,
  MedianGPA, AcceptanceRate, PeerScore, RecruiterScore, GMAT_Combined.

Pre-pipeline feature engineering:
  - Salary-by-Profession indicator: per-occupation ratio
    school_avg / cohort_weighted_avg, weighted by reporters per occupation,
    excluding "Other" and any occupation with <3 reporters.
  - GMAT_Combined: per-year percentile rank for {GMAT_old, GMAT_new, GRE Q,
    GRE V, GRE AW}, GRE-internal 40/40/20 blend, then submission-proportion
    blend across exams (renormalised); scaled to 0-100. Schools with no
    submitted scores get the per-year cohort floor.
  - All other indicators are imputed per-year via KNN (n_neighbors=5).

Pipeline:
  OutlierCapper(5%/95%, excluding GMAT_Combined and SalaryByProfession)
    → log1p(AvgSalaryBonus)
    → logit(EmployedAtGrad, Employed3Mo, AcceptanceRate)
    → StandardScaler
    → bootstrapped ElasticNetCV (10,000 iterations, l1_ratio grid, 5-fold CV)
    → mean coefficients
    → calibrate intercept so rank-1 school = 100

Outputs (public/model_artifacts/):
  model_config, capper_bounds, transformer_config, scaler_params,
  model_weights, data_snapshot, gmat_inference_curves, feature_ranges,
  model_explainability.
"""

import os
import re
import json
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, rankdata
from joblib import Parallel, delayed

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATHS = {
    2024: os.path.join(SCRIPT_DIR, '..', '..', 'all_schools_flat_2024.csv'),
    2025: os.path.join(SCRIPT_DIR, '..', '..', 'all_schools_flat_2025.csv'),
}
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'public', 'model_artifacts')
INFERENCE_CURVE_YEAR = 2025

ALL_FEATURES = [
    'EmployedAtGrad', 'Employed3Mo', 'AvgSalaryBonus', 'SalaryByProfession',
    'MedianGPA', 'AcceptanceRate', 'PeerScore', 'RecruiterScore',
    'GMAT_Combined',
]
TARGET = 'OverallScore'

# Salary-by-profession excluded categories
EXCLUDED_OCCUPATIONS = {'Other'}
MIN_OCCUPATION_REPORTERS = 3

# Regression preprocessing
LOG_VARS = ['AvgSalaryBonus']
LOGIT_VARS = ['EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate']
# Pre-bounded features (already on a percentile / ratio scale) — exclude from outlier capping
OUTLIER_EXCLUDE = {'GMAT_Combined', 'SalaryByProfession'}
N_BOOTSTRAP_ITERATIONS = 10000
N_JOBS = 4

GMAT_INPUT_CONFIG = {
    'gmat_scale_default': 'old',
    'gmat_old_range': {'min': 200, 'max': 800, 'step': 5},
    'gmat_new_range': {'min': 205, 'max': 805, 'step': 5},
    'gre_q_range':    {'min': 130, 'max': 170, 'step': 1},
    'gre_v_range':    {'min': 130, 'max': 170, 'step': 1},
    'gre_aw_range':   {'min': 0.0, 'max': 6.0, 'step': 0.5},
    'gre_default_enabled': False,
}

SBP_OCCUPATIONS = [
    'Consulting', 'Finance / Accounting', 'General Management',
    'Marketing / Sales', 'Operations / Production',
    'Management Information Systems (MIS)', 'Human Resources',
]
SBP_SLIDER = {'min': 60000, 'max': 260000, 'step': 1000, 'format': 'dollar'}

SLIDER_RANGES = {
    'EmployedAtGrad':      {'min': 0.20, 'max': 1.00, 'step': 0.01, 'label': 'Employed at Graduation', 'format': 'percent'},
    'Employed3Mo':         {'min': 0.20, 'max': 1.00, 'step': 0.01, 'label': 'Employed 3 Months After', 'format': 'percent'},
    'AvgSalaryBonus':      {'min': 80000, 'max': 220000, 'step': 1000, 'label': 'Avg Salary + Bonus ($)', 'format': 'dollar'},
    'SalaryByProfession':  {'min': 0.6, 'max': 1.4, 'step': 0.01, 'label': 'Salary by Profession (cohort ratio)', 'format': 'number'},
    'MedianGPA':           {'min': 3.0, 'max': 4.0, 'step': 0.01, 'label': 'Median GPA', 'format': 'number'},
    'AcceptanceRate':      {'min': 0.05, 'max': 1.00, 'step': 0.01, 'label': 'Acceptance Rate', 'format': 'percent'},
    'PeerScore':           {'min': 1.0, 'max': 5.0, 'step': 0.1,  'label': 'Peer Assessment Score', 'format': 'number'},
    'RecruiterScore':      {'min': 1.0, 'max': 5.0, 'step': 0.1,  'label': 'Recruiter Assessment Score', 'format': 'number'},
    'GMAT_Combined':       {'min': 0,   'max': 100, 'step': 1,    'label': 'GMAT/GRE Blended Percentile', 'format': 'number'},
}

COLUMN_RENAME_MAP = {
    'school_info.school_name': 'School',
    'school_info.us_news_rank': 'Rank',
    'school_info.us_news_overall_score': 'OverallScore',
    'ranking_scores_two_year_averages.fulltime_employed_at_graduation_two_yr_avg': 'EmployedAtGrad',
    'ranking_scores_two_year_averages.fulltime_employed_3_months_after_two_yr_avg': 'Employed3Mo',
    'ranking_scores_two_year_averages.avg_starting_salary_and_bonus_two_yr_avg': 'AvgSalaryBonus',
    'ranking_scores_two_year_averages.median_undergraduate_gpa': 'MedianGPA',
    'ranking_scores_two_year_averages.acceptance_rate': 'AcceptanceRate',
    'ranking_scores_two_year_averages.peer_assessment_score_out_of_5': 'PeerScore',
    'ranking_scores_two_year_averages.recruiter_assessment_score_out_of_5': 'RecruiterScore',
    'ranking_scores_two_year_averages.median_gmat_score_fulltime_old': 'GMAT_Old',
    'ranking_scores_two_year_averages.median_gmat_score_fulltime_new': 'GMAT_New',
    'gmat_data.percent_new_entrants_providing_gmat_old': 'Pct_GMAT_Old',
    'gmat_data.percent_new_entrants_providing_gmat_new': 'Pct_GMAT_New',
    'gre_data.percent_new_entrants_providing_gre': 'Pct_GRE',
    'gre_data.gre_score_range_10th_90th': 'GRE_Range_Str',
}


# ============================================================
# GRE PARSING (verbal, quantitative, analytical writing)
# ============================================================

def parse_gre_components(s):
    """Parse 'NN-NN verbal, NN-NN quantitative, F.F-F.F writing' -> (V_mid, Q_mid, AW_mid).
    Returns (None, None, None) on missing/empty/unparsable input.
    """
    if not isinstance(s, str) or not s.strip():
        return (None, None, None)
    text = s.lower()
    out = {'verbal': None, 'quant': None, 'writing': None}
    for m in re.finditer(r'(\d{1,3}(?:\.\d+)?)\s*[-–]\s*(\d{1,3}(?:\.\d+)?)\s*([a-z]+)?', text):
        try:
            lo, hi = float(m.group(1)), float(m.group(2))
        except ValueError:
            continue
        lbl = (m.group(3) or '').lower()
        mid = (lo + hi) / 2.0
        if 'verbal' in lbl: out['verbal'] = mid
        elif 'quant' in lbl: out['quant'] = mid
        elif 'writ' in lbl: out['writing'] = mid
    return (out['verbal'], out['quant'], out['writing'])


# ============================================================
# GMAT/GRE BLENDED PERCENTILE
# ============================================================

def _percentile_rank(values):
    out = np.full(len(values), np.nan)
    mask = ~np.isnan(values)
    if mask.sum() == 0:
        return out
    ranks = rankdata(values[mask], method='average') / float(mask.sum())
    out[mask] = ranks
    return out


def compute_gmat_gre_blend(df, year_col='Year'):
    """Per-year percentile rank for GMAT_Old, GMAT_New, GRE_Q, GRE_V, GRE_AW;
    GRE-internal blend = 0.4Q + 0.4V + 0.2AW; cross-exam blend weighted by
    each school's submission proportions; scaled to 0-100. NaN handled by
    leaving GMAT_Combined as NaN here (cohort-floor fallback is applied later
    in preprocess()).
    """
    df = df.copy()

    if 'GRE_Range_Str' in df.columns:
        comps = df['GRE_Range_Str'].apply(parse_gre_components)
        df['GRE_V'] = comps.apply(lambda t: t[0])
        df['GRE_Q'] = comps.apply(lambda t: t[1])
        df['GRE_AW'] = comps.apply(lambda t: t[2])
    else:
        df['GRE_V'] = np.nan
        df['GRE_Q'] = np.nan
        df['GRE_AW'] = np.nan

    for col in ('Pct_GMAT_Old', 'Pct_GMAT_New', 'Pct_GRE'):
        if col not in df.columns:
            df[col] = 0.0
        m = df[col].dropna().max() if df[col].notna().any() else 0
        if m is not None and m > 1.5:  # treat as percent → fraction
            df[col] = df[col] / 100.0
        df[col] = df[col].fillna(0.0).clip(0.0, 1.0)

    score_to_rank = [
        ('GMAT_Old', 'rank_gmat_old'), ('GMAT_New', 'rank_gmat_new'),
        ('GRE_Q', 'rank_gre_q'), ('GRE_V', 'rank_gre_v'), ('GRE_AW', 'rank_gre_aw'),
    ]
    for _, rc in score_to_rank:
        df[rc] = np.nan

    for year, sub in df.groupby(year_col):
        for sc, rc in score_to_rank:
            if sc not in df.columns:
                continue
            df.loc[sub.index, rc] = _percentile_rank(sub[sc].astype(float).values)

    gre_pct = 0.4 * df['rank_gre_q'] + 0.4 * df['rank_gre_v'] + 0.2 * df['rank_gre_aw']
    fallback = 0.5 * df['rank_gre_q'].fillna(np.nan) + 0.5 * df['rank_gre_v'].fillna(np.nan)
    df['rank_gre'] = gre_pct.where(gre_pct.notna(), fallback)

    p_old = df['Pct_GMAT_Old'].values
    p_new = df['Pct_GMAT_New'].values
    p_gre = df['Pct_GRE'].values
    r_old = df['rank_gmat_old'].values
    r_new = df['rank_gmat_new'].values
    r_gre = df['rank_gre'].values

    p_old_eff = np.where(np.isnan(r_old), 0.0, p_old)
    p_new_eff = np.where(np.isnan(r_new), 0.0, p_new)
    p_gre_eff = np.where(np.isnan(r_gre), 0.0, p_gre)
    r_old_eff = np.where(np.isnan(r_old), 0.0, r_old)
    r_new_eff = np.where(np.isnan(r_new), 0.0, r_new)
    r_gre_eff = np.where(np.isnan(r_gre), 0.0, r_gre)

    total = p_old_eff + p_new_eff + p_gre_eff
    weighted = p_old_eff * r_old_eff + p_new_eff * r_new_eff + p_gre_eff * r_gre_eff
    with np.errstate(divide='ignore', invalid='ignore'):
        blended = np.where(total > 0, weighted / total, np.nan)

    df['GMAT_Combined'] = blended * 100.0
    return df


# ============================================================
# SALARY-BY-PROFESSION INDICATOR
# ============================================================

def compute_salary_by_profession(df, year_col='Year'):
    """Per-school weighted average of (school_avg / cohort_weighted_avg) per
    occupation, weighted by the school's reporters per occupation. Excludes
    'Other' and any occupation with <3 reporters. Computed per-year so 2024
    salaries are compared against the 2024 cohort.
    """
    occ_idx = list(range(8))
    sba = pd.Series(np.nan, index=df.index, dtype=float)

    for year, sub in df.groupby(year_col):
        per_occ = {}
        for i in occ_idx:
            occ_col = f'base_salary_by_occupation[{i}].occupation'
            sal_col = f'base_salary_by_occupation[{i}].average_salary'
            n_col   = f'base_salary_by_occupation[{i}].number_reporting_jobs'
            if occ_col not in sub.columns:
                continue
            for idx, row in sub.iterrows():
                occ = row[occ_col]; sal = row[sal_col]; n = row[n_col]
                if not isinstance(occ, str) or pd.isna(sal) or pd.isna(n): continue
                if occ.strip() in EXCLUDED_OCCUPATIONS: continue
                if n < MIN_OCCUPATION_REPORTERS: continue
                per_occ.setdefault(occ.strip(), []).append((idx, float(sal), float(n)))

        cohort_means = {}
        for k, rows in per_occ.items():
            ntot = sum(r[2] for r in rows)
            if ntot > 0:
                cohort_means[k] = sum(r[1] * r[2] for r in rows) / ntot

        for idx, row in sub.iterrows():
            ratio_sum, n_sum = 0.0, 0.0
            for i in occ_idx:
                occ_col = f'base_salary_by_occupation[{i}].occupation'
                sal_col = f'base_salary_by_occupation[{i}].average_salary'
                n_col   = f'base_salary_by_occupation[{i}].number_reporting_jobs'
                if occ_col not in sub.columns: continue
                occ = row[occ_col]; sal = row[sal_col]; n = row[n_col]
                if not isinstance(occ, str) or pd.isna(sal) or pd.isna(n): continue
                k = occ.strip()
                if k in EXCLUDED_OCCUPATIONS or n < MIN_OCCUPATION_REPORTERS: continue
                if k not in cohort_means or cohort_means[k] <= 0: continue
                ratio = float(sal) / cohort_means[k]
                ratio_sum += ratio * float(n)
                n_sum += float(n)
            if n_sum > 0:
                sba.loc[idx] = ratio_sum / n_sum

    return sba


# ============================================================
# CUSTOM TRANSFORMERS
# ============================================================

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns, limits=(0.05, 0.05)):
        self.columns = columns
        self.limits = limits
        self.caps_ = {}

    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                self.caps_[col] = (
                    float(X[col].quantile(self.limits[0])),
                    float(X[col].quantile(1 - self.limits[1])),
                )
        return self

    def transform(self, X):
        X_t = X.copy()
        for col, (lo, hi) in self.caps_.items():
            if col in X_t.columns:
                X_t[col] = X_t[col].clip(lo, hi)
        return X_t


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_cols, logit_cols):
        self.log_cols = log_cols
        self.logit_cols = logit_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_t = X.copy()
        for c in self.log_cols:
            if c in X_t.columns:
                X_t[c] = np.log1p(X_t[c])
        for c in self.logit_cols:
            if c in X_t.columns:
                p = X_t[c].clip(0.001, 0.999)
                X_t[c] = np.log(p / (1 - p))
        return X_t


# ============================================================
# BOOTSTRAP HELPER
# ============================================================

# Sign constraints by feature. All features default to 'positive' (higher
# value → higher predicted score). AcceptanceRate is 'negative' (lower
# acceptance → more selective → higher predicted score). The bootstrap fits a
# constrained ElasticNet on a sign-flipped copy of the AcceptanceRate column;
# the resulting coefficient is then flipped back to the original sign in the
# final coefficient vector. This prevents tiny collinearity-driven noise from
# producing counterintuitive directions (e.g. GMAT_Combined sliding the rank
# the wrong way).
SIGN_FLIP_FEATURES = ['AcceptanceRate']

def _run_single_bootstrap(X, y):
    Xb, yb = resample(X, y)
    en = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=10000, n_jobs=1, positive=True)
    en.fit(Xb, yb)
    return en.coef_


# ============================================================
# REGRESSION RANKER
# ============================================================

class RegressionRanker:
    def __init__(self, features, n_iterations=N_BOOTSTRAP_ITERATIONS, n_jobs=N_JOBS):
        self.features = features
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs
        cap_cols = [f for f in features if f not in OUTLIER_EXCLUDE]
        self.pipeline = Pipeline([
            ('outliers', OutlierCapper(columns=cap_cols)),
            ('transform', FeatureTransformer(LOG_VARS, LOGIT_VARS)),
            ('scale', StandardScaler()),
        ])
        self.engine = LinearRegression()
        self.bootstrap_history_ = None

    def fit_and_calibrate(self, df, target, top_anchor):
        if isinstance(top_anchor, pd.Series):
            top_anchor = top_anchor.to_frame().T
        Xp = self.pipeline.fit_transform(df[self.features])
        y = df[target].values

        # Sign-flip "lower is better" features before constrained fitting so
        # every coefficient produced by ElasticNet(positive=True) corresponds
        # to a "higher → better" relationship. We flip the resulting
        # coefficients back at the end.
        flip_idx = [i for i, f in enumerate(self.features) if f in SIGN_FLIP_FEATURES]
        Xp_train = Xp.copy()
        for i in flip_idx:
            Xp_train[:, i] = -Xp_train[:, i]

        print(f"  Bootstrapping {self.n_iterations} iterations using {self.n_jobs} cores "
              f"(positive=True, sign-flipped: {[self.features[i] for i in flip_idx]})...")
        coefs = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_single_bootstrap)(Xp_train, y) for _ in range(self.n_iterations)
        )
        self.bootstrap_history_ = np.array(coefs)
        # Flip the sign back for the lower-is-better features.
        for i in flip_idx:
            self.bootstrap_history_[:, i] = -self.bootstrap_history_[:, i]
        self.engine.coef_ = np.mean(self.bootstrap_history_, axis=0)

        x_top = self.pipeline.transform(top_anchor[self.features])
        raw_top = float(np.dot(x_top, self.engine.coef_)[0])
        self.engine.intercept_ = 100.0 - raw_top
        print(f"  Calibration intercept: {self.engine.intercept_:.4f}")

    def predict(self, df):
        Xp = self.pipeline.transform(df[self.features])
        return self.engine.predict(Xp)

    def transform(self, df):
        return self.pipeline.transform(df[self.features])

    def significance_report(self):
        lo = np.percentile(self.bootstrap_history_, 2.5, axis=0)
        hi = np.percentile(self.bootstrap_history_, 97.5, axis=0)
        rows = []
        for i, f in enumerate(self.features):
            rows.append({
                'feature': f,
                'mean_weight': float(self.engine.coef_[i]),
                'lower_95_ci': float(lo[i]),
                'upper_95_ci': float(hi[i]),
                'is_significant': not (lo[i] <= 0 <= hi[i]),
            })
        return rows


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(data_paths):
    print(f"\n{'='*60}\nPHASE 1a: PREPROCESSING (2-YEAR + 9-INDICATOR)\n{'='*60}")
    frames = []
    for year, p in data_paths.items():
        print(f"\n  Loading {year}: {p}")
        d = pd.read_csv(p, low_memory=False)
        d = d.rename(columns=COLUMN_RENAME_MAP)
        d['Year'] = year
        frames.append(d)
    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\n  Stacked shape: {df.shape}")

    df['SalaryByProfession'] = compute_salary_by_profession(df, year_col='Year')
    n_sba = df['SalaryByProfession'].notna().sum()
    print(f"  Salary-by-Profession computed for {n_sba} / {len(df)} rows "
          f"(range: [{df['SalaryByProfession'].min():.3f}, {df['SalaryByProfession'].max():.3f}])")

    df = compute_gmat_gre_blend(df, year_col='Year')
    n_gmat = df['GMAT_Combined'].notna().sum()
    print(f"  GMAT_Combined computed for {n_gmat} / {len(df)} rows "
          f"(range: [{df['GMAT_Combined'].min():.2f}, {df['GMAT_Combined'].max():.2f}])")

    impute_cols = [f for f in ALL_FEATURES + [TARGET] if f in df.columns]
    pre_miss = df[impute_cols].isnull().sum()
    print(f"\n  Missing BEFORE imputation:")
    for c, n in pre_miss.items():
        if n: print(f"    {c}: {n} ({n/len(df)*100:.1f}%)")
    parts = []
    for year, sub in df.groupby('Year'):
        sub = sub.copy()
        cols = [c for c in impute_cols if c in sub.columns]
        if sub[cols].isnull().sum().sum() > 0:
            imp = KNNImputer(n_neighbors=5, weights='distance')
            sub[cols] = imp.fit_transform(sub[cols])
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True, sort=False)

    # Cohort-floor fallback for GMAT_Combined and SalaryByProfession (per-year)
    for year, sub in df.groupby('Year'):
        floor = sub.loc[sub['GMAT_Combined'].notna(), 'GMAT_Combined'].min()
        if pd.isna(floor): floor = 0.0
        idx = sub.index[sub['GMAT_Combined'].isna()]
        df.loc[idx, 'GMAT_Combined'] = floor
        floor2 = sub.loc[sub['SalaryByProfession'].notna(), 'SalaryByProfession'].min()
        if pd.isna(floor2): floor2 = 1.0
        idx2 = sub.index[sub['SalaryByProfession'].isna()]
        df.loc[idx2, 'SalaryByProfession'] = floor2

    miss_after = df[ALL_FEATURES + [TARGET]].isnull().sum().sum()
    print(f"\n  Missing AFTER imputation: {miss_after}")
    assert miss_after == 0
    print(f"  Final shape: {df.shape}")
    return df


# ============================================================
# EXPORT
# ============================================================

def _write_json(data, output_dir, filename):
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [OK] {filename}")


def export_artifacts(df, regression, perf, output_dir):
    print(f"\n{'='*60}\nEXPORTING ARTIFACTS\n{'='*60}")
    os.makedirs(output_dir, exist_ok=True)

    snap_year = INFERENCE_CURVE_YEAR
    snap_df = df[df['Year'] == snap_year].copy()

    # 1. Model config
    _write_json({
        'features': ALL_FEATURES,
        'target': TARGET,
        'log_vars': LOG_VARS,
        'logit_vars': LOGIT_VARS,
        'outlier_exclude': sorted(OUTLIER_EXCLUDE),
    }, output_dir, 'model_config.json')

    # 2. OutlierCapper bounds
    cap = regression.pipeline.named_steps['outliers']
    _write_json({k: {'lower': float(v[0]), 'upper': float(v[1])} for k, v in cap.caps_.items()},
                output_dir, 'capper_bounds.json')

    # 3. Transformer config
    trans = regression.pipeline.named_steps['transform']
    _write_json({'log_cols': trans.log_cols, 'logit_cols': trans.logit_cols},
                output_dir, 'transformer_config.json')

    # 4. Scaler params
    scaler = regression.pipeline.named_steps['scale']
    _write_json({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'feature_names': ALL_FEATURES},
                output_dir, 'scaler_params.json')

    # 5. Model weights
    _write_json({'coef': regression.engine.coef_.tolist(), 'intercept': float(regression.engine.intercept_)},
                output_dir, 'model_weights.json')

    # 6. Data snapshot (most recent year, with imputed indicators)
    snapshot_cols = ['School', 'Rank', TARGET] + ALL_FEATURES
    _write_json(snap_df[snapshot_cols].to_dict(orient='records'),
                output_dir, 'data_snapshot.json')

    # 7. GMAT/GRE inference curves (per-exam sorted scores from snapshot year)
    src = df[df['Year'] == snap_year]
    def _sorted(col):
        if col not in src.columns: return []
        return sorted(float(x) for x in src[col].dropna().values)
    _write_json({
        'year': int(snap_year),
        'gmat_old': _sorted('GMAT_Old'),
        'gmat_new': _sorted('GMAT_New'),
        'gre_q':    _sorted('GRE_Q'),
        'gre_v':    _sorted('GRE_V'),
        'gre_aw':   _sorted('GRE_AW'),
    }, output_dir, 'gmat_inference_curves.json')

    # 8. Feature ranges + GWU current values + SBP cohort + GMAT input config
    feature_ranges = {}
    for f in ALL_FEATURES:
        v = snap_df[f]
        feature_ranges[f] = {
            'data_min': float(v.min()), 'data_max': float(v.max()),
            'data_mean': float(v.mean()), 'data_median': float(v.median()),
            **SLIDER_RANGES.get(f, {}),
        }
    feature_ranges['_gmat_input_config'] = GMAT_INPUT_CONFIG

    sbp_cohort = {}
    for occ in SBP_OCCUPATIONS:
        rows = []
        for i in range(8):
            occ_col = f'base_salary_by_occupation[{i}].occupation'
            sal_col = f'base_salary_by_occupation[{i}].average_salary'
            n_col   = f'base_salary_by_occupation[{i}].number_reporting_jobs'
            if occ_col not in df.columns: continue
            sub = df[(df['Year'] == snap_year) & (df[occ_col] == occ)]
            for _, r in sub.iterrows():
                sal = r.get(sal_col); n = r.get(n_col)
                if pd.isna(sal) or pd.isna(n) or n < MIN_OCCUPATION_REPORTERS: continue
                rows.append((float(sal), float(n)))
        if rows:
            ntot = sum(r[1] for r in rows)
            sbp_cohort[occ] = {
                'cohort_mean': float(sum(r[0]*r[1] for r in rows) / ntot),
                'cohort_n_total': int(ntot),
            }
    feature_ranges['_sbp_cohort'] = sbp_cohort
    feature_ranges['_sbp_occupations'] = SBP_OCCUPATIONS
    feature_ranges['_sbp_slider'] = SBP_SLIDER

    gwu = snap_df[snap_df['School'].str.contains('George Washington', case=False, na=False)]
    if not gwu.empty:
        gwu_full = df[(df['School'].str.contains('George Washington', case=False, na=False))
                      & (df['Year'] == snap_year)].iloc[0]
        gwu_vals = {f: float(gwu.iloc[0][f]) for f in ALL_FEATURES}
        for raw_col, key in (('GMAT_Old', 'gmat_old'), ('GMAT_New', 'gmat_new'),
                             ('GRE_Q', 'gre_q'), ('GRE_V', 'gre_v'), ('GRE_AW', 'gre_aw'),
                             ('Pct_GMAT_Old', 'pct_gmat_old'), ('Pct_GMAT_New', 'pct_gmat_new'),
                             ('Pct_GRE', 'pct_gre'),
                             ('student_body_fulltime_mba.enrollment', 'fulltime_enrollment')):
            v = gwu_full.get(raw_col, np.nan)
            gwu_vals[key] = None if pd.isna(v) else float(v)
        gwu_sbp = {}
        for i in range(8):
            occ_col = f'base_salary_by_occupation[{i}].occupation'
            sal_col = f'base_salary_by_occupation[{i}].average_salary'
            n_col   = f'base_salary_by_occupation[{i}].number_reporting_jobs'
            occ = gwu_full.get(occ_col)
            if not isinstance(occ, str) or occ.strip() not in SBP_OCCUPATIONS: continue
            sal = gwu_full.get(sal_col); n = gwu_full.get(n_col)
            gwu_sbp[occ.strip()] = {
                'salary': None if pd.isna(sal) else float(sal),
                'n_reporting': None if pd.isna(n) else float(n),
            }
        gwu_vals['sbp_per_occupation'] = gwu_sbp
        feature_ranges['_gwu_current'] = gwu_vals
        feature_ranges['_gwu_school_name'] = str(gwu.iloc[0]['School'])
        feature_ranges['_gwu_current_rank'] = int(gwu.iloc[0]['Rank'])
        feature_ranges['_gwu_current_score'] = float(gwu.iloc[0][TARGET])
    _write_json(feature_ranges, output_dir, 'feature_ranges.json')

    # 9. Explainability (regression performance + coefficients + contributions)
    coef_rows = regression.significance_report()
    Xs = regression.transform(df)
    coef = np.asarray(regression.engine.coef_)
    contribs = Xs * coef[np.newaxis, :]
    avg_abs = np.mean(np.abs(contribs), axis=0)
    avg_abs_pct = (avg_abs / avg_abs.sum() * 100.0).tolist() if avg_abs.sum() > 0 else [0.0] * len(coef)

    gwu_pct = []
    gwu_in = df[(df['School'].str.contains('George Washington', case=False, na=False))
                & (df['Year'] == snap_year)]
    if not gwu_in.empty:
        idx_pos = list(df.index).index(gwu_in.index[0])
        signed = contribs[idx_pos]
        denom = np.sum(np.abs(signed))
        if denom > 0:
            gwu_pct = (signed / denom * 100.0).tolist()

    explainability = {
        'regression': {
            'coefficients': coef_rows,
            'intercept': float(regression.engine.intercept_),
            'avg_abs_contribution_pct': [
                {'feature': f, 'pct': float(p)} for f, p in zip(ALL_FEATURES, avg_abs_pct)
            ],
            'gwu_contribution_pct': [
                {'feature': f, 'signed_pct': float(p)} for f, p in zip(ALL_FEATURES, gwu_pct)
            ] if gwu_pct else [],
            'performance': perf,
        },
        'methodology': {
            'regression': (
                f"Bootstrapped ElasticNetCV ({N_BOOTSTRAP_ITERATIONS} iterations) over the 9 "
                "features on stacked 2024+2025 data (~243 observations). Includes log/logit "
                "transforms and outlier capping (excluding GMAT_Combined and SalaryByProfession, "
                "which are already on bounded scales). Calibrated so the rank-1 school = 100."
            ),
            'gmat_blend': (
                "Per-year percentile rank for GMAT_old, GMAT_new, GRE_Q, GRE_V, GRE_AW. "
                "GRE-internal: 0.4*Q + 0.4*V + 0.2*AW. Cross-exam blend weighted by "
                "submission proportions (renormalised). Median GRE Q/V/AW approximated "
                "from 10th-90th range midpoints. No-input fallback: per-year cohort floor."
            ),
        },
    }
    _write_json(explainability, output_dir, 'model_explainability.json')

    print(f"\n  All artifacts exported to: {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("GWSB RANKING PREDICTOR - REGRESSION TRAINING")
    print("=" * 60)

    df = preprocess(DATA_PATHS)

    print(f"\n{'='*60}\nTRAINING REGRESSION (BOOTSTRAPPED ELASTICNET, 9 FEATURES)\n{'='*60}")
    cur_year = max(df['Year'].unique())
    top_anchor = df[(df['Year'] == cur_year) & (df['Rank'] == 1)].head(1)
    if top_anchor.empty:
        top_anchor = df[df['Rank'] == 1].head(1)
    print(f"  Calibration anchor: {top_anchor.iloc[0]['School']} "
          f"(Year {int(top_anchor.iloc[0]['Year'])}, Rank {int(top_anchor.iloc[0]['Rank'])})")

    reg = RegressionRanker(features=ALL_FEATURES)
    reg.fit_and_calibrate(df, target=TARGET, top_anchor=top_anchor)

    preds = reg.predict(df)
    actuals = df[TARGET].values
    perf = {
        'mae': float(mean_absolute_error(actuals, preds)),
        'rmse': float(np.sqrt(mean_squared_error(actuals, preds))),
        'r2': float(r2_score(actuals, preds)),
        'spearman': float(spearmanr(actuals, preds)[0]),
        'n_observations': int(len(df)),
        'n_bootstrap_iterations': int(N_BOOTSTRAP_ITERATIONS),
        'training_years': sorted(int(y) for y in df['Year'].unique()),
    }
    print(f"\n  vs published OverallScore:")
    print(f"    MAE:      {perf['mae']:.4f}")
    print(f"    RMSE:     {perf['rmse']:.4f}")
    print(f"    R^2:      {perf['r2']:.4f}")
    print(f"    Spearman: {perf['spearman']:.4f}")
    print(f"\n  Coefficients:")
    for r in reg.significance_report():
        print(f"    {r['feature']:24s} mean={r['mean_weight']:7.3f}  "
              f"CI=[{r['lower_95_ci']:7.3f}, {r['upper_95_ci']:7.3f}]  "
              f"{'sig' if r['is_significant'] else 'n.s.'}")

    export_artifacts(df, reg, perf, OUTPUT_DIR)
    print(f"\n{'='*60}\nTRAINING COMPLETE\n{'='*60}")


if __name__ == '__main__':
    main()
