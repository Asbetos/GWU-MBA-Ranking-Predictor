"""
GWU Ranking Predictor - Model Training Script
==============================================
Two-year (2024 + 2025) training pipeline with US-News-style GMAT/GRE blended
percentile transformation.

Steps:
  1a. Load 2024 + 2025 CSVs, rename columns, parse GRE ranges, KNN-impute per-year,
      compute blended GMAT_Combined feature (per-year percentile ranks weighted by
      submission percentages, with <25% submission penalty).
  1b. Train USNewsRankingSystem (OutlierCapper -> RankingFeatureTransformer ->
      StandardScaler -> Bootstrapped ElasticNetCV -> Calibration).
  2.  Export 9 model artifacts as JSON for the JS serverless backend, including
      model_explainability.json and gmat_inference_curves.json.
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
from scipy.stats import spearmanr, norm, rankdata
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
INFERENCE_CURVE_YEAR = 2025  # Year used for client-side percentile lookups

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

COLUMNS_TO_DROP = [
    'school_info.us_news_rank_out_of',
    'ranking_scores_two_year_averages.salaries_by_profession_indicator_rank',
]

ALL_FEATURES = [
    'EmployedAtGrad', 'Employed3Mo', 'AvgSalaryBonus',
    'MedianGPA', 'AcceptanceRate', 'PeerScore',
    'RecruiterScore', 'GMAT_Combined'
]
TARGET = 'OverallScore'

LOG_VARS = ['AvgSalaryBonus']  # GMAT_Combined removed - now bounded 0-100
LOGIT_VARS = ['EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate']
INV_NORM_VARS = []
OUTLIER_CAP_FEATURES = [f for f in ALL_FEATURES if f != 'GMAT_Combined']  # blended score already bounded

GMAT_BLEND_THRESHOLD = 0.25  # min total submission % before penalty
N_BOOTSTRAP_ITERATIONS = 10000
N_JOBS = 4

# Inference-side input config (consumed by frontend via feature_ranges.json)
GMAT_INPUT_CONFIG = {
    'gmat_scale_default': 'old',
    'gmat_old_range': {'min': 500, 'max': 800, 'step': 5},
    'gmat_new_range': {'min': 505, 'max': 805, 'step': 5},
    'gre_range':      {'min': 280, 'max': 340, 'step': 1},
    'gre_default_enabled': False,
}

SLIDER_RANGES = {
    'EmployedAtGrad':  {'min': 0.20, 'max': 1.00, 'step': 0.01, 'label': 'Employed at Graduation', 'format': 'percent'},
    'Employed3Mo':     {'min': 0.20, 'max': 1.00, 'step': 0.01, 'label': 'Employed 3 Months After', 'format': 'percent'},
    'AvgSalaryBonus':  {'min': 80000, 'max': 220000, 'step': 1000, 'label': 'Avg Salary + Bonus ($)', 'format': 'dollar'},
    'MedianGPA':       {'min': 3.0,  'max': 4.0,  'step': 0.01, 'label': 'Median GPA', 'format': 'number'},
    'AcceptanceRate':  {'min': 0.05, 'max': 1.00, 'step': 0.01, 'label': 'Acceptance Rate', 'format': 'percent'},
    'PeerScore':       {'min': 1.0,  'max': 5.0,  'step': 0.1,  'label': 'Peer Assessment Score', 'format': 'number'},
    'RecruiterScore':  {'min': 1.0,  'max': 5.0,  'step': 0.1,  'label': 'Recruiter Assessment Score', 'format': 'number'},
    # GMAT_Combined slider is rendered as a composite control on the frontend; the
    # data_min/data_max from feature_ranges.json still describe the blended-score range.
    'GMAT_Combined':   {'min': 0,    'max': 100,  'step': 1,    'label': 'GMAT/GRE Blended Percentile Score', 'format': 'number'},
}

# ============================================================
# 1. CUSTOM TRANSFORMERS
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
                    float(X[col].quantile(1 - self.limits[1]))
                )
        return self

    def transform(self, X):
        X_trans = X.copy()
        for col, (lower, upper) in self.caps_.items():
            if col in X_trans.columns:
                X_trans[col] = X_trans[col].clip(lower, upper)
        return X_trans


class RankingFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_cols, logit_cols, inv_norm_cols):
        self.log_cols = log_cols
        self.logit_cols = logit_cols
        self.inv_norm_cols = inv_norm_cols
        self.rank_counts_ = {}

    def fit(self, X, y=None):
        for col in self.inv_norm_cols:
            if col in X.columns:
                self.rank_counts_[col] = float(X[col].max())
        return self

    def transform(self, X):
        X_trans = X.copy()
        inv_norm_cols = getattr(self, 'inv_norm_cols', [])
        rank_counts = getattr(self, 'rank_counts_', {})

        for col in self.log_cols:
            if col in X_trans.columns:
                X_trans[col] = np.log1p(X_trans[col])

        for col in self.logit_cols:
            if col in X_trans.columns:
                p = X_trans[col].clip(0.001, 0.999)
                X_trans[col] = np.log(p / (1 - p))

        for col in inv_norm_cols:
            if col in X_trans.columns:
                N = rank_counts.get(col, 120)
                percentile = ((X_trans[col] - 0.5) / N).clip(0.001, 0.999)
                X_trans[col] = -1 * norm.ppf(percentile)

        return X_trans


# ============================================================
# 2. PARALLEL BOOTSTRAP HELPER
# ============================================================

def _run_single_bootstrap(X, y):
    X_b, y_b = resample(X, y)
    en = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=10000, n_jobs=1)
    en.fit(X_b, y_b)
    return en.coef_


# ============================================================
# 3. CORE RANKING SYSTEM
# ============================================================

class USNewsRankingSystem:
    def __init__(self, features, target, log_vars, logit_vars, inv_norm_vars,
                 outlier_features=None, n_iterations=100, n_jobs=-1):
        self.features = features
        self.target = target
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs
        cap_cols = outlier_features if outlier_features is not None else features
        self.pipeline = Pipeline([
            ('outliers', OutlierCapper(columns=cap_cols)),
            ('transform', RankingFeatureTransformer(log_vars, logit_vars, inv_norm_vars)),
            ('scale', StandardScaler())
        ])
        self.engine = LinearRegression()
        self.bootstrap_history_ = None

    def fit_and_calibrate(self, df, global_top_ref):
        if isinstance(global_top_ref, pd.Series):
            global_top_ref = global_top_ref.to_frame().T

        X_processed = self.pipeline.fit_transform(df[self.features])
        y = df[self.target].values

        print(f"Bootstrapping {self.n_iterations} iterations using {self.n_jobs} cores...")

        coef_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_single_bootstrap)(X_processed, y) for _ in range(self.n_iterations)
        )

        self.bootstrap_history_ = np.array(coef_list)
        self.engine.coef_ = np.mean(self.bootstrap_history_, axis=0)

        x_global_top = self.pipeline.transform(global_top_ref[self.features])
        raw_top_score = np.dot(x_global_top, self.engine.coef_)[0]
        self.engine.intercept_ = 100 - raw_top_score

        print(f"Calibration Complete. Intercept: {self.engine.intercept_:.4f}")

    def predict(self, df):
        X_p = self.pipeline.transform(df[self.features])
        return self.engine.predict(X_p)

    def transform(self, df):
        return self.pipeline.transform(df[self.features])

    def get_significance_report(self):
        if self.bootstrap_history_ is None:
            return "Model not fitted yet."

        lower_ci = np.percentile(self.bootstrap_history_, 2.5, axis=0)
        upper_ci = np.percentile(self.bootstrap_history_, 97.5, axis=0)
        means = self.engine.coef_

        report = pd.DataFrame({
            'Feature': self.features,
            'Mean_Weight': means,
            'Lower_95_CI': lower_ci,
            'Upper_95_CI': upper_ci
        })
        report['Is_Significant'] = ~((report['Lower_95_CI'] <= 0) & (report['Upper_95_CI'] >= 0))
        return report.sort_values('Mean_Weight', key=lambda s: s.abs(), ascending=False)


# ============================================================
# GRE PARSING
# ============================================================

def parse_gre_range(s):
    """Parse 'NNN-NNN verbal, NNN-NNN quantitative, F.F-F.F writing' -> total midpoint.

    Returns NaN on missing/empty/unparsable input. Skips writing pairs (values < 10).
    Returns (verbal_mid + quant_mid).
    """
    if not isinstance(s, str) or not s.strip():
        return np.nan
    text = s.lower()
    # Find all numeric range pairs (ints or floats); we only keep integer pairs with both >= 10.
    pairs = re.findall(r'(\d{2,3}(?:\.\d+)?)\s*[-–]\s*(\d{2,3}(?:\.\d+)?)', text)
    if not pairs:
        return np.nan

    # Try to label by adjacent words
    labeled = {}
    for m in re.finditer(r'(\d{2,3}(?:\.\d+)?)\s*[-–]\s*(\d{2,3}(?:\.\d+)?)\s*([a-z]+)?', text):
        lo, hi, lbl = m.group(1), m.group(2), (m.group(3) or '')
        try:
            lo_f, hi_f = float(lo), float(hi)
        except ValueError:
            continue
        if lo_f < 10 or hi_f < 10:
            continue  # writing
        mid = (lo_f + hi_f) / 2.0
        if 'verbal' in lbl:
            labeled['verbal'] = mid
        elif 'quant' in lbl:
            labeled['quant'] = mid

    if 'verbal' in labeled and 'quant' in labeled:
        return labeled['verbal'] + labeled['quant']

    # Fallback: first two int pairs (>=10) are verbal then quantitative
    int_pairs = [(float(a), float(b)) for a, b in pairs if float(a) >= 10 and float(b) >= 10]
    if len(int_pairs) >= 2:
        v_mid = (int_pairs[0][0] + int_pairs[0][1]) / 2.0
        q_mid = (int_pairs[1][0] + int_pairs[1][1]) / 2.0
        return v_mid + q_mid
    return np.nan


# ============================================================
# BLENDED GMAT FEATURE
# ============================================================

def compute_blended_gmat_feature(df, year_col='Year'):
    """Compute GMAT_Combined as a 0-100 blended percentile score.

    Per-year percentile ranks for GMAT_Old, GMAT_New, GRE_Total, weighted by each
    school's submission percentages, with <25% total-submission penalty.
    """
    df = df.copy()

    # Coerce submission %s to fractions in [0, 1]; fill missing with 0
    for col in ('Pct_GMAT_Old', 'Pct_GMAT_New', 'Pct_GRE'):
        if col not in df.columns:
            df[col] = 0.0
        # Some sources use percentages (>1); coerce robustly.
        col_max = df[col].dropna().max() if df[col].notna().any() else 0
        if col_max is not None and col_max > 1.5:
            df[col] = df[col] / 100.0
        df[col] = df[col].fillna(0.0).clip(0.0, 1.0)

    # Per-year percentile ranks for each test (rank/N on non-null subset; null -> 0)
    score_cols = {'GMAT_Old': 'rank_old', 'GMAT_New': 'rank_new', 'GRE_Total': 'rank_gre'}
    for rcol in score_cols.values():
        df[rcol] = 0.0

    for year, sub in df.groupby(year_col):
        for sc, rc in score_cols.items():
            if sc not in df.columns:
                continue
            mask = sub[sc].notna()
            if not mask.any():
                continue
            vals = sub.loc[mask, sc].values
            ranks = rankdata(vals, method='average') / float(len(vals))
            df.loc[sub.index[mask], rc] = ranks

    # Blend
    p_old = df['Pct_GMAT_Old'].values
    p_new = df['Pct_GMAT_New'].values
    p_gre = df['Pct_GRE'].values
    r_old = df['rank_old'].values
    r_new = df['rank_new'].values
    r_gre = df['rank_gre'].values

    total_pct = p_old + p_new + p_gre
    weighted = p_old * r_old + p_new * r_new + p_gre * r_gre

    with np.errstate(divide='ignore', invalid='ignore'):
        blended = np.where(total_pct > 0, weighted / total_pct, np.nan)

    penalty = np.where(total_pct < GMAT_BLEND_THRESHOLD, total_pct / GMAT_BLEND_THRESHOLD, 1.0)
    blended = blended * penalty
    blended_score = blended * 100.0

    # NaN fallback: per-year median of blended_score
    df['GMAT_Combined'] = blended_score
    for year, sub in df.groupby(year_col):
        med = np.nanmedian(sub['GMAT_Combined'].values)
        if np.isnan(med):
            med = 50.0
        idx = sub.index[sub['GMAT_Combined'].isna()]
        df.loc[idx, 'GMAT_Combined'] = med

    df['GMAT_Combined'] = df['GMAT_Combined'].clip(0.0, 100.0)
    return df


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_raw_data(data_paths):
    """Load 2 years, rename, parse GRE, KNN-impute per-year, blend GMAT."""
    print(f"\n{'='*60}")
    print("PHASE 1a: PREPROCESSING (2-YEAR)")
    print(f"{'='*60}")

    frames = []
    for year, path in data_paths.items():
        print(f"\n  Loading {year}: {path}")
        d = pd.read_csv(path, low_memory=False)
        print(f"    Raw shape: {d.shape}")
        for col in COLUMNS_TO_DROP:
            if col in d.columns:
                d = d.drop(columns=[col])
        d = d.rename(columns=COLUMN_RENAME_MAP)
        d['Year'] = year
        frames.append(d)

    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\n  Stacked shape: {df.shape}")

    # Parse GRE -> numeric total
    if 'GRE_Range_Str' in df.columns:
        df['GRE_Total'] = df['GRE_Range_Str'].apply(parse_gre_range)
        n_parsed = df['GRE_Total'].notna().sum()
        print(f"  GRE parsed: {n_parsed} / {len(df)} rows have a numeric GRE_Total")
    else:
        df['GRE_Total'] = np.nan
        print("  WARNING: GRE_Range_Str column missing")

    # Numeric columns to KNN-impute (per year): all model features + raw test scores
    impute_cols = list(set([
        'EmployedAtGrad', 'Employed3Mo', 'AvgSalaryBonus',
        'MedianGPA', 'AcceptanceRate', 'PeerScore', 'RecruiterScore',
        'GMAT_Old', 'GMAT_New', 'GRE_Total',
        'Pct_GMAT_Old', 'Pct_GMAT_New', 'Pct_GRE',
        TARGET,
    ]))
    impute_cols = [c for c in impute_cols if c in df.columns]

    print(f"\n  Missing BEFORE imputation:")
    for col in impute_cols:
        miss = df[col].isnull().sum()
        if miss:
            print(f"    {col}: {miss} ({miss/len(df)*100:.1f}%)")

    print(f"\n  KNN imputation per-year (n_neighbors=5)...")
    parts = []
    for year, sub in df.groupby('Year'):
        sub = sub.copy()
        # Submission percentages: missing means 'didn't submit', not unknown -> fill with 0
        for col in ('Pct_GMAT_Old', 'Pct_GMAT_New', 'Pct_GRE'):
            if col in sub.columns:
                sub[col] = sub[col].fillna(0.0)
        # KNN impute the rest of impute_cols within this year
        impute_now = [c for c in impute_cols if c in sub.columns]
        # Need at least 5 non-NaN samples per column for KNN; rely on KNNImputer default behaviour
        if sub[impute_now].isnull().sum().sum() > 0:
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            sub[impute_now] = imputer.fit_transform(sub[impute_now])
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True, sort=False)

    # Compute the blended GMAT feature from imputed inputs (per-year ranks)
    df = compute_blended_gmat_feature(df, year_col='Year')

    # Validate: only the model features need to be non-null
    miss_after = df[ALL_FEATURES + [TARGET]].isnull().sum().sum()
    print(f"\n  Missing AFTER imputation+blend (model features+target): {miss_after}")
    assert miss_after == 0, "Imputation did not fill all model-feature missing values!"

    print(f"\n  Final shape: {df.shape}")
    print(f"  GMAT_Combined range: [{df['GMAT_Combined'].min():.2f}, {df['GMAT_Combined'].max():.2f}]")
    print(f"  GMAT_Combined mean/median: {df['GMAT_Combined'].mean():.2f} / {df['GMAT_Combined'].median():.2f}")

    return df


# ============================================================
# EXPORT ARTIFACTS
# ============================================================

def export_artifacts(ranking_system, df_imputed, output_dir, perf, sig_report):
    """Export all model artifacts as JSON for the JS frontend."""
    print(f"\n{'='*60}")
    print("EXPORTING ARTIFACTS")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    pipeline = ranking_system.pipeline

    # 1. Model config
    config = {
        'features': ranking_system.features,
        'target': ranking_system.target,
        'log_vars': LOG_VARS,
        'logit_vars': LOGIT_VARS,
        'inv_norm_vars': INV_NORM_VARS,
        'outlier_cap_features': OUTLIER_CAP_FEATURES,
    }
    _write_json(config, output_dir, 'model_config.json')

    # 2. OutlierCapper bounds
    capper = pipeline.named_steps['outliers']
    caps = {k: {'lower': float(v[0]), 'upper': float(v[1])} for k, v in capper.caps_.items()}
    _write_json(caps, output_dir, 'capper_bounds.json')

    # 3. RankingFeatureTransformer config
    transformer = pipeline.named_steps['transform']
    trans_config = {
        'log_cols': transformer.log_cols,
        'logit_cols': transformer.logit_cols,
        'inv_norm_cols': transformer.inv_norm_cols,
        'rank_counts': {k: float(v) for k, v in transformer.rank_counts_.items()},
    }
    _write_json(trans_config, output_dir, 'transformer_config.json')

    # 4. StandardScaler params
    scaler = pipeline.named_steps['scale']
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_names': ranking_system.features,
    }
    _write_json(scaler_params, output_dir, 'scaler_params.json')

    # 5. Model weights
    model_weights = {
        'coef': ranking_system.engine.coef_.tolist(),
        'intercept': float(ranking_system.engine.intercept_),
    }
    _write_json(model_weights, output_dir, 'model_weights.json')

    # 6. Data snapshot - use most recent year only for the simulation cohort
    snapshot_year = INFERENCE_CURVE_YEAR
    snap = df_imputed[df_imputed['Year'] == snapshot_year].copy()
    if snap.empty:
        snap = df_imputed.copy()
    snapshot_cols = ['School', 'Rank', TARGET] + ranking_system.features
    snapshot = snap[snapshot_cols].to_dict(orient='records')
    _write_json(snapshot, output_dir, 'data_snapshot.json')

    # 7. Feature ranges + GMAT input config + GWU current values
    feature_ranges = {}
    for feat in ranking_system.features:
        vals = snap[feat]
        feature_ranges[feat] = {
            'data_min': float(vals.min()),
            'data_max': float(vals.max()),
            'data_mean': float(vals.mean()),
            'data_median': float(vals.median()),
            **SLIDER_RANGES.get(feat, {}),
        }

    feature_ranges['_gmat_input_config'] = GMAT_INPUT_CONFIG

    gwu_row = snap[snap['School'].str.contains('George Washington', case=False, na=False)]
    if not gwu_row.empty:
        gwu_values = {feat: float(gwu_row.iloc[0][feat]) for feat in ranking_system.features}
        # Add raw test inputs (or null) so the slider can default to a real value
        gwu_extras = {}
        for raw_col, key in (('GMAT_Old', 'gmat_old'), ('GMAT_New', 'gmat_new'),
                             ('GRE_Total', 'gre_total'),
                             ('Pct_GMAT_Old', 'pct_gmat_old'),
                             ('Pct_GMAT_New', 'pct_gmat_new'),
                             ('Pct_GRE', 'pct_gre')):
            if raw_col in gwu_row.columns:
                v = gwu_row.iloc[0][raw_col]
                gwu_extras[key] = None if pd.isna(v) else float(v)
            else:
                gwu_extras[key] = None
        gwu_values.update(gwu_extras)
        feature_ranges['_gwu_current'] = gwu_values
        feature_ranges['_gwu_school_name'] = str(gwu_row.iloc[0]['School'])
        feature_ranges['_gwu_current_rank'] = int(gwu_row.iloc[0]['Rank'])
        feature_ranges['_gwu_current_score'] = float(gwu_row.iloc[0][TARGET])
    else:
        print("  WARNING: George Washington University not found in dataset!")

    _write_json(feature_ranges, output_dir, 'feature_ranges.json')

    # 8. GMAT inference curves (per-year sorted scores from snapshot year)
    curves_year = INFERENCE_CURVE_YEAR
    src = df_imputed[df_imputed['Year'] == curves_year]
    if src.empty:
        src = df_imputed
    def _sorted_nonnull(col):
        if col not in src.columns:
            return []
        v = src[col].dropna().values
        return sorted(float(x) for x in v)
    curves = {
        'year': curves_year,
        'gmat_old': _sorted_nonnull('GMAT_Old'),
        'gmat_new': _sorted_nonnull('GMAT_New'),
        'gre_total': _sorted_nonnull('GRE_Total'),
    }
    _write_json(curves, output_dir, 'gmat_inference_curves.json')

    # 9. Model performance + explainability
    # Per-row signed contributions: scaled[i] * coef[i]
    X_scaled = ranking_system.transform(df_imputed)  # numpy 2D
    coef = np.asarray(ranking_system.engine.coef_)
    contrib = X_scaled * coef[np.newaxis, :]  # shape (n_rows, n_feats)
    avg_abs = np.mean(np.abs(contrib), axis=0)
    avg_abs_pct = (avg_abs / avg_abs.sum() * 100.0).tolist() if avg_abs.sum() > 0 else [0.0] * len(coef)

    gwu_pct = []
    gwu_idx_local = None
    gwu_in_imputed = df_imputed[(df_imputed['School'].str.contains('George Washington', case=False, na=False)) &
                                (df_imputed['Year'] == INFERENCE_CURVE_YEAR)]
    if not gwu_in_imputed.empty:
        gwu_idx_local = gwu_in_imputed.index[0]
        # Locate in the X_scaled order (df_imputed index is preserved)
        idx_pos = list(df_imputed.index).index(gwu_idx_local)
        signed = contrib[idx_pos]
        denom = np.sum(np.abs(signed))
        if denom > 0:
            gwu_pct = (signed / denom * 100.0).tolist()
        else:
            gwu_pct = [0.0] * len(coef)

    coef_rows = []
    for _, row in sig_report.iterrows():
        coef_rows.append({
            'feature': str(row['Feature']),
            'mean_weight': float(row['Mean_Weight']),
            'lower_95_ci': float(row['Lower_95_CI']),
            'upper_95_ci': float(row['Upper_95_CI']),
            'is_significant': bool(row['Is_Significant']),
        })

    explainability = {
        'performance': {
            'mae': float(perf['mae']),
            'rmse': float(perf['rmse']),
            'r2': float(perf['r2']),
            'spearman': float(perf['spearman']),
            'n_observations': int(len(df_imputed)),
            'n_bootstrap_iterations': int(N_BOOTSTRAP_ITERATIONS),
            'training_years': sorted(df_imputed['Year'].unique().tolist()),
        },
        'intercept': float(ranking_system.engine.intercept_),
        'coefficients': coef_rows,
        'avg_abs_contribution_pct': [
            {'feature': f, 'pct': float(p)}
            for f, p in zip(ranking_system.features, avg_abs_pct)
        ],
        'gwu_contribution_pct': [
            {'feature': f, 'signed_pct': float(p)}
            for f, p in zip(ranking_system.features, gwu_pct)
        ] if gwu_pct else [],
        'methodology': {
            'gmat_blend': (
                "GMAT_Combined is computed per-school as a 0-100 blended percentile score: "
                "(1) per-year percentile rank for each of GMAT_old, GMAT_new, GRE_total; "
                "(2) weighted average using each school's submission percentages; "
                "(3) penalty multiplier of min(1, total_submission/0.25) when total submission < 25%; "
                "(4) z-score standardization in the model pipeline."
            ),
            'training': f"Bootstrapped ElasticNetCV over {N_BOOTSTRAP_ITERATIONS} resamples on {len(df_imputed)} observations from {sorted(df_imputed['Year'].unique().tolist())}.",
        },
    }
    _write_json(explainability, output_dir, 'model_explainability.json')

    print(f"\n  All artifacts exported to: {output_dir}")


def _write_json(data, output_dir, filename):
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [OK] {filename}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("GWU RANKING PREDICTOR - MODEL TRAINING (2-YEAR + BLENDED GMAT/GRE)")
    print("=" * 60)

    df_imputed = preprocess_raw_data(DATA_PATHS)

    print(f"\n{'='*60}")
    print("PHASE 1b: MODEL TRAINING")
    print(f"{'='*60}")

    # Calibrate on the rank-1 school in the most recent year
    cur_year = max(df_imputed['Year'].unique())
    cur_top = df_imputed[(df_imputed['Year'] == cur_year) & (df_imputed['Rank'] == 1)]
    if cur_top.empty:
        # fallback: any year's rank-1
        cur_top = df_imputed[df_imputed['Rank'] == 1].head(1)
    global_top_school = cur_top.head(1)
    print(f"\n  Calibration anchor: {global_top_school.iloc[0]['School']} "
          f"(Year {int(global_top_school.iloc[0]['Year'])}, Rank {int(global_top_school.iloc[0]['Rank'])})")

    ranking_system = USNewsRankingSystem(
        features=ALL_FEATURES,
        target=TARGET,
        log_vars=LOG_VARS,
        logit_vars=LOGIT_VARS,
        inv_norm_vars=INV_NORM_VARS,
        outlier_features=OUTLIER_CAP_FEATURES,
        n_iterations=N_BOOTSTRAP_ITERATIONS,
        n_jobs=N_JOBS,
    )

    ranking_system.fit_and_calibrate(df_imputed, global_top_ref=global_top_school)

    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE")
    print(f"{'='*60}")

    preds = ranking_system.predict(df_imputed)
    actuals = df_imputed[TARGET].values
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    spearman, _ = spearmanr(actuals, preds)
    perf = {'mae': mae, 'rmse': rmse, 'r2': r2, 'spearman': spearman}

    print(f"  MAE:      {mae:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  R^2:      {r2:.4f}")
    print(f"  Spearman: {spearman:.4f}")

    sig_report = ranking_system.get_significance_report()
    print(f"\n  Coefficient Significance:")
    print(sig_report.to_string(index=False))

    export_artifacts(ranking_system, df_imputed, OUTPUT_DIR, perf, sig_report)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
