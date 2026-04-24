"""
GWU Ranking Predictor — Model Training Script
==============================================
Replicates the preprocessing + modeling pipeline from [NEW]us_news_model_script_v2.ipynb.

Steps:
  1a. Load raw CSV, rename columns, combine GMAT, KNN-impute missing values
  1b. Train USNewsRankingSystem (OutlierCapper → RankingFeatureTransformer → StandardScaler → Bootstrapped ElasticNetCV → Calibration)
  2.  Export all model artifacts as JSON for the JS serverless backend
"""

import os
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
from scipy.stats import spearmanr, norm
from joblib import Parallel, delayed

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'us_news_data_2025.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'public', 'model_artifacts')

# Column mapping: raw CSV names → model names
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
}

COLUMNS_TO_DROP = [
    'school_info.us_news_rank_out_of',
    'ranking_scores_two_year_averages.salaries_by_profession_indicator_rank',
]

# 8-feature model (no ProfessionSalaryRank)
ALL_FEATURES = [
    'EmployedAtGrad', 'Employed3Mo', 'AvgSalaryBonus',
    'MedianGPA', 'AcceptanceRate', 'PeerScore',
    'RecruiterScore', 'GMAT_Combined'
]
TARGET = 'OverallScore'
LOG_VARS = ['AvgSalaryBonus', 'GMAT_Combined']
LOGIT_VARS = ['EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate']
INV_NORM_VARS = []  # None for 8-feature model

N_BOOTSTRAP_ITERATIONS = 10000
N_JOBS = 4  # Use 4 cores (avoid issues with -1 on Windows)

# Fixed slider ranges (not data-dependent)
SLIDER_RANGES = {
    'EmployedAtGrad':  {'min': 0.20, 'max': 1.00, 'step': 0.01, 'label': 'Employed at Graduation', 'format': 'percent'},
    'Employed3Mo':     {'min': 0.20, 'max': 1.00, 'step': 0.01, 'label': 'Employed 3 Months After', 'format': 'percent'},
    'AvgSalaryBonus':  {'min': 80000, 'max': 220000, 'step': 1000, 'label': 'Avg Salary + Bonus ($)', 'format': 'dollar'},
    'MedianGPA':       {'min': 3.0,  'max': 4.0,  'step': 0.01, 'label': 'Median GPA', 'format': 'number'},
    'AcceptanceRate':  {'min': 0.05, 'max': 1.00, 'step': 0.01, 'label': 'Acceptance Rate', 'format': 'percent'},
    'PeerScore':       {'min': 1.0,  'max': 5.0,  'step': 0.1,  'label': 'Peer Assessment Score', 'format': 'number'},
    'RecruiterScore':  {'min': 1.0,  'max': 5.0,  'step': 0.1,  'label': 'Recruiter Assessment Score', 'format': 'number'},
    'GMAT_Combined':   {'min': 205,  'max': 805,  'step': 5,    'label': 'GMAT Score (Combined)', 'format': 'number'},
}

# ============================================================
# 1. CUSTOM TRANSFORMERS (from [NEW] notebook)
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

        # 1. Log Transformation
        for col in self.log_cols:
            if col in X_trans.columns:
                X_trans[col] = np.log1p(X_trans[col])

        # 2. Logit Transformation
        for col in self.logit_cols:
            if col in X_trans.columns:
                p = X_trans[col]
                p = p.clip(0.001, 0.999)
                X_trans[col] = np.log(p / (1 - p))

        # 3. Inverse Normal Transformation (Rank -> Z-Score)
        for col in inv_norm_cols:
            if col in X_trans.columns:
                N = rank_counts.get(col, 120)
                percentile = (X_trans[col] - 0.5) / N
                percentile = percentile.clip(0.001, 0.999)
                X_trans[col] = -1 * norm.ppf(percentile)

        return X_trans


# ============================================================
# 2. PARALLEL BOOTSTRAP HELPER
# ============================================================

def _run_single_bootstrap(X, y):
    """Helper function to run one iteration in parallel."""
    X_b, y_b = resample(X, y)
    en = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=10000, n_jobs=1)
    en.fit(X_b, y_b)
    return en.coef_


# ============================================================
# 3. CORE RANKING SYSTEM
# ============================================================

class USNewsRankingSystem:
    def __init__(self, features, target, log_vars, logit_vars, inv_norm_vars, n_iterations=100, n_jobs=-1):
        self.features = features
        self.target = target
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs
        self.pipeline = Pipeline([
            ('outliers', OutlierCapper(columns=features)),
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

        # Calibration: top school gets score = 100
        x_global_top = self.pipeline.transform(global_top_ref[self.features])
        raw_top_score = np.dot(x_global_top, self.engine.coef_)[0]
        self.engine.intercept_ = 100 - raw_top_score

        print(f"Calibration Complete. Intercept: {self.engine.intercept_:.4f}")

    def predict(self, df):
        X_p = self.pipeline.transform(df[self.features])
        return self.engine.predict(X_p)

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
        return report.sort_values('Mean_Weight', ascending=False)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_raw_data(data_path):
    """Load raw CSV, rename columns, combine GMAT, KNN-impute missing values."""
    print(f"\n{'='*60}")
    print("PHASE 1a: PREPROCESSING")
    print(f"{'='*60}")

    # 1. Load
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Raw shape: {df.shape}")
    print(f"  Raw columns: {list(df.columns)}")

    # 2. Drop unused columns
    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"  Dropped: {col}")

    # 3. Rename columns
    df = df.rename(columns=COLUMN_RENAME_MAP)
    print(f"\n  Renamed columns: {list(df.columns)}")

    # 4. Combine GMAT scores (prioritize Old, fallback to New)
    if 'GMAT_Old' in df.columns and 'GMAT_New' in df.columns:
        df['GMAT_Combined'] = df['GMAT_Old'].fillna(df['GMAT_New'])
        df = df.drop(columns=['GMAT_Old', 'GMAT_New'])
        print(f"  Combined GMAT_Old + GMAT_New -> GMAT_Combined")
    elif 'GMAT_Combined' in df.columns:
        print(f"  GMAT_Combined already exists")
    else:
        raise ValueError("Cannot find GMAT columns in the data")

    # 5. Check missing values before imputation
    print(f"\n  Missing values BEFORE imputation:")
    missing = df[ALL_FEATURES + [TARGET]].isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")

    # 6. KNN Imputation (using sklearn KNNImputer, as imported in [NEW] notebook)
    numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    if df[numeric_cols].isnull().sum().sum() > 0:
        print(f"\n  Running KNNImputer (n_neighbors=5) on {len(numeric_cols)} numeric columns...")
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print(f"  Imputation complete.")

    # 7. Validate
    missing_after = df[ALL_FEATURES + [TARGET]].isnull().sum().sum()
    print(f"\n  Missing values AFTER imputation: {missing_after}")
    assert missing_after == 0, "Imputation did not fill all missing values!"

    print(f"\n  Final shape: {df.shape}")
    print(f"  Final columns: {list(df.columns)}")

    return df


# ============================================================
# EXPORT ARTIFACTS
# ============================================================

def export_artifacts(ranking_system, df_imputed, output_dir):
    """Export all model artifacts as JSON for the JS serverless backend."""
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

    # 5. Model weights (LinearRegression)
    model_weights = {
        'coef': ranking_system.engine.coef_.tolist(),
        'intercept': float(ranking_system.engine.intercept_),
    }
    _write_json(model_weights, output_dir, 'model_weights.json')

    # 6. Data snapshot (all schools, all relevant columns)
    snapshot_cols = ['School', 'Rank', TARGET] + ranking_system.features
    snapshot = df_imputed[snapshot_cols].to_dict(orient='records')
    _write_json(snapshot, output_dir, 'data_snapshot.json')

    # 7. Feature ranges for sliders
    feature_ranges = {}
    for feat in ranking_system.features:
        vals = df_imputed[feat]
        feature_ranges[feat] = {
            'data_min': float(vals.min()),
            'data_max': float(vals.max()),
            'data_mean': float(vals.mean()),
            'data_median': float(vals.median()),
            **SLIDER_RANGES.get(feat, {}),
        }

    # Find GWU's current values
    gwu_row = df_imputed[df_imputed['School'].str.contains('George Washington', case=False, na=False)]
    if not gwu_row.empty:
        gwu_values = {}
        for feat in ranking_system.features:
            gwu_values[feat] = float(gwu_row.iloc[0][feat])
        feature_ranges['_gwu_current'] = gwu_values
        feature_ranges['_gwu_school_name'] = gwu_row.iloc[0]['School']
        feature_ranges['_gwu_current_rank'] = int(gwu_row.iloc[0]['Rank'])
        feature_ranges['_gwu_current_score'] = float(gwu_row.iloc[0][TARGET])
    else:
        print("  WARNING: George Washington University not found in dataset!")

    _write_json(feature_ranges, output_dir, 'feature_ranges.json')

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
    print("GWU RANKING PREDICTOR — MODEL TRAINING")
    print("=" * 60)

    # Phase 1a: Preprocessing
    df_imputed = preprocess_raw_data(DATA_PATH)

    # Phase 1b: Model Training
    print(f"\n{'='*60}")
    print("PHASE 1b: MODEL TRAINING")
    print(f"{'='*60}")

    # Identify the global top school (Rank == 1) for calibration
    global_top_idx = df_imputed[df_imputed['Rank'] == 1].index[0]
    global_top_school = df_imputed.loc[[global_top_idx]]
    print(f"\n  Global top school: {global_top_school.iloc[0]['School']} (Rank {int(global_top_school.iloc[0]['Rank'])})")

    # Initialize and train
    ranking_system = USNewsRankingSystem(
        features=ALL_FEATURES,
        target=TARGET,
        log_vars=LOG_VARS,
        logit_vars=LOGIT_VARS,
        inv_norm_vars=INV_NORM_VARS,
        n_iterations=N_BOOTSTRAP_ITERATIONS,
        n_jobs=N_JOBS
    )

    ranking_system.fit_and_calibrate(df_imputed, global_top_ref=global_top_school)

    # Performance Report
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE")
    print(f"{'='*60}")

    preds = ranking_system.predict(df_imputed)
    actuals = df_imputed[TARGET].values

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    spearman, _ = spearmanr(actuals, preds)

    print(f"  MAE:      {mae:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  R²:       {r2:.4f}")
    print(f"  Spearman: {spearman:.4f}")

    # Significance report
    sig_report = ranking_system.get_significance_report()
    print(f"\n  Coefficient Significance:")
    print(sig_report.to_string(index=False))

    # Export artifacts
    export_artifacts(ranking_system, df_imputed, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
