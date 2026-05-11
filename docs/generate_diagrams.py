"""
Render the technical-documentation diagrams as PNG images using matplotlib.
Run once after any architecture changes:

    cd webapp-v2/docs
    python generate_diagrams.py

Outputs into ./img/ — referenced by technical-docs.md.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT_DIR = os.path.join(os.path.dirname(__file__), 'img')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Shared colour palette ----------
C_PYTHON = '#3776AB'
C_DATA   = '#7C3AED'
C_ML     = '#10B981'
C_JSON   = '#F59E0B'
C_JS     = '#EC4899'
C_USER   = '#06B6D4'
C_BG     = '#0F172A'
C_FG     = '#E2E8F0'
C_GRID   = '#1E293B'
C_NOTE   = '#64748B'


def fancy_box(ax, x, y, w, h, text, *, fc, ec=None, text_color='white', fontsize=10, bold=False):
    """Draw a rounded-rectangle box with centred text."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=1.4, edgecolor=ec or fc, facecolor=fc, alpha=0.92,
    )
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            color=text_color, fontsize=fontsize, fontweight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, *, color='#94A3B8', lw=1.6, style='->', text=None):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, shrinkA=4, shrinkB=4,
    )
    ax.add_patch(arr)
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, text, ha='center', va='center',
                color=C_FG, fontsize=8, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', fc=C_BG, ec='none', alpha=0.85))


def stylize(fig, ax, title):
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, color=C_FG, fontsize=13, fontweight='bold', pad=12, loc='left')


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    print(f"  [OK] {name} ({os.path.getsize(path)//1024} KB)")


# ============================================================
# 1. End-to-end architecture
# ============================================================
def draw_architecture():
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.2)
    stylize(fig, ax, 'Fig. 1 — End-to-end architecture')

    # Data
    fancy_box(ax, 0.4, 5.0,  2.4, 1.4, "Raw CSVs\nall_schools_flat\n2024 & 2025",
              fc=C_DATA, fontsize=10, bold=True)
    # Python preprocess
    fancy_box(ax, 3.6, 5.0,  2.6, 1.4, "preprocess()\n• GRE parse\n• SBP ratio\n• GMAT blend\n• KNN impute",
              fc=C_PYTHON, fontsize=9)
    # Pipeline
    fancy_box(ax, 7.0, 5.0,  3.0, 1.4,
              "Sklearn pipeline\nOutlierCapper → log/logit\n→ StandardScaler\n→ Bootstrap ElasticNetCV\n   (positive=True, ×10,000)",
              fc=C_ML, fontsize=8.5)
    # Artifacts
    fancy_box(ax, 10.6, 5.0, 2.0, 1.4, "9 JSON\nartifacts",
              fc=C_JSON, fontsize=10, bold=True)

    # Frontend
    fancy_box(ax, 0.4, 1.8, 3.0, 1.4, "Vite + Tailwind\nstatic site",
              fc=C_JS, fontsize=10, bold=True)
    fancy_box(ax, 4.0, 1.8, 3.0, 1.4, "model.js\n9-feature inference\n+ Monte Carlo (10k)",
              fc=C_JS, fontsize=9)
    fancy_box(ax, 7.6, 1.8, 3.0, 1.4, "Sliders / Lever / Insights\ntabs",
              fc=C_JS, fontsize=9)
    fancy_box(ax, 11.2, 1.8, 1.4, 1.4, "Browser\n(user)",
              fc=C_USER, fontsize=10, bold=True)

    # Arrows top row
    arrow(ax, 2.8, 5.7, 3.6, 5.7)
    arrow(ax, 6.2, 5.7, 7.0, 5.7)
    arrow(ax, 10.0, 5.7, 10.6, 5.7)
    arrow(ax, 11.6, 4.9, 6.0, 3.3, color=C_JSON, style='->', text='fetch on load')

    # Frontend chain
    arrow(ax, 3.4, 2.5, 4.0, 2.5)
    arrow(ax, 7.0, 2.5, 7.6, 2.5)
    arrow(ax, 10.6, 2.5, 11.2, 2.5)
    arrow(ax, 11.9, 1.8, 11.9, 1.2, color=C_USER, style='<->', text='slider')

    # Legend
    legend_y = 0.3
    legend_items = [
        (C_DATA, 'Source data'),
        (C_PYTHON, 'Python / sklearn'),
        (C_ML, 'Model training'),
        (C_JSON, 'JSON artifacts'),
        (C_JS, 'JavaScript / browser'),
        (C_USER, 'End user'),
    ]
    for i, (c, lbl) in enumerate(legend_items):
        x0 = 0.4 + i*2.1
        ax.add_patch(mpatches.Rectangle((x0, legend_y), 0.25, 0.25, facecolor=c, edgecolor='none'))
        ax.text(x0 + 0.32, legend_y + 0.12, lbl, color=C_FG, fontsize=8, va='center')

    save(fig, 'fig01_architecture.png')


# ============================================================
# 2. Preprocessing pipeline
# ============================================================
def draw_preprocessing():
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.6)
    stylize(fig, ax, 'Fig. 2 — Preprocessing pipeline (preprocess() in train_model.py)')

    steps = [
        ("2024 CSV\n121 rows × 437 cols", C_DATA),
        ("2025 CSV\n122 rows × 437 cols", C_DATA),
        ("Stack +\nrename columns\n→ 243 rows", C_PYTHON),
        ("compute_salary_by_profession()\nper-year cohort means\nweighted ratio per school", C_ML),
        ("parse_gre_components()\n→ GRE_V, GRE_Q, GRE_AW\nfrom range strings", C_PYTHON),
        ("compute_gmat_gre_blend()\nper-year percentile\nGRE 40/40/20\nsubmission-weighted", C_ML),
        ("KNN imputation\n(per-year, n=5)\nweights=distance", C_PYTHON),
        ("Cohort-floor fallback\nGMAT_Combined / SBP\n→ lowest per year", C_PYTHON),
        ("Final 9-feature DF\n243 rows × 9 cols\n(zero NaN)", C_JSON),
    ]
    # Layout: stack inputs → arrows → linear pipeline
    fancy_box(ax, 0.3, 5.7, 2.4, 1.0, steps[0][0], fc=steps[0][1], fontsize=9)
    fancy_box(ax, 0.3, 4.3, 2.4, 1.0, steps[1][0], fc=steps[1][1], fontsize=9)
    fancy_box(ax, 3.4, 5.0, 2.4, 1.0, steps[2][0], fc=steps[2][1], fontsize=9)

    fancy_box(ax, 6.4, 5.7, 3.0, 1.2, steps[3][0], fc=steps[3][1], fontsize=8.5)
    fancy_box(ax, 9.8, 5.7, 2.9, 1.2, steps[4][0], fc=steps[4][1], fontsize=8.5)

    fancy_box(ax, 6.4, 3.5, 3.0, 1.3, steps[5][0], fc=steps[5][1], fontsize=8.5)
    fancy_box(ax, 9.8, 3.5, 2.9, 1.3, steps[6][0], fc=steps[6][1], fontsize=8.5)

    fancy_box(ax, 3.4, 2.0, 2.7, 1.1, steps[7][0], fc=steps[7][1], fontsize=8.5)
    fancy_box(ax, 7.4, 1.0, 3.5, 1.2, steps[8][0], fc=steps[8][1], fontsize=10, bold=True)

    # Arrows
    arrow(ax, 2.7, 6.2, 3.4, 5.7)
    arrow(ax, 2.7, 4.8, 3.4, 5.5)
    arrow(ax, 5.8, 5.5, 6.4, 6.0)        # to SBP
    arrow(ax, 5.8, 5.6, 6.4, 6.3)        # arc to SBP top
    arrow(ax, 5.8, 5.4, 6.4, 6.0)
    arrow(ax, 5.8, 5.4, 9.8, 6.0)        # to GRE parse
    arrow(ax, 7.9, 5.7, 7.9, 4.7)        # SBP → blend
    arrow(ax, 11.2, 5.7, 11.2, 4.7)      # GRE → blend
    arrow(ax, 11.2, 4.7, 7.9, 4.5)       # both into blend
    arrow(ax, 9.4, 4.1, 9.8, 4.1)        # blend → impute
    arrow(ax, 9.8, 3.7, 6.1, 2.5)        # impute → floor
    arrow(ax, 6.1, 2.5, 7.4, 1.7)        # floor → final
    arrow(ax, 9.4, 4.1, 9.4, 1.7)        # quick path

    save(fig, 'fig02_preprocessing.png')


# ============================================================
# 3. GMAT/GRE blend
# ============================================================
def draw_gmat_blend():
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.8)
    stylize(fig, ax, 'Fig. 3 — GMAT / GRE blended-percentile score')

    # Step 1: 5 distributions
    for i, (label, color) in enumerate([
        ('GMAT old\n200–800', '#A78BFA'),
        ('GMAT new\n205–805', '#A78BFA'),
        ('GRE Quant\n130–170', '#22D3EE'),
        ('GRE Verbal\n130–170', '#22D3EE'),
        ('GRE AW\n0.0–6.0', '#22D3EE'),
    ]):
        x = 0.3 + i * 2.55
        fancy_box(ax, x, 5.0, 2.2, 1.3, label, fc=color, text_color='#0F172A', fontsize=9, bold=True)

    # Step 2: per-year percentile rank
    fancy_box(ax, 0.3, 3.5, 12.4, 0.8,
              'Step 2 — Per-year percentile rank vs. cohort (rank/N, NaN excluded)',
              fc=C_PYTHON, fontsize=9.5, bold=True)
    for i in range(5):
        x = 0.3 + i * 2.55
        arrow(ax, x + 1.1, 5.0, x + 1.1, 4.3)
        arrow(ax, x + 1.1, 3.5, x + 1.1, 2.95)

    # Step 3: GRE-internal blend
    fancy_box(ax, 5.4, 2.0, 7.3, 0.9,
              '0.4 × pct(Q) + 0.4 × pct(V) + 0.2 × pct(AW)  =  single GRE percentile',
              fc=C_ML, fontsize=9.5, bold=True)
    fancy_box(ax, 0.3, 2.0, 2.2, 0.9, 'pct_GMAT_old', fc='#A78BFA', text_color='#0F172A', fontsize=9, bold=True)
    fancy_box(ax, 2.8, 2.0, 2.2, 0.9, 'pct_GMAT_new', fc='#A78BFA', text_color='#0F172A', fontsize=9, bold=True)
    arrow(ax, 1.4, 2.95, 1.4, 2.9)
    arrow(ax, 3.9, 2.95, 3.9, 2.9)

    # Step 4: cross-exam blend
    fancy_box(ax, 3.3, 0.4, 6.4, 1.0,
              "Step 4 — Cross-exam blend, weighted by each school's submission %s\n"
              "blended = (p_old·r_old + p_new·r_new + p_gre·r_gre) / total_pct,  × 100",
              fc=C_JSON, text_color='#0F172A', fontsize=8.5, bold=True)
    arrow(ax, 1.4, 2.0, 4.0, 1.45)
    arrow(ax, 3.9, 2.0, 4.5, 1.45)
    arrow(ax, 9.1, 2.0, 7.6, 1.45)

    # Missing-data note
    ax.text(10.5, 0.3, "If a school reports nothing → cohort floor",
            color=C_NOTE, fontsize=8, style='italic', ha='center')

    save(fig, 'fig03_gmat_blend.png')


# ============================================================
# 4. Sign-constrained bootstrap loop
# ============================================================
def draw_bootstrap():
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.4)
    stylize(fig, ax, 'Fig. 4 — Sign-constrained bootstrap (10,000 iterations)')

    # Input
    fancy_box(ax, 0.3, 4.5, 3.0, 1.2,
              "X = scaled features\n(243 × 9 matrix)\ny = OverallScore (243,)",
              fc=C_DATA, fontsize=9.5, bold=True)
    # Sign-flip
    fancy_box(ax, 4.0, 4.5, 3.2, 1.2,
              "Sign-flip\nX[:, AcceptanceRate] *= -1\nso every column is\n'higher → better'",
              fc=C_PYTHON, fontsize=8.5)
    arrow(ax, 3.3, 5.1, 4.0, 5.1)

    # Loop box
    fancy_box(ax, 8.0, 4.0, 4.7, 2.0,
              "Loop  ×  N_BOOTSTRAP_ITERATIONS",
              fc='#1E293B', ec=C_ML, fontsize=10, bold=True)
    arrow(ax, 7.2, 5.1, 8.0, 5.1)

    # Loop internals
    fancy_box(ax, 8.2, 4.7, 4.3, 0.6, "resample(X, y)  →  bootstrap sample",
              fc=C_ML, fontsize=8.5)
    fancy_box(ax, 8.2, 4.0, 4.3, 0.6, "ElasticNetCV(positive=True, cv=5).fit(...)",
              fc=C_ML, fontsize=8.5)

    # Collect
    fancy_box(ax, 4.6, 2.0, 5.0, 1.1,
              "Stack coefficient vectors\n→ bootstrap_history_ (10000 × 9)",
              fc=C_JSON, text_color='#0F172A', fontsize=9, bold=True)
    arrow(ax, 10.0, 4.0, 7.4, 3.1)

    # Sign-flip back
    fancy_box(ax, 0.3, 2.0, 4.0, 1.1,
              "Sign-flip back\nbootstrap_history_[:, AR] *= -1",
              fc=C_PYTHON, fontsize=9)
    arrow(ax, 4.6, 2.5, 4.3, 2.5, color='#94A3B8')

    # Summarize
    fancy_box(ax, 4.6, 0.4, 5.0, 1.0,
              "Mean → final coef\n2.5th & 97.5th pct → 95% CI",
              fc=C_JS, fontsize=9.5, bold=True)
    arrow(ax, 7.1, 2.0, 7.1, 1.4)

    # Note
    ax.text(0.3, 0.6, "→ guarantees every 'higher is better' coefficient ≥ 0\n→ AR coefficient ≤ 0 (lower acceptance = better)",
            color=C_NOTE, fontsize=8.5, style='italic')

    save(fig, 'fig04_bootstrap.png')


# ============================================================
# 5. Monte Carlo simulation
# ============================================================
def draw_monte_carlo():
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.6)
    stylize(fig, ax, 'Fig. 5 — Monte Carlo rank simulation (10,000 iterations, in-browser)')

    # User input
    fancy_box(ax, 0.3, 5.0, 2.6, 1.2, "User moves slider\n(direct or via lever)",
              fc=C_USER, text_color='#0F172A', fontsize=10, bold=True)

    # Build sim data
    fancy_box(ax, 3.4, 5.0, 3.0, 1.2,
              "Build simData[]\nOverride target school\nKeep 121 competitors\nat snapshot values",
              fc=C_JS, fontsize=8.5)
    arrow(ax, 2.9, 5.6, 3.4, 5.6)

    # Score all schools (anchored)
    fancy_box(ax, 7.0, 5.0, 5.6, 1.2,
              "For every school:\n predicted = model(features)\n residual = published − model(baseline_features)\n baseScore = predicted + residual   (anchored)",
              fc=C_ML, fontsize=8.5)
    arrow(ax, 6.4, 5.6, 7.0, 5.6)

    # Tiered noise
    fancy_box(ax, 0.3, 2.8, 3.0, 1.2,
              "Tiered Gaussian σ\nTop-20: σ=0.8\nRank 21–50: σ=1.5\nRank 51+: σ=2.5\n(target: σ=0)",
              fc=C_PYTHON, fontsize=8.5)

    # 10k loop
    fancy_box(ax, 3.8, 2.4, 5.8, 1.8,
              "Loop ×10,000\n  scenario = baseScore + N(0,σ)·noise[i]\n  sort descending\n  predictedRanks.push(rank of target)",
              fc=C_ML, fontsize=9)
    arrow(ax, 3.3, 3.4, 3.8, 3.4)
    arrow(ax, 9.8, 5.0, 8.0, 4.2, color='#94A3B8')

    # Output
    fancy_box(ax, 10.0, 2.4, 2.7, 1.8,
              "Outputs\n• median rank\n• 5th & 95th\n  percentile (90% CI)\n• rank distribution",
              fc=C_JSON, text_color='#0F172A', fontsize=9, bold=True)
    arrow(ax, 9.7, 3.3, 10.0, 3.3)

    # Render
    fancy_box(ax, 4.0, 0.4, 5.5, 1.1,
              "results.js renders:\nrank ↗ animation · score · CI · histogram",
              fc=C_JS, fontsize=9, bold=True)
    arrow(ax, 6.8, 2.4, 6.8, 1.5)

    save(fig, 'fig05_monte_carlo.png')


# ============================================================
# 6. Frontend module dependency graph
# ============================================================
def draw_frontend_modules():
    fig, ax = plt.subplots(figsize=(13, 7.0))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.0)
    stylize(fig, ax, 'Fig. 6 — Frontend module graph (webapp-v2/src/)')

    # main.js at the centre
    fancy_box(ax, 5.4, 5.5, 2.2, 1.0, "main.js\n(bootstrap)",
              fc=C_JS, fontsize=10, bold=True)

    # Tab orchestrators
    fancy_box(ax, 0.4, 3.6, 2.5, 1.0, "sliders.js\n(direct 9 inputs)", fc='#8B5CF6', fontsize=9, bold=True)
    fancy_box(ax, 3.2, 3.6, 2.5, 1.0, "lever-predictor.js\n(indirect levers)", fc='#10B981', fontsize=9, bold=True)
    fancy_box(ax, 6.0, 3.6, 2.5, 1.0, "score-model.js\n(direct insights)", fc='#F59E0B', text_color='#0F172A', fontsize=9, bold=True)
    fancy_box(ax, 8.8, 3.6, 2.5, 1.0, "explainability.js\n(indirect insights)", fc='#F43F5E', fontsize=9, bold=True)
    fancy_box(ax, 11.4, 3.6, 1.4, 1.0, "tabs.js\n(nav)", fc=C_NOTE, fontsize=9, bold=True)

    # Lower layer: shared
    fancy_box(ax, 0.6, 1.5, 2.7, 1.2, "model.js\n(score inference\n+ Monte Carlo)", fc=C_ML, fontsize=9, bold=True)
    fancy_box(ax, 3.6, 1.5, 2.7, 1.2, "cfm-models.js\n(indirect-lever\nCFM inference)", fc=C_ML, fontsize=9, bold=True)
    fancy_box(ax, 6.6, 1.5, 2.7, 1.2, "results.js\n(rank/score/chart\nrenderer)", fc=C_JS, fontsize=9, bold=True)

    # Artifact layers
    fancy_box(ax, 0.6, 0.1, 5.5, 0.9,
              "/public/model_artifacts/*.json   ← Python-exported (9 files)",
              fc=C_JSON, text_color='#0F172A', fontsize=9, bold=True)
    fancy_box(ax, 6.6, 0.1, 5.5, 0.9,
              "/public/cfm_artifacts/*.json   ← CFM-exported (1 summary + 8 models)",
              fc=C_JSON, text_color='#0F172A', fontsize=9, bold=True)

    # Arrows: main.js → tab modules
    for x in [1.65, 4.45, 7.25, 10.05, 12.1]:
        arrow(ax, 6.5, 5.5, x, 4.6)
    # tab modules → shared
    arrow(ax, 1.65, 3.6, 2.0, 2.7)         # sliders.js → model.js
    arrow(ax, 4.45, 3.6, 4.95, 2.7)        # lever-predictor → cfm-models
    arrow(ax, 4.45, 3.6, 2.0, 2.7, color='#94A3B8')   # lever-predictor → model.js (for simulateRank)
    arrow(ax, 7.25, 3.6, 7.9, 2.7)         # score-model → results? actually reads model_explainability
    arrow(ax, 10.05, 3.6, 4.95, 2.7)       # explainability → cfm-models

    # Shared → artifacts
    arrow(ax, 2.0, 1.5, 3.4, 1.0, color='#94A3B8')
    arrow(ax, 4.95, 1.5, 4.0, 1.0, color='#94A3B8')
    arrow(ax, 4.95, 1.5, 9.4, 1.0, color='#94A3B8')

    save(fig, 'fig06_frontend_modules.png')


if __name__ == '__main__':
    print("Rendering documentation diagrams to", OUT_DIR)
    draw_architecture()
    draw_preprocessing()
    draw_gmat_blend()
    draw_bootstrap()
    draw_monte_carlo()
    draw_frontend_modules()
    print("Done.")
