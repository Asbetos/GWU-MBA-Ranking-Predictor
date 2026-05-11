---
title: "GWSB Ranking Predictor — User Guide"
subtitle: "A what-if planning tool for GWU's US News MBA ranking"
author: "GWSB Operations & Strategic Initiatives"
toc: true
toc-depth: 2
---

# About this tool

The **GWSB Ranking Predictor** is an interactive what-if calculator. It lets you change one or more ranking inputs and immediately see how GWU's predicted US News MBA rank would move. Two flavours of input are supported:

- **Direct inputs** — the nine ranking indicators US News uses to compute the overall score (e.g. starting salary, peer assessment, GMAT/GRE).
- **Indirect levers** — non-ranking inputs (tuition, demographics, work experience, etc.) that we believe influence the ranking indicators. The tool routes these through internal "indirect models" that translate a lever into a predicted indicator value, then runs the score model on top.

Every prediction is wrapped in a **10,000-run Monte Carlo simulation**, so what you see is not a single rank but a *median rank* with a *90% confidence range*.

The tool is read-only. Nothing you do here changes GWU's actual data or affects the published US News ranking. Use it for exploration, scenario planning, and presentations.

---

# Getting started

Open the deployed URL in any modern browser (Chrome, Edge, Safari, Firefox). Everything runs in the browser — no login, no installation, no data leaves your machine. A typical first session takes 2–3 minutes to get oriented.

You'll land on the **Direct Predictor** tab, which is where most users spend their time. There are four tabs across the top:

| # | Tab | What it's for |
|---|---|---|
| 1 | **Direct Predictor** | Move the 9 ranking inputs directly. Most direct way to ask "what if our salary improved 10%?" |
| 2 | **Indirect Levers Predictor** | Move non-ranking inputs (tuition, demographics, etc.) and watch their downstream effect on rank. |
| 3 | **Direct Model Insights** | How accurate is the score model? Which indicators move the rank the most? |
| 4 | **Indirect Model Insights** | How accurate is each indirect-lever model? Which levers drive each indicator? |

---

# Tab 1 — Direct Predictor

This is the headline tab. You'll see four things stacked at the top, all of which stay visible as you scroll through the input sliders below:

1. **Current Standing** card — GWU's actual published rank and score.
2. **Predicted Rank** — the rank the model predicts under the current slider settings. Click any slider and this updates within ~250 ms.
3. **Scenario Score / 90% CI Range** — the predicted score (0–100) and the 90% confidence interval on the rank (a band of likely outcomes from the Monte Carlo).
4. **Rank Distribution chart** — a histogram showing how often each rank came up across the 10,000 simulations.

### The 9 input sliders

Below the sticky results panel, you'll see one card per ranking indicator:

| Indicator | What it measures | Units |
|---|---|---|
| **Employed at Graduation** | % of job-seeking grads with a job offer at commencement | percent |
| **Employed 3 Months After** | % with a job 3 months after commencement | percent |
| **Avg Salary + Bonus** | Mean starting compensation | US dollars |
| **Salary by Profession** | Per-industry salary vs. cohort average (see below) | ratio (1.0 = average) |
| **Median GPA** | Undergraduate GPA of new entrants | 3.0 – 4.0 |
| **Acceptance Rate** | % of applicants admitted (lower = more selective) | percent |
| **Peer Assessment** | Average score from dean / director peer survey | 1.0 – 5.0 |
| **Recruiter Assessment** | Average score from corporate recruiter survey | 1.0 – 5.0 |
| **GMAT/GRE Blended** | 0–100 percentile combining GMAT-old, GMAT-new, GRE | composite |

#### Click-to-edit values

Each slider's value is shown as a small chip on the right (e.g. `48.5%`). **Click the chip** to type an exact value, then press Enter. Out-of-range values are shown in red and rejected. This is faster than the slider for precise scenarios.

#### Special: Salary by Profession control

US News computes this indicator by comparing each school's salary in each of seven professions to the cohort average for that profession, weighted by how many of its grads ended up in each profession. The control reflects that:

- 7 industry rows (Consulting, Finance/Accounting, General Management, Marketing/Sales, Operations/Production, MIS, Human Resources).
- For each row: a **salary input** on the left (editable number) plus a **slider** in the middle for the salary, and an **n-reporting** field on the right for the number of GWU grads in that profession.
- Industries with **fewer than 3 reporting graduates** are excluded from the calculation, per US News' methodology.
- The headline chip at the top of the panel shows the resulting cohort ratio.

#### Special: GMAT / GRE composite control

The GMAT/GRE indicator is itself a composite built from up to five inputs. The control has two toggles:

- **Include GMAT input** — when ON, you choose Old GMAT (200–800) or New GMAT (205–805) scale and set a single score. When OFF, no GMAT signal is contributed.
- **Include GRE input** — when ON, three separate sliders appear for GRE Quantitative, Verbal, and Analytical Writing. The internal blend is `0.4 × Quant + 0.4 × Verbal + 0.2 × AW`, exactly as US News specifies.

If **both** toggles are OFF, the cohort floor (the lowest GMAT/GRE score in the cohort) is used — matching the official US News missing-data rule.

#### Reset

The **↺ Reset to Current** button at the top right of the slider grid puts every input back to GWU's actual current value.

### Reading the results

The big number is the **median rank** from 10,000 simulations. The "90% CI Range" tells you how stable that median is — for example, "**60 – 75**" means in 90% of simulations the rank landed between 60 and 75.

The wider the CI, the more uncertain the prediction. CIs widen when:

- GWU is near the boundary between two rank tiers (top-50 / 51+);
- The scenario score is close to many competitors' scores;
- The model is extrapolating to a region with little training data.

---

# Tab 2 — Indirect Levers Predictor

This tab is for the question *"if we change something that isn't a ranking indicator, will the ranking move?"*. For example, "what if we cut tuition by $5k?" or "what if we admit more students with prior work experience?".

### Layout

The sticky panel at the top shows:

- **Baseline (current GWU)** — actual rank and score US News published.
- **Lever-Driven Rank** — predicted rank under the current lever settings.
- **Predicted Core Features** — for each of the 9 ranking indicators, what value the indirect models predict given the lever settings, with the delta vs. GWU's actual value.

Below: a grid of slider cards, one per indirect lever (~23 total).

### Colour coding

Every slider is tinted by **how reliably the indirect model can predict its primary downstream indicator**:

- **Green border** — High confidence. A change here is a trustworthy directional signal.
- **Amber/yellow border** — Medium confidence. Direction is reliable, exact magnitudes are approximate.
- **Red/pink border** — Low confidence. Treat as suggestive; small changes may not translate into real ranking movement.

The same colour coding appears on the predicted-core-feature cards: a green border on "Employed at Grad" means the indirect model for that indicator is reliable.

### Reading the deltas

Below each predicted core feature you'll see a delta (e.g. `+1.2pp` or `−$2,500`). This is *the model's predicted change* in that indicator if you applied your current lever scenario. Compare it to the actual value: if the predicted delta is small relative to year-over-year noise, don't expect much rank movement.

### Reset

The **↺ Reset to GWU current** button restores every lever to GWU's actual current value.

---

# Tab 3 — Direct Model Insights

For when you want to understand *how* the score model is making its prediction. Four sections:

### 1. How well does the model match US News?

Four metrics summarising fit quality, evaluated on all 243 school-years (2024 + 2025):

| Label | What it tells you | Good values |
|---|---|---|
| **Avg score error (MAE)** | Average miss on the 0–100 published score | Lower; ours is ~3 points |
| **Worst-case error (RMSE)** | Same as MAE but more sensitive to big misses | Lower; ours is ~4 points |
| **Variance explained (R²)** | Share of score variation the model captures | Closer to 1.0; ours is 0.96 |
| **Rank agreement (ρ)** | How well our rank order matches US News' | Closer to 1.0; ours is 0.98 |

Hover any tile for a one-line definition.

### 2. How much does each indicator move the score?

A table of **impact weights**: how many score points a one-standard-deviation improvement on each indicator adds. For example, "AvgSalaryBonus +7.27" means a 1-σ improvement in average salary (about $20k) adds ~7 score points.

The **Uncertainty range** column shows the range of weights we saw when we re-trained the model 10,000 times on randomly resampled data. A tight range means we're confident in the exact weight. A wide range means the weight could vary depending on which schools are in the sample. The **Confidence** column flags weights whose range stays entirely on one side of zero — those are statistically reliable; the rest are directional only.

### 3. Where does the score come from?

A horizontal bar chart showing, for either *all schools (average)* or *GWU only*, the share of the predicted score driven by each indicator. The GWU view is signed: indigo bars are currently pushing GWU's score up; pink bars are pulling it down. This is the single best place to see where GWU is over- or under-performing relative to the cohort.

### 4. Under the hood

A short methodology blurb describing the model, the GMAT/GRE blend, and why rank uncertainty exists even when the score is precise.

---

# Tab 4 — Indirect Model Insights

This tab has nine cards, one per indirect-lever model (one per ranking indicator). Each card has:

| Element | Reads as |
|---|---|
| **Indicator name** (e.g. "Employed at Graduation") | Which ranking indicator this model predicts. |
| **Confidence badge** (green / amber / red) | How well the model predicts unseen schools. Green = trustworthy; red = suggestive only. |
| **Cross-val fit** | Test on schools the model wasn't trained on. 1.00 = perfect. |
| **Next-year fit** | Train on 2024, test on 2025. Tests year-over-year stability. |
| **Avg error** | Typical prediction error in the indicator's own units. |
| **Trained on** | Number of school-year observations used. |
| **Strongest levers** chart | Top non-method levers that move the indicator. Green = up, pink = down. |

**Hover any card** to pop up a detailed view showing the full top-8 features with percentage contributions. The popover positions itself next to whichever card you're on, never blocking the rest. Press **Esc** to dismiss any open popover.

The four-card metrics block uses tooltips — hover any tile for a definition.

---

# Common scenarios

### "What if we improve average starting salary by $10k?"

1. Open **Tab 1 — Direct Predictor**.
2. Find the *Avg Salary + Bonus* slider.
3. Drag it up by $10,000 (or click the value chip and type the new amount).
4. Watch the **Predicted Rank** update. Compare with **Current Standing** to see the rank delta.

### "What if both salary AND peer assessment improved?"

Same as above but move two sliders. The model is linear and additive on transformed inputs, so two simultaneous moves compose.

### "Will reducing tuition help?"

Tuition isn't a US News ranking indicator — but it might influence demographics, applicant volume, or selectivity. Switch to **Tab 2 — Indirect Levers Predictor**, find a tuition lever, move it, and watch the predicted core features shift. If the change is mostly in red-bordered (low-confidence) indicators, take the result with a grain of salt.

### "Show me where GWU is under-performing"

**Tab 3 — Direct Model Insights**, click **GWU only** on the contribution chart. Pink bars are indicators currently dragging GWU's rank down. Indigo bars are helping. The biggest pink bar is your highest-leverage target.

### "Which input is most worth moving?"

Same chart, but also check the **Indicator** with the largest **Impact weight** in the coefficients table. A big weight + a big pink bar = highest-impact lever to address.

---

# FAQ

**Q. The slider barely moves the rank. Why?**
Either the indicator's impact weight is small (check the Coefficients table on Tab 3) or GWU is far from any rank boundary so it takes a big push to cross one. A 1-point score change near rank #1 means everything; near rank #80 it usually means a 1-rank change at most.

**Q. The 90% CI is very wide. Why?**
Monte Carlo noise scales with rank tier. The model gives top-20 schools tight CIs (σ = 0.8 score points) and rank 51+ schools wide ones (σ = 2.5 score points), because year-over-year volatility is empirically higher down-tier.

**Q. Why does the GMAT/GRE coefficient look so small?**
Statistically, once we control for salary, peer/recruiter assessment, GPA, and salary-by-profession, there isn't much GMAT-specific variance left to explain. This isn't a model error — it's a feature of the published rankings. The coefficient is constrained to be non-negative so moving the slider never *decreases* the predicted rank.

**Q. Can I trust a "low confidence" result?**
For direction, yes — the model knows the sign. For magnitude, treat it as a coin-flip. If the only path forward is through a red-tiered indirect model, frame it as a hypothesis to test rather than a commitment.

**Q. What if data is missing?**
- Test scores not entered (both GMAT and GRE toggles off) → cohort floor, exactly per US News' missing-data rule.
- Industries with < 3 reporting graduates → excluded from the Salary-by-Profession ratio.

**Q. How often is the data updated?**
The training data is whatever's in `all_schools_flat_2024.csv` and `all_schools_flat_2025.csv` at the time the model was last re-trained. The technical team re-runs training annually when US News publishes new data.

**Q. Will my changes persist if I reload the page?**
No. The page state is in-memory only; reloading gives you a fresh session at GWU's current values.

---

# Glossary

- **Cohort** — the set of schools US News ranks in a given year (~134 schools for the 2026 edition).
- **Coefficient / Impact weight** — how many predicted score points a one-standard-deviation change in an indicator adds.
- **CFM (Core-Feature Model)** — an indirect-lever model that predicts one ranking indicator from non-method levers. Tab 4 shows nine of these.
- **Confidence Interval (CI)** — a range of values; "90% CI" means 90% of the simulated outcomes fall inside the band.
- **Monte Carlo simulation** — running the prediction many times with random noise to characterise uncertainty.
- **Indicator / Ranking input** — one of the nine quantities US News uses to compute the overall score.
- **Lever** — a non-ranking input we can change. Tuition, demographic mix, work-experience averages, etc.
- **Percentile** — the position of a value in a sorted cohort. The 80th percentile means 80% of schools score below this value.
- **R² (Variance explained)** — how well a model captures variation in the target. 1.0 = perfect; 0 = no better than predicting the mean.
- **Spearman ρ (Rank agreement)** — how similarly two rankings are ordered. 1.0 = identical order; −1.0 = reversed.
- **Sign-constrained regression** — a regression where coefficients are forced to respect domain direction. Used here so the GMAT slider can never push the rank the wrong way.
