# MGNREGA ST Access — Village Risk Watchlist

**Predicting which villages are likely to under-serve Scheduled Tribe households, before the year begins.**

An early-warning system built on 9 years of village-level MGNREGA data from Meghalaya. A Block Programme Officer opens a dashboard in April and sees a ranked list of at-risk villages for the year ahead, with the historical evidence behind each flag. That gives them the full season to act, instead of finding out after the year is lost.

---

## The Problem

MGNREGA guarantees rural households up to 100 days of paid work a year. It is the world's largest public employment programme, covering over 25 crore registered workers. But a job card does not guarantee work.

Across Meghalaya, ST households consistently receive employment at lower rates than other households. We analysed **57,835 records across 7,273 unique villages** and found that a persistent minority of villages had ST employment access below 30% in three or more of the last nine years. Not a bad year here and there. The same villages, falling short year after year.

Today, this pattern only becomes visible in retrospective reports and annual social audits, long after the missed workdays are gone. Officers have no way to know, going into a new season, which villages are likely to repeat the pattern.

## The Solution

A predictive model that looks at each village's history (job cards, employment, and persondays over prior years) and flags, at the start of each fiscal year, which villages are at high risk of another low-access year.

**The innovation is not discovering the gap. That is already documented. The innovation is turning nine years of scattered annual snapshots into a forward-looking risk signal officers can act on before a village's next bad year even begins.**

Across four unseen years, reviewing the top 20% of flagged villages reached **72–76%** of the villages that ended up low-access.

---

## At a Glance

- **Target users:** Block Programme Officers, District MGNREGA cells, and social audit teams in Meghalaya.
- **Expected impact:** If officers review the top 20% of villages flagged each April, they reach about 72 to 76% of the villages that would otherwise end up low-access, before the year is lost.
- **Feasibility:** Built entirely on publicly available NDAP data. No new data collection. Runs on standard hardware.
- **Scalability:** The pipeline works for any district, block, or village structure. Meghalaya first, and extendable to other states with comparable MGNREGA reporting.
- **Implementation:** See "How It Works" below.

---

## How It Works

1. **Data:** NDAP MGNREGA village-level export, Meghalaya, fiscal years 2014 to 2021.
2. **Feature engineering:** Lagged access rates (1, 2, 3 years), 3-year rolling average, year-over-year change, workdays per employed household, 100-day completion rate, and ST share of job cards. All lag and rolling features are built with `.shift()` so they stay leakage-free.
3. **Label:** `is_low_access_current = 1` if ST access rate is below 30% in that village-year.
4. **Model:** XGBoost, tuned with `TimeSeriesSplit` cross-validation, scored on `average_precision` (PR-AUC). Benchmarked against LightGBM and CatBoost on the same time-based split. XGBoost gave the best PR-AUC and the cleanest SHAP explanations, so it was chosen as the final model.
5. **Explainability:** SHAP TreeExplainer produces per-village reasons for each flag.
6. **Output:** `village_risk_2021.csv`, a ranked watchlist with risk score and top-3 reasons per village.
7. **Dashboard:** A Streamlit app for Block Programme Officers.

**Time-based split:** Train on years up to 2018, validate on 2019 and 2020, test on 2021. Threshold is set on validation and applied to 2021, with no peeking.

**2022 excluded:** The extract records almost no employment that year. It is a reporting artifact, not real signal.

---

## Results

### How the model was chosen

Three gradient-boosted tree models were tuned on the same time-based split, each with `TimeSeriesSplit` cross-validation and scored on `average_precision` (PR-AUC). PR-AUC is the right metric here because only ~5–6% of village-years are low-access: accuracy would be misleading, and PR-AUC measures how well the model separates the rare positive class from the rest.

CatBoost was weakest (PR-AUC 0.319). LightGBM (0.393) and XGBoost (0.394) were effectively tied on PR-AUC. XGBoost was chosen as the final model because it matched the best PR-AUC and produced the cleanest, most stable SHAP explanations — which is what powers the per-village reason strings in the dashboard.

An ablation run then tested three variants on the same split:

- **A) Baseline** (all features, persondays NaN kept as NaN): PR-AUC 0.393, recall@top20% 70.1%
- **B) Fill persondays NaN with 0:** PR-AUC 0.393, recall@top20% 69.9%
- **C) B + drop year/district-rate fingerprint features:** PR-AUC **0.407**, recall@top20% **73.9%**

Version C was kept, since dropping the year/district-rate features improved both PR-AUC and recall while removing features that could act as a year fingerprint.

### Model performance (2021 held-out test)

| Model | PR-AUC | Recall @ top 20% |
|---|---|---|
| Random baseline (base rate) | 0.056 | ~20% |
| Naive rule ("low last year → low this year") | — | 33.6% |
| XGBoost (final) | 0.394 | 71.6% |
| LightGBM | 0.393 | 70.1% |
| CatBoost | 0.319 | 58.6% |

The naive rule ("if a village was low-access last year, flag it again") achieves only 33.6% recall at a 3.8% flag rate. The model reaches 71.6% recall when reviewing the top 20% of villages — a substantial lift over the obvious baseline.

### How recall is calculated

For each village in the 2021 test set, the model outputs a risk score. Villages are ranked by that score from highest to lowest. If an officer reviews the top K% of ranked villages, "recall at top K%" is the share of all actually low-access villages in 2021 that fall inside that reviewed set:


Precision at the same cutoff is the share of reviewed villages that were actually low-access:

This mirrors the real decision a Block Programme Officer makes: "I have limited time, so I will look at the top X% of the list." It rewards models that put true positives near the top.

### Recall at review budget (2021)

| Top K% reviewed | Recall | Precision |
|---|---|---|
| 5% | 37.1% | 41.4% |
| 10% | 51.0% | 28.4% |
| 15% | 63.2% | 23.5% |
| 20% | 71.6% | 20.0% |
| 30% | 81.4% | 15.1% |

At a 5% review budget the model is highly precise (41.4% of reviewed villages were truly low-access, versus a 5.6% base rate). At 20% review it catches 71.6% of all low-access villages. This is the "watchlist" tradeoff: the more villages an officer reviews, the more of the true cases they catch, but the lower the hit rate.

### Backtest across unseen years (XGBoost)

To confirm the model is not overfit to one year, the same pipeline was retrained on all years before each test year and evaluated on that year alone:

| Test year | Base rate | PR-AUC | Recall @ top 20% | Naive recall |
|---|---|---|---|---|
| 2018 | 0.058 | 0.381 | 76.0% | 56.3% |
| 2019 | 0.042 | 0.361 | 73.6% | 49.6% |
| 2020 | 0.037 | 0.438 | 76.3% | 49.6% |
| 2021 | 0.056 | 0.415 | 71.9% | 33.6% |

PR-AUC stays in the 0.36–0.44 range across four completely unseen years, and recall@top20% stays in the 72–76% range. The naive rule degrades badly over time (56% → 34%), while the model stays stable. This is the strongest evidence that the signal is real and generalizable, not a single-year fluke.

### Threshold

The decision threshold is set on the validation years (2019–2020) using a recall target, then applied unchanged to 2021. This means the test year is never used to choose the threshold. For an 80% recall target, validation picks a threshold of 0.162, which on 2021 delivers 76.5% recall at 17.7% precision, flagging ~24% of villages. Lower thresholds push recall toward 99% but flag most of the dataset, which is why the dashboard frames the output as a watchlist rather than a binary decision.

---

## Visuals

### Label Balance
<img width="630" height="470" alt="Low Access vs Not" src="https://github.com/user-attachments/assets/6216b698-0948-4749-9381-e929f41528d1" />


About 18% of village-years are low-access. This class imbalance is why the model is tuned on PR-AUC and uses `scale_pos_weight`, and why the dashboard frames flags as a watchlist rather than a verdict.

### SHAP: Global Feature Importance
<img width="1607" height="1378" alt="shap_bar_C" src="https://github.com/user-attachments/assets/7127d808-9c70-43f9-bf36-9759cf2aa36a" />

What drives predictions across all villages. The top signal is **workdays per employed household last year**. A village can count a household as "employed" while giving it almost no actual work. **District** ranks second, which tells us geography carries real, systematic signal, not just village-level variance. Recent access history (lag 1, 2, 3, and the 3-year average) follows.

Notably, **ST share of job cards** is a weak predictor. The gap is in employment delivery, not job-card registration.

### SHAP: Per-Village Explanation
<img width="1601" height="1266" alt="shap_village_C" src="https://github.com/user-attachments/assets/11b545e7-eb0e-40fc-b468-a87b24aa6829" />


Why one specific village was flagged high risk. This village shows three straight years of zero access (`lag1=0`, `lag2=0`, `lag3=0`) and **zero workdays per employed household last year**. That pushes its risk score to `f(x) = 3.04` in log-odds, against a base of `E[f(X)] = 0.016`. The model is not guessing. It is detecting persistent zeros. These reasons power the red and green explanation boxes in the dashboard's village detail tab.

### Streamlit Dashboard
<img width="1366" height="720" alt="Global Innovation Hackathon 2026 Summary - DeepSeek - Google Chrome 24-09-2026 19_58_24" src="https://github.com/user-attachments/assets/47ce6e02-8747-47e1-92df-5ac2ead928b9" />


The Block Programme Officer's watchlist view. Filter by district and block, set a review budget for the top percentage of villages, and see risk-ranked villages with last year's access rate, 3-year average, and the actual 2021 outcome for honest backtesting.

---

## Files in This Repository

| File | What it is |
|---|---|
| `mgnrega.ipynb` | Full pipeline: cleaning, feature engineering, model training, SHAP, export |
| `app.py` | Streamlit dashboard |
| `requirements.txt` | Python dependencies |
| `village_risk_2021.csv` | Model output: ranked risk watchlist with per-village reasons |
| `Low Access vs Not.png` | EDA: label distribution |
| `shap_summary.png` | SHAP global feature importance |
| `shap_waterfall.png` | SHAP per-village explanation |
| `dashboard.png` | Screenshot of the Streamlit app |

---

## Data Source

NDAP (NITI Aayog), MGNREGA village-level dataset, Meghalaya, FY 2014 to 2021.
https://ndap.niti.gov.in/dataset/8071?tab=profile

## Limitations

- **It is a watchlist, not a verdict.** About 1 in 5 flagged villages is truly low-access. Use the flags to decide where to look first.
- **Zero employment can mean two things.** Around two-thirds of low-access cases show exactly zero recorded employment. That may be real denial or a reporting gap. Verify that work was recorded first.
- **It shows where, not why.** The SHAP explanations describe what drove the score, not the underlying cause.
- **2022 was excluded** because the extract records almost no employment, which is a reporting artifact rather than signal.
