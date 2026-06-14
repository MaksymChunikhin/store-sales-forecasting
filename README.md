# Store Sales — Time Series Forecasting (Kaggle, top 7.1%)

**[🇷🇺 Русская версия](README.ru.md)**

**TL;DR.** End-to-end time series forecasting on the Kaggle [Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition: a 16-day daily sales forecast for 1,782 series (54 stores × 33 product families, Corporación Favorita, Ecuador). A clean, readable LightGBM pipeline with day-by-day iterative forecasting scores **public RMSLE 0.39109**; adding geography features and a small two-model ensemble brings the project's best to **0.38803** (top ~7% as of June 2026; the leaderboard is rolling and refills with public-notebook forks over time). No 25-model stacks — the gains come from honest feature work and disciplined blending.

Beyond the model itself, this project documents a **validation-forensics story** and an honest map of **what transfers to the leaderboard and what doesn't**. The original "honest" iterative validation (RMSLE 0.293) turned out to contain a training-window leak; after the fix the honest estimate became 0.399 — almost exactly the public score. Six experiment notebooks (04–09) then probe per-family models, exogenous features and three different architectures, each through a strict 3-window acceptance gate. Most are **honestly reported negative results**, and they converge on a clear finding: on this competition, validation gains smaller than ~0.003 RMSLE do not transfer to the public test, so the practical ceiling of a principled single-pipeline approach is ≈0.388.

Notebook commentary is bilingual — English first, with Russian notes; this README also has a [Russian version](README.ru.md).

## Results

| Step | RMSLE |
|---|---:|
| Baseline LightGBM (raw target, 22 features), validation 07-31…08-15 | 0.5119 |
| + `log1p` target, 59 features (teacher-forcing validation) | 0.3706 |
| Iterative validation, original version — **leaked**: final models had seen the validation window during training | ~~0.2928~~ |
| Iterative validation after the leak fix (models trained strictly before the window) | **0.3994** |
| **Kaggle Public Leaderboard — clean pipeline (notebook 03)** | **0.39109** |
| **Kaggle Public Leaderboard — best (geo features + two-model ensemble blend)** | **0.38803** |

The near-perfect agreement between the leak-free validation (0.399) and the public score (0.391) is the central result: after the fix, local validation finally *predicts* leaderboard behaviour. The improvement to 0.38803 comes from the `geo` features (experiment 05) plus a log-space blend with a recency-rich model variant — the two changes that proved *transferable*; experiments 06–09 show what does not.

![Actual vs Forecast — daily total sales on the validation window](images/validation_forecast.png)

## Validation strategy — the core of this project

Most time series solutions validate with *teacher forcing*: lag features are computed from **actual** past sales, including the validation period itself. That score is systematically optimistic — at inference time the model never has actual sales for the days it is predicting; it only has its **own predictions**.

This project validates the way the model actually runs on the test set:

1. Sales for the entire validation window (2017-07-31 … 08-15, exactly mirroring the 16-day test horizon) are **hidden**.
2. The forecast is produced **day by day**: after each predicted day, all lag and rolling-mean features are **recomputed from the model's own predictions**, never from the hidden actuals.
3. The zero-rule (series with no sales in the last 21 days are predicted as 0) is computed **strictly on pre-validation data**.
4. Validation models are **trained strictly on data before the window** (`*_val` model copies); full-train copies are used only for the test forecast, where training on everything is leak-free by definition.
5. Ensemble weights for the two final models (trained on data from 2016-01-01 and from 2017-01-01) are selected on this honest iterative validation.

**Leak post-mortem.** In the original version step 4 was missing: the final models were trained on the *full* train with no upper cutoff, so they had seen the validation window during training. The validation loop itself was honest — but the models weren't. That single bug produced RMSLE 0.293 vs the true 0.399 and silently flipped the ensemble-weight choice. It was caught while building the rolling validation of experiment 04.

## Experiments (notebooks 04–09)

All experiments share a stricter methodology: **rolling-origin validation on 3 windows** (June / July / August 2017, 16 days each), all models trained strictly before each window, a fast matrix forecasting engine (validated bit-exact against the notebook-02 feature definitions), and an acceptance **gate**: a candidate must beat its baseline on **all three windows**, not on average.

**Experiment 04 — 33 per-family models vs the global model.** Pure per-family models lose to the global one everywhere (small data, lost cross-family patterns), but help as a *mixture*: `25% per-family + 75% global` and a per-family hybrid pass the gate (mean 0.3904 vs 0.3920). On the public leaderboard, however, the hybrid scored **0.39243 vs 0.39109** — the validation gain did not transfer. The reason is structural and worth knowing: any honest pre-test window handicaps models trained on recent data only (a "from 2017" model validated on a July window has never seen August; the deployed one is trained through Aug 15). This freshness premium is invisible to any validation that ends before the test period — so scheme choices that differ in data recency cannot be made from such windows at all.

**Experiment 05 — exogenous features via a feature-group registry.** Features are toggled by groups (`FEATURE_GROUPS` / `VARIANTS`), model caches are keyed by a hash of the active feature set. Two new groups, both exactly known for the test period: `payday` (the 15th and the last day of the month — wage days in Ecuador) and `geo` (store city/state + local/regional holidays matched to the store's location). Result: **`geo` passes the gate** (better than baseline on all 3 windows, mean +0.0006) and is part of the deployed best; **`payday` fails** (wins June, loses July and August) — an honestly reported negative result.

**Experiment 06 — direct per-series Ridge (a different paradigm).** Replaces the recursive GBM with 1,782 tiny linear models that forecast all 16 days at once. Decisively worse (mean ≈0.59 vs 0.39): without cross-series pooling each linear model overfits its own short, noisy series, and the direct setup gives up the strong short lags. A clear rejection of the "the leaderboard wall is per-series linear regression" hypothesis — pooling (one strong global learner) is doing the heavy lifting.

**Experiment 07 — dynamic features + lighter per-family hyperparameters.** Adds `growth_ratio`, `dow_mean_4`, `rolling_std_28`, `zero_share_28` (recomputed from own predictions inside the loop). They help on June/July but **hurt on the August window** — the recursive train/inference drift compounds with horizon length, biting hardest on the longest, most test-like window. Gate failed; no submission.

**Experiment 08 — direct GBM (no recursion).** Keeps the strong pooled learner but forecasts directly from features safe for the whole horizon (sales lags ≥16, leak-free recent-level aggregates). As a small 20% dose blended with the recursive model it passes the validation gate — but on the leaderboard it scored **0.39417**, worse than the deployed best. The direct model is good on the first half of August (validation window W3) and poor on the true test (Aug 16–31): same month, different half. The decisive validation→public transfer failure.

**Experiment 09 — CatBoost as a third blend parent (last attempt).** A genuinely different boosting library (native categoricals, ordered boosting), trained identically to the LightGBM `geo` model. Weaker on every window and much weaker on the August window, so every blend weight worsens it there; the pre-agreed structural bar (≥0.003 mean gain) is not approached. "Just ensemble LightGBM + CatBoost + XGBoost" does not hold here — diversity helps only when the added model's errors are different *and not systematically worse on the target region*.

**What transferred, and the ceiling.** Only two changes improved the public score: the `geo` features (05) and a file-level blend with a recency-rich model variant — both *structural*. Every sub-~0.003 validation win (the experiment-04 hybrid, the experiment-08 direct blend) turned into a leaderboard loss. The validation–public gap exceeds the gains on offer, so the practical ceiling of this honest single-pipeline approach is **≈0.388** (best: 0.38803). The remaining distance to the ~0.3735 leaderboard "wall" is a heavily-forked, multi-account-seeded tuned ensemble, not a target reachable by principled single-model work.

## Engineering details that matter

- **Leak-free rolling features**: every rolling mean over sales uses `shift(1)` — the current day never leaks into its own features.
- **Safe transaction lags**: `transactions` exist only for the train period, so the model uses `transactions_lag_16 … 23` — for every date of the 16-day test horizon these lags are guaranteed to point back into known train data.
- **Lags justified by analysis, not guessed**: ACF/PACF analysis motivates the lag set — short-term lags 1–6, weekly/monthly 7/14/21/28/42/56, and yearly 364/365.
- **Calendar reconstruction**: missing dates restored in `train`, `oil` (linear interpolation) and `transactions`; December 25 added with zero sales (stores are closed).
- **Event-specific flags instead of one generic holiday flag**: EDA shows different events move sales in different directions, so the model gets separate flags for national holidays, Black Friday, Cyber Monday, the Manabí earthquake period, Navidad, Día de la Madre, football events and more.
- **A matrix forecasting engine for experiments**: the data is rectangular (1,704 dates × 1,782 series), so inside the iterative loop lag features are computed by row slicing of a sales matrix — a validation window runs in seconds instead of minutes, which is what makes 3-window × many-scheme experiments affordable.

## Honest limitations

- **The freshness premium is unvalidatable.** The deployed models are trained through Aug 15; any honest validation window must end before that. Comparisons between schemes that differ in training-data recency therefore do not transfer to the leaderboard (measured directly: a gated validation win of −0.011 RMSLE became a public *loss* of +0.0013). Feature comparisons (same training data, same recency) transfer better.
- **`onpromotion` and oil moving averages are computed without `shift(1)` — and that is legal here.** Promotions and oil prices are known for the test period by the competition design (they are exogenous inputs, not targets), so using their current-day values is not leakage. For *sales*, the same construction would be a leak — which is why every sales-based feature is shifted.

## What's inside

The pipeline is split into notebooks — the first three are the main pipeline, 04–09 are self-contained experiments that do not modify it, and 10 is a visualization:

| Notebook | Contents |
|---|---|
| `notebooks/01_eda.ipynb` | Data audit, calendar gaps, target distribution (31% zeros), ACF/PACF and STL decomposition, holiday/event effect analysis, store & transaction analysis → a feature plan derived from the EDA |
| `notebooks/02_feature_engineering.ipynb` | Calendar reconstruction, lag/rolling/Fourier/event features + candidate groups (payday, geo, dynamic) → saves `artifacts/features.parquet` (a superset; notebooks pick columns via explicit lists) |
| `notebooks/03_modeling_submission.ipynb` | Baseline → tuned `log1p` model → two final models with leak-free `*_val` validation copies, iterative validation, iterative test forecast → `submission.csv` |
| `notebooks/04_experiment_per_family.ipynb` | 3-window rolling validation, matrix engine, 8 schemes (global / per-family / blends) + stitched hybrids, gate, public-score post-mortem |
| `notebooks/05_experiment_features.ipynb` | Feature-group registry with hash-keyed model caches, A/B of `payday` and `geo` groups against the experiment-04 baselines, gate → `submission_05_*.csv` |
| `notebooks/06_experiment_direct_ridge.ipynb` | Direct per-series Ridge (no recursion) — negative result: pooling matters, linear is too weak |
| `notebooks/07_experiment_gbm_boost.ipynb` | Dynamic features (`growth_ratio` etc.) + lighter per-family hyperparameters — negative: dynamic features drift on the August window |
| `notebooks/08_experiment_direct_gbm.ipynb` | Direct GBM on leak-free features ≥16 days; blend passes validation but fails the leaderboard (the key transfer-failure case) |
| `notebooks/09_experiment_catboost.ipynb` | CatBoost as a third diverse blend parent — negative: weaker exactly on the August window, gate rejects it |
| `notebooks/10_submission_viz.ipynb` | Visualizing the submissions by day: forecasts in context, horizon comparison, individual series |

Trained models are cached with a **load-or-train** pattern: if a saved model exists it is loaded, otherwise it is trained and saved. Delete the files to retrain from scratch.

## Project structure

```
├── notebooks/
│   ├── 01_eda.ipynb                   # EDA: calendar, ACF/PACF, STL, events
│   ├── 02_feature_engineering.ipynb   # preprocessing + features → artifacts/
│   ├── 03_modeling_submission.ipynb   # models, leak-free iterative validation, submission
│   ├── 04_experiment_per_family.ipynb # experiment: per-family models, blends, hybrids
│   ├── 05_experiment_features.ipynb   # experiment: payday & geo feature groups
│   ├── 06_experiment_direct_ridge.ipynb # experiment: direct per-series Ridge (negative)
│   ├── 07_experiment_gbm_boost.ipynb  # experiment: dynamic features + light per-family (negative)
│   ├── 08_experiment_direct_gbm.ipynb # experiment: direct GBM, no recursion (transfer failure)
│   ├── 09_experiment_catboost.ipynb   # experiment: CatBoost blend parent (negative)
│   └── 10_submission_viz.ipynb        # visualization: submissions by day
├── artifacts/                         # parquet features + experiment caches (not in git)
├── models/                            # trained LightGBM models, load-or-train cache
├── data/                              # Kaggle competition data (not in git)
├── requirements.txt
├── README.md                          # this file
└── README.ru.md                       # Russian version
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Get the data (Kaggle CLI) and unpack into data/
kaggle competitions download -c store-sales-time-series-forecasting -p data
unzip data/store-sales-time-series-forecasting.zip -d data

# Run the notebooks in order: 01 → 02 → 03 (main pipeline); 04 → 05 (experiments, optional)
jupyter notebook notebooks/
```

Note: `models/*.joblib` artifacts are sensitive to the `lightgbm` / `scikit-learn` versions — use the versions pinned in `requirements.txt`, or delete the saved models and let the notebooks retrain them. The experiment notebooks train ~140 LightGBM models on the first run (several hours); all models and window predictions are cached, so re-runs take minutes.

## Features (main groups)

| Group | Features |
|---|---|
| Calendar | `day_of_week`, `month`, `year`, `is_weekend`, `day_of_year`, `week_of_year`, `date_index` |
| Fourier | `sin_day`, `cos_day`, `sin_week`, `cos_week` |
| Sales history | lags 1–6, 7, 14, 21, 28, 42, 56, 364, 365; rolling means 7/14/28/364 (all via `shift(1)`) |
| External signals | `dcoilwtico` (+ MA 7/28), `onpromotion` (+ MA 7/28) |
| Transactions | `transactions_lag_16 … 23` (leak-free by construction) |
| Events | national holidays, day-before-holiday, Black Friday, Cyber Monday, earthquake period, named holidays, work days |
| Store & product | `store_nbr`, `store_type`, `cluster`, `family` |
| Paydays (exp. 05, failed the gate) | `day_of_month`, `is_payday`, `days_since_payday`, `days_to_payday` |
| Geography (exp. 05, passed the gate) | `city`, `state`, `is_holiday_local`, `is_holiday_regional` |

## Tech stack

Python · pandas · NumPy · statsmodels · LightGBM · scikit-learn · matplotlib · seaborn · joblib · pyarrow

## Key takeaways

- **An honest validation loop is not enough — the models inside it must be honest too.** The iterative loop never touched hidden actuals, yet the score was fantasy because the models had seen the window during training. Leak-free pipelines need *both* honest inference simulation and honest training cutoffs.
- **Validation measures quality under its own information set.** Schemes that differ in training-data recency cannot be compared on pre-test windows: the freshness premium of the deployed model is invisible there, and gated validation wins can become leaderboard losses.
- A scheme is accepted only if it wins on **all** rolling windows, not on average — single-window winners are often noise.
- Different events move sales in different directions — one generic `is_holiday` flag loses signal that separate event flags capture; matching local holidays to each store's city adds a further, gate-passing increment.
- When future covariates are unavailable (transactions), lags can be chosen so that the entire forecast horizon points back into known data — a leak-free feature by construction.
- **Ensembling helps only with the right kind of diversity.** A blend improves on its parents when they make *different* errors that are *not systematically worse on the test region*. The per-family blend and geo features cleared that bar; the direct GBM and CatBoost did not (both weaker exactly on August). "Add more model types" is not automatically useful.
- **Know your ceiling.** Six experiments mapped what transfers (structural changes ≥0.003) and what doesn't (everything below). Recognising the ≈0.388 ceiling — and stopping there rather than chasing a forked-notebook leaderboard wall — is itself a result.
- Negative results are results: per-series linear models, dynamic features, payday features, a CatBoost parent, and two validation-picked schemes that lost on the leaderboard are all documented in the notebooks.

## Author

**Maksym Chunikhin** — [Kaggle](https://www.kaggle.com/maksymchunikhin)
