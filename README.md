# Store Sales — Time Series Forecasting (Kaggle, top 7.1%)

**[🇷🇺 Русская версия](README.ru.md)**

**TL;DR.** End-to-end time series forecasting on the Kaggle [Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition: a 16-day daily sales forecast for 1,782 series (54 stores × 33 product families, Corporación Favorita, Ecuador). A clean, readable LightGBM solution with day-by-day iterative forecasting and a validation scheme that faithfully simulates real inference — **RMSLE 0.39165, rank 66 / 931 (top 7.1%)** vs 0.37687 for the competition leader (as of May 5, 2026). No stacking and no ensembles of dozens of models — the result comes from feature engineering, honest validation and a careful inference loop.

Notebook commentary is bilingual — English first, with Russian notes; this README also has a [Russian version](README.ru.md).

## Results

| Step | RMSLE (validation 2017-07-31 … 08-15) |
|---|---:|
| Baseline LightGBM (raw target, 22 features) | 0.5119 |
| + `log1p` target, 60+ features (teacher-forcing validation) | 0.3706 |
| Final scheme: 2 models + **iterative validation simulating real inference** | 0.2928 |
| **Kaggle Public Leaderboard** | **0.39165 — rank 66 / 931** |

![Actual vs Forecast — daily total sales on the validation window](images/validation_forecast.png)

## Validation strategy — the core of this project

Most time series solutions validate with *teacher forcing*: lag features are computed from **actual** past sales, including the validation period itself. That score is systematically optimistic — at inference time the model never has actual sales for the days it is predicting; it only has its **own predictions**.

This project validates the way the model actually runs on the test set:

1. Sales for the entire validation window (2017-07-31 … 08-15, exactly mirroring the 16-day test horizon) are **hidden**.
2. The forecast is produced **day by day**: after each predicted day, all lag and rolling-mean features are **recomputed from the model's own predictions**, never from the hidden actuals.
3. The zero-rule (series with no sales in the last 21 days are predicted as 0) is computed **strictly on pre-validation data**.
4. Ensemble weights for the two final models (trained on data from 2016-01-01 and from 2017-01-01) are selected on this honest iterative validation — not on teacher-forcing scores.

The same loop then generates the Kaggle submission, so the validation measures exactly what is deployed.

## Engineering details that matter

- **Leak-free rolling features**: every rolling mean over sales uses `shift(1)` — the current day never leaks into its own features.
- **Safe transaction lags**: `transactions` exist only for the train period, so the model uses `transactions_lag_16 … 23` — for every date of the 16-day test horizon these lags are guaranteed to point back into known train data.
- **Lags justified by analysis, not guessed**: ACF/PACF analysis motivates the lag set — short-term lags 1–6, weekly/monthly 7/14/21/28/42/56, and yearly 364/365.
- **Calendar reconstruction**: missing dates restored in `train`, `oil` (linear interpolation) and `transactions`; December 25 added with zero sales (stores are closed).
- **Event-specific flags instead of one generic holiday flag**: EDA shows different events move sales in different directions, so the model gets separate flags for national holidays, Black Friday, Cyber Monday, the Manabí earthquake period, Navidad, Día de la Madre, football events and more.

## Honest limitations

- **A single validation window.** All tuning decisions (feature set, model weights) are made on one window, 2017-07-31 … 08-15. This is a deliberate trade-off: the window exactly mirrors the test horizon, but a rolling-origin evaluation over 2–3 windows would additionally estimate the *stability* of these choices. The gap between local iterative RMSLE (0.293) and the public score (0.392) is consistent with some overfitting to this short window.
- **`onpromotion` and oil moving averages are computed without `shift(1)` — and that is legal here.** Promotions and oil prices are known for the test period by the competition design (they are exogenous inputs, not targets), so using their current-day values is not leakage. For *sales*, the same construction would be a leak — which is why every sales-based feature is shifted.

## What's inside

The pipeline is split into three notebooks executed in order:

| Notebook | Contents |
|---|---|
| `notebooks/01_eda.ipynb` | Data audit, calendar gaps, target distribution (31% zeros), ACF/PACF and STL decomposition, holiday/event effect analysis, store & transaction analysis → a feature plan derived from the EDA |
| `notebooks/02_feature_engineering.ipynb` | Calendar reconstruction, lag/rolling/Fourier/event features → saves `artifacts/features.parquet` |
| `notebooks/03_modeling_submission.ipynb` | Baseline → tuned `log1p` model → two final models, iterative validation, iterative test forecast → `submission.csv` |

Trained models are cached with a **load-or-train** pattern: if `models/*.joblib` exists it is loaded, otherwise the model is trained and saved. Delete the files to retrain from scratch.

## Project structure

```
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA: calendar, ACF/PACF, STL, events
│   ├── 02_feature_engineering.ipynb  # preprocessing + features → artifacts/
│   └── 03_modeling_submission.ipynb  # models, iterative validation, submission
├── artifacts/                        # parquet features (generated by 02, not in git)
├── models/                           # trained LightGBM models, load-or-train cache
├── data/                             # Kaggle competition data (not in git)
├── requirements.txt
├── README.md                         # this file
└── README.ru.md                      # Russian version
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Get the data (Kaggle CLI) and unpack into data/
kaggle competitions download -c store-sales-time-series-forecasting -p data
unzip data/store-sales-time-series-forecasting.zip -d data

# Run the notebooks in order: 01 → 02 → 03
jupyter notebook notebooks/
```

Note: `models/*.joblib` artifacts are sensitive to the `lightgbm` / `scikit-learn` versions — use the versions pinned in `requirements.txt`, or delete the saved models and let notebook 03 retrain them.

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

## Tech stack

Python · pandas · NumPy · statsmodels · LightGBM · scikit-learn · matplotlib · seaborn · joblib · pyarrow

## Key takeaways

- A validation scheme that reproduces inference conditions is worth more than any single feature: teacher-forcing scores and honest iterative scores answer different questions, and only the second one predicts leaderboard behaviour.
- Different events move sales in different directions — one generic `is_holiday` flag loses signal that separate event flags capture.
- When future covariates are unavailable (transactions), lags can be chosen so that the entire forecast horizon points back into known data — a leak-free feature by construction.
- A clean single-model solution with careful features lands in the top 7.1% of a competition where the top is squeezed out by heavy ensembles.

## Author

**Maksym Chunikhin** — [Kaggle](https://www.kaggle.com/maksymchunikhin)
