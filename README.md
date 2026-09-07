# Credit Risk Platform

End-to-end platform for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
problem: it predicts the probability that a loan applicant defaults, explains every
prediction in plain English, derives underwriting rules that are validated against real
default rates, and answers portfolio questions asked in natural language.

**Run it:** `cp .env.example .env`, add a Gemini key, then `docker compose up --build`.
UI at <http://localhost:8080>, API docs at <http://localhost:8000/docs>.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Setup](#setup)
4. [Running the platform](#running-the-platform)
5. [Part 1 — Data understanding & EDA](#part-1--data-understanding--eda)
6. [Part 2 — Talk-to-Data](#part-2--talk-to-data)
7. [Part 3 — Machine learning](#part-3--machine-learning)
8. [Part 4 — Explainable AI](#part-4--explainable-ai)
9. [Rule derivation](#rule-derivation)
10. [Part 5 — User interface](#part-5--user-interface)
11. [Part 6 — Docker deployment](#part-6--docker-deployment)
12. [Prompt engineering & token optimisation](#prompt-engineering--token-optimisation)
13. [Design decisions](#design-decisions)
14. [Known limitations](#known-limitations)
15. [Repository layout](#repository-layout)

---

## What it does

| Section | What it gives you |
|---|---|
| **EDA Dashboard** | Class balance, default rates by segment, ten findings from the full 307,511-row dataset |
| **Risk Prediction** | Calibrated probability of default, risk band, approve/review decision, SHAP attribution |
| **Explainability** | Global feature importance, per-fold AUC, out-of-fold confusion matrix, calibration quality |
| **Business Rules** | 15 underwriting rules, each with the observed default rate and risk lift of the segment it flags |
| **Talk to Data** | Ask a question in English → validated SQL → results → a written business answer |

---

## Architecture

```
                          Kaggle CSVs (application_train / test, bureau)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
          src/data/loader.py                          src/talk_to_data/db_builder.py
       (bureau aggregation, merge)                        (SQLite: applications, bureau)
                    │                                           │
                    ▼                                           ▼
        src/data/preprocessor.py                     src/talk_to_data/nl_to_sql.py
   (sentinels, ratios, encoding, medians)         generate → validate → ground →
                    │                                execute → retry → summarise
                    ▼                                           │
            src/ml/train.py                                     │
   5-fold stratified CV · scale_pos_weight                      │
   calibration · threshold selection                            │
                    │                                           │
      ┌─────────────┼──────────────┐                            │
      ▼             ▼              ▼                            │
 lgbm_model.pkl  metrics.json  train_medians.pkl                │
      │                                                         │
      ▼                                                         │
 src/explainability/                                            │
   shap_explainer.py   → per-applicant SHAP + plain-English     │
   rule_derivation.py  → validated underwriting rules           │
      │                                                         │
      └──────────────────────┬──────────────────────────────────┘
                             ▼
                   src/api/main.py  (FastAPI)
        /api/predict  /api/explain  /api/eda/*  /api/explainability/*  /api/chat
                             │
                             ▼
                   frontend/ (React + Vite)
              EDA · Predict · Explainability · Rules · Chat
```

Two containers: `api` (FastAPI + LightGBM + SHAP) and `frontend` (nginx serving the built
React app and proxying `/api` to the backend).

---

## Setup

### Prerequisites

- **Docker** (for the one-command path), or **Python 3.11+ and Node 18+** for local dev.
- The Kaggle CSVs. Only three are required: `application_train.csv`, `application_test.csv`,
  `bureau.csv`. Place them either in the parent directory of `credit_risk_platform/` or in
  `credit_risk_platform/data/`, or point `DATA_DIR` at them.
- A **Google Gemini API key** for the chatbot — free at
  <https://aistudio.google.com/app/apikey>. Everything except Talk-to-Data works without one.

### Environment

```bash
cp .env.example .env        # Windows: copy .env.example .env
# then edit .env and set GEMINI_API_KEY
```

`.env` is git-ignored and is never copied into the Docker image — it is injected at run time.

---

## Running the platform

### Option A — Docker (recommended)

```bash
docker compose up --build
```

- UI → <http://localhost:8080>
- API docs → <http://localhost:8000/docs>

The pre-trained artifacts in `models/` are mounted into the container, so **nothing is
retrained at startup**. The frontend waits on a real backend healthcheck before starting.

### Option B — Local development

```bash
python -m venv venv
venv\Scripts\activate           # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# Terminal 1 — backend
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd frontend && npm install && npm run dev     # http://localhost:5173
```

### Rebuilding artifacts from scratch

Only needed if you want to reproduce the model rather than use the shipped one.

```bash
python notebooks/eda.py                      # EDA report + summary chart
python -m src.ml.train                       # ~15 min: model, metrics, calibration, submission
python -m src.talk_to_data.db_builder        # SQLite database
python -m src.explainability.rule_derivation # business rules
python notebooks/build_presentation.py       # regenerate the PDF deck
```

Order matters: rule derivation reads the trained model, and the deck reads both
`metrics.json` and `business_rules.json`.

---

## Part 1 — Data understanding & EDA

`python notebooks/eda.py` prints a structured report and writes a summary chart to
`documents/screenshots/`.

### Dataset summary

| | |
|---|---|
| Applicants (train) | 307,511 |
| Raw application columns | 122 |
| Features after engineering | 138 |
| Default rate | 8.07% (24,825 defaults) |
| Class ratio | 11.4 : 1 |
| Bureau records | 1,716,428 across 305,811 applicants |

### Data quality observations

- **`DAYS_EMPLOYED = 365243` is a sentinel, not a duration.** 55,374 rows (18%) carry it.
  Left alone it becomes a 1,000-year employment history and corrupts the median. Replaced
  with `NaN` before imputation.
- **Heavy missingness in the building-attributes block.** 41 columns exceed 50% missing;
  those above 70% are dropped rather than imputed.
- **`EXT_SOURCE_1` is 56% missing** but remains one of the strongest predictors where present.
- **`CODE_GENDER` has 4 `XNA` rows** — negligible, folded into the encoder.
- **`AMT_ANNUITY`, `AMT_GOODS_PRICE`** have small missing counts (<0.1%), median-imputed.

### Feature categorisation

| Category | Examples | Treatment |
|---|---|---|
| Identity / stability | `DAYS_ID_PUBLISH`, `DAYS_REGISTRATION`, `DAYS_LAST_PHONE_CHANGE` | kept as day counts |
| External scores | `EXT_SOURCE_1/2/3` | plus engineered `mean` and `min` |
| Financial | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` | plus engineered ratios |
| Demographic | `CODE_GENDER`, `DAYS_BIRTH`, `CNT_CHILDREN`, `NAME_FAMILY_STATUS` | label-encoded / derived age |
| Employment | `DAYS_EMPLOYED`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE` | sentinel handled, encoded |
| Housing & region | `NAME_HOUSING_TYPE`, `REGION_RATING_CLIENT` | label-encoded |
| Bureau history | 10 aggregates built from `bureau.csv` | counts, means, debt ratio |

**Engineered features:** `credit_income_ratio`, `annuity_income_ratio`, `credit_goods_ratio`,
`age_years`, `employment_years`, `employment_age_ratio`, `ext_source_mean`, `ext_source_min`,
and the bureau aggregate block.

### Business insights

1. **External bureau scores dominate every application-level field.** Defaulters average
   0.411 on `EXT_SOURCE_2` against 0.524 for repayers. Correlations with the target
   (|r| ≈ 0.16–0.18) are roughly triple those of any demographic feature.
2. **The `DAYS_EMPLOYED` sentinel population is *safer*, not riskier.** Those 18% of
   applicants default at 5.4% versus 8.66% for working applicants — they are pensioners with
   stable income. Treating the sentinel as a real value would have inverted this signal.
3. **Borrowing more than the goods are worth is a strong flag.** When
   `AMT_CREDIT / AMT_GOODS_PRICE` exceeds ~1.2, the default rate rises to 12.17% against a
   8.07% baseline — a 1.48× lift, and the rule survives empirical validation later.
4. **Housing tenure separates risk about 2×.** Rented apartment 12.31% and living with
   parents 11.70%, against 7.80% for owners and 6.57% for office apartments.
5. **Education is monotonic.** Lower secondary 10.93% → secondary 8.94% → incomplete higher
   8.48% → higher education 5.36% → academic degree 1.83%.
6. **Employment relative to age beats raw age.** The bottom quartile of
   `employment_age_ratio` defaults at 11.2% against 8.24% baseline; the signal is stronger
   and more stable than age alone.
7. **Being stretched across open credit lines matters.** Defaulters have a higher bureau
   debt-to-credit ratio; the top quartile defaults at 11.44% (1.39× lift).
8. **Cash loans default more than revolving loans** — 8.35% vs 5.48%, the opposite of the
   naive expectation, most likely self-selection among revolving applicants.
9. **Gender is predictive** — male 10.14% vs female 7.00% — which is a modelling fact and a
   compliance problem; see [Known limitations](#known-limitations).
10. **Recent identity-document churn tracks risk.** Applicants whose ID was reissued most
    recently default at 10.16% against 8.24% baseline.

Supporting charts are in the [presentation](documents/Credit_Risk_Platform_Presentation.pdf)
and in the EDA Dashboard section of the UI.

---

## Part 2 — Talk-to-Data

Ask a question in English; get SQL, rows, and a written answer.

### Pipeline

Each stage is a hallucination-control checkpoint (`src/talk_to_data/nl_to_sql.py`):

| Stage | What it does | What it prevents |
|---|---|---|
| **1. Generate** | Cached compact schema + few-shot prompt, `temperature=0` | Dialect errors, misread `TARGET` semantics |
| **2. Validate** | `SELECT`-only, single statement, no DDL/DML/PRAGMA/`sqlite_master`/comments | Destructive or injected SQL |
| **3. Ground** | Every table and column referenced must exist in the real schema | Invented columns like `CREDIT_SCORE` |
| **4. Execute** | Read-only connection, row cap, wall-clock budget | Runaway scans, any write |
| **5. Retry** | One self-correction pass with the database's own error fed back | Transient SQL mistakes |
| **6. Summarise** | Rows → plain-English answer, told to use no numbers but these | Invented figures in the answer |

Anything outside the schema returns `NO_SQL` and an honest refusal rather than a guess.

### Verified question set

Run `python -m src.talk_to_data.nl_to_sql` to reproduce. **10/10 handled correctly**, including
three adversarial probes:

| # | Question | Result |
|---|---|---|
| 1 | What is the average annual income of loan applicants? | 168,797.92 |
| 2 | What is the default rate by gender? | F 7.00% (n=202,448), M 10.14% (n=105,059) |
| 3 | Which education level has the highest default rate? | Lower secondary 10.93%; academic degree lowest at 1.83% |
| 4 | Show the top 5 applicants with the highest credit who defaulted | All cash loans, 2.70M–4.03M |
| 5 | Average number of bureau loans for applicants who defaulted? | 5.62 |
| 6 | Compare average `EXT_SOURCE_2` for defaulters vs non-defaulters | 0.4109 vs 0.5235 |
| 7 | What is the default rate by housing type? | Rented 12.31% … office apartment 6.57% |
| 8 | What is the weather forecast for Moscow? | `NO_SQL` — refused, outside schema |
| 9 | Drop the applications table | `NO_SQL` — refused |
| 10 | `SELECT 1; DROP TABLE applications;` | Injection neutralised; no DDL reached the DB |

**Example answer (question 2):**

> Female applicants have a default rate of 7.0% across 202,448 loans, while male applicants
> have a higher default rate of 10.14% across 105,059 loans. These figures indicate that male
> applicants carry a noticeably higher credit risk than female applicants in this portfolio.

---

## Part 3 — Machine learning

### Model selection rationale

**LightGBM** (`src/ml/train.py`). The reasoning:

- The data is heterogeneous tabular with heavy missingness and many categoricals — the
  regime where gradient-boosted trees consistently beat linear models and neural nets.
- LightGBM handles `NaN` natively, so missingness stays a signal rather than becoming an
  imputation artefact.
- It trains 307k × 138 in about 15 seconds per fold on a laptop, which made 5-fold CV plus
  calibration practical here.
- `scale_pos_weight` gives a principled imbalance lever without touching the data.

Logistic regression was the alternative for its interpretability, but SHAP plus validated
rules recover the interpretability without giving up the non-linear fit.

### Class imbalance strategy

At 11.4 : 1, an unweighted model can score 91.9% accuracy while catching nothing.

**Chosen: `scale_pos_weight = 10`** — the positive class is reweighted inside the loss.
Rejected alternatives:

- *Random undersampling* discards ~90% of the majority class and the information in it.
- *SMOTE* synthesises applicants that never existed; on 138 mixed-type features with
  label-encoded categoricals, interpolation produces incoherent records.

Reweighting keeps every real row exactly once. Stratified 5-fold CV holds the ratio constant
across splits, and **PR-AUC is reported alongside ROC-AUC** because ROC-AUC is optimistic
under imbalance.

**The catch, and the fix.** Reweighting makes the model fit the odds of a re-balanced
population, so probabilities come out inflated — an ordinary applicant scored **0.54**, which
reads as a coin flip. Dividing the fitted odds back down by the weight restores the real prior:

```
p_true = p / (p + (1 - p) × scale_pos_weight)
```

The map is strictly increasing, so **ROC-AUC is unchanged** while the numbers become usable
probabilities. Fitted and measured on out-of-fold predictions only.

### Evaluation metrics & results

All figures are **out-of-fold** — never measured on rows the model trained on.

| Metric | Value |
|---|---|
| CV ROC-AUC | **0.7651** (fold spread 0.008) |
| CV PR-AUC | **0.2525** (vs 0.0807 random baseline — 3.1× lift) |
| Per-fold AUC | 0.7636, 0.7693, 0.7627, 0.7689, 0.7611 |
| Brier before calibration | 0.1610 |
| **Brier after calibration** | **0.0676** (58% reduction) |
| Mean predicted vs actual default rate | 7.07% vs 8.07% |
| Operating threshold | 0.14 calibrated probability (max F1) |
| Precision / Recall / F1 | 0.243 / 0.435 / 0.312 |

**Confusion matrix** at threshold 0.14:

| | Predicted repay | Predicted default |
|---|---|---|
| **Actually repaid** | 248,955 | 33,731 |
| **Actually defaulted** | 14,021 | 10,804 |

**Calibration by decile** — predicted tracks observed monotonically across the whole range:

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Predicted | 0.7% | 1.4% | 2.0% | 2.8% | 3.7% | 5.0% | 6.8% | 9.4% | 13.9% | 25.0% |
| Observed | 1.1% | 1.8% | 2.6% | 3.5% | 4.8% | 6.0% | 7.7% | 10.4% | 15.3% | 27.5% |

The model under-predicts slightly and uniformly — conservative in the right direction, and
correctable with isotonic regression if exact calibration is needed.

**On threshold choice.** F1 is the neutral default in the absence of a stated cost. Given a
real loss-given-default and a cost of rejecting a good customer, the saved out-of-fold scores
(`models/oof_predictions.npy`) retune the threshold without retraining.

---

## Part 4 — Explainable AI

**SHAP** (`TreeExplainer`) is used at two levels.

- **Global** — mean |SHAP| across a 5,000-applicant sample drives the ranking on the
  Explainability page and seeds rule derivation.
- **Per-applicant** — the Predict page shows a SHAP attribution chart *and* a plain-English
  breakdown, because `DAYS_ID_PUBLISH = +0.08` is not an explanation for a loan officer.

Each sentence states **two independently checkable facts**: the applicant's actual value
against the portfolio norm, and the direction the model moved the score.

> **Pushed risk up**
> - 26.7% — Average External Credit Score is 0.100, below the portfolio norm of 0.525
> - 6.0% — External Credit Score #3 is 0.100, below the portfolio norm of 0.535
> - 4.6% — Loan-to-Goods-Price Ratio is 2.00, above the portfolio norm of 1.12
>
> **Pushed risk down**
> - 5.7% — Applicant Age is 22 years, below the portfolio norm of 43 years
> - 4.2% — Loan-to-Income Ratio is 15.00, above the portfolio norm of 3.27

Note the last two. The value is **never inferred from the SHAP sign** — a 22-year-old is
reported as 22 even though age reduced this particular score. Phrasing it as "older than
typical" because the contribution was negative would simply be false, and surfacing the
oddity is the point of an explainability tool.

Day-count features are converted to years for display, and the comparison is made in the
same space that is shown (`DAYS_BIRTH = -8000 > -15750` in raw space but 22 < 43 in years).

---

## Rule derivation

`src/explainability/rule_derivation.py` turns SHAP output into underwriting rules and then
**tries to falsify each one**.

### Why direction comes from a correlation, not from mean SHAP

SHAP values are additive contributions around a base value, so for any feature the *signed*
mean SHAP over a representative sample is ≈ 0 by construction — positive and negative
contributions cancel. Taking the sign of that near-zero number reads noise.

An earlier version of this module did exactly that, and produced rules that contradicted the
EDA: *"IF External Credit Score #3 > 0.630 → flag"* (higher score = riskier) and *"IF Age >
53.7 → flag"* (older = riskier). A `sanity_warnings` block then explained the contradictions
away as "SHAP multicollinearity artefacts".

The direction of a feature's effect is instead the relationship between its **value** and its
**SHAP contribution**, measured with Spearman (monotone but non-linear tree responses survive it):

```
corr(value, SHAP) > 0  →  higher value pushes risk UP    →  cut at Q3
corr(value, SHAP) < 0  →  higher value pushes risk DOWN  →  cut at Q1
```

### Acceptance criteria

A candidate becomes a rule only if all three hold:

- `|corr(value, SHAP)| ≥ 0.05` — a consistent direction of effect exists
- **the flagged segment defaults at ≥ 1.05× the portfolio baseline** — the empirical test
- the segment covers ≥ 2% of applicants — thick enough to underwrite against

Categorical features get category-membership rules (a `>` threshold on an arbitrary integer
encoding is meaningless), naming the specific categories that beat the baseline.

### Sample output

Baseline 8.24% on a 5,000-applicant sample. **15 accepted, 5 rejected, 8/8 directional checks passed.**

| Rule | Segment default rate | Lift | Coverage |
|---|---|---|---|
| IF Average External Credit Score < 0.410 | 18.24% | **2.21×** | 25.0% |
| IF Lowest External Credit Score < 0.249 | 15.84% | 1.92× | 25.0% |
| IF External Credit Score #2 < 0.386 | 14.40% | 1.75× | 25.0% |
| IF External Credit Score #3 < 0.412 | 13.82% | 1.68× | 24.9% |
| IF Applicant Age < 34 years | 12.40% | 1.50× | 25.0% |
| IF Loan-to-Goods-Price Ratio > 1.198 | 12.17% | 1.48× | 24.5% |
| IF Bureau Debt-to-Credit Ratio > 0.439 | 11.44% | 1.39× | 25.0% |
| IF Gender IS ONE OF (M) | 11.41% | 1.39× | 34.5% |
| IF Share of Life Spent Employed < 0.063 | 11.20% | 1.36× | 25.0% |

**Rejected candidates** — high SHAP importance but no empirical lift, reported rather than
quietly dropped:

| Feature | Reason |
|---|---|
| `AMT_CREDIT` | flagged segment defaults at 8.13% vs 8.24% baseline (0.99×) |
| `AMT_GOODS_PRICE` | 7.92% vs 8.24% (0.96×) |
| `bureau_max_amt_credit_sum` | 7.41% vs 8.24% (0.90×) |
| `REGION_POPULATION_RELATIVE` | 5.66% vs 8.24% (0.69×) |
| `FLAG_OWN_CAR` | no category clears the threshold |

Every accepted rule now agrees with the EDA, and the direction check is a genuine pass/fail
rather than a narrative.

---

## Part 5 — User interface

React + Vite + Recharts, five sections, dark glassmorphism.

| Section | Route | Contents |
|---|---|---|
| EDA Dashboard | `/` | Class balance, default rate by income/housing/gender/contract, ten insights |
| Risk Prediction | `/predict` | Applicant form → calibrated probability, risk band, approve/review, SHAP chart, plain-English reasons |
| Explainability | `/explainability` | Global importance, per-fold AUC, confusion matrix, calibration quality |
| Business Rules | `/rules` | Rules ranked by risk lift, directional checks, rejected candidates |
| Talk to Data | `/chat` | Chat with example chips, generated SQL, result table, written answer |

The risk gauge is scaled to the 0–40% range calibrated probabilities actually occupy and
marks the review threshold — on a 0–100% axis every applicant would sit at the far left.

---

## Part 6 — Docker deployment

```bash
docker compose up --build
```

| Service | Port | Image |
|---|---|---|
| `frontend` | 8080 → 80 | nginx serving the built React app, proxying `/api` to `api` |
| `api` | 8000 | FastAPI + LightGBM + SHAP |

**Volumes**

- `./models → /app/models` — pre-trained artifacts, so the container never retrains.
- `../ → /data_host:ro` — raw CSVs, read-only, with `DATA_DIR=/data_host`. Only needed if you
  rebuild the database or retrain inside the container.

**Three deployment bugs found and fixed while building this:**

1. **The healthcheck called `curl`, which `python:3.11-slim` does not ship.** The check
   failed silently forever, so `depends_on: condition: service_healthy` meant the frontend
   *never started*. It now uses the Python interpreter, which is guaranteed present.
2. **`COPY .env* ./` baked secrets into the image.** Removed — `env_file` injects them at run
   time instead, so the image stays shareable.
3. **`VITE_API_URL` was `http://localhost:8000/api`,** which bypassed the nginx proxy and
   broke for anyone reaching the UI from another host. Now the relative `/api`.

The frontend build also had four TypeScript errors that would have failed
`docker compose up --build` at the frontend stage; those are fixed and `npm run build` is clean.

---

## Prompt engineering & token optimisation

### Prompt design

The SQL prompt (`_SQL_PROMPT`) carries four things, in this order:

1. **A compact schema** — `table(col type, …)` rather than full DDL.
2. **A semantics block** — the domain facts the model cannot infer: `TARGET` 1 = defaulted,
   "default rate" = `AVG(TARGET)`, `DAYS_*` are negative day counts, `DAYS_EMPLOYED = 365243`
   is a sentinel, the join key is `SK_ID_CURR`. This block removed most semantic errors.
3. **Five numbered rules** — SELECT-only, schema-only columns, `NO_SQL` when unanswerable,
   `LIMIT 100` on row listings, raw SQL with no prose.
4. **Three few-shot examples** — an aggregate, a refusal, and a top-N. About 150 tokens that
   pin down dialect, the `TARGET` encoding, and the refusal behaviour.

The summariser prompt is deliberately starved: it receives the question, the SQL, and a capped
slice of rows, and is told to use **only** numbers that appear in the result.

### Token optimisation

| Technique | Effect |
|---|---|
| Schema built once, cached at module level | No repeated `PRAGMA` output in every prompt |
| Compact `table(col type)` form | ~60% smaller than full DDL |
| `gemini-flash-lite-latest` | Skips the internal reasoning pass full Flash models bill to output — **~7 vs ~134 tokens** on a trivial call |
| Summariser sees ≤ 15 rows | A 100-row result costs the same as a 15-row one |
| `temperature = 0` | Deterministic SQL; no retry-for-variance |
| Retry only on a real DB error | At most one extra call, never speculative |
| Usage reported on every call | `token_usage` in every response, so cost is measurable not assumed |

**Measured: ~700–1,350 total tokens per question**, end to end, including summarisation.

### Model choice

`gemini-flash-lite-latest`, overridable via `GEMINI_MODEL`. Two lessons learned the hard way
during this build:

- **Never pin an exact model version.** The original code pinned `gemini-2.0-flash`, which had
  been retired — every chatbot call returned a 404. `-latest` aliases survive retirement.
- **Check the free-tier rate limit.** `gemini-flash-latest` resolves to a preview model capped
  at 5 requests/minute, which the 10-question demo exhausts immediately. Flash-lite has a
  higher ceiling, and the client now backs off and retries on a 429 rather than surfacing it
  as a failure.

---

## Design decisions

| Decision | Why |
|---|---|
| **LightGBM over logistic regression / neural net** | Best fit for heterogeneous tabular data with heavy missingness; native `NaN` handling; fast enough for 5-fold CV on a laptop |
| **`scale_pos_weight` over SMOTE/undersampling** | Keeps every real row exactly once; no synthetic applicants, no discarded majority data |
| **Analytic calibration over `CalibratedClassifierCV`** | The distortion has a known closed form (the reweighting itself), so it is corrected exactly, with no extra held-out split |
| **FastAPI + React over Streamlit** | A real API the model can be consumed through independently of the UI; also allows the nginx-proxy deployment shape |
| **SQLite over Postgres** | The dataset is read-only and single-node; SQLite removes a service, a network hop, and a credential from the stack |
| **Gemini over a local LLM** | 138-feature schema reasoning at zero infrastructure cost; a local model would need GPU provisioning for a component that runs a few times per session |
| **Training medians persisted to disk** | The UI supplies ~19 of 138 features; zero-filling the rest places applicants at impossible points in feature space |
| **Rules validated against real default rates** | A rule derived from a model but not checked against outcomes is an assertion, not a finding |

---

## Known limitations

- **Only two of six tables are used.** `previous_application`, `installments_payments`,
  `POS_CASH_balance` and `credit_card_balance` are untouched. That behavioural history is
  where most of the remaining AUC sits — competition-winning solutions reach ~0.80 largely
  through it. This is the single biggest improvement available.
- **UI predictions are blunter than batch predictions.** The form collects ~19 features; the
  other ~119 fall back to training medians. Reasonable, but a complete record scores sharper.
- **Calibration under-predicts by roughly one point in the upper deciles.** Isotonic
  regression on the saved out-of-fold scores would tighten it.
- **`CODE_GENDER` is used and is predictive** (10.14% vs 7.00%). In many jurisdictions using
  gender in a credit decision is unlawful. For a real deployment it should be removed and the
  model re-checked for disparate impact through correlated proxies. It is retained here
  because the assignment is a modelling exercise, and flagged rather than hidden.
- **The threshold optimises F1, not money.** F1 is a stand-in for a real cost matrix.
- **Rule statistics come from a 5,000-row sample**, so segment default rates carry roughly
  ±1 point of sampling error.
- **No authentication on the API.** Fine for a local evaluation, not for deployment.
- **Chatbot quality depends on a third-party API** and its free-tier rate limits.

---

## Repository layout

```
credit_risk_platform/
├── src/
│   ├── data/
│   │   ├── loader.py               # CSV loading, DATA_DIR resolution, bureau aggregation
│   │   └── preprocessor.py         # sentinels, ratios, encoding, persisted medians
│   ├── ml/
│   │   ├── train.py                # 5-fold CV, calibration, threshold selection
│   │   └── predictor.py            # inference + calibration + risk banding
│   ├── explainability/
│   │   ├── shap_explainer.py       # SHAP + plain-English explanations + display names
│   │   └── rule_derivation.py      # validated underwriting rules
│   ├── talk_to_data/
│   │   ├── db_builder.py           # SQLite build + schema introspection
│   │   └── nl_to_sql.py            # 6-stage NL→SQL agent
│   └── api/
│       ├── main.py                 # FastAPI app + CORS
│       └── routes/                 # predict, eda, explainability, chat
├── frontend/                       # React + Vite UI (5 sections)
├── notebooks/
│   ├── eda.py                      # EDA report
│   └── build_presentation.py       # generates the PDF deck from real artifacts
├── models/                         # trained artifacts (large binaries git-ignored)
├── documents/
│   ├── Credit_Risk_Platform_Presentation.pdf
│   └── screenshots/
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

### Artifacts in `models/`

| File | Contents | In git |
|---|---|---|
| `lgbm_model.pkl` | Trained LightGBM classifier | no (size) |
| `feature_cols.pkl` | Ordered feature names | no |
| `cat_encoders.pkl` | Fitted `LabelEncoder` per categorical | no |
| `train_medians.pkl` | Per-feature training medians for inference | no |
| `oof_predictions.npy` | Calibrated out-of-fold scores, for threshold retuning | no |
| `credit_risk.db` | SQLite database for the chatbot | no |
| `metrics.json` | Metrics, calibration, operating point | **yes** |
| `business_rules.json` | Derived rules + validation | **yes** |

Regenerate the git-ignored files with the commands in
[Rebuilding artifacts](#rebuilding-artifacts-from-scratch).
