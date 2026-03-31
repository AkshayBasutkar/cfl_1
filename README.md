# CFL Parallel Meta-Ensemble Forecasting Pipeline

A production-grade, parallel multi-model sales-unit forecasting system for the **CFL Phase 2** dataset. The pipeline runs three independent forecasting models concurrently and fuses their outputs into a single, risk-managed final prediction for **FY26 Q2** bookings.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Models](#models)
  - [Model 1 – Hybrid Risk-Managed Anchor](#model-1--hybrid-risk-managed-anchor)
  - [Model 2 – v1 Human-Ensemble Architecture](#model-2--v1-human-ensemble-architecture)
  - [Model 3 – v8 Oracle Engine](#model-3--v8-oracle-engine)
- [Meta-Fusion Logic](#meta-fusion-logic)
- [Input Data Format](#input-data-format)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Diagnostic Table](#diagnostic-table)
- [Project Structure](#project-structure)

---

## Overview

This pipeline forecasts product-level booking units for **FY26 Q2** across a catalogue of telecom hardware products (e.g. IP Phones, Headsets). It ingests an Excel workbook that holds historical actuals and human forecast inputs, executes three machine-learning / heuristic models in parallel threads, and then merges their predictions with a bifurcated meta-fusion strategy that takes product cost rank into account.

Key design goals:
- **Parallel execution** – all three models run simultaneously via `ThreadPoolExecutor`, minimising wall-clock time.
- **Risk management** – asymmetric loss functions and lifecycle-based adjustments prevent systematic over-forecasting.
- **Human-in-the-loop blending** – Demand Planning (DP), Marketing (Mkt), and Data Science (DS) forecasts are incorporated where available.
- **Auditability** – a per-product diagnostic table is printed to stdout, showing actual vs. forecast for the most recent completed quarter (FY26 Q1).

---

## Architecture

```
Excel Workbook
      │
      ├── Thread 1 ──► Model 1 (LightGBM + XGBoost + Dynamic Blend) ──► M1_Pred
      ├── Thread 2 ──► Model 2 (Human Ensemble + Moving Average)     ──► M2_Pred
      └── Thread 3 ──► Model 3 (Oracle Engine + Business Overrides)  ──► M3_Pred
                                           │
                              Bifurcated Meta-Fusion
                              (Cost Rank–based routing)
                                           │
                              FINAL_META_FORECAST
                                           │
                              ┌────────────┴────────────┐
                         CSV Output              Diagnostic Table
```

---

## Models

### Model 1 – Hybrid Risk-Managed Anchor

**File location:** `execute_model_1()` in `main.py`

Model 1 is the primary ML anchor. It trains two gradient-boosted regressors on log-transformed lag features and then blends the ML prediction with human forecasts using dynamic, accuracy-weighted coefficients.

**Feature engineering:**
| Feature | Description |
|---|---|
| `Log_Lag_1` … `Log_Lag_5` | Natural log of units booked 1–5 quarters ago |
| `Log_Roll_Mean` | Log of the 4-quarter rolling mean |
| `YoY_Growth` | Ratio of last quarter to same quarter one year ago |

**ML layer:**
- **LightGBM** (`n_estimators=100`, `max_depth=3`, `lr=0.05`) – trained with a custom *asymmetric* objective that penalizes over-predictions more heavily, reducing excess-inventory risk.
- **XGBoost** (`n_estimators=100`, `max_depth=2`, `lr=0.05`, `objective=reg:absoluteerror`) – a conservative complementary learner.
- Base forecast = `LightGBM × 0.60 + XGBoost × 0.40`.

**Lifecycle adjustments applied after the ML base:**
| Lifecycle tag | Multiplier |
|---|---|
| Decline | × 0.85 |
| NPI (New Product Introduction) | × 1.15 |

**Dynamic blend with human inputs:**
```
w_DS  = min((1 − DS_bias_error) × 0.45, 0.50)
w_DP  = min((1 − DP_bias_error) × 0.35, 0.50)
w_ML  = max(1 − (w_DS + w_DP), 0.20)

M1_Pred = (DS_forecast × w_DS + DP_forecast × w_DP + ML_forecast × w_ML)
          / (w_DS + w_DP + w_ML)
```
Forecasters with historically lower bias receive higher weights automatically.

---

### Model 2 – v1 Human-Ensemble Architecture

**File location:** `execute_model_2()` in `main.py`

Model 2 is a lightweight human-ensemble blender. It averages all available human forecasts (DP, Mkt, DS) and mixes that average with a 3-period moving average of recent actuals.

```
M2_Pred = mean(available human forecasts) × 0.84
        + 3-period moving average × 0.16
```

If no human forecasts are present for a product, the 3-period moving average is used directly as a fallback.

This model acts as a **conservative, interpretable baseline** that anchors the ensemble against purely data-driven extrapolations.

---

### Model 3 – v8 Oracle Engine

**File location:** `execute_model_3()` in `main.py`

Model 3 is a rule-based and trend-following engine that captures business knowledge and product-specific overrides.

**Business-logic overrides (hardcoded for known products):**
| Product | Override value |
|---|---|
| IP PHONE Enterprise Desk_1 | 10,500 units |
| IP PHONE Enterprise Desk_2 | 5,800 units |

**Trend logic for all other products:**
```
M3_Pred = lag_1 × 1.05   if lag_1 > lag_2  (upward trend → slight uplift)
        = lag_1 × 0.95   otherwise          (flat/downward → slight haircut)
```

This model is the most sensitive to the most recent quarter and acts as a **momentum signal** within the ensemble.

---

## Meta-Fusion Logic

After the three model predictions are collected they are combined using a **bifurcated routing strategy** that depends on the product's **Cost Rank** (lower = higher value / priority product):

| Cost Rank | Fusion formula |
|---|---|
| ≤ 3 (high-priority) | `0.50 × M1 + 0.50 × M2` |
| > 3 (standard) | `min(M1, M2, M3)` |

High-priority products receive an average of the two most robust models (Model 1 and Model 2), whereas standard products take the minimum across all three – this **conservative floor** limits upside bias and protects against over-stocking.

---

## Input Data Format

The pipeline reads a single Excel workbook. Default path (configurable):

```
/content/CFL_External Data Pack_Phase2.xlsx
```

**Required sheet:** `Ph.2 Data Pack-Actual Booking`

**Sheet layout (0-indexed columns):**

| Column index | Content |
|---|---|
| 0 | Cost Rank (integer; lower = higher priority) |
| 1 | Product name |
| 2 | Lifecycle stage (`Sustaining`, `Decline`, `NPI`, `Growth`, `Mature`) |
| 3 – 14 | Historical booking actuals: FY23 Q2 → FY26 Q1 (12 quarters) |
| 15 | (reserved / unused) |
| 16 | Demand Planning (DP) forecast for FY26 Q2 |
| 17 | Marketing (Mkt) forecast for FY26 Q2 |
| 18 | Data Science (DS) forecast for FY26 Q2 |
| 19 | (reserved / unused) |
| 20 | DS historical bias metric |

Rows 0–1 are treated as headers and skipped. Only rows where column 1 (Product) is non-null are processed. Rows where column 2 contains a lifecycle keyword are treated as **unit-booking rows**; all other non-null rows are treated as **metric rows** (bias figures, etc.).

---

## Dependencies

Install all required packages with:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm openpyxl
```

| Package | Purpose |
|---|---|
| `pandas` | Data ingestion, transformation, and output |
| `numpy` | Numerical operations and log transformations |
| `scikit-learn` | `Ridge`, `ElasticNet`, `RandomForestRegressor`, `GradientBoostingRegressor`, `StandardScaler` (imported; available for extension) |
| `xgboost` | Gradient-boosted trees (Model 1) |
| `lightgbm` | Gradient-boosted trees with custom loss (Model 1) |
| `openpyxl` | Excel `.xlsx` reading backend for pandas |
| `concurrent.futures` | Standard-library thread pool for parallel model execution |

**Python version:** 3.8 or higher recommended.

---

## Configuration

Edit the constants at the top of `main.py` before running:

| Constant | Default | Description |
|---|---|---|
| `FILE_PATH` | `'/content/CFL_External Data Pack_Phase2.xlsx'` | Absolute path to the input Excel workbook |

The output CSV path is set inside `run_parallel_pipeline()`:

| Variable | Default | Description |
|---|---|---|
| `OUTPUT_FILE` | `'/content/CFL_Phase2_Final_Pipeline_Submission.csv'` | Absolute path for the submission CSV |

---

## Usage

### Running in Google Colab (recommended)

1. Upload `CFL_External Data Pack_Phase2.xlsx` to `/content/` in your Colab session.
2. Install dependencies:
   ```python
   !pip install xgboost lightgbm openpyxl -q
   ```
3. Run the script:
   ```python
   !python main.py
   ```
   Or paste the contents of `main.py` into a Colab cell and execute it.

### Running locally

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm openpyxl

# 2. Update FILE_PATH in main.py to point to your local workbook

# 3. Execute
python main.py
```

---

## Output

### Submission CSV

Path: `/content/CFL_Phase2_Final_Pipeline_Submission.csv`

| Column | Description |
|---|---|
| `Cost Rank` | Product priority rank (ascending) |
| `Product Name` | Product identifier |
| `Predicted FY26 Q2 Units` | Final ensemble forecast (integer) |

The file is sorted by `Cost Rank` ascending.

### Console output

The pipeline prints progress messages as each thread starts and completes, followed by a detailed diagnostic table (see below).

---

## Diagnostic Table

After producing forecasts, the pipeline back-tests against **FY26 Q1 actuals** (the most recent completed quarter in the dataset) to give an indicative accuracy benchmark:

```
===============================================================================================
 📊 PER-PRODUCT ACCURACY BREAKDOWN (FY26 Q1 BACKTEST)
===============================================================================================
CR  | Product Name                             | Actual   | Forecast | Accuracy | Status
-----------------------------------------------------------------------------------------------
1   | IP PHONE Enterprise Desk_1               |   10,200 |   10,500 |   97.1%  | ✓ Excellent
...
```

**Accuracy formula:**
```
accuracy = max(0, 100 − (|actual − forecast| / actual) × 100)
```

**Status thresholds:**

| Range | Label |
|---|---|
| ≥ 90 % | ✓ Excellent |
| 75 % – 89 % | △ Acceptable |
| < 75 % | ⚠ Critical Miss |

> **Note:** The diagnostic table uses FY26 Q1 actuals as a proxy. The final submission forecasts target FY26 Q2, for which actuals are not yet available.

---

## Project Structure

```
cfl_1/
├── main.py          # Full pipeline: data loading, Models 1–3, meta-fusion, output
└── README.md        # This documentation file
```
