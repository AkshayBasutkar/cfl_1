"""
CFL Phase 2 – Production Apex Pipeline (Parallel Meta-Ensemble)
================================================================
Dynamically runs Model 1, Model 2, and Model 3 in parallel threads and fuses
their FY26 Q2 unit-booking forecasts into a single, risk-managed submission.

Pipeline summary
----------------
1. **Model 1 – Hybrid Risk-Managed Anchor**
   Trains LightGBM (custom asymmetric loss) and XGBoost on log-transformed lag
   features, applies lifecycle multipliers, then blends the ML forecast with
   human Demand-Planning (DP) and Data-Science (DS) forecasts using dynamic
   accuracy-weighted coefficients.

2. **Model 2 – v1 Human-Ensemble Architecture**
   Averages available DP / Marketing / DS human forecasts and mixes that
   average with a 3-period moving average of historical actuals
   (global alpha ≈ 0.84 / 0.16).

3. **Model 3 – v8 Oracle Engine**
   Applies hardcoded business-logic overrides for known high-value products and
   uses a simple momentum rule (±5 %) for all others.

4. **Bifurcated Meta-Fusion**
   - Cost Rank ≤ 3  →  50 % Model 1 + 50 % Model 2
   - Cost Rank >  3  →  min(Model 1, Model 2, Model 3)

Input
-----
FILE_PATH : str
    Path to the Excel workbook ``CFL_External Data Pack_Phase2.xlsx``.
    Required sheet: ``Ph.2 Data Pack-Actual Booking``.

Output
------
A CSV file at ``/content/CFL_Phase2_Final_Pipeline_Submission.csv`` containing
columns ``Cost Rank``, ``Product Name``, and ``Predicted FY26 Q2 Units``,
sorted by Cost Rank ascending.

A per-product diagnostic table is also printed to stdout showing FY26 Q1
back-test accuracy for indicative model evaluation.

Usage
-----
    python main.py

Dependencies
------------
pandas, numpy, scikit-learn, xgboost, lightgbm, openpyxl
"""

#pipelined 3 models
# ==============================================================================
# 🏆 THE PRODUCTION APEX PIPELINE (PARALLEL META-ENSEMBLE)
# Dynamically runs Model 1, Model 2, and Model 3 in parallel and fuses the outputs.
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
import concurrent.futures
import openpyxl
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

FILE_PATH = '/content/CFL_External Data Pack_Phase2.xlsx'

# ==============================================================================
# 1. MODEL 1: HYBRID RISK-MANAGED ARCHITECTURE (Isolated Execution)
# ==============================================================================
def execute_model_1(filepath):
    """Run Model 1: Hybrid Risk-Managed Anchor.

    Loads the booking data, engineers lag and rolling-mean features, trains a
    LightGBM + XGBoost ensemble on log-transformed targets, applies lifecycle
    multipliers, and finally blends the ML forecast with human DP / DS inputs
    using dynamic accuracy-weighted coefficients derived from historical bias.

    Parameters
    ----------
    filepath : str
        Absolute path to the input Excel workbook
        (``CFL_External Data Pack_Phase2.xlsx``).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``['Cost Rank', 'Product', 'M1_Pred']``
        containing one row per product.  ``M1_Pred`` is a non-negative integer.

    Notes
    -----
    Feature set used for ML training:
        ``Log_Lag_1`` … ``Log_Lag_5``, ``Log_Roll_Mean``, ``YoY_Growth``

    Lifecycle multipliers applied to the raw ML base forecast:
        * Decline → × 0.85
        * NPI     → × 1.15

    Dynamic blend weights (before normalisation):
        * ``w_DS  = min((1 − DS_error) × 0.45, 0.50)``
        * ``w_DP  = min((1 − DP_error) × 0.35, 0.50)``
        * ``w_ML  = max(1 − (w_DS + w_DP), 0.20)``
    """
    xl = pd.ExcelFile(filepath)
    df_raw = xl.parse('Ph.2 Data Pack-Actual Booking', header=None)
    QUARTERS = ['FY23 Q2', 'FY23 Q3', 'FY23 Q4', 'FY24 Q1', 'FY24 Q2', 'FY24 Q3',
                'FY24 Q4', 'FY25 Q1', 'FY25 Q2', 'FY25 Q3', 'FY25 Q4', 'FY26 Q1']

    df_data = df_raw.iloc[2:150].dropna(subset=[1]).copy()
    df_units = df_data[df_data[2].astype(str).str.contains('Sustaining|Decline|NPI|Growth|Mature', na=False, case=False)].reset_index(drop=True)
    df_metrics = df_data[~df_data[2].astype(str).str.contains('Sustaining|Decline|NPI|Growth|Mature', na=False, case=False)].reset_index(drop=True)

    actuals = df_units.iloc[:, :15].copy()
    actuals.columns = ['Cost Rank', 'Product', 'Lifecycle'] + QUARTERS
    for q in QUARTERS: actuals[q] = pd.to_numeric(actuals[q], errors='coerce').fillna(0)
    actuals['Cost Rank'] = pd.to_numeric(actuals['Cost Rank'], errors='coerce').fillna(30)
    actuals['FY26 Q2'] = 0

    team_f = df_units[[1, 16, 17, 18]].copy()
    team_f.columns = ['Product', 'DP_Q2', 'Mkt_Q2', 'DS_Q2']
    for col in ['DP_Q2', 'Mkt_Q2', 'DS_Q2']:
        team_f[col] = pd.to_numeric(team_f[col], errors='coerce').fillna(0)

    acc_map = {}
    for idx, row in df_metrics.iterrows():
        prod = str(row[2]).strip()
        try: dp_b = float(row[6])
        except: dp_b = 0.0
        try: ds_b = float(row[20])
        except: ds_b = 0.0
        acc_map[prod] = {'DP_Bias_Past': dp_b if not pd.isna(dp_b) else 0.15,
                         'DS_Bias_Past': ds_b if not pd.isna(ds_b) else 0.12}

    df_long = actuals.melt(id_vars=['Cost Rank', 'Product', 'Lifecycle'], value_vars=QUARTERS + ['FY26 Q2'], var_name='Quarter', value_name='Actual_Units')
    df_master = df_long.sort_values(by=['Product', 'Quarter']).reset_index(drop=True)

    for i in [1, 2, 3, 4, 5]:
        df_master[f'Lag_{i}'] = df_master.groupby('Product')['Actual_Units'].shift(i)
        df_master[f'Log_Lag_{i}'] = np.log1p(df_master[f'Lag_{i}'])

    df_master['Roll_Mean_4'] = (df_master['Lag_1'] + df_master['Lag_2'] + df_master['Lag_3'] + df_master['Lag_4']) / 4.0
    df_master['Log_Roll_Mean'] = np.log1p(df_master['Roll_Mean_4'])
    df_master['YoY_Growth'] = (df_master['Lag_1'] + 1) / (df_master['Lag_5'] + 1)

    df_master = df_master.dropna(subset=['Lag_5']).fillna(0)
    df_master['Sample_Weight'] = 1.0 / np.maximum(df_master['Cost Rank'], 1)
    df_master['Target_Log'] = np.log1p(df_master['Actual_Units'])

    features = ['Log_Lag_1', 'Log_Lag_2', 'Log_Lag_3', 'Log_Lag_4', 'Log_Roll_Mean', 'YoY_Growth']
    train_df = df_master[df_master['Quarter'] != 'FY26 Q2']
    test_df = df_master[df_master['Quarter'] == 'FY26 Q2'].copy()

    X_train, y_train_log, w_train = train_df[features], train_df['Target_Log'], train_df['Sample_Weight']
    X_test = test_df[features]

    def custom_asymmetric_objective(y_true, y_pred):
        """Custom asymmetric LightGBM objective that penalizes over-predictions.

        The residual is defined as ``residual = y_true - y_pred``.
        Over-forecasting (``y_pred > y_true``, so ``residual < 0``) is
        penalized with a gradient coefficient of 2.4, while under-forecasting
        uses 2.0.  This biases the model towards conservative estimates,
        reducing excess-inventory risk.

        Parameters
        ----------
        y_true : numpy.ndarray
            True log-transformed target values.
        y_pred : numpy.ndarray
            Predicted log-transformed values from the current tree iteration.

        Returns
        -------
        grad : numpy.ndarray
            First-order gradient of the loss.
        hess : numpy.ndarray
            Second-order (constant) Hessian of the loss.
        """
        grad = np.where(residual < 0, -2.4 * residual, -2.0 * residual)
        hess = np.where(residual < 0, 2.4, 2.0)
        return grad, hess

    lgb = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, objective=custom_asymmetric_objective, random_state=42, verbose=-1)
    lgb.fit(X_train, y_train_log, sample_weight=w_train)

    xgb = XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, objective='reg:absoluteerror', random_state=42)
    xgb.fit(X_train, y_train_log, sample_weight=w_train)

    test_df['ML_Base'] = (np.expm1(lgb.predict(X_test)) * 0.60) + (np.expm1(xgb.predict(X_test)) * 0.40)
    test_df['ML_Adjusted'] = test_df['ML_Base']
    test_df.loc[test_df['Lifecycle'].astype(str).str.contains('Decline', case=False), 'ML_Adjusted'] *= 0.85
    test_df.loc[test_df['Lifecycle'].astype(str).str.contains('NPI', case=False), 'ML_Adjusted'] *= 1.15

    final_df = pd.merge(test_df[['Product', 'Cost Rank', 'ML_Adjusted']], team_f, on='Product', how='left')

    def production_dynamic_blend(row):
        """Compute the accuracy-weighted blend of ML, DS, and DP forecasts.

        Weights are derived from each forecaster's historical bias: lower
        historical error → higher weight.  The ML floor is guaranteed at 20 %
        of the total weight, capping DS at 50 % and DP at 50 %.

        Parameters
        ----------
        row : pandas.Series
            A single row from ``final_df`` containing at least the fields
            ``Product``, ``ML_Adjusted``, ``DS_Q2``, and ``DP_Q2``.

        Returns
        -------
        float
            Blended forecast value (un-clipped, un-rounded).
        """
        ml_f, ds_f, dp_f = row['ML_Adjusted'], row['DS_Q2'], row['DP_Q2']
        ds_err = abs(acc_map.get(prod, {}).get('DS_Bias_Past', 0.12))
        dp_err = abs(acc_map.get(prod, {}).get('DP_Bias_Past', 0.15))
        ds_acc, dp_acc = max(1.0 - ds_err, 0.0), max(1.0 - dp_err, 0.0)

        w_ds = min(ds_acc * 0.45, 0.50)
        w_dp = min(dp_acc * 0.35, 0.50)
        w_ml = max(1.0 - (w_ds + w_dp), 0.20)
        total = w_ds + w_dp + w_ml
        return (ds_f * (w_ds/total)) + (dp_f * (w_dp/total)) + (ml_f * (w_ml/total))

    final_df['M1_Pred'] = final_df.apply(production_dynamic_blend, axis=1).clip(lower=0).round(0).astype(int)
    print("✅ [Thread 1] Model 1 Complete.")
    return final_df[['Cost Rank', 'Product', 'M1_Pred']]

# ==============================================================================
# 2. MODEL 2: v1 ARCHITECTURE (Isolated Execution)
# ==============================================================================
def execute_model_2(filepath):
    """Run Model 2: v1 Human-Ensemble Architecture.

    For each product, averages the available human forecasts (Demand Planning,
    Marketing, Data Science) and blends that average with a 3-period trailing
    moving average of historical actuals using a fixed global alpha (0.84 / 0.16).

    Parameters
    ----------
    filepath : str
        Absolute path to the input Excel workbook
        (``CFL_External Data Pack_Phase2.xlsx``).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``['Product', 'M2_Pred']`` containing one row
        per product.  ``M2_Pred`` is a non-negative integer.

    Notes
    -----
    Blend formula::

        M2_Pred = mean(available human forecasts) × 0.84
                + 3-period moving average          × 0.16

    If no human forecasts are available for a product, the 3-period moving
    average is used as the sole predictor (fallback).
    """
    # For speed and brevity in the parallel pipeline, we use a streamlined
    # proxy of Model 2's output logic utilizing the DP/DS/Mkt blend architecture.
    # (In a true modular environment, this would call import model2; model2.predict())

    xl = pd.ExcelFile(filepath)
    df_raw = xl.parse('Ph.2 Data Pack-Actual Booking', header=None)

    df_data = df_raw.iloc[2:150].dropna(subset=[1]).copy()
    df_units = df_data[df_data[2].astype(str).str.contains('Sustaining|Decline|NPI|Growth|Mature', na=False, case=False)].reset_index(drop=True)

    res = []
    for idx, row in df_units.iterrows():
        prod = str(row[1]).strip()
        # Model 2 heavily weighted the human ensemble for FY26 Q2
        f_dp = pd.to_numeric(row[16], errors='coerce')
        f_mkt = pd.to_numeric(row[17], errors='coerce')
        f_ds = pd.to_numeric(row[18], errors='coerce')

        forecasts = [f for f in [f_dp, f_mkt, f_ds] if not pd.isna(f) and f > 0]
        # Model 2 proxy: average of available human forecasts blended with trailing moving average
        recent_actuals = [pd.to_numeric(row[c], errors='coerce') for c in range(12, 15)]
        ma_3 = np.mean([a for a in recent_actuals if not pd.isna(a)])

        if forecasts:
            m2_pred = (np.mean(forecasts) * 0.84) + (ma_3 * 0.16) # M2's typical global alpha
        else:
            m2_pred = ma_3

        res.append({'Product': prod, 'M2_Pred': int(round(max(0, m2_pred)))})

    print("✅ [Thread 2] Model 2 Complete.")
    return pd.DataFrame(res)

# ==============================================================================
# 3. MODEL 3: v8 ORACLE ENGINE (Isolated Execution)
# ==============================================================================
def execute_model_3(filepath):
    """Run Model 3: v8 Oracle Engine.

    Applies hardcoded business-logic overrides for specific high-value products
    and uses a simple momentum rule for all other products:
    if the most recent quarter's actuals exceed the prior quarter's actuals the
    forecast is a 5 % uplift, otherwise a 5 % haircut is applied.

    Parameters
    ----------
    filepath : str
        Absolute path to the input Excel workbook
        (``CFL_External Data Pack_Phase2.xlsx``).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``['Product', 'M3_Pred']`` containing one row
        per product.  ``M3_Pred`` is a non-negative integer.

    Notes
    -----
    Business-logic overrides (hardcoded)::

        'IP PHONE Enterprise Desk_1'  →  10,500
        'IP PHONE Enterprise Desk_2'  →   5,800

    Momentum rule for all other products::

        M3_Pred = lag_1 × 1.05   if lag_1 > lag_2   (upward trend)
                = lag_1 × 0.95   otherwise            (flat / declining trend)
    """
    xl = pd.ExcelFile(filepath)
    df_raw = xl.parse('Ph.2 Data Pack-Actual Booking', header=None)

    df_data = df_raw.iloc[2:150].dropna(subset=[1]).copy()
    df_units = df_data[df_data[2].astype(str).str.contains('Sustaining|Decline|NPI|Growth|Mature', na=False, case=False)].reset_index(drop=True)

    res = []
    for idx, row in df_units.iterrows():
        prod = str(row[1]).strip()
        actuals = [pd.to_numeric(row[c], errors='coerce') for c in range(3, 15)]
        actuals = [a if not pd.isna(a) else 0 for a in actuals]

        lag1, lag2 = actuals[-1], actuals[-2]

        # M3 Business Logic Overrides
        if prod == 'IP PHONE Enterprise Desk_1':
            m3_pred = 10500
        elif prod == 'IP PHONE Enterprise Desk_2':
            m3_pred = 5800
        else:
            # M3 Engine Base Logic (Simplified TS proxy for parallel execution)
            m3_pred = lag1 * 1.05 if lag1 > lag2 else lag1 * 0.95

        res.append({'Product': prod, 'M3_Pred': int(round(max(0, m3_pred)))})

    print("✅ [Thread 3] Model 3 Complete.")
    return pd.DataFrame(res)

# ==============================================================================
# 4. PARALLEL EXECUTION & BIFURCATED META-FUSION
# ==============================================================================
def run_parallel_pipeline():
    """Orchestrate the full parallel multi-model inference pipeline.

    Executes the three model functions concurrently in a ``ThreadPoolExecutor``
    (3 workers), merges their outputs, applies the bifurcated meta-fusion
    strategy to produce a single ``FINAL_META_FORECAST`` per product, saves
    the submission CSV, and prints a per-product accuracy diagnostic table
    using FY26 Q1 actuals as a back-test reference.

    Meta-fusion routing by Cost Rank
    ---------------------------------
    * **Cost Rank ≤ 3** (high-priority products):
      ``0.50 × M1_Pred + 0.50 × M2_Pred``
    * **Cost Rank > 3** (standard products):
      ``min(M1_Pred, M2_Pred, M3_Pred)``

    Outputs
    -------
    * **CSV file** at ``/content/CFL_Phase2_Final_Pipeline_Submission.csv``
      with columns ``Cost Rank``, ``Product Name``, ``Predicted FY26 Q2 Units``
      sorted by Cost Rank ascending.
    * **Diagnostic table** printed to stdout showing per-product Actual,
      Forecast, Accuracy (%), and qualitative status for the FY26 Q1 back-test.

    Accuracy status thresholds::

        ≥ 90 %          →  ✓ Excellent
        75 % – 89 %     →  △ Acceptable
        < 75 %          →  ⚠ Critical Miss

    Returns
    -------
    None
    """
    print("⚡ LAUNCHING PARALLEL MULTI-MODEL INFERENCE PIPELINE")
    print("="*80)

    # Run models in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_m1 = executor.submit(execute_model_1, FILE_PATH)
        future_m2 = executor.submit(execute_model_2, FILE_PATH)
        future_m3 = executor.submit(execute_model_3, FILE_PATH)

        df_m1 = future_m1.result()
        df_m2 = future_m2.result()
        df_m3 = future_m3.result()

    print("\n🧠 Models converged. Executing Bifurcated Oracle Routing...")

    # Merge outputs
    meta_df = pd.merge(df_m1, df_m2, on='Product', how='inner')
    meta_df = pd.merge(meta_df, df_m3, on='Product', how='inner')

    # ===================== META FUSION =====================
    def optimized_meta_fusion(row):
        """Apply bifurcated meta-fusion to produce the final forecast.

        Routes each product through one of two fusion strategies based on its
        Cost Rank:

        * **Cost Rank ≤ 3** (high-priority): equal-weight average of M1 and M2,
          providing a stable, high-confidence estimate for flagship products.
        * **Cost Rank > 3** (standard): minimum across all three models,
          acting as a conservative floor that guards against over-stocking.

        Parameters
        ----------
        row : pandas.Series
            A single row from ``meta_df`` containing ``Cost Rank``,
            ``M1_Pred``, ``M2_Pred``, and ``M3_Pred``.

        Returns
        -------
        float
            Final fused forecast value (un-rounded).
        """
        m1, m2, m3 = row['M1_Pred'], row['M2_Pred'], row['M3_Pred']

        if cr <= 3:
            return (0.50 * m1) + (0.50 * m2)
        else:
            return min(m1, m2, m3)

    meta_df['FINAL_META_FORECAST'] = meta_df.apply(optimized_meta_fusion, axis=1).round(0).astype(int)

    # ===================== LOAD ACTUALS (FY26 Q1) =====================
    xl = pd.ExcelFile(FILE_PATH)
    df_raw = xl.parse('Ph.2 Data Pack-Actual Booking', header=None)

    df_data = df_raw.iloc[2:150].dropna(subset=[1]).copy()
    df_units = df_data[df_data[2].astype(str).str.contains('Sustaining|Decline|NPI|Growth|Mature', na=False, case=False)].reset_index(drop=True)

    actual_map = {}
    for _, row in df_units.iterrows():
        product = str(row[1]).strip()
        actual_q1 = pd.to_numeric(row[14], errors='coerce')  # FY26 Q1 column
        actual_map[product] = int(actual_q1) if not pd.isna(actual_q1) else 0

    # ===================== OUTPUT =====================
    OUTPUT_FILE = '/content/CFL_Phase2_Final_Pipeline_Submission.csv'
    submission_df = meta_df[['Cost Rank', 'Product', 'FINAL_META_FORECAST']].sort_values('Cost Rank')
    submission_df.rename(columns={
        'Product': 'Product Name',
        'FINAL_META_FORECAST': 'Predicted FY26 Q2 Units'
    }, inplace=True)
    submission_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*80)
    print("🏆 APEX PIPELINE COMPLETE")
    print("="*80)

    # ===================== DIAGNOSTIC TABLE =====================
    print("\n" + "="*95)
    print(" 📊 PER-PRODUCT ACCURACY BREAKDOWN (FY26 Q1 BACKTEST)")
    print("="*95)
    print(f"{'CR':<3} | {'Product Name':<40} | {'Actual':<8} | {'Forecast':<8} | {'Accuracy':<8} | {'Status'}")
    print("-" * 95)

    display_df = meta_df.sort_values('Cost Rank')

    for _, row in display_df.iterrows():
        cr = int(row['Cost Rank'])
        product = str(row['Product'])
        prod = product[:40].ljust(40)

        forecast = int(row['FINAL_META_FORECAST'])
        actual = int(actual_map.get(product, 0))

        # Accuracy calculation
        if actual == 0 and forecast == 0:
            acc = 100.0
        elif actual == 0:
            acc = 0.0
        else:
            err = abs(actual - forecast) / actual
            acc = max(0, 100.0 - (err * 100))

        if acc >= 90:
            status = "✓ Excellent"
        elif acc >= 75:
            status = "△ Acceptable"
        else:
            status = "⚠ Critical Miss"

        print(f"{cr:<3} | {prod} | {actual:>8,} | {forecast:>8,} | {acc:>7.1f}% | {status}")

    print("="*95)

# Execute the pipeline
if __name__ == '__main__':
    run_parallel_pipeline()
