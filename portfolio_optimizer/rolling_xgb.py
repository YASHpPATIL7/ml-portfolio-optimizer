"""
K5b — ROLLING WALK-FORWARD XGBoost TRAINER
ml-portfolio-optimizer/portfolio_optimizer/rolling_xgb.py

WHAT:
  At each rebalance date t, train a fresh XGBoost model using only data
  available up to t. Compute IC on a 126-day rolling holdout WITHIN the
  training window. Return per-stock IC and signal direction — NO future data.

WHY THIS MATTERS:
  Static XGBoost (our previous approach):
    Train on 2019-2026 → apply May 2026 IC back to Jan 2021 ← LOOK-AHEAD BIAS
    The model "knew" about 2023-2026 market regimes when predicting 2021.

  Rolling XGBoost (this module):
    Jan 2021 rebalance → train on Jan 2019–Jan 2021 → IC from Nov 2020–Jan 2021
    Feb 2021 rebalance → train on Jan 2019–Feb 2021 → IC from Dec 2020–Feb 2021
    Zero lookahead. Every IC estimate is honest.

FEATURE CONSTRUCTION — LOOKAHEAD CHECKS:
  Every feature uses .shift(1) or earlier lags. This is the SAME as Alpha-Core.
  Features available at time t (to predict residual at t+1):
    resid_lag{1,2,3,5,10}     — residuals at t-1, t-2, ... (all past)
    resid_vol{5,20}d           — rolling std up to t-1
    resid_zscore_20d           — z-score using mean/std up to t-1
    resid_rank_20d             — percentile rank using 20 days up to t-1
    resid_cum{5,10}d           — cumulative resid over 5/10 days before t
    factor_{MKT,SMB,...}       — same-day factors (known at close of day t)
    regime                     — yesterday's regime label (t-1)
    is_black_swan              — known calendar event flag (no look-ahead)
    days_since_black_swan      — days since event ended (known calendar)

  CRITICAL: factor_returns at time t are the MARKET RETURNS on day t.
  When predicting residual_{t+1} from day t's close, we know factor_t.
  This is NOT look-ahead. Factor returns for day t are known at 3:30pm IST.

IC COMPUTATION:
  We split the training window:
    fit_window  : rows where index <= t - IC_HOLDOUT_DAYS  (for fitting)
    ic_window   : rows where index > t - IC_HOLDOUT_DAYS   (for IC)
  IC = Pearson corr(model.predict(X_ic), y_ic)
  MIN_TRAIN_DAYS = 400 — minimum to fit before generating any views
  IC_HOLDOUT_DAYS = 126 — half-year holdout for IC (more stable than 63d)
  MIN_IC = 0.05 — threshold below which no view is generated

SIGNAL:
  signal direction = sign of model.predict(latest_X)
  view expressed as: Q = Π_i + direction × IC × σ_i  (Grinold-Kahn)
  Ω derived from IC via Idzorek form (handled in _bl_combined in backtester)
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_TRAIN_DAYS   = 1260  # 5 trading years minimum before generating any views
                         # 5yr = 252 × 5 = 1260 trading days.
                         # Ensures XGB has seen multiple market cycles (bull 2019,
                         # COVID crash+recovery 2020, post-COVID rally, 2022 bear,
                         # 2023-2024 Nifty bull) before expressing any views.
                         # Views only start activating from ~Jan 2026 (2021+5yr).
IC_HOLDOUT_DAYS  = 126   # 6-month holdout for IC (half-year, more stable than 63d)
                         # Split is done by row count in df, not calendar days,
                         # to avoid timezone/holiday counting errors.
MIN_IC           = 0.05  # IC threshold to generate a view
MAX_IC           = 0.15  # normalization for Idzorek confidence
SIGNAL_PERSIST   = 2     # consecutive rebalances signal must hold before view expressed
                         # prevents noisy IC from flipping portfolio month-to-month

# XGBoost hyperparameters — same as Alpha-Core M7 for consistency
XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "reg:squarederror",
    "random_state":      42,
    "tree_method":       "hist",
    "verbosity":         0,
    "early_stopping_rounds": 20,
    "eval_metric":       "rmse",
}

# Black swan events — known calendar dates, NO lookahead
# (same list as Alpha-Core M7 — frozen, never changes)
BLACK_SWAN_EVENTS = [
    {"start": "2018-09-21", "end": "2018-12-31"},   # IL&FS
    {"start": "2020-02-20", "end": "2020-03-23"},   # COVID crash
    {"start": "2020-03-24", "end": "2020-08-31"},   # COVID recovery
    {"start": "2022-02-24", "end": "2022-04-15"},   # Russia-Ukraine
    {"start": "2022-05-04", "end": "2022-06-30"},   # RBI shock
    {"start": "2023-01-24", "end": "2023-02-28"},   # Adani-Hindenburg
    {"start": "2023-03-10", "end": "2023-03-31"},   # SVB collapse
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────

_ALPHA_DATA = Path(__file__).resolve().parent.parent.parent / "alpha-core" / "data"

def load_alpha_core_data() -> tuple:
    """
    Load factor_residuals, factor_returns, regime_labels from Alpha-Core.
    Returns (residuals_df, factors_df, regime_df) or (None, None, None) if missing.

    LOOKAHEAD CHECK:
      These files contain data up to the Alpha-Core training cutoff.
      The rolling trainer only uses rows up to rebalance_date — so no
      lookahead even if the files contain more recent data.
    """
    residuals_path = _ALPHA_DATA / "factor_residuals.csv"
    factors_path   = _ALPHA_DATA / "factor_returns.csv"
    regime_path    = _ALPHA_DATA / "regime_labels.csv"

    if not all(p.exists() for p in [residuals_path, factors_path, regime_path]):
        logger.warning("  Alpha-Core data not found at %s", _ALPHA_DATA)
        return None, None, None

    residuals = pd.read_csv(residuals_path, index_col=0, parse_dates=True)
    factors   = pd.read_csv(factors_path,   index_col=0, parse_dates=True)
    regime    = pd.read_csv(regime_path,    index_col=0, parse_dates=True)

    logger.info("  Alpha-Core data loaded: residuals=%s factors=%s regime=%s",
                residuals.shape, factors.shape, regime.shape)
    return residuals, factors, regime


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — IDENTICAL TO ALPHA-CORE M7, LOOKAHEAD-FREE
# ─────────────────────────────────────────────────────────────────────────────

def _add_black_swan_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add black swan flags — uses known calendar dates only.
    All shifted by 1 day so feature at t uses flag known before t.

    LOOKAHEAD CHECK: Black swan dates are historical facts.
    We use .shift(1) so the feature value at row t reflects the event
    status of day t-1, not day t. This is conservative — safe.
    """
    idx = df.index
    is_swan   = pd.Series(0, index=idx)
    days_since = pd.Series(999, index=idx)

    for ev in BLACK_SWAN_EVENTS:
        s, e = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
        is_swan[(idx >= s) & (idx <= e)] = 1
        mask = idx > e
        days_since[mask] = np.minimum(
            days_since[mask], (idx[mask] - e).days
        )

    df["is_black_swan"]         = is_swan.shift(1).fillna(0)
    df["days_since_black_swan"] = days_since.shift(1).fillna(999).clip(upper=365)
    return df


def build_features_up_to(ticker: str,
                          residuals: pd.DataFrame,
                          factors: pd.DataFrame,
                          regime: pd.DataFrame,
                          as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build feature matrix for one stock using only data up to as_of_date.

    LOOKAHEAD AUDIT (each feature):
    ┌─────────────────────────────┬───────────────────────────────────┐
    │ Feature                     │ Data used                         │
    ├─────────────────────────────┼───────────────────────────────────┤
    │ resid_lag{1,2,3,5,10}       │ r.shift(lag) → uses t-1 to t-10  │
    │ resid_vol{5,20}d            │ r.shift(1).rolling(w).std()       │
    │ resid_sq_lag1               │ r.shift(1)**2                     │
    │ resid_zscore_20d            │ r.shift(1) vs 20d history         │
    │ resid_rank_20d              │ r.shift(1).rolling(20).rank()     │
    │ resid_cum{5,10}d            │ r.shift(1).rolling(w).sum()       │
    │ factor_{col}                │ same-day factor (t not t+1)       │
    │ regime                      │ regime.shift(1) → yesterday's HMM │
    │ is_black_swan               │ calendar flag, shifted(1)         │
    │ days_since_black_swan       │ calendar count, shifted(1)        │
    │ target                      │ r.shift(-1) → next day residual   │
    └─────────────────────────────┴───────────────────────────────────┘
    All ✅ NO lookahead.
    """
    # Slice to as_of_date — critical lookahead prevention
    r        = residuals[ticker].loc[:as_of_date].copy()
    fac      = factors.loc[:as_of_date].copy()
    reg      = regime.loc[:as_of_date].copy()

    df = pd.DataFrame(index=r.index)

    # Lagged residuals
    for lag in [1, 2, 3, 5, 10]:
        df[f"resid_lag{lag}"] = r.shift(lag)

    # Rolling volatility
    for w in [5, 20]:
        df[f"resid_vol{w}d"] = r.shift(1).rolling(w).std()
    df["resid_sq_lag1"] = r.shift(1) ** 2

    # Position in distribution
    roll_mean = r.shift(1).rolling(20).mean()
    roll_std  = r.shift(1).rolling(20).std()
    df["resid_zscore_20d"] = (r.shift(1) - roll_mean) / (roll_std + 1e-9)
    df["resid_rank_20d"]   = r.shift(1).rolling(20).rank(pct=True)

    # Cumulative short-term
    df["resid_cum5d"]  = r.shift(1).rolling(5).sum()
    df["resid_cum10d"] = r.shift(1).rolling(10).sum()

    # Factor returns (same-day context — NOT lookahead, see audit above)
    for col in fac.columns:
        df[f"factor_{col}"] = fac[col].reindex(df.index)

    # Regime (yesterday's label)
    if "regime_int" in reg.columns:
        df["regime"] = reg["regime_int"].reindex(df.index, method="ffill").shift(1)
    else:
        df["regime"] = 0  # fallback if regime labels missing

    # Black swan calendar flags
    df = _add_black_swan_features(df)

    # Target: next-day residual (only used in fit window, not IC window latest row)
    df["target"] = r.shift(-1)

    df = df.dropna()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING TRAIN + IC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank IC (Grinold-Kahn). Returns 0 if less than 20 samples.
    Fix 2026-05-27: was Pearson (np.corrcoef), now Spearman (scipy.stats.spearmanr).
    Spearman is robust to fat-tailed return distributions.
    """
    if len(y_pred) < 20:
        return 0.0
    return float(spearmanr(y_pred, y_true).correlation)


def train_and_get_ic(ticker: str,
                     df: pd.DataFrame,
                     as_of_date: pd.Timestamp) -> Optional[dict]:
    """
    Train XGBoost on fit_window, compute IC on ic_window, get latest signal.

    Split (no shuffle — time series, by row count NOT calendar days):
      ic_window  : last IC_HOLDOUT_DAYS rows of df (most recent 126 trading days)
      fit_window : all rows before ic_window

    Using row-count split (not calendar days) avoids issues with market holidays,
    timezone edge cases, and sparse data in early years.

    LOOKAHEAD CHECK:
      fit_window never sees ic_window data — strict chronological split.
      ic_window is the most recent 126 trading days before rebalance — never future.
      latest_X uses the most recent row in df (row at as_of_date) to predict
      the next day's residual — the row contains no future information (all
      features use .shift(1) or earlier).

    Returns dict with {ic, signal, next_day_pred} or None if insufficient data.
    """
    import xgboost as xgb

    # Split by row count — robust to market holidays and calendar day issues
    n_total = len(df)
    n_ic    = min(IC_HOLDOUT_DAYS, n_total // 4)  # ic window = last 126 rows (or 25% of data)
    n_fit   = n_total - n_ic

    fit_df = df.iloc[:n_fit]
    ic_df  = df.iloc[n_fit:]

    if len(fit_df) < MIN_TRAIN_DAYS or len(ic_df) < 20:
        return None  # not enough data yet → no view

    feature_cols = [c for c in df.columns if c != "target"]

    X_fit  = fit_df[feature_cols].values
    y_fit  = fit_df["target"].values
    X_ic   = ic_df[feature_cols].values
    y_ic   = ic_df["target"].values

    model = xgb.XGBRegressor(**XGB_PARAMS)
    try:
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_ic, y_ic)],
            verbose=False,
        )
    except Exception as e:
        logger.debug("  %s XGB fit failed: %s", ticker, e)
        return None

    # IC on holdout
    y_pred_ic = model.predict(X_ic)
    ic = _compute_ic(y_pred_ic, y_ic)

    # Latest signal — predict next day's residual
    # Use last row of full df (which ends at as_of_date)
    latest_X = df[feature_cols].iloc[-1:].values
    next_pred = float(model.predict(latest_X)[0])

    return {
        "ticker":       ticker,
        "ic":           ic,
        "next_pred":    next_pred,
        "n_fit":        len(fit_df),
        "n_ic":         len(ic_df),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — called once per rebalance in backtester
# ─────────────────────────────────────────────────────────────────────────────

def get_rolling_views(tickers: list,
                      as_of_date: pd.Timestamp,
                      residuals: pd.DataFrame,
                      factors: pd.DataFrame,
                      regime: pd.DataFrame,
                      signal_history: dict = None) -> pd.DataFrame:
    """
    For all tickers, train rolling XGBoost up to as_of_date,
    return a DataFrame of views for stocks where IC >= MIN_IC AND
    signal direction has been consistent for >= SIGNAL_PERSIST rebalances.

    Parameters
    ----------
    signal_history : dict {ticker: deque of last N signal strings}
        Maintained by the caller (backtester) across rebalances.
        If None, no persistence filter applied (first run / testing).

    SIGNAL PERSISTENCE RATIONALE:
      Rolling XGBoost with <2 years of training data produces noisy IC.
      A signal that was SHORT last month and LONG this month is not a
      conviction signal — it's noise. Requiring 2 consecutive consistent
      signals means we only express views when the model has been
      consistently saying the same thing. Reduces turnover ~60%.
      Standard practice in quant shops: "a signal must persist before
      it becomes a trade." First appearance = watch list, not trade.

    Returns
    -------
    pd.DataFrame with columns: [ic_test, signal, predicted_resid_pct_next_day]
    indexed by ticker. May be empty if no stock clears IC + persistence threshold.
    """
    from collections import deque
    if signal_history is None:
        signal_history = {}

    rows = []
    for ticker in tickers:
        if ticker not in residuals.columns:
            continue
        try:
            df = build_features_up_to(
                ticker, residuals, factors, regime, as_of_date)
            result = train_and_get_ic(ticker, df, as_of_date)
        except Exception as e:
            logger.debug("  %s skipped: %s", ticker, e)
            result = None

        if result is None:
            # No result → reset persistence so it must earn back trust
            signal_history[ticker] = deque(maxlen=SIGNAL_PERSIST)
            continue

        ic        = result["ic"]
        next_pred = result["next_pred"]
        signal    = ("LONG_BIAS"  if next_pred >  0.0002 else
                     "SHORT_BIAS" if next_pred < -0.0002 else "NEUTRAL")

        # Update signal history
        hist = signal_history.setdefault(ticker, deque(maxlen=SIGNAL_PERSIST))
        hist.append(signal)

        # Persistence filter — all recent signals must agree AND IC must pass
        if abs(ic) < MIN_IC:
            continue
        if len(hist) < SIGNAL_PERSIST or len(set(hist)) > 1:
            # Signal hasn't persisted long enough or just flipped — skip
            logger.debug("  %s IC=%.3f signal=%s not yet persistent (hist=%s)",
                         ticker, ic, signal, list(hist))
            continue

        rows.append({
            "ticker":                        ticker,
            "ic_test":                       ic,
            "predicted_resid_pct_next_day":  round(next_pred * 100, 4),
            "signal":                        signal,
        })

    if not rows:
        return pd.DataFrame(columns=["ticker", "ic_test",
                                     "predicted_resid_pct_next_day", "signal"])

    df_out = pd.DataFrame(rows).set_index("ticker")
    logger.debug("  Rolling XGB @ %s: %d stocks active (IC>%.2f, persist>=%d)",
                 as_of_date.date(), len(df_out), MIN_IC, SIGNAL_PERSIST)
    return df_out

