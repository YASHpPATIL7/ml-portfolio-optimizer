"""
K7 — WALK-FORWARD BACKTESTER
ml-portfolio-optimizer/portfolio_optimizer/backtester.py

WHAT:
  Simulates live portfolio management from 2021 onwards.
  Every 21 trading days (monthly), recompute weights using trailing 252-day window.
  Apply those weights to the NEXT 21 days of actual returns (out-of-sample).
  Record P&L, compare 5 strategies.

STRATEGIES:
  1. Equal Weight          — 1/N baseline (benchmark)
  2. MVO Max Sharpe        — K2 with Ledoit-Wolf Σ
  3. MVO Min Vol           — K2 minimum variance
  4. HRP                   — K3 hierarchical risk parity
  5. BL-Combined           — K4 (ML views) + K6 (GatiShakti macro views)

WHY WALK-FORWARD (not full-sample backtest):
  Full-sample: fit on 2019-2026, evaluate on 2019-2026 → in-sample, useless.
  Walk-forward: fit on 2019-2021, evaluate 2021. Fit 2019-2022, evaluate 2022.
  Each month's return is truly out-of-sample — the model never saw future data.
  This is the only honest way to compare portfolio methods.

ML VIEW GENERATION — ZERO LOOKAHEAD:
  At each rebalance date t, if Alpha-Core data is available:
    - build_features_up_to(ticker, ..., as_of_date=t)
    - fit XGBoost on rows where row_count <= len(df) - IC_HOLDOUT_DAYS
    - compute IC on last IC_HOLDOUT_DAYS rows
    - generate view only if IC > 0.05 AND signal persisted >= 2 rebalances
  MIN_TRAIN_DAYS = 1260 (5yr). Views require 5yr of fit data before activating.
  Rebalances 2021-2025: BL runs on GatiShakti macro views only (stable, consistent).
  From ~Jan 2026: XGB has 5yr history, IC stabilises, ML views activate.
  This is intentional — better to have no ML view than a noisy early one.

METRICS:
  Annualized Return  — geometric mean of monthly returns × 12
  Annualized Sharpe  — (ret - rf) / vol × √12
  Max Drawdown       — worst peak-to-trough in cumulative wealth
  Calmar Ratio       — annualized return / |max drawdown|
  Monthly Turnover   — average Σ|w_new - w_old| per rebalance
  HHI               — Herfindahl-Hirschman Index (concentration)
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_FIGURES = Path(__file__).resolve().parent.parent / "figures"
_DATA    = Path(__file__).resolve().parent.parent / "data"
_FIGURES.mkdir(exist_ok=True)

TRAIN_DAYS    = 252   # 1-year trailing window
REBAL_DAYS    = 21    # monthly rebalancing
RISK_FREE_ANN = 0.065 # consistent with K2/K4
START_DATE    = "2021-01-01"
RF_DAILY      = RISK_FREE_ANN / 252


# ─────────────────────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    name:           str
    returns:        pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    weights_history: list = field(default_factory=list)
    turnover_history: list = field(default_factory=list)

    @property
    def cum_wealth(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    @property
    def ann_return(self) -> float:
        n = len(self.returns)
        if n < 2: return 0.0
        total = float(self.cum_wealth.iloc[-1])
        years = n / 252
        return total ** (1 / years) - 1

    @property
    def ann_vol(self) -> float:
        return float(self.returns.std() * np.sqrt(252))

    @property
    def sharpe(self) -> float:
        v = self.ann_vol
        return (self.ann_return - RISK_FREE_ANN) / v if v > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        wealth = self.cum_wealth
        peak   = wealth.cummax()
        dd     = (wealth - peak) / peak
        return float(dd.min())

    @property
    def calmar(self) -> float:
        mdd = abs(self.max_drawdown)
        return self.ann_return / mdd if mdd > 0 else 0.0

    @property
    def avg_turnover(self) -> float:
        return float(np.mean(self.turnover_history)) if self.turnover_history else 0.0

    @property
    def avg_hhi(self) -> float:
        if not self.weights_history: return 0.0
        return float(np.mean([(w**2).sum() for w in self.weights_history]))

    def summary(self) -> dict:
        return {
            "Strategy":   self.name,
            "Return%":    round(self.ann_return * 100, 2),
            "Vol%":       round(self.ann_vol * 100, 2),
            "Sharpe":     round(self.sharpe, 3),
            "MaxDD%":     round(self.max_drawdown * 100, 2),
            "Calmar":     round(self.calmar, 3),
            "Turnover%":  round(self.avg_turnover * 100, 2),
            "HHI":        round(self.avg_hhi, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT SOLVERS (self-contained, no side effects)
# ─────────────────────────────────────────────────────────────────────────────

def _equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def _mvo_max_sharpe(mu: np.ndarray, sigma: np.ndarray,
                    w_max: float = 0.20) -> np.ndarray:
    from scipy.optimize import minimize
    n  = len(mu)
    w0 = np.ones(n) / n
    def neg_sharpe(w):
        r = float(w @ mu)
        v = float(np.sqrt(max(w @ sigma @ w, 1e-12)))
        return -(r - RF_DAILY * 252) / v
    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=[(0, w_max)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-10, "maxiter": 500})
    return res.x if res.success else w0



def _mvo_min_vol(sigma: np.ndarray, w_max: float = 0.20) -> np.ndarray:
    from scipy.optimize import minimize
    n  = sigma.shape[0]
    w0 = np.ones(n) / n
    def port_vol(w): return float(np.sqrt(max(w @ sigma @ w, 1e-12)))
    res = minimize(port_vol, w0, method="SLSQP",
                   bounds=[(0, w_max)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-10, "maxiter": 500})
    return res.x if res.success else w0


def _hrp(returns_window: pd.DataFrame) -> np.ndarray:
    """López de Prado HRP — pure numpy, no external deps."""
    corr = returns_window.corr().values
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    n = dist.shape[0]

    # Ward linkage via scipy
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, leaves_list
    condensed = squareform(dist)
    link = linkage(condensed, method="ward")
    order = leaves_list(link)

    # Covariance for allocation
    cov = returns_window.cov().values

    # Recursive bisection
    weights = np.ones(n)
    items   = list(order)

    def _bisect(items):
        if len(items) < 2:
            return
        mid = len(items) // 2
        left, right = items[:mid], items[mid:]
        def _var(idx):
            w = weights[idx]
            c = cov[np.ix_(idx, idx)]
            wn = w / w.sum()
            return float(wn @ c @ wn)
        vl, vr = _var(left), _var(right)
        alloc_l = 1 - vl / (vl + vr)
        weights[left]  *= alloc_l
        weights[right] *= 1 - alloc_l
        _bisect(left)
        _bisect(right)

    _bisect(items)
    return weights / weights.sum()


def _bl_combined(returns_window: pd.DataFrame, tickers: list,
                 w_market: np.ndarray,
                 rebal_date: pd.Timestamp,
                 xgb_valid_from: pd.Timestamp,
                 xgb_preds: pd.DataFrame = None,
                 rolling_xgb_data: tuple = None,
                 signal_history: dict = None,
                 w_prev: np.ndarray = None,
                 blend_alpha: float = 0.35,
                 tau: float = 0.05,
                 w_max: float = 0.20) -> np.ndarray:
    """
    BL with time-gated views + portfolio blending for turnover control.

    TURNOVER CONTROL (portfolio blending):
      After computing the unconstrained BL-optimal weights, blend with
      the previous allocation:

        w_final = blend_alpha × w_BL_optimal + (1 - blend_alpha) × w_prev

      Why blending not L1 penalty:
        L1 in the MVO objective is non-smooth at w = w_prev. SLSQP (gradient
        method) cannot compute a descent direction and freezes at w_prev.
        Portfolio blending is post-optimization, smooth, always moves toward
        the optimal at a controlled pace.

      blend_alpha = 0.35 means:
        35% of the way toward the unconstrained BL optimum per rebalance.
        Turnover reduction: if unconstrained turnover = 40%, blended = ~14%.
        Equivalent interpretation: it takes ~3 months to fully implement a
        new signal, by which time the next signal has updated.

      Production analogy: many quant PM desks use this exact mechanic —
        "phase in 30-40% of the model allocation per month."
        Prevents large single-day order flow that moves the market against you.

    tau = 0.05 (textbook BL standard):
      Controls confidence in views vs the equilibrium prior.
      A = (τΣ)⁻¹ + P'Ω⁻¹P  →  at τ=0.05 the prior term is 20× larger than
      at τ=1.0, so mu_bl stays close to the equilibrium prior (pi) with small
      tilts toward view stocks.
      Rule of thumb: τ = 1/T where T is the number of quarterly return
      observations used to estimate the prior (T≈20 → τ≈0.05).
      At τ=0.50 (previous): views dominated completely → MVO slammed 3 stocks
      to 20% cap every month → 17% vol, 0.009 Sharpe. Broken.
      At τ=0.05: views produce moderate tilts of ±2–4% from equal weight.
      This is the correct BL regime.

    ML view sources (priority order):
      1. Rolling XGBoost (rolling_xgb_data provided) — zero lookahead
      2. Static XGBoost (xgb_preds provided, rebal_date >= xgb_valid_from)
      3. GatiShakti macro views only (BL on macro prior)
    """
    from sklearn.covariance import LedoitWolf
    from portfolio_optimizer.gatishakti_views import refresh_gatishakti_views

    lw = LedoitWolf().fit(returns_window.values)
    sigma_lw = lw.covariance_ * 252      # annualised
    delta    = 2.5
    pi       = delta * sigma_lw @ w_market
    sigma_ann = np.sqrt(np.diag(sigma_lw))
    MAX_IC    = 0.15

    # ── ML Views ──────────────────────────────────────────────────────────────
    P_ml, Q_ml, O_ml = [], [], []

    if rolling_xgb_data is not None:
        # ROLLING MODE — retrain at each rebalance (zero lookahead)
        from portfolio_optimizer.rolling_xgb import get_rolling_views
        residuals, factors, regime = rolling_xgb_data
        live_preds = get_rolling_views(
            tickers, rebal_date, residuals, factors, regime,
            signal_history=signal_history,
        )
        # live_preds: DataFrame indexed by ticker with [ic_test, signal, ...]
        for i, t in enumerate(tickers):
            if t not in live_preds.index: continue
            row = live_preds.loc[t]
            ic  = float(row["ic_test"])
            sig = row["signal"]
            if sig == "NEUTRAL" or abs(ic) < 0.05: continue
            alpha_i   = abs(ic) * sigma_ann[i]
            direction = 1.0 if sig == "LONG_BIAS" else -1.0
            q_i       = pi[i] + direction * alpha_i
            p_k       = min(abs(ic) / MAX_IC, 0.99)
            omega_k   = ((1 - p_k) / p_k) * tau * (sigma_ann[i] ** 2)
            p_row     = np.zeros(len(tickers)); p_row[i] = 1.0
            P_ml.append(p_row); Q_ml.append(q_i); O_ml.append(omega_k)

    elif xgb_preds is not None and rebal_date >= xgb_valid_from:
        # STATIC MODE — May 2026 model, time-gated
        for i, t in enumerate(tickers):
            if t not in xgb_preds.index: continue
            row = xgb_preds.loc[t]
            ic  = float(row["ic_test"])
            sig = row["signal"]
            if sig == "NEUTRAL" or abs(ic) < 0.05: continue
            alpha_i   = abs(ic) * sigma_ann[i]
            direction = 1.0 if sig == "LONG_BIAS" else -1.0
            q_i       = pi[i] + direction * alpha_i
            p_k       = min(abs(ic) / MAX_IC, 0.99)
            omega_k   = ((1 - p_k) / p_k) * tau * (sigma_ann[i] ** 2)
            p_row     = np.zeros(len(tickers)); p_row[i] = 1.0
            P_ml.append(p_row); Q_ml.append(q_i); O_ml.append(omega_k)

    # ── Macro Views (GatiShakti) ───────────────────────────────────────────────
    macro_views = refresh_gatishakti_views(tickers, as_of_date=rebal_date)
    P_gs, Q_gs, O_gs = [], [], []
    for mv in macro_views:
        if not mv.is_active: continue
        idx = [tickers.index(t) for t in mv.tickers if t in tickers]
        if not idx: continue
        p_row    = np.zeros(len(tickers)); p_row[idx] = 1.0 / len(idx)
        avg_pi   = np.mean(pi[idx])
        q_k      = avg_pi + mv.view_decimal
        p_sigma_p = float(p_row @ sigma_lw @ p_row)
        c        = mv.confidence
        omega_k  = ((1 - c) / max(c, 1e-6)) * tau * p_sigma_p
        P_gs.append(p_row); Q_gs.append(q_k); O_gs.append(omega_k)

    # ── Combine and solve BL posterior ────────────────────────────────────────
    P_rows = P_ml + P_gs
    Q_vals = Q_ml + Q_gs
    O_diag = O_ml + O_gs

    if not P_rows:
        # No views → BL posterior = prior → return market portfolio (no tilt).
        # Using MVO here injects spurious concentration from estimation error.
        w_opt = w_market.copy()
    else:
        P = np.array(P_rows); Q = np.array(Q_vals); Omega = np.diag(O_diag)
        tau_sigma_inv = np.linalg.inv(tau * sigma_lw)
        Omega_inv     = np.linalg.inv(Omega)
        A     = tau_sigma_inv + P.T @ Omega_inv @ P
        b     = tau_sigma_inv @ pi + P.T @ Omega_inv @ Q
        mu_bl = np.linalg.solve(A, b)

        # ── Canonical BL analytical portfolio (NOT MVO) ──────────────────────
        # Formula: w_opt = w_market + (δΣ)⁻¹ (μ_bl − π)
        # This is the closed-form BL tilt — weights move from market weights
        # proportional to view strength. Stable and diversified.
        # MVO with mu_bl amplified tiny return differences → 3 stocks at 20%
        # cap → 17% vol, 0.009 Sharpe. Canonical BL avoids this entirely.
        tilt  = np.linalg.solve(delta * sigma_lw, mu_bl - pi)
        w_raw = w_market + tilt
        w_raw = np.clip(w_raw, 0.0, w_max)
        s = w_raw.sum()
        w_opt = w_raw / s if s > 1e-9 else w_market.copy()

    # Portfolio blending — phase in blend_alpha of unconstrained optimum per rebalance
    if w_prev is not None:
        w_final = blend_alpha * w_opt + (1.0 - blend_alpha) * w_prev
        w_final = np.clip(w_final, 0, None)
        w_final /= w_final.sum()
        return w_final
    return w_opt


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(returns_df: pd.DataFrame,
                 w_market: np.ndarray,
                 tickers: list,
                 xgb_preds: pd.DataFrame = None,
                 xgb_valid_from: pd.Timestamp = None,
                 use_rolling_xgb: bool = True) -> dict:
    """
    Walk-forward backtest. Views are time-gated.

    Parameters
    ----------
    use_rolling_xgb : bool (default True)
        If True and Alpha-Core data is available, retrain XGBoost at every
        rebalance using only data up to that date — zero lookahead.
        If False, fall back to static xgb_preds with valid_from gate.

    View priority:
      1. Rolling XGBoost (use_rolling_xgb=True + Alpha-Core data available)
      2. Static XGBoost  (xgb_preds + valid_from gate)
      3. GatiShakti macro views only
    """
    from sklearn.covariance import LedoitWolf
    from portfolio_optimizer.rolling_xgb import load_alpha_core_data

    xgb_vf = xgb_valid_from or pd.Timestamp("2099-01-01")

    # Load Alpha-Core data for rolling XGBoost
    rolling_xgb_data = None
    if use_rolling_xgb:
        residuals, factors, regime = load_alpha_core_data()
        if residuals is not None:
            rolling_xgb_data = (residuals, factors, regime)
            logger.info("  Rolling XGBoost: ENABLED (retrains at each rebalance)")
            from portfolio_optimizer.rolling_xgb import MIN_TRAIN_DAYS as _MTD
            _first_view = pd.Timestamp(START_DATE) + pd.DateOffset(years=5)
            logger.info("  ML views need %d trading days (5yr) fit data → first activation ~%s",
                        _MTD, _first_view.date())
        else:
            logger.info("  Rolling XGBoost: DISABLED (Alpha-Core data not found)")
            logger.info("  Falling back to static XGBoost with valid_from gate")
    else:
        logger.info("  Rolling XGBoost: DISABLED by user flag")

    # Signal history for persistence filter — lives across all rebalances
    signal_history: dict = {}

    start = pd.Timestamp(START_DATE)
    bt_returns = returns_df[returns_df.index >= start - pd.Timedelta(days=TRAIN_DAYS + 30)]
    bt_dates   = returns_df.index[returns_df.index >= start]

    strategies = {
        "Equal Weight": BacktestResult("Equal Weight"),
        "MVO Max Sharpe": BacktestResult("MVO Max Sharpe"),
        "MVO Min Vol":    BacktestResult("MVO Min Vol"),
        "HRP":            BacktestResult("HRP"),
        "BL-Combined":    BacktestResult("BL-Combined"),
    }

    # Rebalance dates (every REBAL_DAYS)
    rebal_dates = bt_dates[::REBAL_DAYS]
    prev_weights = {k: np.ones(len(tickers)) / len(tickers) for k in strategies}

    logger.info("  Backtest: %s → %s  |  %d rebalances",
                bt_dates[0].date(), bt_dates[-1].date(), len(rebal_dates))

    daily_returns = {k: [] for k in strategies}
    daily_index   = []

    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]

        # Training window: TRAIN_DAYS before rebal_date
        train_end   = bt_returns.index[bt_returns.index <= rebal_date]
        if len(train_end) < TRAIN_DAYS:
            continue
        train_slice = bt_returns.loc[train_end[-TRAIN_DAYS:]]

        # Out-of-sample slice: rebal_date → next_rebal
        oos = returns_df.loc[(returns_df.index > rebal_date) &
                              (returns_df.index <= next_rebal)]
        if oos.empty:
            continue

        # Compute covariance on training window
        lw = LedoitWolf().fit(train_slice.values)
        sigma_lw = lw.covariance_ * 252    # annualised
        mu       = train_slice.mean().values * 252  # annualised

        n = len(tickers)
        w = {}
        try:
            w["Equal Weight"]  = _equal_weight(n)
            w["MVO Max Sharpe"] = _mvo_max_sharpe(mu, sigma_lw)
            w["MVO Min Vol"]    = _mvo_min_vol(sigma_lw)
            w["HRP"]            = _hrp(train_slice)
            w["BL-Combined"] = _bl_combined(
                    train_slice, tickers, w_market,
                    rebal_date=rebal_date,
                    xgb_valid_from=xgb_vf,
                    xgb_preds=xgb_preds,
                    rolling_xgb_data=rolling_xgb_data,
                    signal_history=signal_history,
                    w_prev=prev_weights["BL-Combined"],
                    blend_alpha=0.35,   # phase in 35%/month → ~3 months to converge
                    tau=0.05)           # textbook BL: gentle tilt, not view-dominated

        except Exception as e:
            logger.warning("  Rebal %s failed: %s — using equal weight", rebal_date.date(), e)
            for k in strategies:
                w[k] = prev_weights[k]

        # Compute turnover — Fix 2026-05-27: divide by 2 (industry standard).
        # sum(|w_new - w_old|) double-counts: every buy has a matching sell.
        # Correct formula: sum(|Δw|) / 2. Was reporting 2× the true cost.
        for k in strategies:
            turnover = float(np.abs(w[k] - prev_weights[k]).sum()) / 2.0
            strategies[k].turnover_history.append(turnover)
            strategies[k].weights_history.append(w[k].copy())
            prev_weights[k] = w[k].copy()

        # Apply weights to OOS returns
        for date, row in oos.iterrows():
            r = row.values   # daily returns for each stock
            for k in strategies:
                port_ret = float(w[k] @ r)
                daily_returns[k].append(port_ret)
            daily_index.append(date)

        if (i + 1) % 6 == 0:
            logger.info("  Rebal %3d/%d | date=%s | BL weights top: %s",
                        i+1, len(rebal_dates)-1, rebal_date.date(),
                        dict(sorted(zip(tickers, w["BL-Combined"]),
                                   key=lambda x: -x[1])[:3]))

    # Build return series
    idx = pd.DatetimeIndex(daily_index)
    for k in strategies:
        strategies[k].returns = pd.Series(daily_returns[k], index=idx, name=k)

    return strategies


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Equal Weight":   "#475569",
    "MVO Max Sharpe": "#f59e0b",
    "MVO Min Vol":    "#fb923c",
    "HRP":            "#34d399",
    "BL-Combined":    "#38bdf8",
}

def plot_backtest(strategies: dict, out: Path = None):
    out = out or _FIGURES / "k7_backtest.png"

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0a0e1a")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])   # cumulative wealth — full width
    ax2 = fig.add_subplot(gs[1, 0])   # drawdown
    ax3 = fig.add_subplot(gs[1, 1])   # metrics bar chart

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#64748b", labelsize=8)
        ax.spines[:].set_color("#1e293b")

    # ── Panel 1: Cumulative wealth ────────────────────────────────────────
    for name, res in strategies.items():
        wealth = res.cum_wealth
        ax1.plot(wealth.index, wealth.values,
                 color=COLORS.get(name, "#fff"), lw=1.6 if "BL" in name else 1.1,
                 label=f"{name}  (Sharpe={res.sharpe:.2f})",
                 alpha=0.9 if "BL" in name else 0.7)

    ax1.set_title("Walk-Forward Backtest — Cumulative Wealth (₹1 → ?)",
                  color="#f1f5f9", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Portfolio Value (₹)", color="#94a3b8")
    ax1.legend(facecolor="#111827", edgecolor="#1e293b",
               labelcolor="#94a3b8", fontsize=8, loc="upper left")
    ax1.axhline(1, color="#334155", lw=0.8, ls="--")
    ax1.set_xlabel("")

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    for name, res in strategies.items():
        wealth = res.cum_wealth
        peak   = wealth.cummax()
        dd     = (wealth - peak) / peak * 100
        ax2.fill_between(dd.index, dd.values, 0,
                         color=COLORS.get(name, "#fff"),
                         alpha=0.35 if "BL" in name else 0.2,
                         label=name)
        ax2.plot(dd.index, dd.values, color=COLORS.get(name, "#fff"),
                 lw=0.8, alpha=0.8)

    ax2.set_title("Drawdown (%)", color="#f1f5f9", fontweight="bold", fontsize=10)
    ax2.set_ylabel("Drawdown %", color="#94a3b8")
    ax2.legend(facecolor="#111827", edgecolor="#1e293b",
               labelcolor="#94a3b8", fontsize=7)

    # ── Panel 3: Metrics bar ──────────────────────────────────────────────
    names   = list(strategies.keys())
    sharpes = [strategies[n].sharpe for n in names]
    colors  = [COLORS.get(n, "#fff") for n in names]
    bars    = ax3.bar(range(len(names)), sharpes, color=colors, alpha=0.85)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([n.replace(" ", "\n") for n in names],
                        fontsize=7, color="#94a3b8")
    ax3.set_title("Sharpe Ratio Comparison", color="#f1f5f9",
                  fontweight="bold", fontsize=10)
    ax3.set_ylabel("Sharpe Ratio", color="#94a3b8")
    ax3.axhline(0, color="#475569", lw=0.8)
    for bar, val in zip(bars, sharpes):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.2f}", ha="center", va="bottom",
                 color="#f1f5f9", fontsize=8, fontweight="bold")

    plt.suptitle("Kuber K7 — Walk-Forward Backtest  |  2021–2026  |  Monthly Rebalance",
                 color="#f1f5f9", fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("  Backtest chart saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_backtest(strategies: dict):
    rows = [v.summary() for v in strategies.values()]
    df   = pd.DataFrame(rows).set_index("Strategy")

    print(f"\n{'═'*78}")
    print(f" K7 WALK-FORWARD BACKTEST RESULTS  |  {START_DATE} → present")
    print(f" Train: {TRAIN_DAYS}d  |  Rebalance: {REBAL_DAYS}d  |  rf={RISK_FREE_ANN:.1%}")
    print(f"{'═'*78}")
    print(f" {'Strategy':<20} {'Return%':>8} {'Vol%':>7} {'Sharpe':>8} "
          f"{'MaxDD%':>8} {'Calmar':>8} {'Turnover%':>10} {'HHI':>7}")
    print(f"{'─'*78}")
    for _, row in df.iterrows():
        marker = " ◀" if "BL" in row.name else ""
        print(f" {row.name:<20} {row['Return%']:>7.2f}% {row['Vol%']:>6.2f}% "
              f"{row['Sharpe']:>8.3f} {row['MaxDD%']:>7.2f}% "
              f"{row['Calmar']:>8.3f} {row['Turnover%']:>9.2f}% "
              f"{row['HHI']:>7.4f}{marker}")
    print(f"{'═'*78}")
    df.to_csv(_DATA / "k7_backtest_results.csv")
    logger.info("  Results saved → data/k7_backtest_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings as _w; _w.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from portfolio_optimizer.data_loader      import load_data
    from portfolio_optimizer.gatishakti_views import (
        load_gatishakti_config, get_xgb_valid_from
    )

    logger.info("=" * 60)
    logger.info("K7 WALK-FORWARD BACKTESTER  (time-gated views)")
    logger.info("=" * 60)

    bundle = load_data()

    # XGBoost predictions + valid_from date
    xgb_path = Path(__file__).resolve().parent.parent.parent / "alpha-core" / "data" / "xgb_predictions.csv"
    xgb_preds = pd.read_csv(xgb_path, index_col=0) if xgb_path.exists() else None

    gs_cfg = load_gatishakti_config()
    xgb_vf = get_xgb_valid_from(gs_cfg)
    logger.info("  XGBoost valid_from: %s", xgb_vf.date())
    logger.info("  GatiShakti quarters in YAML: %d", len(gs_cfg.get("quarters", [])))

    logger.info("")
    logger.info("  Running walk-forward backtest (views time-gated)...")
    strategies = run_backtest(
        returns_df      = bundle.returns,
        w_market        = bundle.w_market.values,
        tickers         = bundle.tickers,
        xgb_preds       = xgb_preds,
        xgb_valid_from  = xgb_vf,
    )

    print_backtest(strategies)
    plot_backtest(strategies)

    bl  = strategies["BL-Combined"]
    hrp = strategies["HRP"]
    mvo = strategies["MVO Max Sharpe"]
    ew  = strategies["Equal Weight"]

    print(f"\n── Key Takeaways ──")
    print(f"  BL vs Equal Weight:  Sharpe {bl.sharpe:+.3f} vs {ew.sharpe:+.3f}"
          f"  MaxDD {bl.max_drawdown:.1%} vs {ew.max_drawdown:.1%}")
    print(f"  BL vs HRP:           Sharpe {bl.sharpe:+.3f} vs {hrp.sharpe:+.3f}"
          f"  Turnover {bl.avg_turnover:.1%} vs {hrp.avg_turnover:.1%}")
    print(f"  BL vs MVO:           Sharpe {bl.sharpe:+.3f} vs {mvo.sharpe:+.3f}")

