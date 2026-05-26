"""
================================================================================
K4 — BLACK-LITTERMAN MODEL WITH XGBOOST AUTO-VIEWS
ml-portfolio-optimizer/portfolio_optimizer/black_litterman.py

WHAT (3-line summary):
  Start from market equilibrium implied returns (Π).
  Update them with XGBoost signal views (Q) weighted by IC confidence (Ω).
  Apply canonical BL analytical tilt: w = w_mkt + (δΣ)⁻¹(μ_BL − π).

THE 5-STEP BL MATH:
  1. Equilibrium:  Π   = δ × Σ_dcc × w_mkt
  2. Prior:        μ_prior ~ N(Π, τΣ)
  3. Views:        P·μ = Q + ε,  ε ~ N(0, Ω)
  4. Posterior μ:  μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹Π + P'Ω⁻¹Q]
  5. Optimise:     max_w (w'μ_BL - rf) / sqrt(w'Σ_bl·w)

PARAMETERS:
  δ (delta) = 2.5  — market risk aversion (standard institutional value)
  τ (tau)   = 0.05 — uncertainty in the prior (small = trust equilibrium)
  Ω         = diagonal, scaled by 1/IC² per stock (high IC = confident view)

WHY THIS OVER RAW MVO:
  MVO on raw historical μ put 60% in 3 stocks (K2).
  BL starts from what the MARKET implies — the equilibrium is the PM's
  prior. We only deviate where XGBoost has a confident signal (high IC).
  Stocks with low IC stay near equilibrium. No extreme tilts.

PROJECT CONNECTION:
  Vajra DCC  → Σ_dcc  (fresh May 2026 covariance)
  Alpha-Core → XGBoost IC + predicted returns (views Q)
  Kuber K4   → combines them into μ_BL → optimal weights

INTERVIEW:
  Q: "How do you generate the view matrix P and Q?"
  A: "Each XGBoost prediction is an absolute view on that stock's return.
     P is the identity (one view per stock). Q is the XGBoost predicted
     residual annualised. Ω_ii = (σ_i / IC_i)² — stocks with higher
     rolling IC get lower view uncertainty, so they move the posterior more."

  Q: "What is τ and how did you pick it?"
  A: "τ scales the uncertainty in the equilibrium prior. τ=0.05 means
     we're fairly confident in the market equilibrium — typical for
     institutional use. As τ→0 the posterior converges to Π. As τ→∞
     the posterior converges to the raw MVO solution."
================================================================================
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_optimizer.data_loader import DataBundle

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)

_ALPHA_DATA = Path(__file__).resolve().parent.parent.parent / "alpha-core" / "data"
_FIGURES    = Path(__file__).resolve().parent.parent / "figures"
_DATA       = Path(__file__).resolve().parent.parent / "data"
_FIGURES.mkdir(exist_ok=True)
_DATA.mkdir(exist_ok=True)

# ── Model hyperparameters ──────────────────────────────────────────────────────
DELTA        = 2.5    # risk aversion coefficient
TAU          = 0.05   # prior uncertainty scale — trust the equilibrium
                      # τ=0.05: prior dominates views in a stable ratio.
                      # Textbook value (He & Litterman 1999, Idzorek 2005).
                      # Consistent with backtester calibration (K7).
                      # As τ→0: posterior = Π (ignore views entirely).
                      # As τ→∞: posterior = raw MVO (ignore equilibrium).
RISK_FREE    = 0.065  # 10-yr Indian G-Sec (consistent with K2 MVO)
MIN_IC       = 0.05   # IC threshold — only stocks with meaningful predictive
                      # signal become views. IC=0.030 (HDFCBANK) is too weak:
                      # correlation propagation overrides the view anyway.
                      # IC≥0.05 ~ top-half signal quality for daily equity ML.
W_MAX        = 0.20   # per-stock weight cap in final MVO


# ─────────────────────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BLResult:
    # Intermediate quantities
    pi_eq:          pd.Series    # equilibrium implied returns  (N,)
    mu_bl:          pd.Series    # posterior BL returns         (N,)
    sigma_bl:       pd.DataFrame # posterior covariance         (N×N)
    view_matrix_P:  np.ndarray   # view pick matrix             (K×N)
    view_vector_Q:  np.ndarray   # view return vector           (K,)
    view_omega:     np.ndarray   # view uncertainty diagonal    (K,)
    active_views:   list         # tickers with active views
    # Optimal portfolio
    weights:        pd.Series    # final optimal weights        (N,)
    ret_annual:     float
    vol_annual:     float
    sharpe:         float

    def __str__(self):
        top3 = self.weights.nlargest(3)
        s = ", ".join(f"{t}={w:.1%}" for t, w in top3.items())
        return (f"BL: ret={self.ret_annual:.2%}  vol={self.vol_annual:.2%}"
                f"  sharpe={self.sharpe:.3f}  [top3: {s}]")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — EQUILIBRIUM IMPLIED RETURNS
# ─────────────────────────────────────────────────────────────────────────────

def compute_equilibrium(bundle: DataBundle) -> pd.Series:
    """
    Π = δ × Σ_dcc × w_mkt

    WHY this formula:
      In equilibrium every investor is happy holding the market portfolio.
      Back out what expected return vector makes them indifferent:
        ∂/∂w [w'μ - (δ/2)w'Σw] = 0  →  μ = δΣw
      So Π is the return implied by the market's current holdings.

    WHY Σ_dcc not Σ_lw:
      Equilibrium is a CURRENT statement about TODAY's risk. DCC gives
      today's covariance (May 2026 regime), not a 7-year historical average.
      HDFCBANK is currently more volatile (30.9% DCC vs 35.9% LW) — the
      equilibrium should reflect the current regime, not 2019-2026 average.
    """
    w = bundle.w_market.values       # (N,)
    S = bundle.sigma_dcc.values      # (N×N) — Vajra DCC, May 2026
    pi = DELTA * S @ w               # (N,)
    return pd.Series(pi, index=bundle.tickers)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD VIEWS FROM XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

def build_views_from_xgboost(tickers: list, pi: np.ndarray,
                              sigma_lw: np.ndarray) -> tuple:
    """
    Build BL view triplet (P, Q, Ω) using Grinold-Kahn alpha forecasts.

    GRINOLD-KAHN FORMULA (the industry standard for Q):
        alpha_i = IC_i × σ_i

        Where σ_i is the annualised vol of stock i.
        IC measures signal quality; σ scales it to return space.
        From Grinold & Kahn (1999), "Active Portfolio Management" Ch. 6.

    Q construction:
        Q_i = Π_i + sign(view) × IC_i × σ_i

        BL views are RELATIVE to equilibrium. We're saying:
        "I think ONGC will outperform its equilibrium return by IC × σ."
        This keeps Q in the same magnitude as Π (3-10% range), not 32%.

    WHY NOT raw_prediction × 252:
        XGBoost predicts daily residuals in % space. Annualising gives
        32% for ONGC, 60% for DRREDDY. These are predictions of TOTAL
        residual return, not calibrated alpha views. Plugging 32% into
        Q while Ω is sized for a 3-4% signal creates a catastrophic
        signal-to-noise mismatch — BL's prior dominates 99:1.
        The Grinold-Kahn formula produces Q in the 1-5% range
        which is commensurable with Π and Ω.

    Ω construction (He & Litterman 1999 form):
        Ω = diag(P × (τΣ) × P') / IC²
        This scales Ω to the same order as τΣ (the prior uncertainty).
        A stock with IC=0.12 gets 1/0.0144 = 70× less uncertainty —
        i.e. its view is much more confident than the prior.
    """
    xgb_path = _ALPHA_DATA / "xgb_predictions.csv"
    if not xgb_path.exists():
        logger.warning("  xgb_predictions.csv not found — using empty view set")
        return np.zeros((0, len(tickers))), np.zeros(0), np.zeros((0, 0)), []

    xgb  = pd.read_csv(xgb_path, index_col=0)

    # Annualised vol per stock from LW covariance diagonal
    sigma_annual = np.sqrt(np.diag(sigma_lw))   # (N,) in decimal

    active_tickers = []
    P_rows, Q_vals, omega_diag = [], [], []

    for ticker in tickers:
        if ticker not in xgb.index:
            continue
        row    = xgb.loc[ticker]
        ic     = row["ic_test"]
        signal = row["signal"]

        # Filter: skip NEUTRAL or very low IC stocks
        if signal == "NEUTRAL" or abs(ic) < MIN_IC:
            continue

        i      = tickers.index(ticker)
        pi_i   = pi[i]              # equilibrium return for this stock
        sig_i  = sigma_annual[i]    # annualised vol (decimal)

        # ── Grinold-Kahn Q ──────────────────────────────────────────
        # alpha_i = |IC_i| × σ_i  (always positive, sign applied below)
        alpha_i = abs(ic) * sig_i
        direction = +1.0 if signal == "LONG_BIAS" else -1.0
        q_i = pi_i + direction * alpha_i
        # LONG view: we think this stock earns IC×σ above equilibrium
        # SHORT view: we think it earns IC×σ below equilibrium

        # ── Idzorek (2005) confidence-based Ω ───────────────────────
        # p_k = IC_k / max_possible_IC  → view confidence in [0,1]
        # Ω_kk = ((1 - p_k) / p_k) × τ × σ²_i
        # This guarantees: p→1 (IC→max) → Ω→0 → posterior = view
        #                  p→0 (IC→0)   → Ω→∞ → posterior = Π
        # Numerically consistent with τΣ scale by construction.
        # Reference: Idzorek (2005), "A step-by-step guide to BL"
        MAX_IC  = 0.15   # approximate max IC for daily equity signals
        p_k     = min(abs(ic) / MAX_IC, 0.99)
        omega_k = ((1 - p_k) / p_k) * TAU * (sig_i ** 2)

        # Build P row
        p_row = np.zeros(len(tickers))
        p_row[i] = 1.0

        active_tickers.append(ticker)
        P_rows.append(p_row)
        Q_vals.append(q_i)
        omega_diag.append(omega_k)

        logger.info(
            "  View: %-12s  Π=%+.2f%%  alpha=%.2f%%  Q=%+.2f%%  IC=%.3f  Ω=%.4f  [%s]",
            ticker, pi_i * 100, alpha_i * 100, q_i * 100, ic, omega_k, signal)

    if not active_tickers:
        return np.zeros((0, len(tickers))), np.zeros(0), np.zeros((0, 0)), []

    P     = np.array(P_rows)       # (K, N)
    Q     = np.array(Q_vals)       # (K,)
    Omega = np.diag(omega_diag)    # (K, K)

    return P, Q, Omega, active_tickers


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — POSTERIOR RETURNS (THE BL UPDATE)
# ─────────────────────────────────────────────────────────────────────────────

def black_litterman_posterior(pi: np.ndarray, sigma: np.ndarray,
                               P: np.ndarray, Q: np.ndarray,
                               Omega: np.ndarray) -> tuple:
    """
    Bayesian update: combine equilibrium prior with views.

    μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹Π + P'Ω⁻¹Q]

    Posterior covariance of μ:
    M    = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹

    Full posterior portfolio covariance:
    Σ_BL = Σ + M   (Σ captures asset risk, M captures estimation uncertainty)

    WHY this form?
      It's Bayes' theorem applied to a multivariate Gaussian model.
      Prior:    μ ~ N(Π, τΣ)     → precision = (τΣ)⁻¹
      Views:    Q = Pμ + ε       → precision = P'Ω⁻¹P
      Posterior precision = prior precision + view precision (information adds)
      Posterior mean = weighted average of Π and view-implied returns,
      weighted by their respective precisions.

    Special case: no views (K=0)
      With no views, the posterior = the prior = Π (equilibrium).
      This is the correct fallback — BL never extrapolates without evidence.
    """
    tau_sigma     = TAU * sigma
    tau_sigma_inv = np.linalg.inv(tau_sigma)

    if P.shape[0] == 0:
        # No views → posterior = prior
        M   = np.linalg.inv(tau_sigma_inv)
        mu_bl = pi.copy()
        return mu_bl, tau_sigma + M

    Omega_inv     = np.linalg.inv(Omega)

    # Precision-weighted posterior
    A     = tau_sigma_inv + P.T @ Omega_inv @ P        # (N×N)
    b     = tau_sigma_inv @ pi + P.T @ Omega_inv @ Q   # (N,)
    mu_bl = np.linalg.solve(A, b)                      # (N,) posterior μ

    M     = np.linalg.inv(A)                           # posterior uncertainty
    sigma_bl = sigma + M                               # full posterior cov

    return mu_bl, sigma_bl


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — OPTIMISE ON POSTERIOR
# ─────────────────────────────────────────────────────────────────────────────

def _max_sharpe_bl(mu_bl: np.ndarray, sigma_bl: np.ndarray,
                   tickers: list, w_max: float = W_MAX) -> pd.Series:
    """MVO on BL posterior returns. Same SLSQP as K2 but on μ_BL, Σ_BL."""
    n  = len(tickers)
    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(0.0, w_max)] * n

    def neg_sharpe(w):
        r = float(w @ mu_bl)
        v = float(np.sqrt(max(w @ sigma_bl @ w, 1e-12)))
        return -(r - RISK_FREE) / v

    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return pd.Series(res.x, index=tickers)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _plot_bl(result: BLResult, bundle: DataBundle,
             mvo_weights: pd.Series = None):
    """Two-panel: (1) Π vs μ_BL returns, (2) weight comparison."""
    tickers = result.weights.index.tolist()
    x = np.arange(len(tickers))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0a0e1a")
    for ax in axes:
        ax.set_facecolor("#111827")

    # Panel 1 — equilibrium vs posterior returns
    ax = axes[0]
    ax.bar(x - 0.2, result.pi_eq.values * 100,  0.35,
           color="#64748b", alpha=0.85, label="Π Equilibrium")
    ax.bar(x + 0.2, result.mu_bl.values * 100, 0.35,
           color="#38bdf8", alpha=0.85, label="μ_BL Posterior")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8, color="#94a3b8")
    ax.set_ylabel("Annual Return (%)", color="#94a3b8")
    ax.set_title("BL: Equilibrium vs Posterior Returns\n(XGBoost views shift Π → μ_BL)",
                 color="#f1f5f9", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#64748b")
    ax.spines[:].set_color("#1e293b")
    ax.legend(facecolor="#111827", edgecolor="#1e293b",
              labelcolor="#94a3b8", fontsize=8)
    ax.axhline(0, color="#475569", lw=0.8)

    # Mark active views
    for ticker in result.active_views:
        if ticker in tickers:
            xi = tickers.index(ticker)
            ax.axvline(xi, color="#f59e0b", alpha=0.2, lw=12)
    ax.text(0.02, 0.97, f"Yellow = active XGB view ({len(result.active_views)} stocks)",
            transform=ax.transAxes, color="#f59e0b", fontsize=7, va="top")

    # Panel 2 — weight comparison
    ax2 = axes[1]
    bl_w = result.weights.values * 100
    ax2.bar(x, bl_w, 0.35, color="#34d399", alpha=0.85, label="BL Optimal")
    if mvo_weights is not None:
        mvo_w = mvo_weights.reindex(tickers, fill_value=0).values * 100
        ax2.bar(x + 0.37, mvo_w, 0.35, color="#f59e0b", alpha=0.85, label="MVO Max Sharpe")
    ax2.axhline(100 / len(tickers), color="#475569", ls="--",
                lw=1, label=f"Equal weight ({100/len(tickers):.1f}%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8, color="#94a3b8")
    ax2.set_ylabel("Weight (%)", color="#94a3b8")
    ax2.set_title(f"BL vs MVO Weights\nSharpe: BL={result.sharpe:.3f}",
                  color="#f1f5f9", fontsize=10, fontweight="bold")
    ax2.tick_params(colors="#64748b")
    ax2.spines[:].set_color("#1e293b")
    ax2.legend(facecolor="#111827", edgecolor="#1e293b",
               labelcolor="#94a3b8", fontsize=8)

    plt.tight_layout()
    out = _FIGURES / "k4_black_litterman.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("  BL chart saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_black_litterman(bundle: DataBundle,
                        mvo_max_sharpe_weights: pd.Series = None,
                        plot: bool = True) -> BLResult:
    """
    Full K4 Black-Litterman pipeline.

    Parameters
    ----------
    bundle                  : DataBundle from K1
    mvo_max_sharpe_weights  : K2 weights for comparison plot (optional)
    plot                    : save chart
    """
    logger.info("=" * 60)
    logger.info("K4 BLACK-LITTERMAN MODEL")
    logger.info("  δ=%.1f  τ=%.3f  rf=%.1f%%  w_max=%.0f%%  mode=canonical_tilt",
                DELTA, TAU, RISK_FREE * 100, W_MAX * 100)
    logger.info("=" * 60)

    tickers = bundle.tickers
    sigma   = bundle.sigma_dcc.values    # DCC covariance — current regime

    # Step 1: Equilibrium
    pi_series = compute_equilibrium(bundle)
    pi        = pi_series.values
    logger.info("  Equilibrium Π range: [%.2f%%, %.2f%%]",
                pi.min() * 100, pi.max() * 100)

    # Step 2: XGBoost views (Grinold-Kahn calibrated)
    logger.info("  Loading XGBoost views (IC threshold = %.2f)...", MIN_IC)
    P, Q, Omega, active_views = build_views_from_xgboost(
        tickers, pi, bundle.sigma_lw.values)
    logger.info("  Active views: %d / %d stocks", len(active_views), len(tickers))

    # Step 3: Posterior
    mu_bl_arr, sigma_bl_arr = black_litterman_posterior(pi, sigma, P, Q, Omega)
    mu_bl    = pd.Series(mu_bl_arr, index=tickers)
    sigma_bl = pd.DataFrame(sigma_bl_arr, index=tickers, columns=tickers)

    logger.info("  μ_BL range: [%.2f%%, %.2f%%]",
                mu_bl.min() * 100, mu_bl.max() * 100)

    # Step 4: Canonical analytical BL tilt (NOT MVO)
    #
    # Formula:  w = w_market + (δΣ)⁻¹ (μ_BL − π)
    #
    # WHY this instead of MVO on mu_bl:
    #   MVO on mu_bl amplifies tiny return differences → 3 stocks slam to 20%
    #   cap every time, everything else → 0. Not because the views are strong
    #   — because RELIANCE/HDFCBANK always have highest equilibrium Π, so MVO
    #   always finds the same corner solution regardless of XGBoost views.
    #   The canonical formula moves each stock PROPORTIONALLY from its market
    #   weight: view stocks get +2-4% tilt, non-view stocks stay near mkt wt.
    #   This is the closed-form BL solution — identical to what K7 backtester
    #   uses and what He & Litterman (1999) derive as the optimal BL portfolio.
    w_market_arr = bundle.w_market.values
    sigma_for_tilt = bundle.sigma_lw.values   # LW for tilt (more stable than DCC)
    tilt  = np.linalg.solve(DELTA * sigma_for_tilt, mu_bl_arr - pi)
    w_raw = w_market_arr + tilt
    w_raw = np.clip(w_raw, 0.0, W_MAX)        # no shorts, respect 20% cap
    s     = w_raw.sum()
    w_arr = w_raw / s if s > 1e-9 else w_market_arr.copy()
    w_opt = pd.Series(w_arr, index=tickers)

    ret    = float(w_arr @ mu_bl_arr)
    vol    = float(np.sqrt(w_arr @ sigma_bl_arr @ w_arr))
    sharpe = (ret - RISK_FREE) / vol if vol > 0 else 0.0

    logger.info("  Tilt range: [%+.2f%%, %+.2f%%]",
                (tilt * 100).min(), (tilt * 100).max())
    logger.info("  Active view stocks shift: %s",
                {t: f"{(w_opt[t] - bundle.w_market[t]) * 100:+.1f}%"
                 for t in active_views if t in w_opt.index})

    result = BLResult(
        pi_eq=pi_series, mu_bl=mu_bl, sigma_bl=sigma_bl,
        view_matrix_P=P, view_vector_Q=Q, view_omega=Omega,
        active_views=active_views,
        weights=w_opt, ret_annual=ret, vol_annual=vol, sharpe=sharpe,
    )
    logger.info("  %s", result)

    # Save
    w_opt.to_frame("weight").assign(ret=ret, vol=vol, sharpe=sharpe).to_csv(
        _DATA / "k4_bl_weights.csv")
    pd.DataFrame({"pi_eq": pi_series, "mu_bl": mu_bl,
                  "delta": mu_bl - pi_series}).to_csv(_DATA / "k4_bl_returns.csv")

    if plot:
        _plot_bl(result, bundle, mvo_max_sharpe_weights)

    logger.info("=" * 60)
    return result


def print_bl(result: BLResult):
    """Detailed terminal output."""
    print(f"\n{'═'*62}")
    print(f" K4 BLACK-LITTERMAN RESULTS")
    print(f"{'═'*62}")
    print(f" Return: {result.ret_annual:.2%}  |  Vol: {result.vol_annual:.2%}  |  Sharpe: {result.sharpe:.3f}")
    print(f"\n{'─'*62}")
    print(f" {'Stock':<12} {'Π Equil%':>10} {'μ_BL%':>10} {'Δ (view shift)':>16} {'Weight':>8}")
    print(f"{'─'*62}")
    for t in result.weights.index:
        pi  = result.pi_eq[t] * 100
        mbl = result.mu_bl[t] * 100
        w   = result.weights[t] * 100
        tag = " ◀ VIEW" if t in result.active_views else ""
        print(f" {t:<12} {pi:>9.2f}% {mbl:>9.2f}% {mbl-pi:>+15.2f}% {w:>7.2f}%{tag}")
    print(f"{'─'*62}")
    print(f" Active views ({len(result.active_views)}): {', '.join(result.active_views)}")
    print(f" δ={DELTA}  τ={TAU}  IC_threshold={MIN_IC}")
    print(f"{'═'*62}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings as _w; _w.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    from portfolio_optimizer.data_loader import load_data
    from portfolio_optimizer.markowitz   import run_markowitz

    bundle = load_data()
    mvo    = run_markowitz(bundle, plot=False)
    bl     = run_black_litterman(bundle,
                                  mvo_max_sharpe_weights=mvo.max_sharpe.weights)
    print_bl(bl)

    print(f"\n── K2 vs K3 vs K4 Comparison ──")
    print(f"{'Method':<20} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'Max Wt':>8}")
    print("─" * 58)
    ms = mvo.max_sharpe
    print(f"{'MVO Max Sharpe':<20} {ms.ret_annual:>7.2%} {ms.vol_annual:>7.2%}"
          f" {ms.sharpe:>8.3f} {ms.weights.max():>7.2%}")
    print(f"{'BL Posterior':<20} {bl.ret_annual:>7.2%} {bl.vol_annual:>7.2%}"
          f" {bl.sharpe:>8.3f} {bl.weights.max():>7.2%}")

    print(f"\n── View Impact (Π → μ_BL shifts) ──")
    for t in bl.active_views:
        pi  = bl.pi_eq[t] * 100
        mbl = bl.mu_bl[t] * 100
        print(f"  {t:<12}: Π={pi:+.2f}%  →  μ_BL={mbl:+.2f}%  (Δ={mbl-pi:+.2f}%)")
