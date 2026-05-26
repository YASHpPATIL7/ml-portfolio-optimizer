"""
================================================================================
K2 — MARKOWITZ MEAN-VARIANCE OPTIMISATION (MVO)
ml-portfolio-optimizer/portfolio_optimizer/markowitz.py

WHAT:
  Implements Harry Markowitz's (1952, Nobel 1990) portfolio theory.
  Three outputs:
    1. Maximum Sharpe portfolio   — best risk-adjusted return
    2. Minimum volatility portfolio — lowest possible portfolio vol
    3. Efficient frontier          — 100-point curve of optimal portfolios

HOW:
  Uses scipy SLSQP (Sequential Least Squares Programming) constrained
  optimiser. Two constraints:
    - weights sum to 1.0   (fully invested)
    - 0 ≤ w_i ≤ 0.20      (no short-selling, max 20% per stock)
  Covariance input: Σ_lw (Ledoit-Wolf) from K1.

WHY Ledoit-Wolf not sample Σ:
  MVO inverts Σ numerically. Sample Σ with near-zero eigenvalues produces
  huge weight swings on tiny return changes. LW stabilises eigenvalues →
  stable weights → lower turnover across rebalancing periods.

WHY the 20% cap:
  Unconstrained MVO routinely puts 60-80% in one stock. No institutional
  PM runs 80% AXISBANK. SEBI LODR also limits concentration. The cap
  forces real-world diversification.

THE HONEST LIMITATION (say this in interviews):
  MVO is exquisitely sensitive to μ. A 0.5% error in ONGC's expected
  return can swing its weight from 0% to 20%. This is why Black-Litterman
  (K4) replaces raw μ with the market-equilibrium prior + XGBoost views.
  MVO is the baseline to beat, not the production answer.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_optimizer.data_loader import DataBundle

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.065   # 10-yr Indian G-Sec yield ≈ 6.5% annualised
_FIGURES = Path(__file__).resolve().parent.parent / "figures"
_DATA    = Path(__file__).resolve().parent.parent / "data"
_FIGURES.mkdir(exist_ok=True)
_DATA.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────────────────────

@dataclass
class MVOResult:
    """Holds a single optimal portfolio's outputs."""
    label:       str
    weights:     pd.Series          # ticker → weight
    ret_annual:  float              # annualised return
    vol_annual:  float              # annualised volatility
    sharpe:      float              # Sharpe ratio (excess / vol)

    def __str__(self):
        top3 = self.weights.nlargest(3)
        top3_str = ", ".join(f"{t}={w:.1%}" for t, w in top3.items())
        return (f"{self.label}: ret={self.ret_annual:.2%}  "
                f"vol={self.vol_annual:.2%}  sharpe={self.sharpe:.3f}  "
                f"[top3: {top3_str}]")


@dataclass
class MVOOutput:
    """Full K2 output bundle."""
    max_sharpe:   MVOResult
    min_vol:      MVOResult
    frontier:     pd.DataFrame      # columns: ret, vol, sharpe, w_TICKER...
    label:        str = "K2 Markowitz MVO"


# ─────────────────────────────────────────────────────────────
# CORE MATH HELPERS
# ─────────────────────────────────────────────────────────────

def _portfolio_stats(w: np.ndarray, mu: np.ndarray,
                     sigma: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute annualised return, vol, Sharpe for a weight vector.

    Math:
      ret = w' μ              (dot product: weighted sum of returns)
      var = w' Σ w            (quadratic form: portfolio variance)
      vol = sqrt(var)
      sharpe = (ret - rf) / vol

    These three lines are the entire foundation of modern portfolio theory.
    Everything else — Black-Litterman, HRP, Kelly — is built on top of this.
    """
    ret    = float(w @ mu)
    var    = float(w @ sigma @ w)
    vol    = float(np.sqrt(max(var, 1e-12)))   # clip to avoid sqrt(0)
    sharpe = (ret - RISK_FREE_RATE) / vol
    return ret, vol, sharpe


def _build_problem(n: int, w_max: float = 0.20):
    """
    Build scipy constraints and bounds shared by all optimisations.

    Constraints:
      sum(w) = 1.0  — fully invested (no cash, no leverage at portfolio level)

    Bounds:
      0 ≤ w_i ≤ w_max  — no short-selling (0 floor), concentration cap (w_max)

    WHY SLSQP?
      Sequential Least-Squares Programming handles nonlinear objectives
      with linear and nonlinear equality/inequality constraints natively.
      It's the industry default for small-to-medium MVO (N < 1000).
    """
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds      = [(0.0, w_max)] * n
    w0          = np.ones(n) / n          # equal-weight starting point
    return constraints, bounds, w0


# ─────────────────────────────────────────────────────────────
# OPTIMISERS
# ─────────────────────────────────────────────────────────────

def _max_sharpe(mu: np.ndarray, sigma: np.ndarray,
                tickers: List[str], w_max: float = 0.20) -> MVOResult:
    """
    Maximum Sharpe Ratio (Tangency Portfolio).

    Objective: maximise (w'μ - rf) / sqrt(w'Σw)
    Equivalently: minimise -(w'μ - rf) / sqrt(w'Σw)

    WHY "tangency"?
      On the efficient frontier, the tangency portfolio is the one where
      the Capital Market Line (CML) — drawn from the risk-free asset —
      is tangent to the frontier curve. It's the single portfolio you'd
      hold if you could also invest in a risk-free asset (G-Sec / FD).

    INTERVIEW: "What's the tangency portfolio?"
      "The portfolio with the highest Sharpe ratio. If you can combine
       it with a risk-free asset, you can reach any point on the Capital
       Market Line — more efficient than any other risky portfolio."
    """
    n = len(tickers)
    constraints, bounds, w0 = _build_problem(n, w_max)

    def neg_sharpe(w):
        r, v, _ = _portfolio_stats(w, mu, sigma)
        return -(r - RISK_FREE_RATE) / max(v, 1e-12)

    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})
    w_opt = res.x
    ret, vol, sharpe = _portfolio_stats(w_opt, mu, sigma)
    return MVOResult("Max Sharpe", pd.Series(w_opt, index=tickers), ret, vol, sharpe)


def _min_volatility(mu: np.ndarray, sigma: np.ndarray,
                    tickers: List[str], w_max: float = 0.20) -> MVOResult:
    """
    Minimum Variance Portfolio.

    Objective: minimise w'Σw   (portfolio variance)
    Note: μ is NOT in this objective — min-vol ignores expected returns.

    WHY this matters:
      Min-vol is useful when you distrust μ entirely (which is often
      the right call — see the L1 limitation above). It's also the
      leftmost point on the efficient frontier.

      In live trading, min-vol strategies historically deliver 70-80% of
      market returns with 30-40% less volatility. HDFC AMC Low-Vol fund
      is essentially this.

    INTERVIEW: "What's on the left end of the efficient frontier?"
      "The Global Minimum Variance portfolio — the lowest possible
       portfolio volatility achievable given the covariance structure,
       regardless of returns."
    """
    n = len(tickers)
    constraints, bounds, w0 = _build_problem(n, w_max)

    def portfolio_var(w):
        return float(w @ sigma @ w)

    res = minimize(portfolio_var, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})
    w_opt = res.x
    ret, vol, sharpe = _portfolio_stats(w_opt, mu, sigma)
    return MVOResult("Min Volatility", pd.Series(w_opt, index=tickers), ret, vol, sharpe)


def _efficient_frontier(mu: np.ndarray, sigma: np.ndarray,
                         tickers: List[str], n_points: int = 100,
                         w_max: float = 0.20) -> pd.DataFrame:
    """
    Trace the efficient frontier: 100 optimal portfolios.

    Method:
      1. Find the achievable return range [ret_minvol, ret_max]
      2. For each target return in that range, solve:
           min  w'Σw
           s.t. w'μ = target_return
                sum(w) = 1
                0 ≤ w ≤ w_max

      3. Record (ret, vol, sharpe, weights) for each point

    WHY 100 points?
      The frontier is a hyperbola in (vol, ret) space. 100 points gives
      a smooth curve for plotting and is computationally trivial (< 0.5s).

    WHY parametrize by return not risk?
      Fixing target return and minimising variance is a convex QP —
      guaranteed to find the global minimum. Fixing target vol and
      maximising return is also valid but numerically less stable.
    """
    # Anchor points
    mv = _min_volatility(mu, sigma, tickers, w_max)
    ret_min = mv.ret_annual
    # Max achievable return = max individual stock return (trivially, 100% allocation)
    ret_max = float(mu.max())
    if w_max < 1.0:
        ret_max = min(ret_max, w_max * float(mu.max()) + (1 - w_max) * float(np.sort(mu)[-2]))

    target_returns = np.linspace(ret_min, ret_max * 0.98, n_points)
    n = len(tickers)
    records = []

    for target_ret in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=target_ret: float(w @ mu) - tr},
        ]
        bounds = [(0.0, w_max)] * n
        w0     = np.ones(n) / n

        res = minimize(lambda w: float(w @ sigma @ w),
                       w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-10, "maxiter": 500})
        if not res.success:
            continue

        w_opt = res.x
        ret, vol, sharpe = _portfolio_stats(w_opt, mu, sigma)
        row = {"ret": ret, "vol": vol, "sharpe": sharpe}
        row.update({f"w_{t}": float(w_opt[i]) for i, t in enumerate(tickers)})
        records.append(row)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────

def _plot_frontier(frontier: pd.DataFrame,
                   max_sharpe: MVOResult,
                   min_vol:    MVOResult,
                   bundle:     DataBundle,
                   save_path:  Path = None):
    """Efficient frontier with individual stocks overlaid."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#111827")

    # Frontier curve
    ax.plot(frontier["vol"] * 100, frontier["ret"] * 100,
            color="#38bdf8", lw=2.5, label="Efficient Frontier", zorder=3)

    # Colour frontier by Sharpe
    sc = ax.scatter(frontier["vol"] * 100, frontier["ret"] * 100,
                    c=frontier["sharpe"], cmap="viridis",
                    s=15, zorder=4, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio", pad=0.01)

    # Max Sharpe point
    ax.scatter(max_sharpe.vol_annual * 100, max_sharpe.ret_annual * 100,
               color="#f59e0b", s=200, zorder=5, marker="*",
               label=f"Max Sharpe ({max_sharpe.sharpe:.2f})")
    ax.annotate(f"Max Sharpe\n{max_sharpe.sharpe:.2f}",
                (max_sharpe.vol_annual * 100, max_sharpe.ret_annual * 100),
                textcoords="offset points", xytext=(12, 5),
                color="#f59e0b", fontsize=8)

    # Min Vol point
    ax.scatter(min_vol.vol_annual * 100, min_vol.ret_annual * 100,
               color="#34d399", s=150, zorder=5, marker="D",
               label=f"Min Vol ({min_vol.vol_annual:.1%})")
    ax.annotate(f"Min Vol\n{min_vol.vol_annual:.1%}",
                (min_vol.vol_annual * 100, min_vol.ret_annual * 100),
                textcoords="offset points", xytext=(8, -15),
                color="#34d399", fontsize=8)

    # Individual stocks
    mu_a  = bundle.mu_annual
    vols  = np.sqrt(np.diag(bundle.sigma_lw.values)) * 100
    for i, t in enumerate(bundle.tickers):
        ax.scatter(vols[i], mu_a[t] * 100,
                   color="#a78bfa", s=60, zorder=4, alpha=0.7)
        ax.annotate(t, (vols[i], mu_a[t] * 100),
                    textcoords="offset points", xytext=(4, 3),
                    color="#94a3b8", fontsize=6.5)

    # Risk-free rate line
    vol_range = np.linspace(0, frontier["vol"].max() * 100 * 1.1, 100)
    cml_ret   = RISK_FREE_RATE * 100 + max_sharpe.sharpe * vol_range
    ax.plot(vol_range, cml_ret, "--", color="#f87171",
            alpha=0.5, lw=1.2, label="Capital Market Line")

    ax.set_xlabel("Annual Volatility (%)", color="#94a3b8")
    ax.set_ylabel("Annual Return (%)", color="#94a3b8")
    ax.set_title("K2 — Efficient Frontier | 14 NSE Stocks",
                 color="#f1f5f9", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#64748b")
    ax.spines[:].set_color("#1e293b")
    ax.legend(facecolor="#111827", edgecolor="#1e293b",
              labelcolor="#94a3b8", fontsize=8)
    plt.tight_layout()

    out = save_path or (_FIGURES / "k2_efficient_frontier.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("  Frontier plot saved → %s", out)
    return out


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def run_markowitz(bundle: DataBundle, w_max: float = 0.20,
                  plot: bool = True) -> MVOOutput:
    """
    Run full K2 pipeline on a DataBundle.

    Parameters
    ----------
    bundle  : DataBundle from K1 load_data()
    w_max   : per-stock weight cap (default 20%)
    plot    : save efficient frontier chart

    Returns
    -------
    MVOOutput with max_sharpe, min_vol, frontier DataFrame
    """
    logger.info("=" * 60)
    logger.info("K2 MARKOWITZ MVO")
    logger.info("=" * 60)

    mu    = bundle.mu_annual.values          # shape (14,)
    sigma = bundle.sigma_lw.values           # shape (14,14) — Ledoit-Wolf
    tickers = bundle.tickers

    logger.info("  Covariance: Ledoit-Wolf (annualised)")
    logger.info("  Risk-free rate: %.2f%%", RISK_FREE_RATE * 100)
    logger.info("  Weight cap: %.0f%% per stock", w_max * 100)

    # Solve
    ms  = _max_sharpe(mu, sigma, tickers, w_max)
    mv  = _min_volatility(mu, sigma, tickers, w_max)
    ef  = _efficient_frontier(mu, sigma, tickers, w_max=w_max)

    logger.info("  %s", ms)
    logger.info("  %s", mv)
    logger.info("  Frontier: %d points computed", len(ef))

    # Save
    ms.weights.to_frame("weight").assign(
        ret=ms.ret_annual, vol=ms.vol_annual, sharpe=ms.sharpe
    ).to_csv(_DATA / "k2_max_sharpe.csv")

    mv.weights.to_frame("weight").assign(
        ret=mv.ret_annual, vol=mv.vol_annual, sharpe=mv.sharpe
    ).to_csv(_DATA / "k2_min_vol.csv")

    ef.to_csv(_DATA / "k2_frontier.csv", index=False)

    if plot:
        _plot_frontier(ef, ms, mv, bundle)

    out = MVOOutput(max_sharpe=ms, min_vol=mv, frontier=ef)
    logger.info("=" * 60)
    return out


def print_weights(result: MVOResult, threshold: float = 0.005):
    """Pretty-print weights above threshold."""
    print(f"\n── {result.label} ──")
    print(f"   Return: {result.ret_annual:.2%}  |  "
          f"Vol: {result.vol_annual:.2%}  |  "
          f"Sharpe: {result.sharpe:.3f}")
    print("   Weights:")
    for t, w in result.weights.sort_values(ascending=False).items():
        if w > threshold:
            bar = "█" * int(w * 50)
            print(f"   {t:<12} {w:6.2%}  {bar}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from portfolio_optimizer.data_loader import load_data

    bundle = load_data()
    output = run_markowitz(bundle)

    print_weights(output.max_sharpe)
    print_weights(output.min_vol)

    print(f"\n── Efficient Frontier ──")
    print(f"   Points computed: {len(output.frontier)}")
    print(f"   Sharpe range: [{output.frontier['sharpe'].min():.3f}, "
          f"{output.frontier['sharpe'].max():.3f}]")
    print(f"   Vol range:    [{output.frontier['vol'].min():.2%}, "
          f"{output.frontier['vol'].max():.2%}]")

    print("\n── What MVO tells us ──")
    ms = output.max_sharpe
    mv = output.min_vol
    print(f"   Nifty 50 benchmark: ~12% return, ~18% vol, ~Sharpe 0.36")
    print(f"   Our Max Sharpe:     {ms.ret_annual:.1%} return, "
          f"{ms.vol_annual:.1%} vol, Sharpe {ms.sharpe:.2f}")
    print(f"   Our Min Vol:        {mv.ret_annual:.1%} return, "
          f"{mv.vol_annual:.1%} vol, Sharpe {mv.sharpe:.2f}")
    print(f"\n   ⚠️  Remember: μ from 7-year history has ~12% annual error.")
    print(f"   These weights are the BASELINE. BL (K4) will improve them.")
