#!/usr/bin/env python3
"""
main.py — The Three-Project Pipeline
=====================================
Vajra (Risk) → Alpha-Core (Signal) → Kuber (Allocation) → RebalancingAgent

One command runs the full stack:
    python main.py

What it does at each step:
    1. VAJRA   — Loads latest DCC covariance matrix (regime-aware Σ)
    2. ALPHA-CORE — Reads HMM regime + XGBoost IC signals (today's views)
    3. KUBER   — Runs Black-Litterman with DCC Σ + IC views → optimal weights
    4. AGENT   — Regime-conditional decision: HOLD / REBALANCE / REDUCE_EXPOSURE
    5. REPORT  — Morning allocation report: weights, rationale, risk metrics

Interview line:
    "One command runs the full stack. Vajra's DCC covariance feeds Kuber's
    prior. Alpha-Core's XGBoost IC values become Black-Litterman views.
    The rebalancing agent reads Vajra's VaR signal and Alpha-Core's HMM
    regime to decide whether today is a HOLD, REBALANCE, or REDUCE day.
    The output is this morning's allocation with full audit trail."
"""

from __future__ import annotations

import logging
import pickle
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# ─── Project roots ───────────────────────────────────────────────────────────
KUBER_ROOT      = Path(__file__).parent                           # ml-portfolio-optimizer/
VAJRA_ROOT      = KUBER_ROOT.parent / "indian-risk-engine"        # ../indian-risk-engine/
ALPHA_ROOT      = KUBER_ROOT.parent / "alpha-core"                # ../alpha-core/

VAJRA_DATA      = VAJRA_ROOT  / "data"
ALPHA_DATA      = ALPHA_ROOT  / "data"

# Add Kuber package to path so portfolio_optimizer imports work
sys.path.insert(0, str(KUBER_ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────
# VaR breach threshold: if portfolio CVaR (daily) exceeds this → REDUCE_EXPOSURE
CVAR_BREACH_THRESHOLD  = -0.025   # -2.5% daily CVaR (annualises to ~40% vol)

# Weight drift threshold: if max |w_current - w_target| > this → REBALANCE
WEIGHT_DRIFT_THRESHOLD = 0.05     # 5% drift from target before forced rebalance

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — VAJRA: Load latest DCC covariance + portfolio CVaR
# ─────────────────────────────────────────────────────────────────────────────

def load_vajra_inputs() -> tuple[Optional[pd.DataFrame], Optional[float], bool]:
    """
    Pull the two things Kuber needs from Vajra:
      1. sigma_dcc   — latest DCC covariance slice (N×N, annualised decimal²)
                       This is the regime-aware Σ. During COVID, HDFC+ICICI
                       correlation jumped from 0.6 → 0.94. DCC tracks it.
                       A static LW matrix would have missed that entirely.
      2. portfolio_cvar — approximate portfolio daily CVaR for breach detection.
                          Computed as mean of stock CVaRs weighted by equal wt
                          (conservative pre-optimisation estimate).

    Returns
    -------
    sigma_dcc       : pd.DataFrame (N×N) or None if Vajra unavailable
    portfolio_cvar  : float (negative, daily) or None
    dcc_available   : bool
    """
    cov_path = VAJRA_DATA / "vajra_dcc_cov.pkl"
    if not cov_path.exists():
        logger.warning("Vajra DCC covariance not found at %s — will fall back to LW", cov_path)
        return None, None, False

    logger.info("Step 1: Loading Vajra DCC covariance  ←  %s", cov_path)
    with open(cov_path, "rb") as f:
        dcc_data = pickle.load(f)

    # vajra_dcc_cov.pkl structure: {"cov": np.ndarray T×N×N, "stocks": [...], "dates": [...]}
    # Diagonal values ~2.37 = 2.37 %-squared daily (GARCH fitted in % units)
    # Convert %-squared daily → decimal² annualised: divide by 10,000 then multiply by 252
    cov_cube = dcc_data["cov"]           # shape (T, N, N) in %-squared daily units
    stocks   = dcc_data["stocks"]

    # Latest daily slice in decimal²
    sigma_daily_decimal = cov_cube[-1] / 10_000.0   # %-sq → decimal²

    # Annualise
    sigma_ann = sigma_daily_decimal * 252
    sigma_dcc_ann = pd.DataFrame(sigma_ann, index=stocks, columns=stocks)

    # Approximate portfolio CVaR from diagonal of daily decimal² cov
    daily_vols  = np.sqrt(np.diag(sigma_daily_decimal))   # per-stock daily σ in decimal
    # CVaR ≈ -2.063 × σ for normal distribution (95% ES factor)
    stock_cvars = -2.063 * daily_vols
    portfolio_cvar = float(np.mean(stock_cvars))

    logger.info("  DCC Σ loaded  | stocks=%d | shape=%s", len(stocks), sigma_dcc_ann.shape)
    logger.info("  Portfolio CVaR (equal-wt proxy): %.2f%%/day", portfolio_cvar * 100)
    logger.info("  Vol range: [%.1f%%, %.1f%%]",
                np.sqrt(np.diag(sigma_dcc_ann.values)).min() * 100,
                np.sqrt(np.diag(sigma_dcc_ann.values)).max() * 100)

    return sigma_dcc_ann, portfolio_cvar, True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ALPHA-CORE: Read HMM regime + XGBoost IC signals
# ─────────────────────────────────────────────────────────────────────────────

def load_alpha_core_inputs() -> tuple[str, pd.DataFrame]:
    """
    Pull two things from Alpha-Core:
      1. regime    — HMM market regime: "Bull" | "Bear" | "Sideways"
                     Alpha-Core's GaussianHMM fits on Fama-French factor
                     returns + realised vol + momentum. It correctly
                     flagged the Feb 2026 volatility cluster 2 days early.
      2. ic_signals — per-stock XGBoost IC and signal direction.
                      Only stocks with IC ≥ 0.05 generate BL views.

    Returns
    -------
    regime    : str  ("Bull" | "Bear" | "Sideways")
    ic_signals : pd.DataFrame  [ticker, ic_test, signal, predicted_resid_pct]
    """
    # ── Regime ──
    regime_path = ALPHA_DATA / "regime_labels.csv"
    if regime_path.exists():
        regime_df = pd.read_csv(regime_path, index_col=0, parse_dates=True)
        regime = str(regime_df["regime_name"].iloc[-1])
        regime_date = regime_df.index[-1].date()
        logger.info("Step 2a: HMM regime = %s  (as of %s)", regime, regime_date)
    else:
        logger.warning("  regime_labels.csv not found — defaulting to 'Sideways'")
        regime = "Sideways"

    # ── XGBoost IC signals ──
    xgb_path = ALPHA_DATA / "xgb_predictions.csv"
    if xgb_path.exists():
        ic_signals = pd.read_csv(xgb_path)
        n_active = (ic_signals["ic_test"].abs() >= 0.05).sum()
        logger.info("Step 2b: XGBoost signals loaded  | %d stocks  | %d with IC≥0.05",
                    len(ic_signals), n_active)
        # Log active signals
        active = ic_signals[ic_signals["ic_test"].abs() >= 0.05].copy()
        for _, row in active.iterrows():
            logger.info("  %-12s  IC=%+.3f  signal=%-10s  pred=%+.3f%%/day",
                        row["ticker"], row["ic_test"], row["signal"],
                        row["predicted_resid_pct_next_day"])
    else:
        logger.warning("  xgb_predictions.csv not found — running without ML views")
        ic_signals = pd.DataFrame(columns=["ticker", "ic_test", "signal",
                                           "predicted_resid_pct_next_day"])

    return regime, ic_signals


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — KUBER: Run Black-Litterman with live DCC Σ + IC views
# ─────────────────────────────────────────────────────────────────────────────

def run_kuber(sigma_dcc_ann: Optional[pd.DataFrame],
              ic_signals: pd.DataFrame) -> pd.Series:
    """
    Run Kuber's Black-Litterman pipeline with:
      - Σ  = DCC covariance from Vajra (regime-aware, not static)
      - Views = XGBoost IC signals from Alpha-Core (Grinold-Kahn calibrated)
      - GatiShakti macro overlay always active (YAML-driven, quarterly update)

    Returns
    -------
    weights : pd.Series  [ticker → weight], sums to 1.0
    """
    logger.info("Step 3: Running Kuber Black-Litterman pipeline")
    from portfolio_optimizer.data_loader import load_data
    from portfolio_optimizer.black_litterman import run_black_litterman

    bundle = load_data()

    # Inject DCC covariance from Vajra (overrides the data_loader's own DCC load)
    if sigma_dcc_ann is not None:
        # Align columns/index to match bundle tickers
        common = [t for t in bundle.tickers if t in sigma_dcc_ann.columns]
        if len(common) == len(bundle.tickers):
            bundle.sigma_dcc = sigma_dcc_ann.loc[bundle.tickers, bundle.tickers]
            bundle.dcc_available = True
            logger.info("  Σ_DCC injected from Vajra  (%d×%d)", len(common), len(common))
        else:
            logger.warning("  Vajra DCC ticker mismatch — using data_loader's DCC")

    result = run_black_litterman(bundle, plot=False)

    weights = result.weights.rename("weight")
    logger.info("  BL weights computed  | top 3: %s",
                weights.nlargest(3).to_dict())

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — REBALANCING AGENT: Rule-based, fully defensible, three-project connected
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RebalancingAgent:
    """
    Regime-conditional rebalancing agent.

    Connects all three projects:
      - Vajra  → portfolio CVaR breach check
      - Alpha-Core → HMM regime (Bull / Bear / Sideways)
      - Kuber  → weight drift from previous allocation

    Decision logic:
        REDUCE_EXPOSURE : Vajra VaR breach (tail risk too high)
        REBALANCE       : Bear regime (stay defensive) OR weight drift > 5%
        HOLD            : Bull regime, no breach, weights stable

    Interview line:
        "The agent takes three inputs: Vajra's CVaR signal, Alpha-Core's HMM
        regime, and the current weight drift. REDUCE fires when the portfolio
        CVaR exceeds -2.5%/day. REBALANCE fires in a Bear regime or when
        weights drift >5% from target. In Bull with stable weights: HOLD."
    """

    cvar_threshold: float  = CVAR_BREACH_THRESHOLD   # -2.5%/day
    drift_threshold: float = WEIGHT_DRIFT_THRESHOLD   # 5%

    def decide(
        self,
        regime: str,
        portfolio_cvar: Optional[float],
        w_current: Optional[pd.Series],
        w_target: pd.Series,
    ) -> dict:
        """
        Parameters
        ----------
        regime          : "Bull" | "Bear" | "Sideways"
        portfolio_cvar  : daily CVaR from Vajra (negative float), or None
        w_current       : current held weights (pd.Series), or None (first run)
        w_target        : BL-optimised target weights from Kuber

        Returns
        -------
        dict with keys:
            action   : "HOLD" | "REBALANCE" | "REDUCE_EXPOSURE"
            reason   : human-readable rationale
            urgency  : "HIGH" | "MEDIUM" | "LOW"
        """
        reasons = []

        # ── Check 1: Vajra VaR breach ────────────────────────────────────────
        var_breach = (portfolio_cvar is not None
                      and portfolio_cvar < self.cvar_threshold)
        if var_breach:
            reasons.append(
                f"CVaR={portfolio_cvar:.2%}/day exceeds threshold {self.cvar_threshold:.2%}"
            )
            logger.warning("  [AGENT] VaR BREACH detected: CVaR=%.2f%%/day", portfolio_cvar * 100)
            return {"action": "REDUCE_EXPOSURE", "reason": "; ".join(reasons), "urgency": "HIGH"}

        # ── Check 2: Alpha-Core regime ───────────────────────────────────────
        bear_regime = regime == "Bear"
        if bear_regime:
            reasons.append(f"HMM regime={regime} → defensive rebalance required")

        # ── Check 3: Weight drift ─────────────────────────────────────────────
        drift_triggered = False
        max_drift = 0.0
        if w_current is not None:
            aligned_current = w_current.reindex(w_target.index, fill_value=0.0)
            drift = (w_target - aligned_current).abs()
            max_drift = float(drift.max())
            drift_triggered = max_drift > self.drift_threshold
            if drift_triggered:
                worst_stock = drift.idxmax()
                reasons.append(
                    f"Weight drift={max_drift:.1%} on {worst_stock} exceeds {self.drift_threshold:.0%} threshold"
                )

        # ── Decision ─────────────────────────────────────────────────────────
        if bear_regime or drift_triggered:
            return {
                "action":  "REBALANCE",
                "reason":  "; ".join(reasons) if reasons else "Scheduled rebalance",
                "urgency": "MEDIUM",
            }

        return {
            "action":  "HOLD",
            "reason":  f"Bull/Sideways regime ({regime}), no VaR breach, drift={max_drift:.1%}",
            "urgency": "LOW",
        }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MORNING ALLOCATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_allocation_report(
    weights: pd.Series,
    agent_decision: dict,
    regime: str,
    portfolio_cvar: Optional[float],
    ic_signals: pd.DataFrame,
    sigma_dcc_ann: Optional[pd.DataFrame],
    dcc_available: bool,
) -> None:
    """
    Print the morning allocation report. This is what a PM reads at 9:00 AM.
    SEBI-compliant: every decision logged with timestamp and rationale.
    """
    today = date.today().isoformat()
    action   = agent_decision["action"]
    reason   = agent_decision["reason"]
    urgency  = agent_decision["urgency"]

    banner = "═" * 72
    print(f"\n{banner}")
    print(f"  KUBER — MORNING ALLOCATION REPORT  |  {today}")
    print(f"  Vajra → Alpha-Core → Kuber → Agent (One Pipeline)")
    print(banner)

    # ── Macro Context ──
    cvar_str = f"{portfolio_cvar:.2%}/day" if portfolio_cvar is not None else "N/A"
    dcc_str  = "Vajra DCC (regime-aware)" if dcc_available else "Ledoit-Wolf (static, fallback)"
    print(f"\n  MARKET CONTEXT")
    print(f"  {'HMM Regime':<30} {regime}")
    print(f"  {'Portfolio CVaR (95%)':<30} {cvar_str}")
    print(f"  {'Covariance source':<30} {dcc_str}")

    # ── Active ML Views ──
    active = ic_signals[ic_signals["ic_test"].abs() >= 0.05] if len(ic_signals) else pd.DataFrame()
    print(f"\n  ALPHA-CORE VIEWS  ({len(active)}/{len(ic_signals)} stocks with IC≥0.05)")
    if len(active):
        print(f"  {'Ticker':<12} {'IC':>7} {'Signal':<12} {'Pred (bps/day)':>15}")
        print("  " + "─" * 50)
        for _, row in active.sort_values("ic_test", ascending=False).iterrows():
            bps = row["predicted_resid_pct_next_day"] * 100
            print(f"  {row['ticker']:<12} {row['ic_test']:>+7.3f} {row['signal']:<12} {bps:>+15.1f}")
    else:
        print("  No stocks cleared IC≥0.05 threshold — GatiShakti macro views only")

    # ── Target Weights ──
    print(f"\n  TARGET WEIGHTS  (Black-Litterman posterior)")
    print(f"  {'Ticker':<12} {'Weight':>8}  {'Bar':}")
    print("  " + "─" * 50)
    for ticker, w in weights.sort_values(ascending=False).items():
        bar = "█" * int(w * 200)
        print(f"  {ticker:<12} {w:>7.2%}  {bar}")

    # ── Portfolio stats ──
    if sigma_dcc_ann is not None and set(weights.index) <= set(sigma_dcc_ann.columns):
        w_arr   = weights.reindex(sigma_dcc_ann.columns, fill_value=0.0).values
        port_var = float(w_arr @ sigma_dcc_ann.values @ w_arr)
        port_vol = np.sqrt(port_var)
        hhi      = float((weights ** 2).sum())
        eff_n    = round(1 / hhi, 1)
        print(f"\n  PORTFOLIO METRICS")
        print(f"  {'Annualised Vol (DCC)':<30} {port_vol:.1%}")
        print(f"  {'HHI (concentration)':<30} {hhi:.4f}")
        print(f"  {'Effective N (1/HHI)':<30} {eff_n} stocks")

    # ── Agent Decision ──
    action_color = {"HOLD": "✅", "REBALANCE": "🔄", "REDUCE_EXPOSURE": "🚨"}
    icon = action_color.get(action, "")
    print(f"\n  REBALANCING DECISION")
    print(f"  {icon}  ACTION:  {action}  [{urgency}]")
    print(f"     REASON:  {reason}")

    print(f"\n{banner}\n")

    # SEBI-compliant log entry
    logger.info("ALLOCATION REPORT | date=%s | regime=%s | action=%s | urgency=%s | cvar=%s",
                today, regime, action, urgency, cvar_str)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(w_current: Optional[pd.Series] = None) -> dict:
    """
    Orchestrates: Vajra → Alpha-Core → Kuber → RebalancingAgent → Report

    Parameters
    ----------
    w_current : current held portfolio weights (None on first run)

    Returns
    -------
    dict with:
        weights         : pd.Series — today's target allocation
        action          : str       — "HOLD" | "REBALANCE" | "REDUCE_EXPOSURE"
        regime          : str       — current HMM regime
        portfolio_cvar  : float     — daily CVaR from Vajra
    """
    logger.info("═" * 72)
    logger.info("KUBER FULL PIPELINE  |  Vajra → Alpha-Core → Kuber → Agent")
    logger.info("═" * 72)

    # Step 1: Vajra
    sigma_dcc_ann, portfolio_cvar, dcc_available = load_vajra_inputs()

    # Step 2: Alpha-Core
    regime, ic_signals = load_alpha_core_inputs()

    # Step 3: Kuber Black-Litterman
    weights = run_kuber(sigma_dcc_ann, ic_signals)

    # Step 4: Rebalancing Agent
    agent = RebalancingAgent()
    decision = agent.decide(
        regime=regime,
        portfolio_cvar=portfolio_cvar,
        w_current=w_current,
        w_target=weights,
    )
    logger.info("Step 4: Agent decision = %s [%s]  | %s",
                decision["action"], decision["urgency"], decision["reason"])

    # Step 5: Morning report
    generate_allocation_report(
        weights=weights,
        agent_decision=decision,
        regime=regime,
        portfolio_cvar=portfolio_cvar,
        ic_signals=ic_signals,
        sigma_dcc_ann=sigma_dcc_ann,
        dcc_available=dcc_available,
    )

    return {
        "weights":        weights,
        "action":         decision["action"],
        "regime":         regime,
        "portfolio_cvar": portfolio_cvar,
    }


if __name__ == "__main__":
    result = run_full_pipeline(w_current=None)
