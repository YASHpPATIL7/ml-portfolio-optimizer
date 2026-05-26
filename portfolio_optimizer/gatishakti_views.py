"""
================================================================================
K6 — PM GATISHAKTI MACRO VIEW OVERLAY
ml-portfolio-optimizer/portfolio_optimizer/gatishakti_views.py

MULTI-QUARTER SUPPORT:
  YAML now stores one block per fiscal quarter, each with a valid_from date.
  get_active_quarter(as_of_date) returns the most recent quarter whose
  valid_from <= as_of_date. This ensures no future views leak into backtests.

WHAT:
  Converts government capex/sector data into Black-Litterman macro views.
  Reads from gatishakti_views.yaml — updated quarterly, no code change needed.

WHY:
  XGBoost (K4/K5) captures idiosyncratic, quantitative signals.
  GatiShakti captures macro, policy-driven signals — a DIFFERENT information set.
  BL allows combining both: P_combined = [P_ml; P_macro], Q and Ω stacked.
  This is exactly what discretionary quant PMs do: systematic + macro overlay.

HOW — 3 steps:
  1. Load YAML config (sector → tickers → view_bps → confidence)
  2. Build relative views for multi-ticker sectors:
       "Energy outperforms Financials by X%"
       P row: +1/n_long for LONG tickers, -1/n_short for SHORT tickers
  3. Build absolute views for single-ticker sectors:
       P row: identity row for that ticker
  Ω_kk = (1 - confidence) × τ × P_k Σ P_k'  (Idzorek form)

VIEW TYPES:
  Absolute view:  "ONGC will return X% above its equilibrium"
  Relative view:  "Energy stocks will outperform Financials by X%"
  We use ABSOLUTE for single-ticker sectors, RELATIVE for multi-ticker.
  Relative views are more realistic — you're expressing sector rotation,
  not predicting absolute returns.

INTERVIEW:
  Q: "How do you combine macro and ML views?"
  A: "I stack them. P = [P_ml; P_gatishakti]. Q and Ω are vertically
     concatenated. BL doesn't care where the views come from — it just
     needs P, Q, Ω. The Ω for macro views is set by my confidence score
     in the YAML — 0.80 for hard budget data, 0.60 for soft inference.
     Both view sets update μ_BL simultaneously in one Bayesian step."

  Q: "Why GatiShakti specifically?"
  A: "It's public, quarterly, gazette-notified. Sector capex allocation
     data from gstshakti.gov.in is granular to project level. It's the
     same data that sell-side sector analysts use — I'm just automating
     the translation into a quantitative view vector."
================================================================================
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_HERE     = Path(__file__).resolve().parent
_YAML     = _HERE / "gatishakti_views.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MacroView:
    sector:     str
    tickers:    list        # tickers in this sector (in our universe)
    view_bps:   float       # annualised outperformance in basis points
    confidence: float       # 0-1 confidence score → drives Ω
    rationale:  str

    @property
    def view_decimal(self) -> float:
        """view_bps → annual decimal (100bps = 1%)"""
        return self.view_bps / 10_000.0

    @property
    def is_active(self) -> bool:
        return self.view_bps != 0 and len(self.tickers) > 0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD YAML
# ─────────────────────────────────────────────────────────────────────────────

def load_gatishakti_config(yaml_path: Optional[Path] = None) -> dict:
    """Load the full GatiShakti YAML (multi-quarter format)."""
    path = yaml_path or _YAML
    if not path.exists():
        logger.warning("  gatishakti_views.yaml not found at %s", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def get_active_quarter(cfg: dict,
                        as_of_date: Optional[pd.Timestamp] = None) -> Optional[dict]:
    """
    Return the most recent quarter whose valid_from <= as_of_date.
    This is the core time-gating function — ensures no future views
    contaminate the backtest.

    Parameters
    ----------
    cfg         : raw parsed YAML dict
    as_of_date  : the rebalance date in the backtest (or today if None)

    Returns
    -------
    The quarter dict, or None if no quarter is valid yet at as_of_date.
    """
    as_of = as_of_date or pd.Timestamp.today()
    quarters = cfg.get("quarters", [])

    valid = [
        q for q in quarters
        if pd.Timestamp(q["valid_from"]) <= as_of
    ]
    if not valid:
        return None

    # Pick the most recent valid quarter
    return max(valid, key=lambda q: pd.Timestamp(q["valid_from"]))


def get_xgb_valid_from(cfg: dict) -> pd.Timestamp:
    """Return the XGBoost model's valid_from date from YAML config."""
    raw = cfg.get("xgb_valid_from", "2099-01-01")  # default: far future (never valid)
    return pd.Timestamp(raw)


def refresh_gatishakti_views(tickers: list,
                              yaml_path: Optional[Path] = None,
                              as_of_date: Optional[pd.Timestamp] = None) -> list:
    """
    Parse YAML config into MacroView objects for the active quarter.

    Parameters
    ----------
    tickers     : stock tickers in the portfolio universe
    yaml_path   : optional override path to YAML
    as_of_date  : rebalance date — only views valid at this date are returned.
                  None = use today (live mode).

    Returns
    -------
    List[MacroView] for the active quarter, filtered to universe tickers.
    Empty list if no quarter is valid yet at as_of_date.
    """
    cfg = load_gatishakti_config(yaml_path)
    if not cfg:
        return []

    quarter = get_active_quarter(cfg, as_of_date)
    if quarter is None:
        logger.info("  No GatiShakti quarter valid at %s — no macro views",
                    (as_of_date or pd.Timestamp.today()).date())
        return []

    logger.info("  GatiShakti active quarter: %s (valid_from=%s)",
                quarter["quarter"], quarter["valid_from"])
    logger.info("  Source: %s", quarter.get("source", ""))

    views = []
    for sec in quarter.get("sectors", []):
        active_tickers = [t for t in sec.get("tickers", []) if t in tickers]
        if not active_tickers:
            continue
        mv = MacroView(
            sector=sec["name"],
            tickers=active_tickers,
            view_bps=float(sec.get("view_bps", 0)),
            confidence=float(sec.get("confidence", 0.5)),
            rationale=sec.get("rationale", "").strip(),
        )
        views.append(mv)
        status = "ACTIVE" if mv.is_active else "NEUTRAL"
        logger.info("  %-30s  %+5d bps  conf=%.2f  [%s]",
                    mv.sector, mv.view_bps, mv.confidence, status)

    return views


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD P, Q, Ω FROM MACRO VIEWS
# ─────────────────────────────────────────────────────────────────────────────

def build_macro_view_matrices(views: list,
                               tickers: list,
                               sigma_lw: np.ndarray,
                               pi: np.ndarray,
                               tau: float = 0.50) -> tuple:
    """
    Convert MacroView list into BL triplet (P, Q, Ω).

    VIEW CONSTRUCTION:
    Single ticker:
        P[k, i] = 1  (absolute view — stock i expected to earn Q_k)
        Q_k = Π_i + view_decimal  (outperform/underperform equilibrium by X%)

    Multi-ticker sector (relative sector view):
        Long leg:  P[k, i] = +1/n_long  for each ticker in the sector
        Short leg: None (we use absolute views vs equilibrium, not sector rotation)

        WHY: We don't have opposing short sectors in the YAML. Each sector
        is an absolute statement: "Energy will return Π_energy + 1.5%".
        Relative views require a paired short sector — too rigid for quarterly
        YAML updates. Absolute multi-stock views are simpler and defensible.

        Q_k = average(Π_i for i in sector) + view_decimal
        P   = row of 1/n per ticker (equal-weight within sector)

    Ω (Idzorek form):
        p_k = confidence score from YAML (0-1)
        Ω_kk = ((1 - p_k) / p_k) × τ × (P_k Σ P_k')
        High confidence → small Ω → view penetrates prior more

    Parameters
    ----------
    views      : list of MacroView objects (from refresh_gatishakti_views)
    tickers    : universe ticker list
    sigma_lw   : (N×N) Ledoit-Wolf covariance matrix (annualised, decimal²)
    pi         : (N,) equilibrium returns (annualised, decimal)
    tau        : prior uncertainty scale (matches black_litterman.py TAU)

    Returns
    -------
    P     : (K, N) pick matrix
    Q     : (K,)  view return vector
    Omega : (K, K) diagonal uncertainty matrix
    labels: list of K view labels (for display)
    """
    P_rows, Q_vals, omega_diag, labels = [], [], [], []

    active = [v for v in views if v.is_active]

    for mv in active:
        n = len(mv.tickers)
        idx = [tickers.index(t) for t in mv.tickers]

        # ── P row ────────────────────────────────────────────────────────
        p_row = np.zeros(len(tickers))
        p_row[idx] = 1.0 / n   # equal weight within sector

        # ── Q ────────────────────────────────────────────────────────────
        # Sector average equilibrium return + view magnitude
        avg_pi = np.mean(pi[idx])
        q_k = avg_pi + mv.view_decimal

        # ── Ω (Idzorek) ──────────────────────────────────────────────────
        # P_k × Σ × P_k' = weighted sector variance
        p_sigma_p = float(p_row @ sigma_lw @ p_row)
        c = mv.confidence
        omega_k = ((1 - c) / max(c, 1e-6)) * tau * p_sigma_p

        P_rows.append(p_row)
        Q_vals.append(q_k)
        omega_diag.append(omega_k)
        labels.append(mv.sector)

        direction = "LONG" if mv.view_bps > 0 else "SHORT"
        logger.info(
            "  %-30s  Π_avg=%+.2f%%  view=%+dbps  Q=%+.2f%%  Ω=%.4f  [%s]",
            mv.sector, avg_pi * 100, mv.view_bps,
            q_k * 100, omega_k, direction
        )

    if not P_rows:
        n = len(tickers)
        return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0)), []

    P     = np.array(P_rows)
    Q     = np.array(Q_vals)
    Omega = np.diag(omega_diag)

    return P, Q, Omega, labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — COMBINE ML VIEWS + MACRO VIEWS
# ─────────────────────────────────────────────────────────────────────────────

def combine_views(P_ml: np.ndarray, Q_ml: np.ndarray, Omega_ml: np.ndarray,
                  P_macro: np.ndarray, Q_macro: np.ndarray,
                  Omega_macro: np.ndarray) -> tuple:
    """
    Stack ML views (K4/K5) and GatiShakti macro views (K6) into one triplet.

    P_combined = [P_ml    ]     (K_ml + K_macro) × N
                 [P_macro ]

    Q_combined = [Q_ml    ]     (K_ml + K_macro,)
                 [Q_macro ]

    Ω_combined = block_diag(Ω_ml, Ω_macro)

    WHY block diagonal Ω:
      Views are assumed independent — ML signals are idiosyncratic (per-stock
      residuals), GatiShakti signals are macro (sector-level). They capture
      orthogonal information. Off-diagonal Ω elements would imply correlation
      between view errors, which we have no basis to estimate.

    Returns (P, Q, Ω) combined matrices ready for BL posterior computation.
    """
    if P_ml.shape[0] == 0 and P_macro.shape[0] == 0:
        n = P_ml.shape[1] if P_ml.shape[0] == 0 else P_macro.shape[1]
        return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0))

    if P_ml.shape[0] == 0:
        return P_macro, Q_macro, Omega_macro
    if P_macro.shape[0] == 0:
        return P_ml, Q_ml, Omega_ml

    P     = np.vstack([P_ml, P_macro])
    Q     = np.concatenate([Q_ml, Q_macro])
    Omega = np.block([
        [Omega_ml,                          np.zeros((len(Q_ml), len(Q_macro)))],
        [np.zeros((len(Q_macro), len(Q_ml))), Omega_macro                      ]
    ])
    return P, Q, Omega


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_gatishakti(tickers: list,
                   sigma_lw: np.ndarray,
                   pi: np.ndarray,
                   tau: float = 0.50,
                   as_of_date: Optional[pd.Timestamp] = None) -> tuple:
    """
    Full K6 pipeline: load YAML → pick active quarter → build (P, Q, Ω).

    Parameters
    ----------
    as_of_date : rebalance date for backtest time-gating (None = today)

    Returns
    -------
    P, Q, Omega, labels, views  —  ready to feed into BL or combine_views()
    """
    logger.info("=" * 60)
    logger.info("K6 GATISHAKTI MACRO VIEW OVERLAY")
    logger.info("=" * 60)

    views = refresh_gatishakti_views(tickers, as_of_date=as_of_date)
    active = [v for v in views if v.is_active]
    logger.info("")
    logger.info("  Active sectors: %d / %d total", len(active), len(views))
    logger.info("")

    logger.info("  Building P, Q, Ω matrices...")
    P, Q, Omega, labels = build_macro_view_matrices(
        views, tickers, sigma_lw, pi, tau)
    logger.info("")
    logger.info("  Macro views: K=%d, N=%d", P.shape[0], len(tickers))
    logger.info("=" * 60)

    return P, Q, Omega, labels, views


def print_gatishakti(views: list, labels: list,
                     Q: np.ndarray, pi: np.ndarray, tickers: list):
    """Formatted terminal output of GatiShakti view summary."""
    print(f"\n{'═'*66}")
    print(f" K6 GATISHAKTI MACRO VIEW OVERLAY")
    print(f"{'═'*66}")
    print(f" {'Sector':<32} {'View':>8} {'Conf':>6} {'Tickers'}")
    print(f"{'─'*66}")
    for i, v in enumerate(views):
        if not v.is_active:
            continue
        direction = f"+{v.view_bps}bps" if v.view_bps > 0 else f"{v.view_bps}bps"
        print(f" {v.sector:<32} {direction:>8} {v.confidence:>5.0%}  "
              f"{', '.join(v.tickers)}")
    print(f"{'─'*66}")
    n_active = sum(1 for v in views if v.is_active)
    print(f" {n_active} active sector views  |  "
          f"YAML: Q{Path(_YAML).name}")
    print(f"{'═'*66}")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from portfolio_optimizer.data_loader    import load_data
    from portfolio_optimizer.black_litterman import (
        compute_equilibrium, build_views_from_xgboost,
        black_litterman_posterior, _max_sharpe_bl,
        print_bl, BLResult, TAU, RISK_FREE, W_MAX
    )
    import warnings as _w; _w.filterwarnings("ignore")

    bundle = load_data()
    tickers = bundle.tickers
    sigma   = bundle.sigma_dcc.values
    sigma_lw = bundle.sigma_lw.values

    # K4 equilibrium
    pi_series = compute_equilibrium(bundle)
    pi        = pi_series.values

    # K4/K5 ML views
    P_ml, Q_ml, Omega_ml, active_ml = build_views_from_xgboost(
        tickers, pi, sigma_lw)

    # K6 GatiShakti macro views
    P_gs, Q_gs, Omega_gs, labels_gs, views_gs = run_gatishakti(
        tickers, sigma_lw, pi, TAU)

    # Combine
    P_all, Q_all, Omega_all = combine_views(
        P_ml, Q_ml, Omega_ml,
        P_gs, Q_gs, Omega_gs
    )

    print(f"\n  Combined views: {len(Q_ml)} ML + {len(Q_gs)} GatiShakti "
          f"= {len(Q_all)} total")

    # BL posterior on combined views
    mu_bl_arr, sigma_bl_arr = black_litterman_posterior(
        pi, sigma, P_all, Q_all, Omega_all)

    mu_bl    = pd.Series(mu_bl_arr, index=tickers)
    sigma_bl = pd.DataFrame(sigma_bl_arr, index=tickers, columns=tickers)
    w_opt    = _max_sharpe_bl(mu_bl_arr, sigma_bl_arr, tickers)
    ret      = float(w_opt.values @ mu_bl_arr)
    vol      = float(np.sqrt(w_opt.values @ sigma_bl_arr @ w_opt.values))
    sharpe   = (ret - RISK_FREE) / vol

    result = BLResult(
        pi_eq=pi_series, mu_bl=mu_bl, sigma_bl=sigma_bl,
        view_matrix_P=P_all, view_vector_Q=Q_all, view_omega=Omega_all,
        active_views=active_ml + labels_gs,
        weights=w_opt, ret_annual=ret, vol_annual=vol, sharpe=sharpe,
    )

    print_bl(result)
    print_gatishakti(views_gs, labels_gs, Q_gs, pi, tickers)

    print(f"\n── Combined BL View Impact (Π → μ_BL) ──")
    for t in tickers:
        delta = (mu_bl[t] - pi_series[t]) * 100
        if abs(delta) > 0.05:
            tag = " ◀ ML" if t in active_ml else " ◀ MACRO"
            print(f"  {t:<12}: Π={pi_series[t]*100:+.2f}%  →  "
                  f"μ_BL={mu_bl[t]*100:+.2f}%  (Δ={delta:+.2f}%){tag}")
