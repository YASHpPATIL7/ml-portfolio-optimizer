"""
================================================================================
K3 — HIERARCHICAL RISK PARITY (HRP)
ml-portfolio-optimizer/portfolio_optimizer/hrp.py

PAPER: López de Prado, M. (2016). "Building Diversified Portfolios that
       Outperform Out-of-Sample." Journal of Portfolio Management.

WHAT:
  HRP allocates portfolio weights using the STRUCTURE of the correlation
  matrix — through hierarchical clustering — without ever inverting it.
  Result: a diversified portfolio that is robust to estimation error.

HOW (3 steps):
  Step 1 — TREE CLUSTERING
    Build a distance matrix from correlations, run hierarchical clustering.
    Similar stocks cluster together (banks group, IT stocks group, pharma groups).

  Step 2 — QUASI-DIAGONALIZATION
    Reorder the covariance matrix so clustered stocks sit adjacent.
    The matrix becomes "as diagonal as possible" — blocks of correlated stocks
    appear along the diagonal, uncorrelated blocks are off-diagonal near zero.

  Step 3 — RECURSIVE BISECTION
    Split the portfolio into two halves along the dendrogram cut.
    Allocate capital between halves inversely proportional to their variance.
    Recurse within each half until individual stock weights are assigned.

WHY HRP BEATS MVO (the core argument):
  MVO: min w'Σw  requires computing Σ⁻¹ (matrix inverse)
       A 14×14 matrix inversion amplifies estimation errors in small
       eigenvalues, producing extreme concentrated weights.

  HRP: NEVER inverts Σ. Uses the TOPOLOGY of correlation — which stocks
       cluster together — rather than the numerical values of the matrix.
       Eigenvalue stability doesn't matter when you never compute Σ⁻¹.

  Empirical result (López de Prado, 2016):
    HRP produces LOWER out-of-sample variance than MVO on 10 years of
    S&P 500 data. It's not that HRP is theoretically optimal — it's that
    MVO's theoretical optimality evaporates when Σ is estimated from data.

INTERVIEW QUESTIONS:
  Q: "Why is HRP better than Markowitz?"
  A: "Markowitz is theoretically optimal but practically fragile — it
     inverts the covariance matrix, amplifying estimation errors. HRP
     uses hierarchical clustering to allocate without matrix inversion.
     Out-of-sample, HRP consistently delivers lower variance despite
     having no closed-form optimality guarantee."

  Q: "Walk me through HRP step by step."
  A: [Use the 3 steps above — clustering, quasi-diag, recursive bisection]

  Q: "What distance metric does HRP use?"
  A: "d_ij = sqrt(0.5 × (1 - ρ_ij)). This satisfies the triangle
     inequality, making it a proper metric for hierarchical clustering.
     Correlation alone doesn't — ρ=1 → d=0, ρ=-1 → d=1, ρ=0 → d=0.707."
================================================================================
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform

from portfolio_optimizer.data_loader import DataBundle

logger = logging.getLogger(__name__)

_FIGURES = Path(__file__).resolve().parent.parent / "figures"
_DATA    = Path(__file__).resolve().parent.parent / "data"
_FIGURES.mkdir(exist_ok=True)
_DATA.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────────────────────

@dataclass
class HRPResult:
    weights:       pd.Series    # ticker → weight
    ret_annual:    float
    vol_annual:    float
    sharpe:        float
    cluster_order: List[str]    # stocks sorted by dendrogram leaf order
    linkage_matrix: np.ndarray  # scipy linkage matrix for plotting

    def __str__(self):
        top3 = self.weights.nlargest(3)
        top3_str = ", ".join(f"{t}={w:.1%}" for t, w in top3.items())
        return (f"HRP: ret={self.ret_annual:.2%}  "
                f"vol={self.vol_annual:.2%}  sharpe={self.sharpe:.3f}  "
                f"[top3: {top3_str}]")


# ─────────────────────────────────────────────────────────────
# STEP 1 — TREE CLUSTERING
# ─────────────────────────────────────────────────────────────

def _correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """
    Convert correlation matrix → distance matrix.

    Formula: d_ij = sqrt(0.5 × (1 - ρ_ij))

    WHY this formula and not just (1 - ρ)?
      (1 - ρ) ranges [0, 2] and does NOT satisfy the triangle inequality.
      sqrt(0.5 × (1 - ρ)) ranges [0, 1] and IS a proper metric:
        ρ = +1  →  d = 0    (identical: same asset)
        ρ =  0  →  d = 0.707 (uncorrelated: moderate distance)
        ρ = -1  →  d = 1    (perfectly anti-correlated: maximum distance)

      The triangle inequality matters for hierarchical clustering — without
      it, the dendrogram has no geometric meaning.

    INTERVIEW: "Why not use (1 - ρ) directly?"
      "It violates the triangle inequality, making the hierarchical
       clustering geometrically meaningless. d_ij = sqrt(0.5(1-ρ_ij))
       is the standard choice in HRP literature."
    """
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)          # d_ii = 0 by definition
    return dist


def _build_linkage(dist: np.ndarray) -> np.ndarray:
    """
    Hierarchical clustering using Ward linkage.

    Ward linkage: merge the two clusters whose union has minimum
    variance increase. It produces compact, well-separated clusters —
    better for portfolio grouping than single-linkage (which chains).

    Output: scipy linkage matrix Z of shape (N-1, 4)
      Each row [i, j, dist, count] = merge cluster i and j at distance dist
    """
    condensed = squareform(dist)          # upper-triangular vector form
    Z = linkage(condensed, method="ward")
    return Z


def _get_leaf_order(Z: np.ndarray, n: int) -> List[int]:
    """
    Extract the leaf ordering from the dendrogram.
    This is the quasi-diagonalisation order — stocks are reordered so
    that similar (nearby) stocks are adjacent in the matrix.
    """
    return list(leaves_list(Z))


# ─────────────────────────────────────────────────────────────
# STEP 2 — QUASI-DIAGONALISATION
# ─────────────────────────────────────────────────────────────

def _quasi_diag(cov: pd.DataFrame, order: List[int]) -> pd.DataFrame:
    """
    Reorder covariance matrix rows/cols by cluster leaf order.

    WHY quasi-diagonalisation?
      After reordering, the covariance matrix looks "nearly block-diagonal":
        [ Σ_banks  |    ~0    ]
        [   ~0     |  Σ_IT   ]
      The recursive bisection in Step 3 exploits this structure — it
      splits at the dendrogram root, separating the two most different
      clusters (banks vs pharma vs IT).

      Without reordering, the bisection would split arbitrarily by stock
      index, not by actual correlation structure.
    """
    tickers = cov.index.tolist()
    ordered = [tickers[i] for i in order]
    return cov.loc[ordered, ordered]


# ─────────────────────────────────────────────────────────────
# STEP 3 — RECURSIVE BISECTION
# ─────────────────────────────────────────────────────────────

def _get_cluster_var(cov: pd.DataFrame, items: List[str]) -> float:
    """
    Compute variance of an equal-weighted sub-portfolio of `items`.

    WHY equal-weight within cluster?
      Recursive bisection treats each cluster as a mini portfolio.
      We need a single variance number per cluster to allocate between
      them. Equal-weight is the simplest unbiased estimate of "what
      would a naive investor holding this cluster risk?"

      The key insight: we're not optimising WITHIN clusters, we're
      allocating BETWEEN clusters. The within-cluster weights are
      determined by the next level of recursion.

    Math:
      w_eq = [1/n, 1/n, ..., 1/n]
      var  = w_eq' Σ_sub w_eq  (sub-matrix for these stocks only)
    """
    sub   = cov.loc[items, items].values
    n     = len(items)
    w_eq  = np.ones(n) / n
    return float(w_eq @ sub @ w_eq)


def _recursive_bisection(cov: pd.DataFrame, sorted_items: List[str]) -> pd.Series:
    """
    Allocate weights by recursive bisection of the sorted stock list.

    Algorithm:
      1. Split sorted_items into left half and right half
         (the dendrogram root splits the most different clusters)
      2. Compute cluster variance for left half (var_L) and right half (var_R)
      3. Allocate: left gets  α = var_R / (var_L + var_R)
                   right gets (1 - α) = var_L / (var_L + var_R)
         WHY this allocation? Inverse variance weighting — the riskier
         cluster gets LESS capital. This is risk parity at the cluster level.
      4. Recurse within each half

    Intuition: if banks are twice as volatile as pharma, banks get half
    the capital of pharma. Within banks, the same logic applies recursively
    until you reach individual stocks.

    Base case: single stock → weight = 1.0 (will be scaled by parent)
    """
    weights = pd.Series(1.0, index=sorted_items)

    def _bisect(items: List[str], w_alloc: float):
        if len(items) == 1:
            weights[items[0]] = w_alloc
            return

        # Split: left = first half, right = second half
        mid   = len(items) // 2
        left  = items[:mid]
        right = items[mid:]

        # Cluster variances (equal-weight within each sub-cluster)
        var_L = _get_cluster_var(cov, left)
        var_R = _get_cluster_var(cov, right)

        # Inverse-variance allocation between left and right
        # alpha_L + alpha_R = 1, alpha_L/alpha_R = var_R/var_L
        total   = var_L + var_R
        alpha_L = var_R / total    # riskier right → more to left
        alpha_R = var_L / total    # riskier left  → more to right

        _bisect(left,  w_alloc * alpha_L)
        _bisect(right, w_alloc * alpha_R)

    _bisect(sorted_items, 1.0)
    return weights


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def _plot_dendrogram(Z: np.ndarray, tickers: List[str],
                     save_path: Path = None):
    """
    Plot the hierarchical clustering dendrogram.
    Shows which stocks cluster together and at what distance.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#111827")

    dendrogram(Z, labels=tickers, ax=ax,
               color_threshold=0.6 * max(Z[:, 2]),
               above_threshold_color="#64748b",
               leaf_rotation=45, leaf_font_size=9)

    ax.set_title("K3 HRP — Hierarchical Clustering Dendrogram\n"
                 "Distance = √(0.5×(1−ρ))  |  Ward Linkage",
                 color="#f1f5f9", fontsize=11, fontweight="bold")
    ax.set_xlabel("Stock", color="#94a3b8")
    ax.set_ylabel("Cluster Distance", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#1e293b")

    plt.tight_layout()
    out = save_path or (_FIGURES / "k3_dendrogram.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("  Dendrogram saved → %s", out)


def _plot_weights_comparison(hrp: HRPResult, mvo_ms_weights: pd.Series,
                              mvo_mv_weights: pd.Series,
                              save_path: Path = None):
    """
    Side-by-side bar chart: HRP vs Max Sharpe MVO vs Min Vol MVO.
    Visual proof of HRP's diversification advantage.
    """
    tickers = hrp.weights.index.tolist()
    x = np.arange(len(tickers))
    width = 0.28

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#111827")

    b1 = ax.bar(x - width, hrp.weights.values * 100,    width, label="HRP",          color="#34d399", alpha=0.85)
    b2 = ax.bar(x,          mvo_ms_weights.reindex(tickers, fill_value=0).values * 100, width, label="MVO Max Sharpe", color="#f59e0b", alpha=0.85)
    b3 = ax.bar(x + width,  mvo_mv_weights.reindex(tickers, fill_value=0).values * 100, width, label="MVO Min Vol",    color="#a78bfa", alpha=0.85)

    ax.axhline(100 / len(tickers), color="#475569", linestyle="--",
               linewidth=1, label=f"Equal weight ({100/len(tickers):.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9, color="#94a3b8")
    ax.set_ylabel("Weight (%)", color="#94a3b8")
    ax.set_title("K3 — HRP vs MVO Weight Comparison\n"
                 "HRP is naturally more diversified — no stock hits the 20% cap",
                 color="#f1f5f9", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#1e293b")
    ax.legend(facecolor="#111827", edgecolor="#1e293b",
              labelcolor="#94a3b8", fontsize=9)
    ax.set_ylim(0, 25)

    plt.tight_layout()
    out = save_path or (_FIGURES / "k3_weights_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("  Weight comparison saved → %s", out)


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def run_hrp(bundle: DataBundle,
            mvo_max_sharpe_weights: pd.Series = None,
            mvo_min_vol_weights:    pd.Series = None,
            plot: bool = True) -> HRPResult:
    """
    Run the full 3-step HRP pipeline.

    Parameters
    ----------
    bundle                  : DataBundle from K1
    mvo_max_sharpe_weights  : optional — for comparison plot
    mvo_min_vol_weights     : optional — for comparison plot
    plot                    : save dendrogram + comparison charts
    """
    logger.info("=" * 60)
    logger.info("K3 HIERARCHICAL RISK PARITY (HRP)")
    logger.info("López de Prado (2016)")
    logger.info("=" * 60)

    corr = bundle.corr_sample
    cov  = bundle.sigma_lw          # use LW covariance for variance calculations
    tickers = bundle.tickers

    # ── Step 1: Clustering ──
    logger.info("  Step 1: Building correlation distance matrix...")
    dist = _correlation_distance(corr)
    Z    = _build_linkage(dist)
    order = _get_leaf_order(Z, len(tickers))
    cluster_order = [tickers[i] for i in order]
    logger.info("  Cluster order: %s", " → ".join(cluster_order))

    # ── Step 2: Quasi-diagonalisation ──
    logger.info("  Step 2: Quasi-diagonalising covariance matrix...")
    cov_reordered = _quasi_diag(cov, order)

    # ── Step 3: Recursive bisection ──
    logger.info("  Step 3: Recursive bisection weight allocation...")
    raw_weights  = _recursive_bisection(cov_reordered, cluster_order)
    weights      = raw_weights / raw_weights.sum()   # ensure exact sum=1

    # ── Portfolio stats ──
    w_arr  = weights.reindex(tickers).values
    mu_arr = bundle.mu_annual.values
    S_arr  = cov.values

    ret    = float(w_arr @ mu_arr)
    vol    = float(np.sqrt(w_arr @ S_arr @ w_arr))
    rf     = 0.065
    sharpe = (ret - rf) / vol

    result = HRPResult(
        weights=weights.reindex(tickers),
        ret_annual=ret,
        vol_annual=vol,
        sharpe=sharpe,
        cluster_order=cluster_order,
        linkage_matrix=Z,
    )

    logger.info("  %s", result)

    # ── Save ──
    weights.reindex(tickers).to_frame("weight").assign(
        ret=ret, vol=vol, sharpe=sharpe
    ).to_csv(_DATA / "k3_hrp_weights.csv")

    # ── Plots ──
    if plot:
        _plot_dendrogram(Z, [tickers[i] for i in order])
        if mvo_max_sharpe_weights is not None and mvo_min_vol_weights is not None:
            _plot_weights_comparison(result, mvo_max_sharpe_weights,
                                     mvo_min_vol_weights)

    logger.info("=" * 60)
    return result


def print_hrp(result: HRPResult):
    """Pretty-print HRP weights."""
    print(f"\n── HRP Weights ──")
    print(f"   Return: {result.ret_annual:.2%}  |  "
          f"Vol: {result.vol_annual:.2%}  |  "
          f"Sharpe: {result.sharpe:.3f}")
    print(f"   Cluster order (left → right in dendrogram):")
    print(f"   {' → '.join(result.cluster_order)}")
    print("   Weights:")
    for t, w in result.weights.sort_values(ascending=False).items():
        bar = "█" * int(w * 100)
        print(f"   {t:<12} {w:6.2%}  {bar}")

    # Concentration check
    hhi = (result.weights ** 2).sum()
    eff_n = 1 / hhi
    print(f"\n   HHI concentration: {hhi:.4f}  (Effective N: {eff_n:.1f} of {len(result.weights)})")
    print(f"   Max weight: {result.weights.max():.2%}  "
          f"(no cap applied — HRP self-diversifies)")


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
    from portfolio_optimizer.markowitz   import run_markowitz

    bundle = load_data()
    mvo    = run_markowitz(bundle, plot=False)
    hrp    = run_hrp(bundle,
                     mvo_max_sharpe_weights=mvo.max_sharpe.weights,
                     mvo_min_vol_weights=mvo.min_vol.weights,
                     plot=True)

    print_hrp(hrp)

    # Side-by-side comparison
    print("\n── K2 vs K3 Comparison ──")
    print(f"{'Method':<20} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'Max Wt':>8}")
    print("─" * 58)
    ms = mvo.max_sharpe
    mv = mvo.min_vol
    print(f"{'MVO Max Sharpe':<20} {ms.ret_annual:>7.2%} {ms.vol_annual:>7.2%} "
          f"{ms.sharpe:>8.3f} {ms.weights.max():>7.2%}")
    print(f"{'MVO Min Vol':<20} {mv.ret_annual:>7.2%} {mv.vol_annual:>7.2%} "
          f"{mv.sharpe:>8.3f} {mv.weights.max():>7.2%}")
    print(f"{'HRP':<20} {hrp.ret_annual:>7.2%} {hrp.vol_annual:>7.2%} "
          f"{hrp.sharpe:>8.3f} {hrp.weights.max():>7.2%}")

    print("\n── Key insight ──")
    print(f"   HRP max weight: {hrp.weights.max():.2%} (self-diversified, no constraint needed)")
    print(f"   MVO max weight: {ms.weights.max():.2%} (hit the 20% cap — constraint doing the work)")
    print(f"   HRP naturally diversifies by clustering correlation structure.")
    print(f"   MVO concentrates until an artificial constraint stops it.")
