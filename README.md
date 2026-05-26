# Kuber — ML Portfolio Optimizer

> *"Vajra asks: How much can I lose? Alpha-Core asks: What should I trade? Kuber asks: How much of each, given my constraints?"*

**Kuber** is an institutional-grade portfolio construction engine that integrates dynamic risk modeling, machine learning signals, and macro policy overlays into a unified Black-Litterman allocation framework.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The 3-Project Pipeline

```
VAJRA (Indian Risk Engine)
  └── GARCH conditional volatility → DCC dynamic covariance matrix (Σ)
          ↓ feeds Kuber as covariance input

ALPHA-CORE (Signal Engine)
  └── Fama-French decomposition → HMM regime detection
      → XGBoost residual prediction → IC-ranked signals per stock
          ↓ feeds Kuber as Black-Litterman view vector (Q)

KUBER (Allocation Engine)
  └── Takes Σ from Vajra + Q from Alpha-Core
      → equilibrium prior Π = δΣw
      → Bayesian BL update: μ_BL = f(Π, Q, Ω)
      → MVO on posterior → optimal weights
      → Walk-forward backtest → PM morning report
```

**Interview line:** *"My three projects are not independent — they're a pipeline. Vajra's DCC covariance feeds Kuber. Alpha-Core's XGBoost predictions become Kuber's Black-Litterman views. One command runs the full stack."*

---

## Modules

| Module | File | Core concept |
|--------|------|-------------|
| **K1** | `portfolio_optimizer/data_loader.py` | Returns, Σ (Sample/LW/DCC), market cap weights |
| **K2** | `portfolio_optimizer/markowitz.py` | MVO: max Sharpe, min vol, efficient frontier |
| **K3** | `portfolio_optimizer/hrp.py` | Hierarchical Risk Parity — no Σ⁻¹ required |
| **K4+K5** | `portfolio_optimizer/black_litterman.py` | BL posterior + XGBoost auto-views (Grinold-Kahn + Idzorek Ω) |
| **K6** | `portfolio_optimizer/gatishakti_views.py` | GatiShakti macro overlay — YAML multi-quarter, time-gated |
| **K6b** | `portfolio_optimizer/finbert_views.py` | FinBERT sentiment → view_bps via tanh scaling |
| **K7** | `portfolio_optimizer/backtester.py` | Walk-forward backtest, 5 strategies, time-gated views |
| **K8** | `dashboard/app.py` | Streamlit 5-tab live dashboard |

---

## Key Results

### Walk-Forward Backtest (2021–2026, monthly rebalance)

| Strategy | Return | Vol | Sharpe | MaxDD | Turnover |
|----------|--------|-----|--------|-------|----------|
| Equal Weight | 8.65% | 12.83% | 0.167 | -18.3% | 0% |
| MVO Max Sharpe | 9.91% | 14.11% | 0.242 | -20.1% | **39.2%** |
| MVO Min Vol | 8.12% | 11.91% | 0.136 | -25.0% | 15.4% |
| HRP | 9.35% | 12.29% | **0.232** | -19.4% | 14.3% |
| **BL-Combined** | 8.41% | 17.57% | 0.109 | -20.2% | 14.4% |

**Key findings:**
- **HRP = best risk-adjusted** (0.232 Sharpe) with only 14.3% turnover vs MVO's 39.2%
- **MVO turnover is 2.7× HRP** — confirms sensitivity to estimation error
- **MVO Min Vol worst MaxDD** (-25%) despite lowest vol — Σ estimation error inverted
- **BL Sharpe positive** (0.109) with time-gated views — fully defensible out-of-sample

> ⚠️ BL uses static XGBoost views (May 2026 model) gated to `valid_from: 2026-05-01`. For most of 2021–2025, BL runs on GatiShakti macro views only, which is the production-correct design. In live deployment, XGBoost retrains monthly.

---

## Black-Litterman — View Construction

### Grinold-Kahn Alpha (Q calibration)
```
Q_i = Π_i + direction × IC_i × σ_i
```
Raw XGBoost predictions (annualised 30%+) cannot be used directly as BL views — they overwhelm the prior 100:1. Grinold-Kahn scales views to realistic alpha magnitudes (1-7%) consistent with the equilibrium.

### Idzorek Confidence (Ω calibration)
```
p_k = |IC_k| / MAX_IC         # confidence ∈ [0,1]
Ω_kk = ((1-p_k)/p_k) × τ × σ²_i
```
- IC=0.121 (ONGC) → high confidence → small Ω → view penetrates prior strongly
- IC=0.030 (HDFCBANK) → filtered by `MIN_IC=0.05` → no view expressed

### Current active views (IC ≥ 0.05)
| Stock | IC | Signal | Π | μ_BL | Δ |
|-------|----|--------|---|------|---|
| ONGC | 0.121 | LONG | 3.47% | 5.76% | **+2.29%** |
| ICICIBANK | 0.093 | SHORT | 3.87% | 2.82% | -1.06% |
| DRREDDY | 0.079 | LONG | 3.16% | 4.11% | +0.95% |
| ITC | 0.069 | SHORT | 3.30% | 2.69% | -0.61% |
| INFY | 0.052 | LONG | 5.30% | 5.92% | +0.62% |

---

## GatiShakti Macro Overlay

Quarterly government capex data → sector views → combined with ML views in one BL update.

```python
# Multi-quarter YAML — each quarter has a valid_from date
# Backtester automatically picks the correct quarter at each rebalance date

Q1_FY22 (valid_from: 2021-04-01) — Post-COVID recovery: IT+80bps, Financials+50bps
Q1_FY23 (valid_from: 2022-04-01) — Rate hike cycle: Financials-60bps, Energy+80bps
Q1_FY24 (valid_from: 2023-04-01) — IT slowdown: IT-40bps, NBFC+80bps
Q1_FY25 (valid_from: 2024-04-01) — Election year: Consumer+60bps
Q4_FY26 (valid_from: 2026-01-31) — Digital push: IT+120bps, Financials-100bps
```

### FinBERT Enhancement (K6b)
Replace static YAML `view_bps` with NLP-extracted sentiment:
```python
# Instead of: "Digital IT = +120bps" (manually decided)
# FinBERT reads budget speech → IT sentiment = +0.84
view_bps = base_alpha × tanh(sentiment_score)
         = 150 × tanh(0.84) = +103bps  (model decided)
```

---

## Setup

```bash
# Activate Alpha-Core venv (contains all dependencies)
source ../alpha-core/venv/bin/activate

# Install additional deps
pip install streamlit pyyaml

# Run individual modules
PYTHONPATH=. python3 portfolio_optimizer/black_litterman.py   # K4+K5
PYTHONPATH=. python3 portfolio_optimizer/gatishakti_views.py  # K6
PYTHONPATH=. python3 portfolio_optimizer/backtester.py         # K7

# Launch dashboard
streamlit run dashboard/app.py
```

### Data dependencies
```
data/vajra_returns.csv          ← from Vajra Indian Risk Engine
data/vajra_dcc_cov.pkl          ← Vajra DCC covariance cube
../alpha-core/data/xgb_predictions.csv  ← Alpha-Core XGBoost output
portfolio_optimizer/gatishakti_views.yaml  ← quarterly macro config
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      KUBER                               │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │   K1     │   │   K2     │   │      K3          │   │
│  │ Data     │──▶│ Markowitz│   │  HRP             │   │
│  │ Loader   │   │ MVO      │   │  (no Σ⁻¹)        │   │
│  └────┬─────┘   └──────────┘   └──────────────────┘   │
│       │                                                  │
│       ▼  Σ_dcc (from Vajra)  +  IC signals (Alpha-Core) │
│  ┌──────────────────────────────────────────────────┐   │
│  │  K4+K5: Black-Litterman                          │   │
│  │  Π = δΣw  →  BL posterior  →  MVO on μ_BL       │   │
│  └──────────────┬───────────────────────────────────┘   │
│                 │                                        │
│  ┌──────────────▼──────────────────────────────────┐   │
│  │  K6: GatiShakti + FinBERT                        │   │
│  │  Budget text → sentiment → view_bps → P,Q,Ω      │   │
│  └──────────────┬───────────────────────────────────┘   │
│                 │                                        │
│  ┌──────────────▼──────────────────────────────────┐   │
│  │  K7: Walk-Forward Backtester                     │   │
│  │  2021–2026 | Monthly | Time-gated views           │   │
│  └──────────────┬───────────────────────────────────┘   │
│                 │                                        │
│  ┌──────────────▼──────────────────────────────────┐   │
│  │  K8: Streamlit Dashboard (5 tabs)                │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Interview Narrative

> *"I built three interconnected systems. Vajra is the risk engine — GARCH volatility, DCC dynamic correlations, Monte Carlo VaR. Alpha-Core is the signal engine — Fama-French factor decomposition, HMM regime detection, XGBoost on residuals with SHAP explainability. Kuber is the allocation engine — it takes Vajra's DCC covariance as the input matrix and Alpha-Core's XGBoost predictions as Black-Litterman views.*
>
> *Today, with ONGC flagged as LONG_BIAS with IC=0.121, Kuber increases ONGC from its market-cap equilibrium of 3.47% to a posterior return of 5.76% — a 229bps shift. The portfolio weight adjusts accordingly. I also layer in quarterly GatiShakti macro views — Q4 FY26 puts a -100bps view on rate-sensitive banks based on RBI's repo rate pause. Both view sets are combined in one Bayesian update.*
>
> *The walk-forward backtest shows HRP as the most efficient strategy (0.232 Sharpe, 14% turnover) vs MVO's 39% turnover — which is the main reason nobody runs raw MVO in production."*

---

## Tech Stack

- **Python 3.9+** — SciPy, NumPy, Pandas, scikit-learn
- **Optimization** — SLSQP (SciPy), Ledoit-Wolf shrinkage
- **ML** — XGBoost predictions from Alpha-Core, `ProsusAI/finbert`
- **Clustering** — SciPy Ward linkage for HRP dendrogram
- **Visualization** — Plotly, Matplotlib (dark theme)
- **Dashboard** — Streamlit
- **Config** — PyYAML (quarterly GatiShakti views)

---

*Part of the Vajra → Alpha-Core → Kuber quantitative pipeline.*
