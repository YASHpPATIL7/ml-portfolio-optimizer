"""
K8 — KUBER PORTFOLIO DASHBOARD
dashboard/app.py

5-tab Streamlit app:
  Tab 1: Portfolio Builder   — choose method, see weights + efficient frontier
  Tab 2: Black-Litterman     — P/Q/Ω matrices, Π vs μ_BL shifts, BL weights
  Tab 3: GatiShakti + FinBERT — live view generation, sentiment scores
  Tab 4: Backtest            — 4-method cumulative return + Sharpe comparison
  Tab 5: Risk Bridge         — Vajra DCC covariance heatmap, vol term structure

Run: streamlit run dashboard/app.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kuber — Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .stApp { background-color: #0a0e1a; color: #f1f5f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #111827; padding: 8px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { background: #1e293b; color: #94a3b8; border-radius: 6px; padding: 8px 16px; }
    .stTabs [aria-selected="true"] { background: #38bdf8 !important; color: #0a0e1a !important; font-weight: bold; }
    .metric-card { background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; margin: 4px; }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .stDataFrame { background: #111827; }
    div[data-testid="stMetricValue"] { color: #38bdf8; font-size: 1.6rem; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ── Cached data loader ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def load_bundle():
    from portfolio_optimizer.data_loader import load_data
    return load_data()


@st.cache_data(ttl=3600, show_spinner="Running Black-Litterman...")
def load_bl_result():
    from portfolio_optimizer.data_loader      import load_data
    from portfolio_optimizer.markowitz        import run_markowitz
    from portfolio_optimizer.black_litterman  import run_black_litterman
    bundle = load_data()
    mvo    = run_markowitz(bundle, plot=False)
    bl     = run_black_litterman(bundle,
                                  mvo_max_sharpe_weights=mvo.max_sharpe.weights,
                                  plot=False)
    return bl, mvo


@st.cache_data(ttl=3600, show_spinner="Running HRP...")
def load_hrp_result():
    from portfolio_optimizer.data_loader import load_data
    from portfolio_optimizer.hrp        import run_hrp
    bundle = load_data()
    return run_hrp(bundle, plot=False)


@st.cache_data(ttl=86400, show_spinner="Running walk-forward backtest (~30s)...")
def load_backtest():
    from portfolio_optimizer.data_loader      import load_data
    from portfolio_optimizer.backtester       import run_backtest
    from portfolio_optimizer.gatishakti_views import load_gatishakti_config, get_xgb_valid_from
    bundle  = load_data()
    xgb_path = Path(__file__).resolve().parent.parent.parent / "alpha-core" / "data" / "xgb_predictions.csv"
    xgb_preds = pd.read_csv(xgb_path, index_col=0) if xgb_path.exists() else None
    gs_cfg  = load_gatishakti_config()
    xgb_vf  = get_xgb_valid_from(gs_cfg)
    return run_backtest(bundle.returns, bundle.w_market.values,
                        bundle.tickers, xgb_preds, xgb_vf)


# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "Equal Weight":   "#475569",
    "MVO Max Sharpe": "#f59e0b",
    "MVO Min Vol":    "#fb923c",
    "HRP":            "#34d399",
    "BL-Combined":    "#38bdf8",
}
ACCENT = "#38bdf8"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Kuber")
    st.markdown("*ML Portfolio Optimizer*")
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("```\nVajra  → DCC Covariance\nAlpha  → XGBoost Views\nKuber  → BL Allocation\n```")
    st.markdown("---")
    st.caption("Refresh data → reload page")
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Universe:** 14 NSE stocks")
    st.markdown("**Period:** 2019 – 2026")
    st.markdown("**Rebalance:** Monthly")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Portfolio Builder",
    "🎯 Black-Litterman",
    "🏛️ GatiShakti + FinBERT",
    "📉 Backtest",
    "🔗 Risk Bridge",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO BUILDER
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Portfolio Builder")
    st.caption("Compare optimization methods — weights, risk, and efficient frontier")

    bundle = load_bundle()
    bl, mvo = load_bl_result()
    hrp_result = load_hrp_result()

    col1, col2 = st.columns([1, 2])

    with col1:
        method = st.radio("Method", [
            "MVO Max Sharpe", "MVO Min Vol", "HRP", "Black-Litterman"
        ], index=3)

        method_map = {
            "MVO Max Sharpe":   mvo.max_sharpe.weights,
            "MVO Min Vol":      mvo.min_vol.weights,
            "HRP":              hrp_result.weights,
            "Black-Litterman":  bl.weights,
        }
        w = method_map[method]

        ret_map = {
            "MVO Max Sharpe":   mvo.max_sharpe.ret_annual,
            "MVO Min Vol":      mvo.min_vol.ret_annual,
            "HRP":              hrp_result.ret_annual,
            "Black-Litterman":  bl.ret_annual,
        }
        vol_map = {
            "MVO Max Sharpe":   mvo.max_sharpe.vol_annual,
            "MVO Min Vol":      mvo.min_vol.vol_annual,
            "HRP":              hrp_result.vol_annual,
            "Black-Litterman":  bl.vol_annual,
        }
        sh_map = {
            "MVO Max Sharpe":   mvo.max_sharpe.sharpe,
            "MVO Min Vol":      mvo.min_vol.sharpe,
            "HRP":              hrp_result.sharpe,
            "Black-Litterman":  bl.sharpe,
        }

        st.metric("Annual Return", f"{ret_map[method]:.2%}")
        st.metric("Annual Vol",    f"{vol_map[method]:.2%}")
        st.metric("Sharpe Ratio",  f"{sh_map[method]:.3f}")
        st.metric("Max Weight",    f"{w.max():.1%}")
        hhi = float((w**2).sum())
        st.metric("HHI (concentration)", f"{hhi:.4f}")

    with col2:
        # Weight bar chart
        df_w = pd.DataFrame({"Weight": w * 100}).reset_index()
        df_w.columns = ["Stock", "Weight"]
        fig = px.bar(df_w, x="Stock", y="Weight",
                     color="Weight",
                     color_continuous_scale=[[0, "#1e293b"], [0.5, "#38bdf8"], [1, "#7c3aed"]],
                     title=f"{method} — Portfolio Weights")
        fig.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font=dict(color="#94a3b8"),
            title_font=dict(color="#f1f5f9", size=14),
            showlegend=False,
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", title="Weight (%)"),
        )
        fig.add_hline(y=100/len(bundle.tickers), line_dash="dash",
                      line_color="#475569",
                      annotation_text=f"Equal ({100/len(bundle.tickers):.1f}%)",
                      annotation_font_color="#64748b")
        st.plotly_chart(fig, use_container_width=True)

    # Efficient frontier
    st.subheader("Efficient Frontier")
    ef = mvo.frontier
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ef["vol"] * 100, y=ef["ret"] * 100,
        mode="lines", name="Frontier",
        line=dict(color=ACCENT, width=2)
    ))
    for name, label_r, label_v, color in [
        ("MVO Max Sharpe", mvo.max_sharpe.ret_annual, mvo.max_sharpe.vol_annual, "#f59e0b"),
        ("MVO Min Vol",    mvo.min_vol.ret_annual,    mvo.min_vol.vol_annual,    "#fb923c"),
        ("HRP",            hrp_result.ret_annual,     hrp_result.vol_annual,     "#34d399"),
        ("BL",             bl.ret_annual,             bl.vol_annual,             "#38bdf8"),
    ]:
        fig2.add_trace(go.Scatter(
            x=[label_v * 100], y=[label_r * 100],
            mode="markers+text", name=name,
            marker=dict(size=12, color=color, symbol="diamond"),
            text=[name], textposition="top center",
            textfont=dict(color=color, size=9)
        ))
    fig2.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#94a3b8"),
        title="Efficient Frontier — Risk vs Return",
        title_font=dict(color="#f1f5f9"),
        xaxis=dict(title="Volatility (%)", gridcolor="#1e293b"),
        yaxis=dict(title="Return (%)", gridcolor="#1e293b"),
        legend=dict(bgcolor="#111827", bordercolor="#1e293b"),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — BLACK-LITTERMAN
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Black-Litterman Model")
    st.caption("Π (equilibrium) → XGBoost views → μ_BL (posterior) → optimal weights")

    bl, mvo = load_bl_result()

    # Π vs μ_BL comparison
    tickers = bl.pi_eq.index.tolist()
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="Π Equilibrium", x=tickers, y=bl.pi_eq.values * 100,
        marker_color="#64748b", opacity=0.8
    ))
    fig3.add_trace(go.Bar(
        name="μ_BL Posterior", x=tickers, y=bl.mu_bl.values * 100,
        marker_color=ACCENT, opacity=0.85
    ))
    # Mark view stocks
    view_x = [t for t in tickers if t in bl.active_views]
    view_y = [bl.mu_bl[t] * 100 for t in view_x]
    fig3.add_trace(go.Scatter(
        x=view_x, y=view_y, mode="markers",
        marker=dict(size=14, color="#f59e0b", symbol="star"),
        name="Active XGB Views"
    ))
    fig3.update_layout(
        barmode="group",
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#94a3b8"),
        title="BL: Equilibrium vs Posterior Returns",
        title_font=dict(color="#f1f5f9"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(title="Annual Return (%)", gridcolor="#1e293b"),
        legend=dict(bgcolor="#111827", bordercolor="#1e293b"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Δ shift table
    st.subheader("View Impact: Π → μ_BL")
    rows = []
    for t in tickers:
        pi_  = bl.pi_eq[t] * 100
        mbl_ = bl.mu_bl[t] * 100
        w_   = bl.weights[t] * 100
        has_view = t in bl.active_views
        rows.append({
            "Stock": t,
            "Π Equil%": f"{pi_:.2f}%",
            "μ_BL%":    f"{mbl_:.2f}%",
            "Δ":        f"{mbl_-pi_:+.2f}%",
            "Weight":   f"{w_:.2f}%",
            "View":     "★" if has_view else "",
        })
    df_bl = pd.DataFrame(rows)
    st.dataframe(df_bl, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Views",   len(bl.active_views))
    col2.metric("BL Sharpe",      f"{bl.sharpe:.3f}")
    col3.metric("δ / τ",          "2.5 / 0.50")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — GATISHAKTI + FINBERT
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("GatiShakti Macro Views")
    st.caption("Government capex → sector views → BL overlay")

    bundle = load_bundle()
    from portfolio_optimizer.gatishakti_views import (
        refresh_gatishakti_views, load_gatishakti_config, get_active_quarter
    )
    cfg = load_gatishakti_config()
    q   = get_active_quarter(cfg)  # today

    if q:
        st.info(f"**Active quarter:** {q['quarter']} | valid_from: {q['valid_from']} | Source: {q.get('source','')}")
    else:
        st.warning("No GatiShakti quarter loaded.")

    views = refresh_gatishakti_views(bundle.tickers)

    rows = []
    for v in views:
        rows.append({
            "Sector":     v.sector,
            "View (bps)": v.view_bps,
            "Confidence": f"{v.confidence:.0%}",
            "Tickers":    ", ".join(v.tickers),
            "Direction":  "📈 LONG" if v.view_bps > 0 else ("📉 SHORT" if v.view_bps < 0 else "⚖️ NEUTRAL"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # FinBERT section
    st.markdown("---")
    st.subheader("🤖 FinBERT Sentiment Engine")
    st.caption("Paste a budget speech or GatiShakti document to auto-generate views")

    sample_text = st.text_area(
        "Policy document text",
        value="The government allocates ₹2.1 lakh crore for digital infrastructure "
              "with strong focus on AI and cloud. Energy transition faces headwinds "
              "as renewable targets are accelerated. Healthcare PLI scheme extended "
              "benefiting pharma manufacturers. Banking sector faces NIM pressure "
              "from RBI rate pause.",
        height=150,
    )

    if st.button("🚀 Run FinBERT Analysis", type="primary"):
        with st.spinner("Running ProsusAI/finbert..."):
            from portfolio_optimizer.finbert_views import run_finbert_views
            results = run_finbert_views(sample_text, quarter="dashboard_run")

        if results:
            fb_rows = []
            for sec, v in results.items():
                fb_rows.append({
                    "Sector":       sec,
                    "Sentiment":    f"{v['sentiment']:+.3f}",
                    "FinBERT bps":  v["view_bps"],
                    "Direction":    "📈" if v["view_bps"] > 0 else "📉",
                })
            st.dataframe(pd.DataFrame(fb_rows), use_container_width=True, hide_index=True)

            fig_fb = px.bar(
                pd.DataFrame(fb_rows), x="Sector", y="FinBERT bps",
                color="FinBERT bps",
                color_continuous_scale=[[0, "#ef4444"], [0.5, "#475569"], [1, "#22c55e"]],
                title="FinBERT-Derived View Magnitudes (bps)"
            )
            fig_fb.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font=dict(color="#94a3b8"),
                title_font=dict(color="#f1f5f9"),
                xaxis=dict(gridcolor="#1e293b", tickangle=-30),
                yaxis=dict(gridcolor="#1e293b"),
            )
            fig_fb.add_hline(y=0, line_color="#475569", line_width=1)
            st.plotly_chart(fig_fb, use_container_width=True)
        else:
            st.error("FinBERT returned no results. Check model availability.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Walk-Forward Backtest")
    st.caption("2021–2026 | Monthly rebalance | Views time-gated to valid_from dates")

    with st.spinner("Running backtest (cached after first run)..."):
        strategies = load_backtest()

    # Metrics table
    rows = [v.summary() for v in strategies.values()]
    df_bt = pd.DataFrame(rows)
    st.dataframe(df_bt.style.highlight_max(
        subset=["Sharpe", "Return%", "Calmar"], color="#0f2d1e"
    ).highlight_min(
        subset=["MaxDD%", "Turnover%"], color="#0f2d1e"
    ), use_container_width=True, hide_index=True)

    # Cumulative wealth
    fig4 = go.Figure()
    for name, res in strategies.items():
        wealth = res.cum_wealth
        fig4.add_trace(go.Scatter(
            x=wealth.index, y=wealth.values,
            name=f"{name} ({res.sharpe:.2f})",
            line=dict(color=COLORS.get(name, "#fff"),
                      width=3 if "BL" in name else 1.5),
            opacity=0.95 if "BL" in name else 0.7,
        ))
    fig4.add_hline(y=1, line_dash="dash", line_color="#334155")
    fig4.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#94a3b8"),
        title="Cumulative Wealth — ₹1 Invested Jan 2021",
        title_font=dict(color="#f1f5f9", size=14),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(title="Portfolio Value", gridcolor="#1e293b"),
        legend=dict(bgcolor="#111827", bordercolor="#1e293b"),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Drawdown
    fig5 = go.Figure()
    for name, res in strategies.items():
        wealth = res.cum_wealth
        peak   = wealth.cummax()
        dd     = (wealth - peak) / peak * 100
        fig5.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=name, fill="tozeroy",
            line=dict(color=COLORS.get(name, "#fff"), width=1),
            opacity=0.5,
        ))
    fig5.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#94a3b8"),
        title="Drawdown (%)",
        title_font=dict(color="#f1f5f9"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(title="Drawdown %", gridcolor="#1e293b"),
        legend=dict(bgcolor="#111827", bordercolor="#1e293b"),
    )
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — RISK BRIDGE (VAJRA DCC)
# ═══════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Risk Bridge — Vajra DCC Covariance")
    st.caption("Live DCC covariance from Indian Risk Engine · Powers BL equilibrium prior Π = δΣw")

    bundle = load_bundle()
    tickers = bundle.tickers

    # DCC correlation heatmap
    dcc_cov = bundle.sigma_dcc.values
    dcc_std  = np.sqrt(np.diag(dcc_cov))
    dcc_corr = dcc_cov / np.outer(dcc_std, dcc_std)
    np.fill_diagonal(dcc_corr, 1.0)

    fig6 = px.imshow(
        dcc_corr, x=tickers, y=tickers,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="DCC Covariance → Correlation Matrix (current regime)",
        text_auto=".2f",
    )
    fig6.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#94a3b8", size=8),
        title_font=dict(color="#f1f5f9"),
        coloraxis_colorbar=dict(title="ρ", tickfont=dict(color="#94a3b8")),
    )
    st.plotly_chart(fig6, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Vol comparison: DCC vs LW
        st.subheader("DCC vs Ledoit-Wolf Volatility")
        dcc_vols = np.sqrt(np.diag(bundle.sigma_dcc.values)) * 100
        lw_vols  = np.sqrt(np.diag(bundle.sigma_lw.values)) * 100
        df_vol = pd.DataFrame({
            "Stock":   tickers,
            "DCC Vol": dcc_vols,
            "LW Vol":  lw_vols,
        })
        fig7 = go.Figure()
        fig7.add_trace(go.Bar(name="DCC (current)", x=tickers, y=dcc_vols,
                              marker_color=ACCENT, opacity=0.85))
        fig7.add_trace(go.Bar(name="Ledoit-Wolf (historical)", x=tickers, y=lw_vols,
                              marker_color="#64748b", opacity=0.7))
        fig7.update_layout(
            barmode="group",
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font=dict(color="#94a3b8"),
            title="Annualised Volatility: DCC vs LW",
            title_font=dict(color="#f1f5f9"),
            xaxis=dict(gridcolor="#1e293b", tickangle=-45),
            yaxis=dict(title="Vol (%)", gridcolor="#1e293b"),
            legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        # Market cap weights + equilibrium returns
        st.subheader("Market Cap Weights & Π Equilibrium")
        pi_series = bundle.returns  # just need tickers
        from portfolio_optimizer.black_litterman import compute_equilibrium
        pi_eq = compute_equilibrium(bundle)
        df_eq = pd.DataFrame({
            "Stock":     tickers,
            "Mkt Cap %": bundle.w_market.values * 100,
            "Π Equil %": pi_eq.values * 100,
        })
        fig8 = go.Figure()
        fig8.add_trace(go.Bar(name="Market Cap Weight", x=tickers,
                              y=df_eq["Mkt Cap %"], marker_color="#7c3aed", opacity=0.8))
        fig8.add_trace(go.Bar(name="Π Equilibrium Return", x=tickers,
                              y=df_eq["Π Equil %"], marker_color=ACCENT, opacity=0.8))
        fig8.update_layout(
            barmode="group",
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font=dict(color="#94a3b8"),
            title="Π = δ × Σ_dcc × w_mkt",
            title_font=dict(color="#f1f5f9"),
            xaxis=dict(gridcolor="#1e293b", tickangle=-45),
            yaxis=dict(title="%", gridcolor="#1e293b"),
            legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig8, use_container_width=True)

    st.caption("📌 DCC covariance computed by Vajra Indian Risk Engine "
               "(GARCH conditional vol → DCC dynamic correlation). "
               "Refreshed with each run of `garch_model.py`.")
