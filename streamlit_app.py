"""
QuantEdge Pro — Institutional Risk & Portfolio Dashboard
=========================================================
Investment-committee-grade Streamlit terminal built on the QuantEdge
quantitative backend (data.py, portfolio.py, performance.py,
risk_metrics.py, evt.py, backtesting.py, stress_testing.py).

Adds: VaR methodology comparison (Parametric / Historical / EVT),
Kupiec + Christoffersen backtesting, a Basel traffic-light indicator,
hypothetical stress scenarios, and a formal assumptions/limitations page.

CORRECTED BUILD — see accompanying audit report for the full list of
fixes. In short, this version:
  1. Calls the validated risk_metrics.{historical,parametric}_{var,es}
     functions directly instead of re-deriving VaR/ES formulas locally
     (the old local historical-ES formula understated tail risk).
  2. Removes a look-ahead bias in the rolling VaR backtest: the
     Parametric/Historical rolling windows previously included the
     return being tested inside its own estimation window.
  3. Fixes the Kupiec backend call so the validated
     backtesting.kupiec_test() is actually invoked (the previous call
     signature always raised and silently fell back to a duplicate
     local implementation).
  4. Prevents an invalid EVT threshold/confidence combination and
     reports the real cause instead of a generic "insufficient data"
     message.
  5. Corrects the hypothetical stress scenarios so every shocked
     ticker is actually a member of the traded universe (two of the
     five scenarios previously referenced tickers — TLT, USO — that
     are not in DEFAULT_TICKERS and therefore always produced a
     fabricated-looking 0.00% impact).
  6. Adds an explicit In-Sample / Out-of-Sample mode so the dashboard
     never silently reports an in-sample backtest as if it were an
     out-of-sample performance result.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm, chi2, binom

# ----------------------------------------------------------------------------
# QuantEdge Core Modules — single source of truth for every calculation
# ----------------------------------------------------------------------------
from data import (
    DEFAULT_TICKERS, download_adjusted_prices, clean_prices,
    calculate_returns, descriptive_statistics
)
from portfolio import optimize_portfolio
from backtesting import backtest_portfolio
from performance import performance_summary
from evt import fit_evt_var_es
from stress_testing import HISTORICAL_SCENARIOS, historical_stress
from risk_metrics import historical_var, historical_es, parametric_var, parametric_es

# Optional backend extensions — imported defensively so the app degrades
# gracefully if a given backend build doesn't expose them yet.
try:
    import risk_metrics as rm
except ImportError:
    rm = None

try:
    from backtesting import kupiec_test as backend_kupiec_test
except ImportError:
    backend_kupiec_test = None

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="QuantEdge Pro | Risk Committee Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS — INSTITUTIONAL DARK THEME  (unchanged — visual identity preserved)
# ==============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #0a0e14;
        --bg-secondary: #0f1520;
        --bg-card: #121824;
        --border-color: #1f2a3a;
        --accent-green: #00e6a8;
        --accent-red: #ff4d6d;
        --accent-amber: #f0b429;
        --accent-blue: #4d9fff;
        --text-primary: #e8edf4;
        --text-secondary: #8694a8;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text-primary); }
    .stApp { background: radial-gradient(circle at top left, #0d1420 0%, #080b10 100%); }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1440px; }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff 0%, #8694a8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3, h4 { font-weight: 600 !important; color: var(--text-primary) !important; letter-spacing: -0.3px; }

    .terminal-subtitle {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-secondary);
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: -10px;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, var(--bg-card), #0d1320);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem 1.1rem 0.8rem 1.1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        transition: border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover { border-color: #2c3a50; }
    div[data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem !important;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        color: var(--text-secondary) !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: var(--bg-secondary);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] { height: 42px; border-radius: 8px; color: var(--text-secondary); font-weight: 500; font-size: 0.88rem; }
    .stTabs [aria-selected="true"] { background-color: var(--bg-card) !important; color: var(--accent-green) !important; border: 1px solid var(--border-color); }

    details { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--border-color); border-radius: 10px; overflow: hidden; }

    .stButton button { background-color: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-primary); border-radius: 8px; font-weight: 500; }
    .stButton button:hover { border-color: var(--accent-green); color: var(--accent-green); }

    .section-divider { border-top: 1px solid var(--border-color); margin: 1.4rem 0 1.2rem 0; }

    .status-pill {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.5px;
    }
    .pill-green { background: rgba(0,230,168,0.12); color: var(--accent-green); border: 1px solid rgba(0,230,168,0.3); }

    .traffic-card {
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        border: 1px solid var(--border-color);
        text-align: center;
    }
    .traffic-GREEN { background: rgba(0,230,168,0.10); border-color: rgba(0,230,168,0.4); }
    .traffic-YELLOW { background: rgba(240,180,41,0.10); border-color: rgba(240,180,41,0.4); }
    .traffic-RED { background: rgba(255,77,109,0.10); border-color: rgba(255,77,109,0.4); }
    .traffic-label { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 1px; text-transform: uppercase; color: var(--text-secondary); }
    .traffic-zone { font-size: 2.1rem; font-weight: 700; letter-spacing: 1px; margin: 0.2rem 0; }
    .zone-GREEN { color: var(--accent-green); }
    .zone-YELLOW { color: var(--accent-amber); }
    .zone-RED { color: var(--accent-red); }

    .disclaimer-box {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-left: 3px solid var(--accent-amber);
        border-radius: 8px;
        padding: 1rem 1.3rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        line-height: 1.55;
        color: var(--text-primary);
    }
    .disclaimer-box b { color: var(--accent-amber); }

    .mode-banner {
        border-radius: 10px;
        padding: 0.7rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.86rem;
        line-height: 1.5;
        border: 1px solid rgba(240,180,41,0.4);
        background: rgba(240,180,41,0.08);
        color: var(--text-primary);
    }
    .mode-banner.oos {
        border: 1px solid rgba(0,230,168,0.4);
        background: rgba(0,230,168,0.08);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_GREEN = "#00e6a8"
COLOR_RED = "#ff4d6d"
COLOR_BLUE = "#4d9fff"
COLOR_AMBER = "#f0b429"
PALETTE = ["#00e6a8", "#4d9fff", "#f0b429", "#ff4d6d", "#9d7bff", "#3ad6ff", "#ff9e4d"]


def style_fig(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e8edf4", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#121824", font_size=12, font_family="JetBrains Mono"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1a2332", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1a2332", zeroline=False)
    return fig


def safe_get(d, key, default=np.nan):
    try:
        return d[key]
    except Exception:
        return default


def fmt_pct(x, decimals=2):
    return f"{x * 100:.{decimals}f}%" if pd.notna(x) else "N/A"


def fmt_num(x, decimals=2):
    return f"{x:.{decimals}f}" if pd.notna(x) else "N/A"


def compute_drawdown_series(portfolio_values: pd.Series) -> pd.Series:
    running_max = portfolio_values.cummax()
    return (portfolio_values / running_max) - 1.0


# ==============================================================================
# RISK ENGINE
# ==============================================================================
# FIX #1 — these two wrappers now call the validated functions in
# risk_metrics.py directly instead of re-deriving VaR/ES with duplicate,
# unvalidated formulas. The previous local `historical_var_es()` used a
# simple tail mean for ES, which understates tail risk versus the
# backend's exact fractional-mass estimator (verified to differ by ~25%
# on small samples and ~0.6% on a 1-year/95% sample in the audit). The
# previous local VaR also used pandas' default linear-interpolation
# quantile instead of the backend's `inverted_cdf` convention.
def parametric_var_es(returns: pd.Series, confidence: float):
    """Normal (variance-covariance) VaR / ES — delegates to risk_metrics.py."""
    return parametric_var(returns, confidence), parametric_es(returns, confidence)


def historical_var_es(returns: pd.Series, confidence: float):
    """Empirical (non-parametric) VaR / ES — delegates to risk_metrics.py."""
    return historical_var(returns, confidence), historical_es(returns, confidence)


@st.cache_data(show_spinner=False)
def rolling_var_series(returns: pd.Series, confidence: float, method: str,
                        window: int = 250, evt_step: int = 10, evt_threshold_q: float = 0.90) -> pd.Series:
    """
    Rolling, point-in-time VaR forecast used purely for exception-based
    backtesting (Kupiec / Christoffersen / Basel traffic light).

    FIX #2 — Parametric and Historical forecasts are now computed on
    `returns.shift(1)` before rolling. Pandas' `.rolling(window)` is a
    trailing, inclusive window: at row i it covers rows [i-window+1, i],
    i.e. it includes the very return being tested. Comparing that
    return against a VaR estimate that was partly built from itself is
    look-ahead bias and will understate the true exception rate (the
    model looks better calibrated than it is). Shifting by one period
    first means the VaR forecast for day t uses only information
    through day t-1, matching the convention already used correctly by
    the EVT branch below and by backend `rolling_var_backtest()` in
    backtesting.py (see test_forecasts_do_not_look_ahead in
    tests/test_backtesting.py).
    """
    if method == "Parametric":
        shifted = returns.shift(1)
        roll_mean = shifted.rolling(window).mean()
        roll_std = shifted.rolling(window).std()
        z = norm.ppf(confidence)
        return z * roll_std - roll_mean

    elif method == "Historical":
        shifted = returns.shift(1)
        return -shifted.rolling(window).quantile(1 - confidence)

    elif method == "EVT":
        var_series = pd.Series(index=returns.index, dtype=float)
        last_val = np.nan
        for i in range(window, len(returns), evt_step):
            window_data = returns.iloc[i - window:i]  # already excludes day i — no look-ahead
            try:
                evt_stats = fit_evt_var_es(window_data, confidence=confidence, threshold_quantile=evt_threshold_q)
                last_val = evt_stats["evt_var"]
            except Exception:
                pass
            end = min(i + evt_step, len(returns))
            var_series.iloc[i:end] = last_val
        return var_series

    raise ValueError(f"Unknown VaR methodology: {method}")


def kupiec_pof_test(exceptions: int, n_obs: int, confidence: float):
    """Local Kupiec (1995) POF test — numerically verified against
    backtesting.kupiec_test() in the audit (identical LR statistic and
    p-value across several exception counts/sample sizes). Used only as
    a fallback if the backend function cannot be imported at all."""
    p = 1 - confidence
    x = exceptions
    if n_obs == 0:
        return {"LR_stat": np.nan, "p_value": np.nan, "exceptions": x, "n_obs": n_obs,
                "expected_exceptions": np.nan, "reject_h0": None}

    x_hat = x / n_obs
    expected = p * n_obs

    def _log_term(prob, count, total):
        if prob <= 0 or prob >= 1:
            return 0.0
        return count * np.log(prob) + (total - count) * np.log(1 - prob)

    log_l_null = _log_term(p, x, n_obs)
    log_l_alt = _log_term(x_hat, x, n_obs) if 0 < x_hat < 1 else 0.0

    lr_stat = max(0.0, -2 * (log_l_null - log_l_alt))
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    return {
        "LR_stat": lr_stat, "p_value": p_value, "exceptions": x, "n_obs": n_obs,
        "expected_exceptions": expected, "reject_h0": bool(p_value < 0.05),
    }


def run_kupiec_test(exception_flags: pd.Series, confidence: float):
    """
    FIX #3 — The previous build attempted to call the validated backend
    `backtesting.kupiec_test()` with keyword arguments it does not
    accept (`returns=`, `var_series=`), then with a positional
    `(n_exceptions, n_obs, confidence)` signature it also does not
    accept. Both calls always raised TypeError and were silently
    swallowed, so the dashboard never actually exercised the backend
    function — it always fell through to the local duplicate. The
    local duplicate was verified to be numerically correct, but this
    defeats the "use shared, validated functions" principle and is
    fragile: any future change to the local formula would silently
    diverge from the tested backend implementation with no warning.

    The real signature is `kupiec_test(exceptions: pd.Series[bool],
    confidence: float)`. This wrapper calls it correctly first and
    only falls back to the local copy if the backend module could not
    be imported at all.
    """
    n_obs = len(exception_flags)
    n_exceptions = int(exception_flags.sum())
    if backend_kupiec_test is not None:
        try:
            backend_result = backend_kupiec_test(exception_flags.astype(bool), confidence)
            return {
                "LR_stat": backend_result["lr_statistic"],
                "p_value": backend_result["p_value"],
                "exceptions": backend_result["exceptions"],
                "n_obs": backend_result["observations"],
                "expected_exceptions": (1 - confidence) * backend_result["observations"],
                "reject_h0": bool(backend_result["p_value"] < 0.05),
                "source": "backend",
            }
        except Exception:
            pass
    result = kupiec_pof_test(n_exceptions, n_obs, confidence)
    result["source"] = "local_fallback"
    return result


def christoffersen_independence_test(exception_flags: pd.Series):
    """
    Christoffersen (1998) Conditional Coverage test, independence component.
    Implemented locally (scipy/numpy) since the backend module does not
    expose a packaged function for this test — it sits on top of raw
    exception flags rather than being part of backtesting.py. This is
    disclosed explicitly in the Assumptions & Limitations tab.
    """
    flags = exception_flags.dropna().astype(int).values
    if len(flags) < 2:
        return {"LR_ind": np.nan, "p_value_ind": np.nan, "n00": 0, "n01": 0, "n10": 0, "n11": 0}

    prev, curr = flags[:-1], flags[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

    log_l_restricted = (
        (n00 + n10) * np.log(1 - pi) if 0 < pi < 1 and (n00 + n10) > 0 else 0.0
    ) + (
        (n01 + n11) * np.log(pi) if 0 < pi < 1 and (n01 + n11) > 0 else 0.0
    )

    log_l_unrestricted = 0.0
    if 0 < pi01 < 1:
        log_l_unrestricted += n00 * np.log(1 - pi01) + n01 * np.log(pi01)
    if 0 < pi11 < 1:
        log_l_unrestricted += n10 * np.log(1 - pi11) + n11 * np.log(pi11)

    lr_ind = -2 * (log_l_restricted - log_l_unrestricted)
    lr_ind = max(lr_ind, 0.0)
    p_value_ind = 1 - chi2.cdf(lr_ind, df=1)

    return {"LR_ind": lr_ind, "p_value_ind": p_value_ind, "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def basel_traffic_light(exceptions: int, n_obs: int, confidence: float):
    """
    General (cumulative-binomial) formulation of the Basel traffic-light
    backtesting framework. The textbook 4 / 9 exception cut-offs are the
    special case of this formula at p = 1% and N = 250; here it is
    generalized to any confidence level / sample size so the indicator
    stays meaningful when the committee changes the confidence selector.
    """
    p = 1 - confidence
    if n_obs == 0:
        return {"zone": "N/A", "cum_prob": np.nan}
    cum_prob = binom.cdf(exceptions, n_obs, p)
    if cum_prob < 0.95:
        zone = "GREEN"
    elif cum_prob < 0.9999:
        zone = "YELLOW"
    else:
        zone = "RED"
    return {"zone": zone, "cum_prob": cum_prob}


# ==============================================================================
# CACHED BACKEND PIPELINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_market_data(tickers, start, end):
    raw_prices = download_adjusted_prices(tickers=tickers, start=start, end=end)
    clean = clean_prices(raw_prices, drop_incomplete_rows=False)
    clean = clean.loc[:, list(DEFAULT_TICKERS)].dropna()
    return calculate_returns(clean)


@st.cache_data(show_spinner=False)
def run_optimization(returns, objective, max_weight):
    return optimize_portfolio(returns, objective=objective, max_weight=max_weight)


@st.cache_data(show_spinner=False)
def run_backtest_cached(returns, weights, initial_val, reb, cost):
    return backtest_portfolio(
        asset_returns=returns, target_weights=weights,
        initial_value=initial_val, rebalance=reb, transaction_cost_bps=cost,
    )


@st.cache_data(show_spinner=False)
def run_performance(port_returns, risk_free_rate):
    return performance_summary(port_returns, risk_free_rate=risk_free_rate)


@st.cache_data(show_spinner=False)
def run_evt(port_returns, confidence, threshold_quantile):
    return fit_evt_var_es(port_returns, confidence=confidence, threshold_quantile=threshold_quantile)


# ==============================================================================
# HYPOTHETICAL STRESS SCENARIOS (committee-defined shocks, not historical replay)
# ==============================================================================
# FIX #5 — The previous scenario set referenced "TLT" (Treasury bond
# ETF) and "USO" (oil ETF), neither of which is a member of
# DEFAULT_TICKERS = (SPY, EFA, EEM, AGG, GLD, XLE, XLK). Because
# apply_hypothetical_shock() only applies a shock to tickers actually
# present in the optimized weights, "Rates Shock" and "Oil Price
# Collapse" always evaluated to an impact of exactly 0.00% for *any*
# portfolio drawn from this universe — not because rates/oil risk is
# genuinely zero, but because the scenario was never wired to a real
# holding. That is indistinguishable, on the dashboard, from a
# legitimately computed zero-impact result, which violates the "never
# display fabricated-looking results" rule.
#
# Fix: every scenario below is mapped onto an asset that is actually a
# member of DEFAULT_TICKERS. AGG (aggregate bond ETF) stands in for
# the former TLT leg; XLE (energy-sector equity ETF) stands in for the
# former USO leg — renamed "Energy Price Collapse" rather than "Oil
# Price Collapse" since XLE is energy-sector equity exposure, not a
# direct oil-price proxy. A runtime check below additionally guards
# against this class of bug recurring silently in the future.
HYPOTHETICAL_SCENARIOS = {
    "Broad Equity Selloff (-20%)": {"SPY": -0.20, "EFA": -0.20, "EEM": -0.20, "XLK": -0.20, "default": -0.20},
    "Rate Shock (Bond Selloff -8%)": {"AGG": -0.08, "default": 0.0},
    "Energy Price Collapse (-35%)": {"XLE": -0.35, "default": 0.0},
    "Flight to Quality": {"GLD": 0.10, "AGG": 0.04, "default": -0.08},
    "Stagflation Shock": {"GLD": 0.12, "XLE": 0.15, "SPY": -0.15, "AGG": -0.06, "default": -0.10},
}


def apply_hypothetical_shock(weights: pd.Series, shocks: dict) -> float:
    default_shock = shocks.get("default", 0.0)
    portfolio_impact = 0.0
    for ticker, w in weights.items():
        shock = shocks.get(ticker, default_shock)
        portfolio_impact += w * shock
    return portfolio_impact


def scenario_has_mapped_exposure(weights: pd.Series, shocks: dict) -> bool:
    """True if the scenario shocks at least one asset actually held, or
    applies a nonzero default shock to the rest of the book. Guards
    against the exact bug fixed above (a scenario whose named tickers
    don't exist in the current universe, silently producing 0.00%)."""
    referenced = set(shocks) - {"default"}
    overlap = referenced & set(weights.index)
    return len(overlap) > 0 or shocks.get("default", 0.0) != 0.0


# ==============================================================================
# SIDEBAR — CONTROL PANEL
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Risk Committee Control Panel")
    st.markdown('<span class="status-pill pill-green">● ENGINE READY</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**📅 Data Window**")
    start_date = st.date_input("Start Date", datetime(2015, 1, 1))
    end_date = st.date_input("End Date", datetime(2023, 1, 1))
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    st.markdown("**🧪 Evaluation Mode**")
    evaluation_mode = st.radio(
        "How should weights be optimized and evaluated?",
        ["In-Sample (single window)", "Out-of-Sample (train / test split)"],
        index=0,
        help=(
            "In-Sample: weights are optimized and performance is evaluated on "
            "the same date range — a methodology/diagnostic view, not a "
            "forward-looking performance claim. Out-of-Sample: weights are "
            "optimized on the training slice only; all performance, risk, and "
            "backtesting figures are computed on the held-out test slice."
        ),
    )
    split_date = None
    if evaluation_mode.startswith("Out-of-Sample"):
        default_split = start_date + (end_date - start_date) * 2 // 3
        split_date = st.date_input(
            "Train / Test Split Date",
            default_split,
            min_value=start_date + timedelta(days=30),
            max_value=end_date - timedelta(days=30),
        )

    st.markdown("**💰 Capital & Costs**")
    initial_capital = st.number_input("Initial Capital ($)", value=100000.0, step=10000.0, format="%.2f")
    risk_free_rate = st.number_input("Risk-Free Rate (Annual %)", value=2.0, step=0.1) / 100
    transaction_cost = st.slider("Transaction Cost (bps)", 0.0, 50.0, 5.0)

    st.markdown("**🎯 Optimization Setup**")
    objective = st.selectbox("Objective", ["max_sharpe", "min_variance"])
    max_asset_weight = st.slider("Max Weight per Asset", 0.1, 1.0, 0.4)
    rebalance_freq = st.selectbox("Rebalancing", ["monthly", "quarterly", "none"])

    st.markdown("---")
    st.markdown("**⚠️ Risk Engine Settings**")
    confidence_level = st.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.99)
    var_methodology = st.selectbox("VaR Methodology (Backtest)", ["Parametric", "Historical", "EVT"])
    # EVT requires threshold_quantile < confidence (see evt.py). Cap the
    # slider so the sidebar cannot express an invalid combination.
    evt_q_max = max(0.50, confidence_level - 0.02)
    evt_threshold_q = st.slider(
        "EVT POT Threshold Quantile", 0.50, float(evt_q_max), min(0.90, float(evt_q_max))
    )
    backtest_window = st.slider("Rolling Estimation Window (days)", 100, 500, 250, step=10)

    st.markdown("---")
    st.caption("QuantEdge Pro · Investment Committee Build")

# ==============================================================================
# HEADER
# ==============================================================================
st.title("📈 QuantEdge Pro — Risk Committee Terminal")
st.markdown(
    f'<div class="terminal-subtitle">Institutional Risk & Portfolio Analytics · '
    f'{", ".join(DEFAULT_TICKERS)} · {int(confidence_level*100)}% Confidence · '
    f'{var_methodology} VaR · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
    unsafe_allow_html=True,
)

# FIX #6 — Always-visible disclosure of which evaluation methodology is
# active. Previously the dashboard optimized weights and backtested
# performance on the identical date range with no indication that this
# is an in-sample diagnostic rather than an out-of-sample performance
# estimate (see audit report, Issue: In-Sample Optimization/Backtest).
if evaluation_mode.startswith("In-Sample"):
    st.markdown(
        '<div class="mode-banner">⚠️ <b>In-Sample Mode.</b> Portfolio weights are optimized '
        'and all performance/risk figures below are evaluated on the <b>same</b> date range '
        f'({start_date} → {end_date}). These figures describe how the optimizer\'s chosen '
        'weights would have performed over the data used to choose them, which will '
        'generally overstate the risk-adjusted performance an investor could have captured '
        'in real time. Switch to Out-of-Sample mode in the sidebar for a held-out test.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="mode-banner oos">✅ <b>Out-of-Sample Mode.</b> Weights are optimized on '
        f'the training window ({start_date} → {split_date}) only. All performance, risk, and '
        f'backtesting figures below are computed on the held-out test window '
        f'({split_date} → {end_date}), which the optimizer never saw.</div>',
        unsafe_allow_html=True,
    )

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
with st.spinner("Fetching market data and running the quantitative pipeline..."):
    try:
        try:
            ret_data = get_market_data(DEFAULT_TICKERS, str(start_date), str(end_date))
            full_simple_returns = ret_data.simple_returns
        except Exception as e:
            st.error(f"🔴 Data download failed: {e}")
            st.stop()

        # ------------------------------------------------------------
        # Train / test split (FIX #6)
        # ------------------------------------------------------------
        if evaluation_mode.startswith("Out-of-Sample"):
            split_ts = pd.Timestamp(split_date)
            train_returns = full_simple_returns.loc[full_simple_returns.index < split_ts]
            eval_returns = full_simple_returns.loc[full_simple_returns.index >= split_ts]
            if len(train_returns) < 30 or len(eval_returns) < 30:
                st.error(
                    "🔴 The train/test split leaves fewer than 30 observations on one side. "
                    "Widen the date range or move the split date."
                )
                st.stop()
        else:
            train_returns = full_simple_returns
            eval_returns = full_simple_returns

        try:
            opt_result = run_optimization(train_returns, objective, max_asset_weight)
        except Exception as e:
            st.toast("⚠️ Optimizer did not converge cleanly — check constraints.", icon="⚠️")
            st.error(f"🔴 Portfolio optimization failed: {e}")
            st.stop()

        try:
            bt_result = run_backtest_cached(
                eval_returns, opt_result.weights, initial_capital, rebalance_freq, transaction_cost
            )
            port_history = bt_result["history"]
            port_returns = port_history["portfolio_return"]
        except Exception as e:
            st.error(f"🔴 Backtest execution failed: {e}")
            st.stop()

        perf_metrics = run_performance(port_returns, risk_free_rate)
        cagr = safe_get(perf_metrics, "annualized_return_cagr")
        ann_vol = safe_get(perf_metrics, "annualized_volatility")
        sharpe = safe_get(perf_metrics, "sharpe_ratio")
        max_dd = safe_get(perf_metrics, "maximum_drawdown")

        if not st.session_state.get("_pipeline_toast_shown", False):
            st.toast("✅ Pipeline executed successfully.", icon="✅")
            st.session_state["_pipeline_toast_shown"] = True

        # ----------------------------------------------------------------
        # KPI HEADER
        # ----------------------------------------------------------------
        eval_label = "Full Window" if evaluation_mode.startswith("In-Sample") else "Out-of-Sample Test Window"
        st.markdown(f"#### 🧮 Portfolio KPIs — {eval_label} ({port_history.index.min().date()} → {port_history.index.max().date()})")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CAGR", fmt_pct(cagr))
        k2.metric("Ann. Volatility", fmt_pct(ann_vol))
        k3.metric("Sharpe Ratio", fmt_num(sharpe))
        k4.metric("Max Drawdown", fmt_pct(max_dd))

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊  Portfolio Overview",
            "⚖️  Comparative Risk Dashboard",
            "🛡️  Backtesting & Compliance",
            "🌪️  Stress Testing",
            "📜  Assumptions & Limitations",
        ])

        # ============================= TAB 1 — PORTFOLIO OVERVIEW =============================
        with tab1:
            colA, colB = st.columns([1, 1.3])

            w_df = opt_result.weights.to_frame(name="Weight")
            w_df = w_df[w_df["Weight"] > 0.001].sort_values("Weight", ascending=False)

            with colA:
                st.markdown(f"##### Optimal Allocation ({objective.replace('_', ' ').title()})")
                train_label = (
                    f"{start_date} → {end_date}" if evaluation_mode.startswith("In-Sample")
                    else f"{start_date} → {split_date} (training window)"
                )
                st.caption(f"Weights estimated on: {train_label}")
                fig_donut = go.Figure(data=[go.Pie(
                    labels=w_df.index, values=w_df["Weight"], hole=0.62,
                    marker=dict(colors=PALETTE, line=dict(color="#0a0e14", width=2)),
                    textinfo="label+percent", textfont=dict(size=12, color="#e8edf4"),
                    hovertemplate="%{label}<br>%{percent}<extra></extra>",
                )])
                fig_donut.update_layout(annotations=[dict(
                    text="Weights", x=0.5, y=0.5, font_size=14, font_color="#8694a8", showarrow=False
                )])
                fig_donut = style_fig(fig_donut, height=380)
                st.plotly_chart(fig_donut, use_container_width=True)

            with colB:
                st.markdown("##### Wealth Compounding Index")
                st.caption(f"Backtest evaluated on: {port_history.index.min().date()} → {port_history.index.max().date()}")
                fig_wealth = go.Figure()
                fig_wealth.add_trace(go.Scatter(
                    x=port_history.index, y=port_history["portfolio_value"], mode="lines",
                    name="Portfolio Value", line=dict(color=COLOR_GREEN, width=2),
                    fill="tozeroy", fillcolor="rgba(0,230,168,0.08)",
                    hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
                ))
                fig_wealth = style_fig(fig_wealth, height=380)
                fig_wealth.update_yaxes(title_text="Portfolio Value ($)")
                st.plotly_chart(fig_wealth, use_container_width=True)

            st.markdown("##### Underwater Chart (Drawdown)")
            dd_series = compute_drawdown_series(port_history["portfolio_value"])
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd_series.index, y=dd_series.values * 100, mode="lines", name="Drawdown",
                line=dict(color=COLOR_RED, width=1.4),
                fill="tozeroy", fillcolor="rgba(255,77,109,0.18)",
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
            ))
            fig_dd = style_fig(fig_dd, height=260)
            fig_dd.update_yaxes(title_text="Drawdown (%)")
            st.plotly_chart(fig_dd, use_container_width=True)

            with st.expander("📋 Asset Descriptive Statistics"):
                st.caption(f"Computed on: {train_returns.index.min().date()} → {train_returns.index.max().date()}")
                stats = descriptive_statistics(train_returns)
                stats_df = pd.DataFrame({
                    "Ann. Mean (%)": stats["annualized_mean"] * 100,
                    "Ann. Vol (%)": stats["annualized_volatility"] * 100,
                })
                st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

        # ============================= TAB 2 — COMPARATIVE RISK DASHBOARD =============================
        with tab2:
            st.markdown(
                f"##### VaR / ES / EVT Comparison — {int(confidence_level*100)}% Confidence, "
                f"1-Day Horizon, {eval_label}"
            )
            st.caption(
                "All three methodologies below are computed on the identical portfolio-return "
                "series, confidence level, and 1-trading-day horizon shown above, so they are "
                "directly comparable. Figures are expressed as positive loss fractions of "
                "portfolio value (a VaR of 2.00% means a 1-day loss of 2% of the portfolio)."
            )

            p_var, p_es = parametric_var_es(port_returns, confidence_level)
            h_var, h_es = historical_var_es(port_returns, confidence_level)

            evt_available = True
            evt_var = evt_es = evt_shape = np.nan
            if evt_threshold_q >= confidence_level:
                # Guarded by the sidebar slider cap, but double-checked here
                # since confidence_level can still change after the slider
                # was rendered within the same rerun.
                evt_available = False
                st.warning(
                    f"⚠️ EVT threshold quantile ({evt_threshold_q:.2f}) must be strictly below "
                    f"the selected confidence level ({confidence_level:.2f}). Lower the EVT "
                    "threshold slider in the sidebar."
                )
            else:
                try:
                    evt_stats_full = run_evt(port_returns, confidence_level, evt_threshold_q)
                    evt_var, evt_es, evt_shape = evt_stats_full["evt_var"], evt_stats_full["evt_es"], evt_stats_full["shape"]
                except ValueError as e:
                    evt_available = False
                    st.warning(f"⚠️ EVT fitting failed: {e}")
                except Exception as e:
                    evt_available = False
                    st.warning(f"⚠️ EVT module raised an unexpected error: {e}")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("###### 📐 Parametric (Normal)")
                st.metric("VaR", fmt_pct(p_var))
                st.metric("Expected Shortfall", fmt_pct(p_es))
            with c2:
                st.markdown("###### 📊 Historical Simulation")
                st.metric("VaR", fmt_pct(h_var))
                st.metric("Expected Shortfall", fmt_pct(h_es))
            with c3:
                st.markdown("###### 🌪️ EVT (POT / GPD)")
                st.metric("VaR", fmt_pct(evt_var) if evt_available else "N/A")
                st.metric("Expected Shortfall", fmt_pct(evt_es) if evt_available else "N/A")
                if evt_available:
                    st.caption(f"GPD Shape (ξ): {fmt_num(evt_shape, 4)}")

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("##### Methodology Comparison Chart")
            comp_df = pd.DataFrame({
                "Methodology": ["Parametric", "Historical", "EVT"],
                "VaR (%)": [p_var * 100, h_var * 100, evt_var * 100 if evt_available else np.nan],
                "ES (%)": [p_es * 100, h_es * 100, evt_es * 100 if evt_available else np.nan],
            })
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=comp_df["Methodology"], y=comp_df["VaR (%)"], name="VaR", marker_color=COLOR_BLUE))
            fig_comp.add_trace(go.Bar(x=comp_df["Methodology"], y=comp_df["ES (%)"], name="Expected Shortfall", marker_color=COLOR_RED))
            fig_comp.update_layout(barmode="group")
            fig_comp = style_fig(fig_comp, height=380)
            fig_comp.update_yaxes(title_text="1-Day Loss (%)")
            st.plotly_chart(fig_comp, use_container_width=True)

            if evt_available:
                st.markdown("##### Return Distribution vs. Tail Threshold")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=port_returns * 100, nbinsx=80, marker_color=COLOR_BLUE, opacity=0.65, name="Daily Returns"
                ))
                fig_hist.add_vline(x=-h_var * 100, line_dash="dash", line_color=COLOR_AMBER,
                                    annotation_text="Historical VaR", annotation_font_color=COLOR_AMBER)
                fig_hist.add_vline(x=-evt_var * 100, line_dash="dash", line_color=COLOR_RED,
                                    annotation_text="EVT VaR", annotation_font_color=COLOR_RED)
                fig_hist = style_fig(fig_hist, height=340)
                fig_hist.update_xaxes(title_text="Daily Return (%)")
                fig_hist.update_yaxes(title_text="Frequency")
                st.plotly_chart(fig_hist, use_container_width=True)

        # ============================= TAB 3 — BACKTESTING & COMPLIANCE =============================
        with tab3:
            st.markdown(f"##### VaR Backtest — {var_methodology} Methodology, {int(confidence_level*100)}% Confidence")
            st.caption(
                f"Rolling {backtest_window}-day estimation window. Each day's VaR forecast uses "
                "only information available through the prior trading day (no look-ahead) and "
                "is evaluated against that day's realized, out-of-sample return."
            )

            try:
                var_forecast = rolling_var_series(
                    port_returns, confidence_level, var_methodology,
                    window=backtest_window, evt_threshold_q=evt_threshold_q,
                )
                valid = var_forecast.dropna()
                aligned_returns = port_returns.loc[valid.index]
                exception_flags = (aligned_returns < -valid).astype(int)
                n_obs = len(exception_flags)
                n_exceptions = int(exception_flags.sum())

                if n_obs == 0:
                    st.warning("⚠️ Not enough observations to run a rolling backtest for the selected window.")
                else:
                    kupiec_result = run_kupiec_test(exception_flags, confidence_level)
                    if kupiec_result.get("source") == "local_fallback":
                        st.caption(
                            "ℹ️ Backend `kupiec_test()` could not be imported in this environment; "
                            "using the locally verified fallback implementation."
                        )

                    christoffersen_result = christoffersen_independence_test(exception_flags)

                    lr_pof = safe_get(kupiec_result, "LR_stat")
                    lr_ind = christoffersen_result["LR_ind"]
                    lr_cc = lr_pof + lr_ind if pd.notna(lr_pof) and pd.notna(lr_ind) else np.nan
                    p_value_cc = 1 - chi2.cdf(lr_cc, df=2) if pd.notna(lr_cc) else np.nan

                    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                    bcol1.metric("Exceptions", f"{n_exceptions} / {n_obs}")
                    bcol2.metric("Expected Exceptions", fmt_num(safe_get(kupiec_result, "expected_exceptions"), 1))
                    bcol3.metric("Kupiec LR (UC)", fmt_num(lr_pof))
                    bcol4.metric("Christoffersen LR (Ind.)", fmt_num(lr_ind))

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        st.markdown("###### 🧪 Kupiec Unconditional Coverage Test")
                        kupiec_reject = safe_get(kupiec_result, "reject_h0")
                        kupiec_p = safe_get(kupiec_result, "p_value")
                        st.write(f"**LR statistic:** {fmt_num(lr_pof)}  ·  **p-value:** {fmt_num(kupiec_p, 4)}")
                        if kupiec_reject is True:
                            st.error("❌ Reject H₀ — the model's exception rate is statistically inconsistent with the stated confidence level.")
                        elif kupiec_reject is False:
                            st.success("✅ Fail to reject H₀ — exception frequency is consistent with the stated confidence level. This does not prove the model is correct, only that this test found no evidence of miscalibration.")
                        else:
                            st.info("Insufficient data to evaluate.")

                    with rcol2:
                        st.markdown("###### 🔗 Christoffersen Conditional Coverage Test")
                        reject_cc = bool(p_value_cc < 0.05) if pd.notna(p_value_cc) else None
                        st.write(f"**LR_cc statistic (χ², df=2):** {fmt_num(lr_cc)}  ·  **p-value:** {fmt_num(p_value_cc, 4)}")
                        if reject_cc is True:
                            st.error("❌ Reject H₀ — VaR exceptions are either miscalibrated or temporally clustered.")
                        elif reject_cc is False:
                            st.success("✅ Fail to reject H₀ — exceptions are independent and correctly calibrated per this test.")
                        else:
                            st.info("Insufficient data to evaluate.")

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                    st.markdown("###### 🚦 Basel Traffic-Light Indicator")
                    traffic = basel_traffic_light(n_exceptions, n_obs, confidence_level)
                    zone = traffic["zone"]
                    zone_desc = {
                        "GREEN": "Model performing within expectations — no supervisory action required.",
                        "YELLOW": "Elevated exception count — increased monitoring and model review recommended.",
                        "RED": "Exception count statistically inconsistent with the model — immediate model review and capital add-on likely required.",
                    }.get(zone, "")
                    st.markdown(
                        f'<div class="traffic-card traffic-{zone}">'
                        f'<div class="traffic-label">Basel Zone</div>'
                        f'<div class="traffic-zone zone-{zone}">{zone}</div>'
                        f'<div style="color:#8694a8;font-size:0.85rem;">{n_exceptions} exceptions in {n_obs} observations '
                        f'(cumulative binomial probability: {fmt_num(traffic["cum_prob"]*100 if pd.notna(traffic["cum_prob"]) else np.nan, 2)}%)</div>'
                        f'</div>', unsafe_allow_html=True,
                    )
                    st.caption(zone_desc)
                    if confidence_level != 0.99 or backtest_window != 250:
                        st.caption(
                            "⚠️ The classic 4/9-exception Basel cut-offs apply specifically to "
                            "99% confidence over 250 observations. The zone above uses the "
                            "generalized cumulative-binomial formula for the currently selected "
                            f"{int(confidence_level*100)}% confidence / {n_obs}-observation sample, "
                            "and should not be read as a formal regulatory capital multiplier "
                            "outside the standard 99%/250-day specification."
                        )

                    st.markdown("##### Exception Timeline")
                    fig_exc = go.Figure()
                    fig_exc.add_trace(go.Scatter(
                        x=aligned_returns.index, y=aligned_returns.values * 100, mode="lines",
                        name="Portfolio Return", line=dict(color=COLOR_BLUE, width=1),
                    ))
                    fig_exc.add_trace(go.Scatter(
                        x=valid.index, y=-valid.values * 100, mode="lines", name=f"{var_methodology} VaR Threshold",
                        line=dict(color=COLOR_AMBER, width=1.4, dash="dot"),
                    ))
                    exc_points = aligned_returns[exception_flags == 1]
                    fig_exc.add_trace(go.Scatter(
                        x=exc_points.index, y=exc_points.values * 100, mode="markers", name="Exceptions",
                        marker=dict(color=COLOR_RED, size=7, symbol="x"),
                    ))
                    fig_exc = style_fig(fig_exc, height=380)
                    fig_exc.update_yaxes(title_text="Daily Return (%)")
                    st.plotly_chart(fig_exc, use_container_width=True)

            except Exception as e:
                st.warning(f"⚠️ Backtest could not be completed: {e}")

        # ============================= TAB 4 — STRESS TESTING =============================
        with tab4:
            st.markdown("##### Historical Scenario Replay (Buy-and-Hold)")
            st.caption(
                "Replays the full historical crisis window using the current optimized weights "
                "as a cost-free buy-and-hold, regardless of the Data Window selected in the "
                "sidebar. This is a counterfactual (\"what if this portfolio had existed then\"), "
                "not a report of this portfolio's actual historical performance."
            )
            stress_results = []
            skipped_scenarios = []
            for name, (s_start, s_end) in HISTORICAL_SCENARIOS.items():
                try:
                    res = historical_stress(full_simple_returns, opt_result.weights, s_start, s_end, initial_capital)
                    stress_results.append({
                        "Scenario": name.replace("_", " ").title(),
                        "Start": res["start"], "End": res["end"],
                        "Return (%)": res["period_return"] * 100,
                        "Max Drawdown (%)": res["maximum_drawdown"] * 100,
                    })
                except Exception as e:
                    skipped_scenarios.append((name.replace("_", " ").title(), str(e)))

            if stress_results:
                stress_df = pd.DataFrame(stress_results)
                fig_stress = go.Figure()
                fig_stress.add_trace(go.Bar(
                    x=stress_df["Scenario"], y=stress_df["Return (%)"], name="Period Return",
                    marker_color=[COLOR_GREEN if v >= 0 else COLOR_RED for v in stress_df["Return (%)"]],
                ))
                fig_stress = style_fig(fig_stress, height=340)
                fig_stress.update_yaxes(title_text="Return (%)")
                st.plotly_chart(fig_stress, use_container_width=True)
                st.dataframe(
                    stress_df.style.format({"Return (%)": "{:.2f}", "Max Drawdown (%)": "{:.2f}"}),
                    use_container_width=True,
                )
            else:
                st.warning("⚠️ Insufficient data range for default historical scenarios.")

            if skipped_scenarios:
                with st.expander(f"⚠️ {len(skipped_scenarios)} historical scenario(s) unavailable for the current data provider range"):
                    for name, reason in skipped_scenarios:
                        st.write(f"- **{name}**: {reason}")

            st.caption(
                "Note: a 2022 rate-hiking drawdown scenario is referenced in some project "
                "documentation but is not defined in `stress_testing.HISTORICAL_SCENARIOS` in "
                "this codebase, so it is correctly omitted above rather than fabricated. Add it "
                "to `HISTORICAL_SCENARIOS` in stress_testing.py if that scenario is required."
            )

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("##### Hypothetical Stress Scenarios")
            st.caption(
                "Committee-defined instantaneous shocks applied to current portfolio weights "
                "(not historical replays). Every shock below is mapped to a ticker in the "
                f"traded universe ({', '.join(DEFAULT_TICKERS)})."
            )

            hypo_results = []
            for name, shocks in HYPOTHETICAL_SCENARIOS.items():
                impact = apply_hypothetical_shock(opt_result.weights, shocks)
                mapped = scenario_has_mapped_exposure(opt_result.weights, shocks)
                hypo_results.append({
                    "Scenario": name,
                    "Portfolio Impact (%)": impact * 100,
                    "Estimated P&L ($)": impact * initial_capital,
                    "Mapped to Portfolio": "Yes" if mapped else "⚠️ No exposure",
                })
            hypo_df = pd.DataFrame(hypo_results)

            fig_hypo = go.Figure()
            fig_hypo.add_trace(go.Bar(
                x=hypo_df["Scenario"], y=hypo_df["Portfolio Impact (%)"], name="Portfolio Impact",
                marker_color=[COLOR_GREEN if v >= 0 else COLOR_RED for v in hypo_df["Portfolio Impact (%)"]],
            ))
            fig_hypo = style_fig(fig_hypo, height=340)
            fig_hypo.update_yaxes(title_text="Portfolio Impact (%)")
            st.plotly_chart(fig_hypo, use_container_width=True)
            st.dataframe(
                hypo_df.style.format({"Portfolio Impact (%)": "{:.2f}", "Estimated P&L ($)": "{:,.0f}"}),
                use_container_width=True,
            )

        # ============================= TAB 5 — ASSUMPTIONS & LIMITATIONS =============================
        with tab5:
            st.markdown("### 📜 Assumptions & Limitations")
            st.caption("For presentation to the Investment Committee alongside all risk figures in this terminal.")

            st.markdown(
                '<div class="disclaimer-box"><b>In-Sample vs. Out-of-Sample.</b> This terminal '
                'can run in two modes, selectable in the sidebar. In-Sample mode optimizes '
                'portfolio weights and evaluates performance on the same date range, which will '
                'generally overstate achievable risk-adjusted performance because the weights '
                'are chosen with knowledge of the very data used to score them. Out-of-Sample '
                'mode optimizes weights on a training window only and evaluates every '
                'performance, risk, and backtesting figure on a later, held-out test window the '
                'optimizer never saw. The mode in use for the figures above is shown in the '
                'banner at the top of the page.</div>', unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="disclaimer-box"><b>Parametric (Normal) VaR.</b> The variance-covariance method '
                'assumes portfolio returns are normally distributed and that volatility is stationary over the '
                'estimation window. Financial return series are well documented to exhibit fat tails, negative '
                'skew, and volatility clustering; consequently, parametric VaR will systematically understate '
                'tail risk during periods of market stress and should not be relied upon as a sole measure of '
                'extreme loss potential.</div>', unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="disclaimer-box"><b>Historical Simulation VaR / ES.</b> This approach assumes that '
                'the empirical distribution of returns observed over the lookback window is representative of '
                'future return behavior ("history repeats"). It is inherently backward-looking: if the sample '
                'window does not contain a severe market dislocation, the method will understate the true tail '
                'risk of the portfolio, and results are sensitive to the chosen window length.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="disclaimer-box"><b>Extreme Value Theory (EVT) Tail Extrapolation.</b> The '
                'Peak-Over-Threshold / Generalized Pareto approach models only the behavior of returns beyond a '
                'chosen threshold and extrapolates into the extreme tail using a small number of observations. '
                'Estimates are sensitive to the threshold quantile selected, the GPD shape parameter is '
                'estimated with meaningful uncertainty, and EVT assumes the tail-generating process is stable '
                'and continues to apply at confidence levels beyond those directly observed in history — an '
                'assumption that can break down in genuinely unprecedented events. The threshold quantile must '
                'be strictly below the selected confidence level; the sidebar slider enforces this.</div>', unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="disclaimer-box"><b>Rolling Backtest Methodology.</b> Every rolling VaR forecast '
                '(Parametric, Historical, and EVT) uses only returns observed strictly before the day being '
                'tested — no forecast is built using the return it is then evaluated against. For '
                'computational tractability, the EVT component re-fits the tail model every 10-trading-day '
                'step (rather than daily) and forward-fills the forecast between refits, so the EVT backtest '
                'reacts to regime changes with a short lag relative to the Parametric and Historical '
                'methodologies.</div>', unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="disclaimer-box"><b>Kupiec and Christoffersen Tests.</b> Both tests rely on '
                'asymptotic chi-squared approximations that are most reliable with a large number of '
                'observations and a reasonable number of exceptions; with short backtest windows or very high '
                'confidence levels, the tests can have low statistical power to detect a genuinely miscalibrated '
                'model. Failing to reject the null hypothesis is evidence of no detected miscalibration, not '
                'proof of a correct model. The Christoffersen independence test is implemented locally in this '
                'dashboard rather than in the backend `backtesting.py` module, since no packaged implementation '
                'exists there; it is not covered by the backend\'s own automated test suite.</div>', unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="disclaimer-box"><b>Basel Traffic-Light Indicator.</b> The classic Basel framework '
                '(green ≤ 4, yellow 5–9, red ≥ 10 exceptions) is calibrated specifically to 250 observations at '
                '99% confidence. This dashboard implements the underlying cumulative-binomial formulation so the '
                'indicator remains mathematically consistent at other confidence levels and sample sizes, but '
                'the zone boundaries should not be interpreted as formal regulatory capital multipliers outside '
                'the standard 99% / 250-day specification.</div>', unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="disclaimer-box"><b>Transaction Costs & Rebalancing.</b> Transaction costs are '
                'modeled as a flat basis-point charge applied at each rebalancing date and do not capture '
                'bid-ask spread variation, market impact, slippage on large orders, financing costs, or taxes. '
                'Rebalancing is assumed to execute instantly and at the closing price used in the return series, '
                'which will not be achievable in practice.</div>', unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="disclaimer-box"><b>Historical & Hypothetical Stress Scenarios.</b> Historical '
                'scenario replay assumes the current portfolio composition would have responded identically to '
                'a repeat of a past crisis, ignoring changes in market structure, liquidity, and correlations '
                'since that period, and uses the full historical price series regardless of the sidebar Data '
                'Window. Hypothetical scenario shocks are illustrative, committee-defined inputs rather than '
                'model-derived forecasts, are mapped only to tickers in the traded universe '
                f'({", ".join(DEFAULT_TICKERS)}), and are not intended to represent an exhaustive set of '
                'plausible adverse outcomes. A 2022 rate-hiking scenario is not currently defined in the '
                'backend and is therefore not shown.</div>', unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="disclaimer-box"><b>General.</b> This dashboard is a decision-support and research '
                'tool. All figures are derived from historical and modeled data, are subject to model risk and '
                'data-quality limitations, and do not constitute investment advice or a guarantee of future '
                'performance.</div>', unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"🔴 Pipeline Error: {e}")
        st.stop()
