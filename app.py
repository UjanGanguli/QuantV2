"""
╔══════════════════════════════════════════════════════════════════╗
║   QUANTITATIVE TRADING STRATEGY ENGINE  v2.0                     ║
║   Production-Grade Multi-Factor Signal + Portfolio Construction  ║
║   Walk-Forward Validation · Vol-Targeting · Regime Detection     ║
║   Cross-Sectional Ranking · Monte Carlo · Significance Testing   ║
║   Author: Ujan | Built for NSE Equities                          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats as sp_stats
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
import time

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="QuantEngine v2",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-card: #12121a;
        --bg-card-hover: #1a1a28;
        --accent-green: #00e88f;
        --accent-red: #ff4757;
        --accent-blue: #5b7fff;
        --accent-amber: #ffb347;
        --accent-purple: #a855f7;
        --text-primary: #e8e8ef;
        --text-muted: #6b6b80;
        --border: #1e1e30;
    }

    .stApp { font-family: 'Outfit', sans-serif; }

    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0;
        background: linear-gradient(135deg, #fff 0%, #5b7fff 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #6b6b80;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: -8px;
    }

    .metric-card {
        background: linear-gradient(145deg, #12121a, #1a1a28);
        border: 1px solid #1e1e30;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 6px 0;
    }

    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #6b6b80;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 4px;
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #e8e8ef;
    }

    .metric-positive { color: #00e88f; }
    .metric-negative { color: #ff4757; }
    .metric-neutral { color: #5b7fff; }
    .metric-highlight { color: #a855f7; }

    .signal-buy {
        background: rgba(0, 232, 143, 0.12);
        color: #00e88f;
        padding: 4px 14px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(0, 232, 143, 0.25);
        display: inline-block;
    }

    .signal-hold {
        background: rgba(107, 107, 128, 0.12);
        color: #6b6b80;
        padding: 4px 14px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(107, 107, 128, 0.25);
        display: inline-block;
    }

    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e8e8ef;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e1e30;
        margin: 24px 0 16px 0;
    }

    .explain-box {
        background: rgba(91, 127, 255, 0.06);
        border-left: 3px solid #5b7fff;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.88rem;
        color: #a0a0b8;
        line-height: 1.6;
    }

    .explain-box-warn {
        background: rgba(255, 179, 71, 0.06);
        border-left: 3px solid #ffb347;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.88rem;
        color: #a0a0b8;
        line-height: 1.6;
    }

    .explain-box-purple {
        background: rgba(168, 85, 247, 0.06);
        border-left: 3px solid #a855f7;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.88rem;
        color: #a0a0b8;
        line-height: 1.6;
    }

    .regime-bull {
        background: rgba(0, 232, 143, 0.15);
        color: #00e88f;
        padding: 3px 12px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
    }

    .regime-bear {
        background: rgba(255, 71, 87, 0.15);
        color: #ff4757;
        padding: 3px 12px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
    }

    .regime-neutral {
        background: rgba(91, 127, 255, 0.15);
        color: #5b7fff;
        padding: 3px 12px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
    }

    .stat-sig {
        background: rgba(0, 232, 143, 0.1);
        border: 1px solid rgba(0, 232, 143, 0.3);
        padding: 8px 16px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }

    .stat-insig {
        background: rgba(255, 71, 87, 0.1);
        border: 1px solid rgba(255, 71, 87, 0.3);
        padding: 8px 16px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    div[data-testid="stSidebar"] {
        background: #0d0d14;
        border-right: 1px solid #1e1e30;
    }

    div[data-testid="stSidebar"] .stMarkdown p {
        font-family: 'Outfit', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STOCK UNIVERSE
# ══════════════════════════════════════════════════════════════════

STOCK_UNIVERSE = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "BHARTIARTL.NS": "Bharti Airtel",
    "SBIN.NS": "State Bank of India",
    "ITC.NS": "ITC Limited",
    "LT.NS": "Larsen & Toubro",
    "WIPRO.NS": "Wipro",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "MARUTI.NS": "Maruti Suzuki",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "AXISBANK.NS": "Axis Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "SUNPHARMA.NS": "Sun Pharma",
    "TITAN.NS": "Titan Company",
    "ADANIENT.NS": "Adani Enterprises",
}

st.markdown("""
<div class="sub-header">BY UJAN GANGULI</div>

<div class="explain-box-purple">
    <b>QuantEngine v2</b> is a systematic analytics framework designed to decode market structure, 
    identify probabilistic trade setups, and translate complex quantitative signals into 
    actionable intelligence.

    <br><br>

    It blends <b>statistical rigor</b>, <b>volatility modeling</b>, and <b>regime detection</b> to 
    provide a structured view of price behavior — removing noise, reducing bias, and enhancing 
    decision clarity.

    <br><br>

    <span style="color:#a855f7;"><b>Core Philosophy:</b></span><br>
    • Markets are probabilistic, not deterministic<br>
    • Risk management dominates signal accuracy<br>
    • Consistency > prediction

    <br><br>

    <span style="color:#6b6b80;">Built for disciplined traders, not gamblers.</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def fetch_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_multi(tickers: list, period: str = "5y") -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple tickers."""
    data = {}
    for t in tickers:
        df = fetch_data(t, period)
        if not df.empty and len(df) > 100:
            data[t] = df
    return data


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def compute_features(
    df: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 50,
    rsi_window: int = 14,
) -> pd.DataFrame:
    """
    Feature Engineering Pipeline
    ─────────────────────────────
    1.  Log returns
    2.  Dual moving averages (trend)
    3.  RSI (momentum oscillator)
    4.  Realized volatility — 20d and 60d (risk)
    5.  Volume Z-score (participation)
    6.  MACD (trend confirmation)
    7.  Bollinger Band width (volatility regime)
    8.  Average True Range (execution cost proxy)
    9.  Rolling Sharpe (quality filter)
    10. Return skewness (tail risk)
    """
    out = df.copy()

    # Log returns
    out["returns"] = np.log(out["Close"] / out["Close"].shift(1))

    # Moving averages
    out["ma_fast"] = out["Close"].rolling(ma_fast).mean()
    out["ma_slow"] = out["Close"].rolling(ma_slow).mean()
    out["ma_spread"] = (out["ma_fast"] - out["ma_slow"]) / out["ma_slow"] * 100

    # RSI
    delta = out["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(rsi_window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_window).mean()
    rs = gain / loss
    out["RSI"] = 100 - (100 / (1 + rs))

    # Realized Volatility — dual horizon
    out["volatility_20d"] = out["returns"].rolling(20).std() * np.sqrt(252)
    out["volatility_60d"] = out["returns"].rolling(60).std() * np.sqrt(252)
    out["volatility"] = out["volatility_20d"]  # backward compat

    # Vol regime ratio (short/long vol — >1 means vol expanding)
    out["vol_regime_ratio"] = out["volatility_20d"] / out["volatility_60d"]

    # Volume Z-Score
    vol_mean = out["Volume"].rolling(20).mean()
    vol_std = out["Volume"].rolling(20).std()
    out["vol_z"] = (out["Volume"] - vol_mean) / vol_std

    # MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    # Bollinger Bands
    bb_ma = out["Close"].rolling(20).mean()
    bb_std = out["Close"].rolling(20).std()
    out["BB_upper"] = bb_ma + 2 * bb_std
    out["BB_lower"] = bb_ma - 2 * bb_std
    out["BB_width"] = (out["BB_upper"] - out["BB_lower"]) / bb_ma * 100
    out["BB_pctB"] = (out["Close"] - out["BB_lower"]) / (out["BB_upper"] - out["BB_lower"])

    # Average True Range (14-period) — for execution cost modeling
    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift(1)).abs()
    low_close = (out["Low"] - out["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR"] = true_range.rolling(14).mean()
    out["ATR_pct"] = out["ATR"] / out["Close"] * 100

    # Rolling 60-day Sharpe (quality filter)
    out["rolling_sharpe"] = (
        out["returns"].rolling(60).mean() / out["returns"].rolling(60).std()
    ) * np.sqrt(252)

    # Rolling 60-day skewness (tail risk)
    out["return_skew"] = out["returns"].rolling(60).skew()

    # Average daily volume (for capacity/impact)
    out["adv_20d"] = out["Volume"].rolling(20).mean()

    out.dropna(inplace=True)
    return out


# ══════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ══════════════════════════════════════════════════════════════════

@dataclass
class RegimeState:
    label: str          # "bull", "bear", "neutral"
    vol_regime: str     # "low_vol", "normal", "high_vol"
    confidence: float   # 0-1

def detect_regime(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Multi-dimensional regime classification.

    Trend regime (based on MA structure + returns):
      • Bull:    price > MA_slow AND 60d return > 0
      • Bear:    price < MA_slow AND 60d return < 0
      • Neutral: everything else

    Volatility regime (based on percentile rank of realized vol):
      • Low vol:  vol_20d in bottom 25th percentile of 1-year window
      • High vol: vol_20d in top 25th percentile
      • Normal:   middle 50%

    Returns a df with regime columns appended.
    """
    out = df.copy()

    # Rolling 60-day return
    out["ret_60d"] = out["Close"].pct_change(60)

    # Trend regime
    bull = (out["Close"] > out["ma_slow"]) & (out["ret_60d"] > 0)
    bear = (out["Close"] < out["ma_slow"]) & (out["ret_60d"] < 0)
    out["trend_regime"] = np.where(bull, "bull", np.where(bear, "bear", "neutral"))

    # Volatility regime — expanding percentile rank
    out["vol_pctile"] = out["volatility_20d"].rolling(252, min_periods=60).apply(
        lambda x: sp_stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False
    )
    out["vol_regime"] = np.where(
        out["vol_pctile"] < 0.25, "low_vol",
        np.where(out["vol_pctile"] > 0.75, "high_vol", "normal")
    )

    # Regime confidence — how clearly defined is the regime
    # Use distance from MA as % + trend consistency
    ma_dist = ((out["Close"] - out["ma_slow"]) / out["ma_slow"]).abs()
    trend_consistency = out["returns"].rolling(20).apply(
        lambda x: abs(np.mean(np.sign(x))), raw=True
    )
    out["regime_confidence"] = (ma_dist.clip(0, 0.1) / 0.1 * 0.5 + trend_consistency * 0.5).clip(0, 1)

    return out


# ══════════════════════════════════════════════════════════════════
# SIGNAL GENERATION — ENHANCED
# ══════════════════════════════════════════════════════════════════

def generate_signals(
    df: pd.DataFrame,
    momentum_wt: float = 1.0,
    mean_rev_wt: float = 1.0,
    volume_wt: float = 1.0,
    quality_filter: bool = True,
    regime_aware: bool = True,
) -> pd.DataFrame:
    """
    Multi-Factor Signal Generator v2
    ─────────────────────────────────
    Scores each bar on 3 base factors:
      • Momentum:       Price > Fast MA AND Fast MA > Slow MA AND MACD > signal
      • Mean Reversion: RSI < 35 AND BB %B < 0.2 (oversold + near lower band)
      • Volume Breakout: Vol Z-score > 1.5 (institutional interest)

    Additional filters:
      • Quality gate:   Rolling Sharpe > 0 (don't buy into a losing trend)
      • Regime scaling:  Scale signal confidence by regime (reduce in bear/high-vol)

    Output: continuous composite score [0, max_weight_sum] and binary signal.
    """
    out = df.copy()

    # ── Base factor scores (continuous, not binary where possible) ──

    # Momentum — graded: partial credit for weaker momentum
    strong_mom = (out["Close"] > out["ma_fast"]) & (out["ma_fast"] > out["ma_slow"]) & (out["MACD_hist"] > 0)
    weak_mom = (out["Close"] > out["ma_fast"]) & (out["ma_fast"] > out["ma_slow"])
    out["score_momentum"] = np.where(strong_mom, momentum_wt, np.where(weak_mom, momentum_wt * 0.5, 0.0))

    # Mean Reversion — graded by RSI depth
    rsi_score = np.where(out["RSI"] < 25, 1.0, np.where(out["RSI"] < 35, 0.6, 0.0))
    bb_confirm = np.where(out["BB_pctB"] < 0.2, 1.0, 0.5)
    out["score_mean_rev"] = rsi_score * bb_confirm * mean_rev_wt

    # Volume — graded by z-score magnitude
    vol_score = np.where(out["vol_z"] > 2.5, 1.0, np.where(out["vol_z"] > 1.5, 0.6, 0.0))
    out["score_volume"] = vol_score * volume_wt

    # ── Quality filter ──
    if quality_filter:
        quality_gate = np.where(out["rolling_sharpe"] > 0, 1.0, 0.5)
    else:
        quality_gate = 1.0

    # ── Regime scaling ──
    if regime_aware and "trend_regime" in out.columns:
        regime_scale = np.where(
            out["trend_regime"] == "bull", 1.0,
            np.where(out["trend_regime"] == "neutral", 0.7, 0.3)
        )
        vol_scale = np.where(
            out["vol_regime"] == "low_vol", 1.2,
            np.where(out["vol_regime"] == "normal", 1.0, 0.6)
        )
    else:
        regime_scale = 1.0
        vol_scale = 1.0

    # ── Composite score ──
    raw_score = out["score_momentum"] + out["score_mean_rev"] + out["score_volume"]
    out["composite_score"] = raw_score * quality_gate * regime_scale * vol_scale

    # Threshold — need at least 1 full factor equivalent
    threshold = max(momentum_wt, mean_rev_wt, volume_wt)
    out["signal"] = np.where(out["composite_score"] >= threshold, 1, 0)

    # Store the continuous score for cross-sectional ranking
    out["signal_strength"] = out["composite_score"]

    return out


# ══════════════════════════════════════════════════════════════════
# VECTORIZED MONTE CARLO
# ══════════════════════════════════════════════════════════════════

def monte_carlo_exit_vectorized(
    returns: np.ndarray,
    horizon: int = 5,
    n_sims: int = 500,
    exit_pct: float = 80,
) -> float:
    """
    Vectorized bootstrap Monte Carlo for exit decisions.
    ~50-100x faster than the loop-based version.
    """
    if len(returns) < 10:
        return 1.0
    sampled_indices = np.random.randint(0, len(returns), size=(n_sims, horizon))
    sampled_returns = returns[sampled_indices]
    terminal_wealth = np.exp(sampled_returns.sum(axis=1))
    return np.percentile(terminal_wealth, exit_pct)


def monte_carlo_cone(
    returns: np.ndarray,
    last_price: float,
    horizon: int = 30,
    n_sims: int = 1000,
) -> Dict[str, np.ndarray]:
    """Vectorized forward projection cone."""
    sampled_indices = np.random.randint(0, len(returns), size=(n_sims, horizon))
    sampled_returns = returns[sampled_indices]
    cum_returns = np.cumsum(sampled_returns, axis=1)
    paths = last_price * np.exp(cum_returns)

    return {
        "p5": np.percentile(paths, 5, axis=0),
        "p10": np.percentile(paths, 10, axis=0),
        "p25": np.percentile(paths, 25, axis=0),
        "p50": np.percentile(paths, 50, axis=0),
        "p75": np.percentile(paths, 75, axis=0),
        "p90": np.percentile(paths, 90, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
        "mean": np.mean(paths, axis=0),
        "paths": paths[:50],  # store subset for fan chart
    }


# ══════════════════════════════════════════════════════════════════
# POSITION SIZING — VOL-TARGETED
# ══════════════════════════════════════════════════════════════════

def compute_position_size(
    capital: float,
    price: float,
    current_vol: float,
    target_vol: float = 0.15,
    max_position_pct: float = 1.0,
    signal_strength: float = 1.0,
) -> float:
    """
    Volatility-targeted position sizing.

    Allocates capital such that the position's contribution to portfolio
    volatility ≈ target_vol. Scales linearly with signal strength.

    position_pct = (target_vol / realized_vol) * signal_strength
    capped at max_position_pct of capital.
    """
    if current_vol <= 0 or price <= 0:
        return 0.0

    raw_pct = (target_vol / current_vol) * signal_strength
    capped_pct = min(raw_pct, max_position_pct)

    shares = (capital * capped_pct) / price
    return max(shares, 0.0)


# ══════════════════════════════════════════════════════════════════
# EXECUTION COST MODEL
# ══════════════════════════════════════════════════════════════════

def estimate_execution_cost(
    trade_value: float,
    adv: float,
    atr_pct: float,
    base_slippage_bps: float = 5,
    commission_bps: float = 10,
) -> float:
    """
    Realistic execution cost model.

    Components:
    1. Fixed commission (brokerage)
    2. Base slippage (bid-ask spread proxy)
    3. Market impact: scales with trade_value/ADV and volatility
       impact_bps = atr_pct * 100 * sqrt(trade_value / adv_value)
       (simplified Almgren-Chriss square-root model)
    """
    commission = trade_value * commission_bps / 10000

    slippage = trade_value * base_slippage_bps / 10000

    # Market impact — sqrt model
    if adv > 0:
        participation_rate = trade_value / (adv * 100)  # rough price * volume
        impact_bps = atr_pct * 100 * np.sqrt(min(participation_rate, 0.1))
        impact = trade_value * impact_bps / 10000
    else:
        impact = 0.0

    return commission + slippage + impact


# ══════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE — ENHANCED
# ══════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    target_vol: float = 0.15
    max_position_pct: float = 1.0
    exit_mc_threshold: float = 1.005
    rsi_exit: float = 70
    base_slippage_bps: float = 5
    commission_bps: float = 10
    use_vol_sizing: bool = True
    mc_exit_horizon: int = 5
    mc_exit_sims: int = 500
    lookback: int = 60


def run_backtest(df: pd.DataFrame, config: BacktestConfig) -> dict:
    """
    Enhanced Event-Driven Backtesting Engine
    ─────────────────────────────────────────
    Improvements over v1:
    • Vol-targeted position sizing
    • Execution cost model with market impact
    • Regime-aware exit thresholds
    • Detailed trade analytics
    """
    capital = config.initial_capital
    position = 0.0
    entry_price = 0.0
    entry_date = None
    equity_curve = []
    trades = []
    signals_log = []
    daily_positions = []  # for turnover calc
    total_costs = 0.0

    returns_arr = df["returns"].values
    close_arr = df["Close"].values
    rsi_arr = df["RSI"].values
    signal_arr = df["signal"].values
    vol_arr = df["volatility_20d"].values
    atr_pct_arr = df["ATR_pct"].values
    adv_arr = df["adv_20d"].values
    signal_strength_arr = df["signal_strength"].values if "signal_strength" in df.columns else signal_arr.astype(float)
    dates = df.index

    # Regime-adjusted exit thresholds
    has_regime = "trend_regime" in df.columns
    if has_regime:
        regime_arr = df["trend_regime"].values
    else:
        regime_arr = np.full(len(df), "neutral")

    lookback = config.lookback

    for i in range(lookback, len(df)):
        sig = signal_arr[i]
        price = close_arr[i]
        rsi_val = rsi_arr[i]
        current_vol = vol_arr[i]
        regime = regime_arr[i]

        # Regime-adjusted exit threshold
        if regime == "bear":
            exit_threshold = config.exit_mc_threshold * 1.01  # tighter exit in bear
            rsi_exit = config.rsi_exit - 5
        elif regime == "bull":
            exit_threshold = config.exit_mc_threshold * 0.99  # looser in bull
            rsi_exit = config.rsi_exit + 5
        else:
            exit_threshold = config.exit_mc_threshold
            rsi_exit = config.rsi_exit

        # ── ENTRY ──
        if sig == 1 and position == 0:
            if config.use_vol_sizing:
                position = compute_position_size(
                    capital, price, current_vol,
                    config.target_vol, config.max_position_pct,
                    signal_strength=min(signal_strength_arr[i] / 2.0, 1.0),
                )
            else:
                position = capital / price

            trade_value = position * price
            cost = estimate_execution_cost(
                trade_value, adv_arr[i], atr_pct_arr[i],
                config.base_slippage_bps, config.commission_bps,
            )
            capital -= (trade_value + cost)
            total_costs += cost
            entry_price = price
            entry_date = dates[i]

            trades.append({
                "date": dates[i], "type": "BUY", "price": price,
                "shares": round(position, 2), "cost": round(cost, 2),
                "regime": regime,
            })
            signals_log.append({"date": dates[i], "action": "BUY", "price": price})

        # ── EXIT ──
        elif position > 0:
            hist_returns = returns_arr[max(0, i - lookback):i]
            expected_move = monte_carlo_exit_vectorized(
                hist_returns, config.mc_exit_horizon, config.mc_exit_sims,
            )

            if expected_move < exit_threshold or rsi_val > rsi_exit:
                trade_value = position * price
                cost = estimate_execution_cost(
                    trade_value, adv_arr[i], atr_pct_arr[i],
                    config.base_slippage_bps, config.commission_bps,
                )
                capital += (trade_value - cost)
                total_costs += cost

                pnl_pct = (price / entry_price - 1) * 100
                hold_days = (dates[i] - entry_date).days if entry_date else 0

                trades.append({
                    "date": dates[i], "type": "SELL", "price": price,
                    "shares": round(position, 2), "cost": round(cost, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "hold_days": hold_days,
                    "regime": regime,
                })
                signals_log.append({"date": dates[i], "action": "SELL", "price": price})
                position = 0.0
                entry_price = 0.0
                entry_date = None

        equity = (position * price + capital) if position > 0 else capital
        equity_curve.append(equity)
        daily_positions.append(position * price if position > 0 else 0.0)

    equity_series = pd.Series(equity_curve, index=dates[lookback:], name="Equity")

    # Turnover calculation
    position_series = pd.Series(daily_positions, index=dates[lookback:])
    daily_turnover = position_series.diff().abs()
    avg_turnover = daily_turnover.mean() / config.initial_capital if config.initial_capital > 0 else 0

    return {
        "equity": equity_series,
        "trades": trades,
        "signals": signals_log,
        "final_capital": equity_curve[-1] if equity_curve else config.initial_capital,
        "total_costs": total_costs,
        "avg_daily_turnover": avg_turnover,
    }


# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════

def run_walk_forward(
    df: pd.DataFrame,
    config: BacktestConfig,
    n_folds: int = 5,
    train_pct: float = 0.7,
    params_to_test: dict = None,
) -> dict:
    """
    Walk-Forward Validation Engine
    ──────────────────────────────
    Splits data into n rolling windows. For each:
    1. Train on first train_pct of the window
    2. Test on remaining (1-train_pct) — NEVER SEEN during training
    3. Aggregate OOS (out-of-sample) performance

    This prevents overfitting from manual slider tuning.
    """
    n = len(df)
    fold_size = n // n_folds
    results = []
    oos_equities = []

    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = min(start_idx + fold_size + config.lookback, n)

        if end_idx - start_idx < config.lookback + 60:
            continue

        fold_data = df.iloc[start_idx:end_idx].copy()
        split_idx = int(len(fold_data) * train_pct)

        train_data = fold_data.iloc[:split_idx]
        test_data = fold_data.iloc[split_idx:]

        if len(test_data) < config.lookback + 20:
            continue

        # Run backtest on TEST data only (OOS)
        bt = run_backtest(test_data, config)

        if len(bt["equity"]) < 5:
            continue

        returns = bt["equity"].pct_change().dropna()
        if len(returns) < 5:
            continue

        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        total_ret = bt["equity"].iloc[-1] / bt["equity"].iloc[0] - 1

        dd = bt["equity"] / bt["equity"].cummax() - 1
        max_dd = dd.min()

        results.append({
            "fold": fold + 1,
            "train_start": train_data.index[0].strftime("%Y-%m-%d"),
            "train_end": train_data.index[-1].strftime("%Y-%m-%d"),
            "test_start": test_data.index[config.lookback if config.lookback < len(test_data) else 0].strftime("%Y-%m-%d"),
            "test_end": test_data.index[-1].strftime("%Y-%m-%d"),
            "oos_return": round(total_ret * 100, 2),
            "oos_sharpe": round(sharpe, 2),
            "oos_max_dd": round(max_dd * 100, 2),
            "n_trades": len([t for t in bt["trades"] if t["type"] == "SELL"]),
        })

        oos_equities.append(bt["equity"])

    # Aggregate OOS metrics
    if results:
        avg_sharpe = np.mean([r["oos_sharpe"] for r in results])
        avg_return = np.mean([r["oos_return"] for r in results])
        avg_dd = np.mean([r["oos_max_dd"] for r in results])
        sharpe_std = np.std([r["oos_sharpe"] for r in results])
        consistency = sum(1 for r in results if r["oos_sharpe"] > 0) / len(results)
    else:
        avg_sharpe = avg_return = avg_dd = sharpe_std = consistency = 0

    return {
        "folds": results,
        "oos_equities": oos_equities,
        "avg_oos_sharpe": round(avg_sharpe, 2),
        "avg_oos_return": round(avg_return, 2),
        "avg_oos_max_dd": round(avg_dd, 2),
        "sharpe_stability": round(sharpe_std, 2),
        "consistency": round(consistency * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE TESTING
# ══════════════════════════════════════════════════════════════════

def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> dict:
    """
    Bootstrap confidence interval for Sharpe ratio.
    Resamples daily returns with replacement, computes Sharpe
    for each sample, returns percentile CI.
    """
    returns_arr = returns.values
    n = len(returns_arr)
    sharpe_samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(returns_arr, size=n, replace=True)
        mu = sample.mean()
        sigma = sample.std()
        sharpe_samples[i] = np.sqrt(252) * mu / sigma if sigma > 0 else 0

    alpha = (1 - confidence) / 2
    ci_low = np.percentile(sharpe_samples, alpha * 100)
    ci_high = np.percentile(sharpe_samples, (1 - alpha) * 100)

    # p-value: proportion of bootstrap samples with Sharpe <= 0
    p_value = np.mean(sharpe_samples <= 0)

    return {
        "sharpe_mean": round(np.mean(sharpe_samples), 3),
        "sharpe_median": round(np.median(sharpe_samples), 3),
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
        "p_value": round(p_value, 4),
        "is_significant": p_value < 0.05,
        "distribution": sharpe_samples,
    }


def permutation_test_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_perms: int = 2000,
) -> dict:
    """
    Permutation test for strategy alpha.
    Randomly shuffles strategy returns to create a null distribution
    of alpha, then checks if observed alpha is statistically significant.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {"p_value": 1.0, "is_significant": False, "observed_alpha": 0}

    aligned.columns = ["strategy", "benchmark"]
    observed_excess = (aligned["strategy"] - aligned["benchmark"]).mean() * 252

    null_alphas = np.empty(n_perms)
    excess = (aligned["strategy"] - aligned["benchmark"]).values

    for i in range(n_perms):
        # Randomly flip signs (equivalent to random assignment)
        signs = np.random.choice([-1, 1], size=len(excess))
        null_alphas[i] = (excess * signs).mean() * 252

    p_value = np.mean(np.abs(null_alphas) >= abs(observed_excess))

    return {
        "observed_alpha": round(observed_excess * 100, 3),
        "p_value": round(p_value, 4),
        "is_significant": p_value < 0.05,
        "null_distribution": null_alphas,
    }


# ══════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════

def rank_universe_cross_sectional(
    universe_data: Dict[str, pd.DataFrame],
    top_n: int = 5,
    ma_fast: int = 20,
    ma_slow: int = 50,
    weights: tuple = (1.0, 1.0, 1.0),
) -> pd.DataFrame:
    """
    Cross-sectional ranking of the entire stock universe.
    For each stock: compute features → score → rank.
    Returns ranked DataFrame with signal strengths.
    """
    rankings = []

    for ticker, raw_df in universe_data.items():
        try:
            feat_df = compute_features(raw_df, ma_fast, ma_slow)
            if len(feat_df) < 60:
                continue
            feat_df = detect_regime(feat_df)
            sig_df = generate_signals(feat_df, weights[0], weights[1], weights[2])

            latest = sig_df.iloc[-1]
            recent_5d = sig_df.tail(5)

            rankings.append({
                "ticker": ticker,
                "name": STOCK_UNIVERSE.get(ticker, ticker),
                "price": round(float(latest["Close"]), 2),
                "signal_strength": round(float(latest["signal_strength"]), 3),
                "signal": int(latest["signal"]),
                "RSI": round(float(latest["RSI"]), 1),
                "volatility": round(float(latest["volatility_20d"]) * 100, 1),
                "regime": str(latest["trend_regime"]),
                "vol_regime": str(latest["vol_regime"]),
                "ma_spread": round(float(latest["ma_spread"]), 2),
                "rolling_sharpe": round(float(latest["rolling_sharpe"]), 2),
                "5d_avg_signal": round(float(recent_5d["signal_strength"].mean()), 3),
                "return_20d": round(float(raw_df["Close"].pct_change(20).iloc[-1]) * 100, 2),
            })
        except Exception:
            continue

    if not rankings:
        return pd.DataFrame()

    rank_df = pd.DataFrame(rankings)
    rank_df = rank_df.sort_values("signal_strength", ascending=False).reset_index(drop=True)
    rank_df["rank"] = range(1, len(rank_df) + 1)
    return rank_df


def construct_portfolio_weights(
    rank_df: pd.DataFrame,
    universe_data: Dict[str, pd.DataFrame],
    method: str = "inverse_vol",
    top_n: int = 5,
) -> Dict[str, float]:
    """
    Portfolio construction from cross-sectional rankings.

    Methods:
    • equal_weight: 1/N allocation to top-N
    • inverse_vol: Weight inversely proportional to realized vol
    • signal_weighted: Weight proportional to signal strength
    """
    top = rank_df.head(top_n)
    tickers = top["ticker"].tolist()

    if method == "equal_weight":
        w = {t: 1.0 / len(tickers) for t in tickers}

    elif method == "inverse_vol":
        vols = {}
        for t in tickers:
            if t in universe_data:
                ret = universe_data[t]["Close"].pct_change().dropna()
                vols[t] = ret.tail(60).std() * np.sqrt(252)
        if not vols:
            return {t: 1.0 / len(tickers) for t in tickers}
        inv_vols = {t: 1.0 / v for t, v in vols.items() if v > 0}
        total = sum(inv_vols.values())
        w = {t: v / total for t, v in inv_vols.items()}

    elif method == "signal_weighted":
        strengths = dict(zip(top["ticker"], top["signal_strength"]))
        total = sum(strengths.values())
        if total > 0:
            w = {t: s / total for t, s in strengths.items()}
        else:
            w = {t: 1.0 / len(tickers) for t in tickers}
    else:
        w = {t: 1.0 / len(tickers) for t in tickers}

    return w


# ══════════════════════════════════════════════════════════════════
# PERFORMANCE ANALYTICS
# ══════════════════════════════════════════════════════════════════

def compute_metrics(
    equity: pd.Series,
    initial_capital: float = 100000.0,
    benchmark_returns: pd.Series = None,
) -> dict:
    """Comprehensive performance analytics."""
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return {}

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = len(returns) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

    downside = returns[returns < 0]
    sortino_denom = downside.std() if len(downside) > 0 else returns.std()
    sortino = np.sqrt(252) * returns.mean() / sortino_denom if sortino_denom > 0 else 0

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    win_rate = (returns > 0).sum() / len(returns) * 100

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Tail ratio (95th percentile gain / 5th percentile loss)
    p95 = np.percentile(returns, 95)
    p5 = abs(np.percentile(returns, 5))
    tail_ratio = p95 / p5 if p5 > 0 else float("inf")

    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(252)

    metrics = {
        "Total Return (%)": round(total_ret * 100, 2),
        "CAGR (%)": round(cagr * 100, 2),
        "Ann. Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Calmar Ratio": round(calmar, 2),
        "Daily Win Rate (%)": round(win_rate, 1),
        "Profit Factor": round(profit_factor, 2),
        "Tail Ratio": round(tail_ratio, 2),
    }

    if benchmark_returns is not None:
        aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
        if len(aligned) > 10:
            aligned.columns = ["strategy", "benchmark"]
            cov = np.cov(aligned["strategy"], aligned["benchmark"])
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
            alpha = (returns.mean() - beta * benchmark_returns.mean()) * 252

            # Information ratio
            active_returns = aligned["strategy"] - aligned["benchmark"]
            tracking_error = active_returns.std() * np.sqrt(252)
            info_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0

            metrics["Beta"] = round(beta, 2)
            metrics["Alpha (ann.)"] = round(alpha * 100, 2)
            metrics["Information Ratio"] = round(info_ratio, 2)
            metrics["Tracking Error (%)"] = round(tracking_error * 100, 2)

    return metrics


# ══════════════════════════════════════════════════════════════════
# CHARTING
# ══════════════════════════════════════════════════════════════════

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,15,0.6)",
    font=dict(family="Outfit, sans-serif", color="#a0a0b8"),
    margin=dict(l=40, r=20, t=40, b=30),
    xaxis=dict(gridcolor="#1e1e30", zerolinecolor="#1e1e30"),
    yaxis=dict(gridcolor="#1e1e30", zerolinecolor="#1e1e30"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)


def plot_candlestick_with_signals(df, signals, title="Price Action & Signals"):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[title, "RSI", "Volume"],
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#00e88f", decreasing_line_color="#ff4757",
        increasing_fillcolor="#00e88f", decreasing_fillcolor="#ff4757",
        name="Price", line=dict(width=1),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["ma_fast"], mode="lines",
        line=dict(color="#5b7fff", width=1.2), name="MA Fast", opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_slow"], mode="lines",
        line=dict(color="#ffb347", width=1.2), name="MA Slow", opacity=0.8), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], mode="lines",
        line=dict(color="rgba(91,127,255,0.2)", width=0.8), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], mode="lines",
        line=dict(color="rgba(91,127,255,0.2)", width=0.8),
        fill="tonexty", fillcolor="rgba(91,127,255,0.04)", showlegend=False), row=1, col=1)

    buys = [s for s in signals if s["action"] == "BUY"]
    sells = [s for s in signals if s["action"] == "SELL"]
    if buys:
        fig.add_trace(go.Scatter(
            x=[b["date"] for b in buys], y=[b["price"] for b in buys],
            mode="markers", marker=dict(symbol="triangle-up", size=12, color="#00e88f",
            line=dict(width=1, color="#fff")), name="BUY"), row=1, col=1)
    if sells:
        fig.add_trace(go.Scatter(
            x=[s["date"] for s in sells], y=[s["price"] for s in sells],
            mode="markers", marker=dict(symbol="triangle-down", size=12, color="#ff4757",
            line=dict(width=1, color="#fff")), name="SELL"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines",
        line=dict(color="#5b7fff", width=1.2), showlegend=False), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ff4757", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00e88f", opacity=0.5, row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(91,127,255,0.04)", line_width=0, row=2, col=1)

    colors = ["#00e88f" if c >= o else "#ff4757" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, opacity=0.6,
        showlegend=False), row=3, col=1)

    fig.update_layout(**CHART_LAYOUT, height=680, showlegend=True)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def plot_equity_curve(equity, benchmark_equity=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines",
        line=dict(color="#00e88f", width=2.2),
        fill="tozeroy", fillcolor="rgba(0,232,143,0.06)", name="Strategy"))
    if benchmark_equity is not None:
        fig.add_trace(go.Scatter(x=benchmark_equity.index, y=benchmark_equity.values, mode="lines",
            line=dict(color="#6b6b80", width=1.5, dash="dot"), name="Buy & Hold"))
    fig.update_layout(**CHART_LAYOUT, height=350, title="Equity Curve (₹)")
    return fig


def plot_drawdown(equity):
    dd = (equity / equity.cummax() - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines",
        line=dict(color="#ff4757", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,71,87,0.1)", name="Drawdown"))
    fig.update_layout(**CHART_LAYOUT, height=250, title="Drawdown (%)")
    return fig


def plot_monthly_returns(equity):
    monthly = equity.resample("ME").last().pct_change().dropna()
    monthly_df = pd.DataFrame({"Month": monthly.index.strftime("%b %Y"), "Return": monthly.values * 100})
    colors = ["#00e88f" if r >= 0 else "#ff4757" for r in monthly_df["Return"]]
    fig = go.Figure(go.Bar(x=monthly_df["Month"], y=monthly_df["Return"], marker_color=colors, opacity=0.85))
    fig.update_layout(**CHART_LAYOUT, height=280, title="Monthly Returns (%)")
    return fig


def plot_mc_cone(returns_arr, last_price, horizon=30, n_sims=1000):
    mc = monte_carlo_cone(returns_arr, last_price, horizon, n_sims)
    days = list(range(1, horizon + 1))

    fig = go.Figure()

    # Fan chart layers
    fig.add_trace(go.Scatter(x=days, y=mc["p95"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=mc["p5"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(91,127,255,0.05)", name="5-95th pctl"))

    fig.add_trace(go.Scatter(x=days, y=mc["p90"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=mc["p10"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(91,127,255,0.08)", name="10-90th pctl"))

    fig.add_trace(go.Scatter(x=days, y=mc["p75"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=mc["p25"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(91,127,255,0.15)", name="25-75th pctl"))

    fig.add_trace(go.Scatter(x=days, y=mc["p50"], mode="lines",
        line=dict(color="#5b7fff", width=2), name="Median"))
    fig.add_trace(go.Scatter(x=days, y=mc["mean"], mode="lines",
        line=dict(color="#a855f7", width=1.5, dash="dash"), name="Mean"))

    # Sample paths
    for path in mc["paths"][:15]:
        fig.add_trace(go.Scatter(x=days, y=path, mode="lines",
            line=dict(color="rgba(91,127,255,0.08)", width=0.5), showlegend=False))

    fig.add_hline(y=last_price, line_dash="dot", line_color="#6b6b80", opacity=0.5)
    fig.update_layout(**CHART_LAYOUT, height=380, title=f"Monte Carlo {horizon}-Day Projection (₹)",
        xaxis_title="Days Forward")
    return fig


def plot_factor_heatmap(df):
    recent = df.tail(60)[["score_momentum", "score_mean_rev", "score_volume"]].copy()
    recent.columns = ["Momentum", "Mean Reversion", "Volume"]
    fig = go.Figure(go.Heatmap(
        z=recent.values.T, x=recent.index.strftime("%d %b"),
        y=["Momentum", "Mean Rev.", "Volume"],
        colorscale=[[0, "#12121a"], [0.3, "#1a1a28"], [0.6, "#2a4a3a"], [1, "#00e88f"]],
        showscale=False,
    ))
    fig.update_layout(**CHART_LAYOUT, height=180, title="Factor Activation (Last 60 Days)")
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def plot_regime_timeline(df):
    """Regime timeline visualization."""
    recent = df.tail(252).copy()  # last year

    regime_colors = {"bull": "#00e88f", "bear": "#ff4757", "neutral": "#5b7fff"}
    vol_colors = {"low_vol": "#00e88f", "normal": "#ffb347", "high_vol": "#ff4757"}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.6, 0.4], subplot_titles=["Price + Trend Regime", "Volatility Regime"])

    fig.add_trace(go.Scatter(x=recent.index, y=recent["Close"], mode="lines",
        line=dict(color="#e8e8ef", width=1.5), name="Price"), row=1, col=1)

    # Color background by regime
    for regime, color in regime_colors.items():
        mask = recent["trend_regime"] == regime
        if mask.any():
            fig.add_trace(go.Scatter(
                x=recent.index[mask], y=recent["Close"][mask],
                mode="markers", marker=dict(color=color, size=3), name=regime.title(),
            ), row=1, col=1)

    # Vol regime
    fig.add_trace(go.Scatter(x=recent.index, y=recent["volatility_20d"] * 100, mode="lines",
        line=dict(color="#a855f7", width=1.5), name="20d Vol"), row=2, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent["volatility_60d"] * 100, mode="lines",
        line=dict(color="#6b6b80", width=1, dash="dot"), name="60d Vol"), row=2, col=1)

    fig.update_layout(**CHART_LAYOUT, height=450, showlegend=True)
    return fig


def plot_sharpe_bootstrap(sharpe_dist, ci_low, ci_high, observed_sharpe):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sharpe_dist, nbinsx=60, marker_color="rgba(91,127,255,0.5)",
        marker_line=dict(color="#5b7fff", width=0.5), name="Bootstrap Sharpe",
    ))
    fig.add_vline(x=0, line_dash="solid", line_color="#ff4757", line_width=2,
        annotation_text="Sharpe = 0", annotation_position="top")
    fig.add_vline(x=observed_sharpe, line_dash="solid", line_color="#00e88f", line_width=2,
        annotation_text=f"Observed: {observed_sharpe:.2f}", annotation_position="top")
    fig.add_vrect(x0=ci_low, x1=ci_high, fillcolor="rgba(168,85,247,0.1)",
        line_width=0, annotation_text="95% CI")
    fig.update_layout(**CHART_LAYOUT, height=300, title="Bootstrap Sharpe Ratio Distribution",
        xaxis_title="Sharpe Ratio", yaxis_title="Frequency")
    return fig


def plot_walk_forward_results(wf_results):
    folds = wf_results["folds"]
    if not folds:
        return None

    fig = make_subplots(rows=1, cols=2, subplot_titles=["OOS Return by Fold", "OOS Sharpe by Fold"])

    fold_nums = [f["fold"] for f in folds]
    oos_returns = [f["oos_return"] for f in folds]
    oos_sharpes = [f["oos_sharpe"] for f in folds]

    colors_ret = ["#00e88f" if r >= 0 else "#ff4757" for r in oos_returns]
    colors_sh = ["#00e88f" if s >= 0 else "#ff4757" for s in oos_sharpes]

    fig.add_trace(go.Bar(x=fold_nums, y=oos_returns, marker_color=colors_ret, name="OOS Return %"), row=1, col=1)
    fig.add_trace(go.Bar(x=fold_nums, y=oos_sharpes, marker_color=colors_sh, name="OOS Sharpe"), row=1, col=2)

    fig.update_layout(**CHART_LAYOUT, height=300, showlegend=False)
    fig.update_xaxes(title_text="Fold", row=1, col=1)
    fig.update_xaxes(title_text="Fold", row=1, col=2)
    return fig


# ══════════════════════════════════════════════════════════════════
# INFORMATION COEFFICIENT (IC) ANALYSIS
# ══════════════════════════════════════════════════════════════════

def compute_rolling_ic(
    universe_data: Dict[str, pd.DataFrame],
    ma_fast: int = 20,
    ma_slow: int = 50,
    weights: tuple = (1.0, 1.0, 1.0),
    forward_days: list = None,
    rolling_window: int = 60,
) -> dict:
    """
    Rolling cross-sectional Information Coefficient.

    For each date:
    1. Compute signal_strength for every stock in the universe
    2. Compute forward N-day return for every stock
    3. Spearman rank correlation between signal and forward return

    IC > 0.05 with t-stat > 2 ≈ exploitable signal.
    IC > 0.10 is exceptional for daily equity signals.
    """
    if forward_days is None:
        forward_days = [5, 10, 20]

    # Pre-compute features + signals for all stocks
    processed = {}
    for ticker, raw_df in universe_data.items():
        try:
            feat = compute_features(raw_df, ma_fast, ma_slow)
            if len(feat) < 100:
                continue
            feat = detect_regime(feat)
            sig = generate_signals(feat, weights[0], weights[1], weights[2])
            processed[ticker] = sig
        except Exception:
            continue

    if len(processed) < 5:
        return {"error": "Not enough stocks with valid data"}

    # Find common date range
    all_indices = [df.index for df in processed.values()]
    common_start = max(idx.min() for idx in all_indices)
    common_end = min(idx.max() for idx in all_indices)

    # Build date range for IC computation
    ref_dates = processed[list(processed.keys())[0]].loc[common_start:common_end].index

    results = {}
    for fwd in forward_days:
        ic_series = []
        ic_dates = []

        for i in range(rolling_window, len(ref_dates) - fwd):
            date = ref_dates[i]

            signals = []
            fwd_returns = []

            for ticker, df in processed.items():
                if date in df.index:
                    loc = df.index.get_loc(date)
                    if loc + fwd < len(df):
                        sig_val = df.iloc[loc]["signal_strength"]
                        fwd_ret = np.log(df.iloc[loc + fwd]["Close"] / df.iloc[loc]["Close"])
                        signals.append(sig_val)
                        fwd_returns.append(fwd_ret)

            if len(signals) >= 5:
                # Spearman rank correlation
                corr, p_val = sp_stats.spearmanr(signals, fwd_returns)
                if not np.isnan(corr):
                    ic_series.append(corr)
                    ic_dates.append(date)

        if ic_series:
            ic_arr = np.array(ic_series)
            avg_ic = np.mean(ic_arr)
            ic_std = np.std(ic_arr)
            t_stat = avg_ic / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
            hit_rate = np.mean(ic_arr > 0) * 100

            results[f"{fwd}d"] = {
                "ic_series": pd.Series(ic_arr, index=ic_dates),
                "avg_ic": round(avg_ic, 4),
                "ic_std": round(ic_std, 4),
                "t_stat": round(t_stat, 2),
                "hit_rate": round(hit_rate, 1),
                "is_significant": abs(t_stat) > 2.0,
                "n_observations": len(ic_arr),
            }

    return results


def plot_rolling_ic(ic_results: dict) -> go.Figure:
    """Plot rolling IC for multiple forward horizons."""
    fig = make_subplots(rows=len(ic_results), cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"{k} Forward IC (avg={v['avg_ic']:.4f}, t={v['t_stat']:.1f})"
                        for k, v in ic_results.items()])

    colors = ["#5b7fff", "#00e88f", "#a855f7"]
    for i, (horizon, data) in enumerate(ic_results.items()):
        ic_s = data["ic_series"]
        # Rolling 20-period average for smoothing
        ic_smooth = ic_s.rolling(20, min_periods=5).mean()

        fig.add_trace(go.Scatter(
            x=ic_s.index, y=ic_s.values, mode="lines",
            line=dict(color=colors[i % 3], width=0.6), opacity=0.3,
            name=f"{horizon} IC (raw)", showlegend=False,
        ), row=i+1, col=1)

        fig.add_trace(go.Scatter(
            x=ic_smooth.index, y=ic_smooth.values, mode="lines",
            line=dict(color=colors[i % 3], width=2),
            name=f"{horizon} IC (20d avg)",
        ), row=i+1, col=1)

        fig.add_hline(y=0, line_dash="solid", line_color="#ff4757", line_width=1,
            opacity=0.5, row=i+1, col=1)
        fig.add_hline(y=data["avg_ic"], line_dash="dash", line_color="#ffb347",
            line_width=1, opacity=0.7, row=i+1, col=1)

    height = max(300, 200 * len(ic_results))
    fig.update_layout(**CHART_LAYOUT, height=height, showlegend=True)
    return fig


# ══════════════════════════════════════════════════════════════════
# FACTOR CORRELATION DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════

def compute_factor_correlations(df: pd.DataFrame) -> dict:
    """
    Compute static and rolling correlations between factor scores.
    Exposes whether your 'multi-factor' model is actually one bet.
    """
    factor_cols = ["score_momentum", "score_mean_rev", "score_volume"]
    factors = df[factor_cols].copy()
    factors.columns = ["Momentum", "Mean Rev.", "Volume"]

    # Static correlation
    static_corr = factors.corr(method="spearman")

    # Rolling 60-day correlation (pairwise)
    rolling_corrs = {}
    pairs = [("Momentum", "Mean Rev."), ("Momentum", "Volume"), ("Mean Rev.", "Volume")]
    for a, b in pairs:
        rolling_corrs[f"{a} × {b}"] = factors[a].rolling(60, min_periods=30).corr(factors[b])

    return {
        "static": static_corr,
        "rolling": rolling_corrs,
        "factor_data": factors,
    }


def plot_factor_correlation_matrix(corr_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of factor correlations."""
    labels = corr_matrix.columns.tolist()
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=labels, y=labels,
        colorscale=[[0, "#ff4757"], [0.5, "#12121a"], [1, "#00e88f"]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 3),
        texttemplate="%{text}",
        textfont=dict(size=14, family="JetBrains Mono"),
        showscale=True,
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(**CHART_LAYOUT, height=320, title="Factor Rank Correlation Matrix (Spearman)")
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_rolling_factor_correlations(rolling_corrs: dict) -> go.Figure:
    """Time series of rolling pairwise factor correlations."""
    fig = go.Figure()
    colors = ["#5b7fff", "#00e88f", "#a855f7"]

    for i, (pair_name, series) in enumerate(rolling_corrs.items()):
        clean = series.dropna()
        fig.add_trace(go.Scatter(
            x=clean.index, y=clean.values, mode="lines",
            line=dict(color=colors[i % 3], width=1.5),
            name=pair_name,
        ))

    fig.add_hline(y=0, line_dash="solid", line_color="#6b6b80", line_width=1, opacity=0.5)
    fig.add_hline(y=0.7, line_dash="dot", line_color="#ff4757", line_width=1, opacity=0.5,
        annotation_text="High correlation danger zone")
    fig.add_hline(y=-0.7, line_dash="dot", line_color="#ff4757", line_width=1, opacity=0.5)

    fig.update_layout(**CHART_LAYOUT, height=320, title="Rolling 60-Day Factor Correlations",
        yaxis_title="Spearman ρ")
    return fig


# ══════════════════════════════════════════════════════════════════
# COST SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════

def cost_sensitivity_analysis(
    df: pd.DataFrame,
    base_config: BacktestConfig,
    cost_range_bps: list = None,
) -> pd.DataFrame:
    """
    Run backtest at varying round-trip cost assumptions.
    Returns Sharpe, return, and max DD for each cost level.
    Identifies the breakeven cost where Sharpe → 0.
    """
    if cost_range_bps is None:
        cost_range_bps = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

    results = []
    for total_cost in cost_range_bps:
        config = BacktestConfig(
            initial_capital=base_config.initial_capital,
            target_vol=base_config.target_vol,
            max_position_pct=base_config.max_position_pct,
            exit_mc_threshold=base_config.exit_mc_threshold,
            rsi_exit=base_config.rsi_exit,
            base_slippage_bps=total_cost // 2,
            commission_bps=total_cost - total_cost // 2,
            use_vol_sizing=base_config.use_vol_sizing,
            mc_exit_horizon=base_config.mc_exit_horizon,
            mc_exit_sims=base_config.mc_exit_sims,
            lookback=base_config.lookback,
        )

        bt = run_backtest(df, config)
        equity = bt["equity"]
        returns = equity.pct_change().dropna()

        if len(returns) < 10:
            continue

        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        dd = (equity / equity.cummax() - 1).min() * 100
        n_trades = len([t for t in bt["trades"] if t["type"] == "SELL"])

        results.append({
            "cost_bps": total_cost,
            "sharpe": round(sharpe, 3),
            "total_return_pct": round(total_ret, 2),
            "max_dd_pct": round(dd, 2),
            "n_trades": n_trades,
            "total_costs": round(bt["total_costs"], 0),
        })

    return pd.DataFrame(results)


def find_breakeven_cost(sensitivity_df: pd.DataFrame) -> float:
    """Find the cost level where Sharpe crosses zero (linear interpolation)."""
    df = sensitivity_df.sort_values("cost_bps")
    for i in range(len(df) - 1):
        s1 = df.iloc[i]["sharpe"]
        s2 = df.iloc[i + 1]["sharpe"]
        c1 = df.iloc[i]["cost_bps"]
        c2 = df.iloc[i + 1]["cost_bps"]
        if s1 >= 0 and s2 < 0:
            # Linear interpolation
            breakeven = c1 + (0 - s1) * (c2 - c1) / (s2 - s1)
            return round(breakeven, 1)
    # If never crosses zero
    if df["sharpe"].iloc[-1] > 0:
        return float("inf")
    return 0.0


def plot_cost_sensitivity(sensitivity_df: pd.DataFrame, breakeven: float) -> go.Figure:
    """Plot Sharpe ratio as a function of transaction costs."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Sharpe vs. Cost", "Return vs. Cost"])

    # Sharpe curve
    colors = ["#00e88f" if s >= 0 else "#ff4757" for s in sensitivity_df["sharpe"]]
    fig.add_trace(go.Scatter(
        x=sensitivity_df["cost_bps"], y=sensitivity_df["sharpe"],
        mode="lines+markers", line=dict(color="#5b7fff", width=2.5),
        marker=dict(size=8, color=colors, line=dict(width=1, color="#fff")),
        name="Sharpe Ratio",
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="solid", line_color="#ff4757", line_width=1.5,
        opacity=0.7, row=1, col=1)

    if breakeven < float("inf") and breakeven > 0:
        fig.add_vline(x=breakeven, line_dash="dash", line_color="#ffb347", line_width=1.5,
            annotation_text=f"Breakeven: {breakeven:.0f} bps", row=1, col=1)

    # Return curve
    ret_colors = ["#00e88f" if r >= 0 else "#ff4757" for r in sensitivity_df["total_return_pct"]]
    fig.add_trace(go.Scatter(
        x=sensitivity_df["cost_bps"], y=sensitivity_df["total_return_pct"],
        mode="lines+markers", line=dict(color="#a855f7", width=2.5),
        marker=dict(size=8, color=ret_colors, line=dict(width=1, color="#fff")),
        name="Total Return %",
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="solid", line_color="#ff4757", line_width=1.5,
        opacity=0.7, row=1, col=2)

    fig.update_xaxes(title_text="Round-Trip Cost (bps)", row=1, col=1)
    fig.update_xaxes(title_text="Round-Trip Cost (bps)", row=1, col=2)
    fig.update_layout(**CHART_LAYOUT, height=350, showlegend=False)
    return fig


# ══════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL LONG-SHORT BACKTEST
# ══════════════════════════════════════════════════════════════════

def run_long_short_backtest(
    universe_data: Dict[str, pd.DataFrame],
    ma_fast: int = 20,
    ma_slow: int = 50,
    weights: tuple = (1.0, 1.0, 1.0),
    top_n: int = 5,
    bottom_n: int = 5,
    rebalance_freq: int = 20,
    initial_capital: float = 100000.0,
) -> dict:
    """
    Cross-Sectional Long-Short Backtest
    ────────────────────────────────────
    Each rebalance period:
    1. Score all stocks in universe
    2. Go long top_n (highest signal strength)
    3. Go short bottom_n (lowest signal strength)
    4. Equal-weight within each leg
    5. Compute dollar-neutral L/S returns

    This isolates signal quality from market direction.
    """
    # Pre-compute all signals
    processed = {}
    for ticker, raw_df in universe_data.items():
        try:
            feat = compute_features(raw_df, ma_fast, ma_slow)
            if len(feat) < 100:
                continue
            feat = detect_regime(feat)
            sig = generate_signals(feat, weights[0], weights[1], weights[2])
            processed[ticker] = sig
        except Exception:
            continue

    if len(processed) < top_n + bottom_n:
        return {"error": "Not enough stocks"}

    # Common date range
    all_indices = [df.index for df in processed.values()]
    common_start = max(idx.min() for idx in all_indices)
    common_end = min(idx.max() for idx in all_indices)

    ref_ticker = list(processed.keys())[0]
    ref_dates = processed[ref_ticker].loc[common_start:common_end].index

    if len(ref_dates) < rebalance_freq + 10:
        return {"error": "Not enough common dates"}

    # Run L/S backtest
    ls_returns = []
    long_returns = []
    short_returns = []
    ls_dates = []
    rebalance_log = []

    for i in range(0, len(ref_dates) - rebalance_freq, rebalance_freq):
        rebal_date = ref_dates[i]

        # Get signal strength on rebalance date
        scores = {}
        for ticker, df in processed.items():
            if rebal_date in df.index:
                scores[ticker] = float(df.loc[rebal_date, "signal_strength"])

        if len(scores) < top_n + bottom_n:
            continue

        sorted_tickers = sorted(scores, key=scores.get, reverse=True)
        longs = sorted_tickers[:top_n]
        shorts = sorted_tickers[-bottom_n:]

        rebalance_log.append({
            "date": rebal_date.strftime("%Y-%m-%d"),
            "longs": [t.replace(".NS", "") for t in longs],
            "shorts": [t.replace(".NS", "") for t in shorts],
            "long_avg_score": round(np.mean([scores[t] for t in longs]), 3),
            "short_avg_score": round(np.mean([scores[t] for t in shorts]), 3),
        })

        # Compute returns over holding period
        for j in range(i, min(i + rebalance_freq, len(ref_dates) - 1)):
            date_today = ref_dates[j]
            date_tomorrow = ref_dates[j + 1]

            long_ret = []
            short_ret = []

            for t in longs:
                df = processed[t]
                if date_today in df.index and date_tomorrow in df.index:
                    r = np.log(df.loc[date_tomorrow, "Close"] / df.loc[date_today, "Close"])
                    long_ret.append(r)

            for t in shorts:
                df = processed[t]
                if date_today in df.index and date_tomorrow in df.index:
                    r = np.log(df.loc[date_tomorrow, "Close"] / df.loc[date_today, "Close"])
                    short_ret.append(r)

            if long_ret and short_ret:
                avg_long = np.mean(long_ret)
                avg_short = np.mean(short_ret)
                ls_ret = avg_long - avg_short  # dollar neutral

                ls_returns.append(ls_ret)
                long_returns.append(avg_long)
                short_returns.append(avg_short)
                ls_dates.append(date_tomorrow)

    if not ls_returns:
        return {"error": "No valid trading periods"}

    ls_series = pd.Series(ls_returns, index=ls_dates, name="L/S Returns")
    long_series = pd.Series(long_returns, index=ls_dates, name="Long Returns")
    short_series = pd.Series(short_returns, index=ls_dates, name="Short Returns")

    # Build equity curves
    ls_equity = initial_capital * np.exp(np.cumsum(ls_returns))
    long_equity = initial_capital * np.exp(np.cumsum(long_returns))
    short_equity = initial_capital * np.exp(np.cumsum([-r for r in short_returns]))

    ls_equity_s = pd.Series(ls_equity, index=ls_dates)
    long_equity_s = pd.Series(long_equity, index=ls_dates)

    # Metrics
    sharpe = np.sqrt(252) * ls_series.mean() / ls_series.std() if ls_series.std() > 0 else 0
    total_ret = (ls_equity[-1] / initial_capital - 1) * 100
    dd = pd.Series(ls_equity, index=ls_dates)
    max_dd = (dd / dd.cummax() - 1).min() * 100

    n_years = len(ls_returns) / 252
    cagr = ((ls_equity[-1] / initial_capital) ** (1 / max(n_years, 0.01)) - 1) * 100

    return {
        "ls_equity": ls_equity_s,
        "long_equity": long_equity_s,
        "ls_returns": ls_series,
        "long_returns": long_series,
        "short_returns": short_series,
        "rebalance_log": rebalance_log,
        "sharpe": round(sharpe, 2),
        "total_return": round(total_ret, 2),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "n_rebalances": len(rebalance_log),
        "avg_daily_ls_return": round(ls_series.mean() * 10000, 2),  # in bps
    }


def plot_long_short_equity(ls_result: dict) -> go.Figure:
    """Plot long-short equity curve with long-only comparison."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ls_result["ls_equity"].index, y=ls_result["ls_equity"].values,
        mode="lines", line=dict(color="#a855f7", width=2.5),
        fill="tozeroy", fillcolor="rgba(168,85,247,0.06)",
        name="Long-Short (market neutral)",
    ))

    fig.add_trace(go.Scatter(
        x=ls_result["long_equity"].index, y=ls_result["long_equity"].values,
        mode="lines", line=dict(color="#00e88f", width=1.5, dash="dot"),
        name="Long Leg Only",
    ))

    fig.add_hline(y=ls_result["ls_equity"].iloc[0], line_dash="dot",
        line_color="#6b6b80", opacity=0.5)

    fig.update_layout(**CHART_LAYOUT, height=380,
        title="Long-Short Equity Curve (₹) — Tests Signal Quality Independent of Market Direction")
    return fig


def plot_ls_monthly_returns(ls_returns: pd.Series) -> go.Figure:
    """Monthly L/S return bars."""
    monthly = ls_returns.resample("ME").sum() * 100  # log returns sum
    monthly_df = pd.DataFrame({"Month": monthly.index.strftime("%b %Y"), "Return": monthly.values})
    colors = ["#a855f7" if r >= 0 else "#ff4757" for r in monthly_df["Return"]]
    fig = go.Figure(go.Bar(x=monthly_df["Month"], y=monthly_df["Return"],
        marker_color=colors, opacity=0.85))
    fig.update_layout(**CHART_LAYOUT, height=280, title="Monthly Long-Short Returns (%)")
    return fig


# ══════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════

def render_metric_card(label, value, css_class=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════

def main():
    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown('<div class="main-header">◈ QuantEngine v2</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">production-grade strategy lab</div>', unsafe_allow_html=True)
        st.markdown("---")

        mode = st.radio("Mode", ["Single Stock", "Cross-Sectional"], horizontal=True)

        if mode == "Single Stock":
            st.markdown("##### 📌 Select Stock")
            selected_ticker = st.selectbox(
                "Ticker", options=list(STOCK_UNIVERSE.keys()),
                format_func=lambda x: f"{STOCK_UNIVERSE[x]}  ({x.replace('.NS','')})",
                label_visibility="collapsed",
            )
        else:
            st.markdown("##### 📌 Portfolio Settings")
            top_n = st.slider("Top N Stocks", 3, 10, 5)
            port_method = st.selectbox("Weighting Method", ["inverse_vol", "equal_weight", "signal_weighted"])
            run_long_short = st.toggle("Long-Short Backtest", value=True)
            if run_long_short:
                bottom_n = st.slider("Short Bottom N", 3, 10, 5)
                rebal_freq = st.slider("Rebalance Frequency (days)", 5, 60, 20)
            else:
                bottom_n = 5
                rebal_freq = 20
            run_ic = st.toggle("Information Coefficient Analysis", value=True)

        st.markdown("##### 📅 Data Period")
        period = st.select_slider("Period", options=["1y", "2y", "5y"], value="5y", label_visibility="collapsed")

        st.markdown("---")
        st.markdown("##### ⚙️ Strategy Parameters")

        col_a, col_b = st.columns(2)
        with col_a:
            ma_fast = st.number_input("Fast MA", min_value=5, max_value=50, value=20, step=5)
        with col_b:
            ma_slow = st.number_input("Slow MA", min_value=20, max_value=200, value=50, step=10)

        st.markdown("###### Factor Weights")
        wt_mom = st.slider("Momentum", 0.0, 2.0, 1.0, 0.1)
        wt_mr = st.slider("Mean Reversion", 0.0, 2.0, 1.0, 0.1)
        wt_vol = st.slider("Volume Breakout", 0.0, 2.0, 1.0, 0.1)

        st.markdown("---")
        st.markdown("##### 🎛️ Risk Management")
        use_vol_sizing = st.toggle("Vol-Targeted Sizing", value=True)
        if use_vol_sizing:
            target_vol = st.slider("Target Volatility (%)", 5, 30, 15, 1) / 100
            max_position = st.slider("Max Position (%)", 20, 100, 100, 5) / 100
        else:
            target_vol = 0.15
            max_position = 1.0

        regime_aware = st.toggle("Regime-Aware Signals", value=True)
        quality_filter = st.toggle("Quality Gate (Rolling Sharpe)", value=True)

        st.markdown("---")
        st.markdown("##### 💰 Execution & Costs")
        initial_capital = st.number_input("Starting Capital (₹)", value=100000, step=10000)
        slippage = st.number_input("Base Slippage (bps)", value=5, step=1)
        commission = st.number_input("Commission (bps)", value=10, step=1)

        st.markdown("---")
        st.markdown("##### 🔬 Validation")
        run_wf = st.toggle("Walk-Forward Validation", value=False)
        if run_wf:
            wf_folds = st.slider("Number of Folds", 3, 8, 5)
        else:
            wf_folds = 5

        run_sig_test = st.toggle("Statistical Significance Tests", value=True)

        st.markdown("---")
        run_btn = st.button("🚀  Run Strategy", use_container_width=True, type="primary")

    # ── MAIN AREA ──
    st.markdown('<div class="main-header">Quantitative Strategy Engine v2</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">NSE Equity · Multi-Factor · Regime-Aware · Walk-Forward Validated</div>', unsafe_allow_html=True)

    if not run_btn and "results_v2" not in st.session_state:
        st.markdown("###")
        st.markdown("""<div class="explain-box-purple">
            <strong>What's New in v2:</strong><br>
            ① <strong>Walk-Forward Validation</strong> — No more overfitting. Rolling OOS testing across multiple folds.<br>
            ② <strong>Vol-Targeted Sizing</strong> — Position size scales with realized volatility to maintain consistent risk.<br>
            ③ <strong>Regime Detection</strong> — Bull/bear/neutral trend classification + volatility regime.<br>
            ④ <strong>Cross-Sectional Mode</strong> — Rank the entire universe, construct weighted portfolios.<br>
            ⑤ <strong>Statistical Significance</strong> — Bootstrap Sharpe CI + permutation test for alpha.<br>
            ⑥ <strong>Execution Cost Model</strong> — Market impact scales with ADV (Almgren-Chriss square-root).<br>
            ⑦ <strong>Enhanced Signals</strong> — Graded factor scores, quality gate, regime scaling.
        </div>""", unsafe_allow_html=True)
        st.markdown("###")
        st.markdown('<div class="section-title">📊 Available Universe</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (tick, name) in enumerate(STOCK_UNIVERSE.items()):
            with cols[i % 4]:
                st.markdown(f"**{name}**  \n`{tick.replace('.NS','')}`")
        return

    # ══════════════════════════════════════════════════════════
    # SINGLE STOCK MODE
    # ══════════════════════════════════════════════════════════
    if mode == "Single Stock":
        if run_btn:
            config = BacktestConfig(
                initial_capital=initial_capital,
                target_vol=target_vol,
                max_position_pct=max_position,
                base_slippage_bps=slippage,
                commission_bps=commission,
                use_vol_sizing=use_vol_sizing,
            )

            with st.spinner(f"Fetching {STOCK_UNIVERSE[selected_ticker]} data..."):
                raw_df = fetch_data(selected_ticker, period)

            if raw_df.empty:
                st.error("Could not fetch data. Check your internet connection or try another ticker.")
                return

            with st.spinner("Computing features, regime & signals..."):
                featured_df = compute_features(raw_df, ma_fast, ma_slow)
                featured_df = detect_regime(featured_df)
                signal_df = generate_signals(featured_df, wt_mom, wt_mr, wt_vol,
                    quality_filter=quality_filter, regime_aware=regime_aware)

            with st.spinner("Running backtest with vol-targeted sizing..."):
                bt_result = run_backtest(signal_df, config)

            with st.spinner("Computing benchmark..."):
                bench_equity = raw_df["Close"].iloc[-len(bt_result["equity"]):].copy()
                bench_equity = bench_equity / bench_equity.iloc[0] * initial_capital
                bench_returns = bench_equity.pct_change().dropna()

            metrics = compute_metrics(bt_result["equity"], initial_capital, bench_returns)
            strat_returns = bt_result["equity"].pct_change().dropna()

            # Walk-forward
            wf_results = None
            if run_wf:
                with st.spinner(f"Running {wf_folds}-fold walk-forward validation..."):
                    wf_results = run_walk_forward(signal_df, config, n_folds=wf_folds)

            # Significance tests
            sig_results = None
            perm_results = None
            if run_sig_test and len(strat_returns) > 30:
                with st.spinner("Running significance tests (bootstrap + permutation)..."):
                    sig_results = bootstrap_sharpe_ci(strat_returns)
                    perm_results = permutation_test_alpha(strat_returns, bench_returns)

            # Factor correlation diagnostics
            with st.spinner("Computing factor correlation diagnostics..."):
                factor_corr = compute_factor_correlations(signal_df)

            # Cost sensitivity analysis
            with st.spinner("Running cost sensitivity analysis (11 cost levels)..."):
                cost_sensitivity = cost_sensitivity_analysis(signal_df, config)
                breakeven_cost = find_breakeven_cost(cost_sensitivity)

            st.session_state["results_v2"] = {
                "signal_df": signal_df,
                "bt_result": bt_result,
                "metrics": metrics,
                "bench_equity": bench_equity,
                "bench_returns": bench_returns,
                "strat_returns": strat_returns,
                "ticker": selected_ticker,
                "raw_df": raw_df,
                "config": config,
                "wf_results": wf_results,
                "sig_results": sig_results,
                "perm_results": perm_results,
                "factor_corr": factor_corr,
                "cost_sensitivity": cost_sensitivity,
                "breakeven_cost": breakeven_cost,
                "mode": "single",
            }

        if "results_v2" in st.session_state and st.session_state["results_v2"].get("mode") == "single":
            res = st.session_state["results_v2"]
            signal_df = res["signal_df"]
            bt = res["bt_result"]
            metrics = res["metrics"]
            bench_equity = res["bench_equity"]
            ticker = res["ticker"]
            config = res["config"]

            # ── TOP METRICS ──
            st.markdown('<div class="section-title">📈 Performance Summary</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)

            total_ret = metrics.get("Total Return (%)", 0)
            sharpe = metrics.get("Sharpe Ratio", 0)
            max_dd = metrics.get("Max Drawdown (%)", 0)
            cagr = metrics.get("CAGR (%)", 0)

            with m1:
                cls = "metric-positive" if total_ret >= 0 else "metric-negative"
                st.markdown(render_metric_card("Total Return", f"{total_ret:+.1f}%", cls), unsafe_allow_html=True)
            with m2:
                cls = "metric-positive" if sharpe >= 1 else ("metric-neutral" if sharpe >= 0.5 else "metric-negative")
                st.markdown(render_metric_card("Sharpe Ratio", f"{sharpe:.2f}", cls), unsafe_allow_html=True)
            with m3:
                st.markdown(render_metric_card("Max Drawdown", f"{max_dd:.1f}%", "metric-negative"), unsafe_allow_html=True)
            with m4:
                cls = "metric-positive" if cagr >= 0 else "metric-negative"
                st.markdown(render_metric_card("CAGR", f"{cagr:+.1f}%", cls), unsafe_allow_html=True)

            m5, m6, m7, m8 = st.columns(4)
            with m5:
                st.markdown(render_metric_card("Sortino", f"{metrics.get('Sortino Ratio', 0):.2f}", "metric-neutral"), unsafe_allow_html=True)
            with m6:
                st.markdown(render_metric_card("Calmar", f"{metrics.get('Calmar Ratio', 0):.2f}", "metric-neutral"), unsafe_allow_html=True)
            with m7:
                ann_vol = metrics.get("Ann. Volatility (%)", 0)
                st.markdown(render_metric_card("Ann. Vol", f"{ann_vol:.1f}%", "metric-neutral"), unsafe_allow_html=True)
            with m8:
                st.markdown(render_metric_card("Tail Ratio", f"{metrics.get('Tail Ratio', 0):.2f}", "metric-neutral"), unsafe_allow_html=True)

            # Alpha/Beta/IR row
            if "Beta" in metrics:
                ab1, ab2, ab3, ab4 = st.columns(4)
                with ab1:
                    st.markdown(render_metric_card("Beta", f"{metrics['Beta']:.2f}", "metric-neutral"), unsafe_allow_html=True)
                with ab2:
                    alpha = metrics.get("Alpha (ann.)", 0)
                    cls = "metric-positive" if alpha >= 0 else "metric-negative"
                    st.markdown(render_metric_card("Alpha (ann.)", f"{alpha:+.2f}%", cls), unsafe_allow_html=True)
                with ab3:
                    ir = metrics.get("Information Ratio", 0)
                    cls = "metric-positive" if ir >= 0.5 else "metric-neutral"
                    st.markdown(render_metric_card("Info Ratio", f"{ir:.2f}", cls), unsafe_allow_html=True)
                with ab4:
                    te = metrics.get("Tracking Error (%)", 0)
                    st.markdown(render_metric_card("Track. Error", f"{te:.1f}%", "metric-neutral"), unsafe_allow_html=True)

            # Cost summary
            cost_col1, cost_col2, cost_col3, _ = st.columns(4)
            with cost_col1:
                st.markdown(render_metric_card("Total Costs", f"₹{bt['total_costs']:,.0f}", "metric-negative"), unsafe_allow_html=True)
            with cost_col2:
                cost_drag = bt['total_costs'] / config.initial_capital * 100
                st.markdown(render_metric_card("Cost Drag", f"{cost_drag:.2f}%", "metric-negative"), unsafe_allow_html=True)

            st.markdown("###")

            # ── TABS ──
            tabs = st.tabs([
                "📊 Price & Signals",
                "💰 Equity & Drawdown",
                "🌡️ Regime Analysis",
                "🎲 Monte Carlo",
                "🧩 Factor Analysis",
                "🔗 Factor Correlations",
                "💸 Cost Sensitivity",
                "🔬 Walk-Forward",
                "📐 Significance Tests",
                "📋 Trade Log",
                "📖 How It Works",
            ])

            with tabs[0]:
                fig = plot_candlestick_with_signals(signal_df, bt["signals"],
                    title=f"{STOCK_UNIVERSE[ticker]} — Price Action")
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                col_eq, col_dd = st.columns([1, 1])
                with col_eq:
                    st.plotly_chart(plot_equity_curve(bt["equity"], bench_equity), use_container_width=True)
                with col_dd:
                    st.plotly_chart(plot_drawdown(bt["equity"]), use_container_width=True)
                st.plotly_chart(plot_monthly_returns(bt["equity"]), use_container_width=True)

            with tabs[2]:
                st.markdown("""<div class="explain-box">
                    <strong>Regime Detection:</strong> The system classifies each day into trend regimes
                    (bull/bear/neutral) based on price vs. slow MA and 60-day returns, and volatility regimes
                    (low/normal/high) based on percentile rank of 20-day realized vol within a 1-year window.
                    Signals are scaled down in bear markets and high-vol environments.
                </div>""", unsafe_allow_html=True)

                st.plotly_chart(plot_regime_timeline(signal_df), use_container_width=True)

                latest = signal_df.iloc[-1]
                trend_cls = {"bull": "regime-bull", "bear": "regime-bear", "neutral": "regime-neutral"}
                vol_cls = {"low_vol": "regime-bull", "normal": "regime-neutral", "high_vol": "regime-bear"}

                st.markdown(f"""
                <div style="display:flex; gap:20px; margin:16px 0;">
                    <div>Current Trend: <span class="{trend_cls.get(latest['trend_regime'], 'regime-neutral')}">{latest['trend_regime'].upper()}</span></div>
                    <div>Vol Regime: <span class="{vol_cls.get(latest['vol_regime'], 'regime-neutral')}">{latest['vol_regime'].upper().replace('_',' ')}</span></div>
                    <div style="color:#6b6b80;">Confidence: {latest['regime_confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

                # Regime breakdown table
                regime_counts = signal_df["trend_regime"].value_counts()
                r1, r2, r3 = st.columns(3)
                r1.metric("Bull Days", int(regime_counts.get("bull", 0)))
                r2.metric("Neutral Days", int(regime_counts.get("neutral", 0)))
                r3.metric("Bear Days", int(regime_counts.get("bear", 0)))

            with tabs[3]:
                st.markdown("""<div class="explain-box">
                    <strong>Monte Carlo Simulation:</strong> Bootstrapped from historical daily returns.
                    The cone shows probability-weighted ranges of future price paths. Sample paths
                    (thin lines) show individual simulated trajectories. Mean vs. median divergence
                    indicates return distribution skewness.
                </div>""", unsafe_allow_html=True)

                mc_horizon = st.slider("Projection Horizon (days)", 10, 90, 30, key="mc_slider")
                last_price = float(signal_df["Close"].iloc[-1])
                fig_mc = plot_mc_cone(signal_df["returns"].values, last_price, mc_horizon)
                st.plotly_chart(fig_mc, use_container_width=True)

            with tabs[4]:
                st.plotly_chart(plot_factor_heatmap(signal_df), use_container_width=True)

                st.markdown("""<div class="explain-box">
                    <strong>Factor Scores in v2:</strong> Unlike v1's binary on/off, factors now produce
                    <em>graded</em> scores. Momentum gives partial credit for weaker trends. Mean reversion
                    scores scale with RSI depth and Bollinger %B confirmation. Volume scores grade by
                    z-score magnitude. This continuous scoring feeds into cross-sectional ranking.
                </div>""", unsafe_allow_html=True)

                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    st.metric("Days Momentum Active", int(signal_df["score_momentum"].gt(0).sum()))
                with fcol2:
                    st.metric("Days Mean Rev. Active", int(signal_df["score_mean_rev"].gt(0).sum()))
                with fcol3:
                    st.metric("Days Volume Active", int(signal_df["score_volume"].gt(0).sum()))
                st.metric("Total BUY Signals Generated", int(signal_df["signal"].sum()))

            with tabs[5]:
                factor_corr = res.get("factor_corr")
                if factor_corr is None:
                    st.info("Factor correlation data not available.")
                else:
                    st.markdown("""<div class="explain-box">
                        <strong>Why This Matters:</strong> If your factors are highly correlated, your "multi-factor"
                        model is really just one bet wearing multiple hats. Ideal factor correlations are near zero
                        or slightly negative — meaning each factor provides independent information. Correlations
                        above ±0.7 mean two factors are essentially redundant.
                    </div>""", unsafe_allow_html=True)

                    corr_col1, corr_col2 = st.columns([1, 1])
                    with corr_col1:
                        st.plotly_chart(plot_factor_correlation_matrix(factor_corr["static"]),
                            use_container_width=True)
                    with corr_col2:
                        st.plotly_chart(plot_rolling_factor_correlations(factor_corr["rolling"]),
                            use_container_width=True)

                    # Interpretation
                    static = factor_corr["static"]
                    max_corr = static.where(np.triu(np.ones(static.shape), k=1).astype(bool)).stack()
                    worst_pair = max_corr.abs().idxmax()
                    worst_val = max_corr.loc[worst_pair]

                    if abs(worst_val) > 0.7:
                        st.markdown(f"""<div class="explain-box-warn">
                            <strong>⚠️ High correlation detected:</strong> {worst_pair[0]} × {worst_pair[1]}
                            has ρ = {worst_val:.3f}. These factors are providing overlapping information.
                            Consider removing one or orthogonalizing them.
                        </div>""", unsafe_allow_html=True)
                    elif abs(worst_val) > 0.4:
                        st.markdown(f"""<div class="explain-box">
                            <strong>Moderate correlation:</strong> {worst_pair[0]} × {worst_pair[1]}
                            has ρ = {worst_val:.3f}. Acceptable but not ideal — some information overlap exists.
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="explain-box-purple">
                            <strong>✓ Good factor independence:</strong> Maximum pairwise correlation is
                            ρ = {worst_val:.3f} ({worst_pair[0]} × {worst_pair[1]}).
                            Your factors are providing substantially independent signals.
                        </div>""", unsafe_allow_html=True)

            with tabs[6]:
                cost_sensitivity = res.get("cost_sensitivity")
                breakeven_cost = res.get("breakeven_cost", 0)

                if cost_sensitivity is None or cost_sensitivity.empty:
                    st.info("Cost sensitivity data not available.")
                else:
                    st.markdown("""<div class="explain-box">
                        <strong>What This Shows:</strong> Strategy Sharpe ratio and total return as a function
                        of round-trip transaction costs. The breakeven cost is where Sharpe hits zero — if your
                        real-world costs exceed this, the strategy destroys value. A strategy that breaks even
                        below 20 bps is fragile; above 50 bps is robust.
                    </div>""", unsafe_allow_html=True)

                    st.plotly_chart(plot_cost_sensitivity(cost_sensitivity, breakeven_cost),
                        use_container_width=True)

                    # Breakeven metric
                    be_cls = "metric-positive" if breakeven_cost > 30 else ("metric-neutral" if breakeven_cost > 15 else "metric-negative")
                    be_text = f"{breakeven_cost:.0f} bps" if breakeven_cost < float("inf") else "∞ (always positive)"
                    be_verdict = ("Robust" if breakeven_cost > 50 else
                                  "Acceptable" if breakeven_cost > 25 else
                                  "Fragile" if breakeven_cost > 10 else "Not viable")

                    cs1, cs2, cs3, cs4 = st.columns(4)
                    with cs1:
                        st.markdown(render_metric_card("Breakeven Cost", be_text, be_cls), unsafe_allow_html=True)
                    with cs2:
                        zero_cost_sharpe = cost_sensitivity[cost_sensitivity["cost_bps"] == 0]["sharpe"].values
                        if len(zero_cost_sharpe) > 0:
                            st.markdown(render_metric_card("Zero-Cost Sharpe", f"{zero_cost_sharpe[0]:.2f}", "metric-neutral"), unsafe_allow_html=True)
                    with cs3:
                        st.markdown(render_metric_card("Verdict", be_verdict, be_cls), unsafe_allow_html=True)
                    with cs4:
                        current_cost = slippage + commission
                        current_sharpe = cost_sensitivity[cost_sensitivity["cost_bps"].between(current_cost - 3, current_cost + 3)]["sharpe"].values
                        if len(current_sharpe) > 0:
                            st.markdown(render_metric_card(f"Sharpe @ {current_cost}bps", f"{current_sharpe[0]:.2f}", "metric-neutral"), unsafe_allow_html=True)

                    st.markdown("##### Full Cost Schedule")
                    st.dataframe(cost_sensitivity, use_container_width=True, hide_index=True)

            with tabs[7]:
                wf_results = res.get("wf_results")
                if wf_results is None:
                    st.markdown("""<div class="explain-box-warn">
                        Walk-forward validation is disabled. Enable it in the sidebar to see out-of-sample
                        performance across multiple time folds. This is the most important test for determining
                        if your strategy has real predictive power vs. being overfit to historical data.
                    </div>""", unsafe_allow_html=True)
                elif not wf_results["folds"]:
                    st.warning("Not enough data for walk-forward validation. Try using a longer data period (5y).")
                else:
                    st.markdown("""<div class="explain-box-purple">
                        <strong>Walk-Forward Validation:</strong> The data is split into rolling folds. For each fold,
                        the strategy is evaluated ONLY on the out-of-sample (unseen) portion. If OOS performance is
                        consistently positive, the strategy likely has real edge. If OOS results are erratic or negative,
                        in-sample results are likely overfit.
                    </div>""", unsafe_allow_html=True)

                    wf_fig = plot_walk_forward_results(wf_results)
                    if wf_fig:
                        st.plotly_chart(wf_fig, use_container_width=True)

                    wm1, wm2, wm3, wm4 = st.columns(4)
                    wm1.metric("Avg OOS Sharpe", f"{wf_results['avg_oos_sharpe']:.2f}")
                    wm2.metric("Avg OOS Return", f"{wf_results['avg_oos_return']:+.1f}%")
                    wm3.metric("Sharpe Stability (σ)", f"{wf_results['sharpe_stability']:.2f}")
                    wm4.metric("Consistency (% +ve folds)", f"{wf_results['consistency']:.0f}%")

                    st.markdown("##### Fold Details")
                    st.dataframe(pd.DataFrame(wf_results["folds"]), use_container_width=True, hide_index=True)

            with tabs[8]:
                sig_results = res.get("sig_results")
                perm_results = res.get("perm_results")

                if sig_results is None:
                    st.markdown("""<div class="explain-box-warn">
                        Statistical significance tests are disabled. Enable them in the sidebar to check
                        if your Sharpe ratio and alpha are statistically different from zero.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="explain-box-purple">
                        <strong>Why This Matters:</strong> A positive Sharpe ratio doesn't mean you have alpha —
                        it could be noise. These tests quantify the probability that your results occurred by chance.
                        A p-value < 0.05 means there's less than a 5% chance the result is random.
                    </div>""", unsafe_allow_html=True)

                    # Sharpe bootstrap
                    st.markdown("##### Bootstrap Sharpe Ratio")
                    sig_class = "stat-sig" if sig_results["is_significant"] else "stat-insig"
                    sig_text = "STATISTICALLY SIGNIFICANT (p < 0.05)" if sig_results["is_significant"] else "NOT SIGNIFICANT (p ≥ 0.05)"
                    sig_color = "#00e88f" if sig_results["is_significant"] else "#ff4757"

                    st.markdown(f'<div class="{sig_class}" style="color:{sig_color};">{sig_text} &nbsp;—&nbsp; p = {sig_results["p_value"]:.4f}</div>', unsafe_allow_html=True)

                    bs1, bs2, bs3, bs4 = st.columns(4)
                    bs1.metric("Bootstrap Mean Sharpe", f"{sig_results['sharpe_mean']:.3f}")
                    bs2.metric("95% CI Lower", f"{sig_results['ci_low']:.3f}")
                    bs3.metric("95% CI Upper", f"{sig_results['ci_high']:.3f}")
                    bs4.metric("p-value", f"{sig_results['p_value']:.4f}")

                    st.plotly_chart(plot_sharpe_bootstrap(
                        sig_results["distribution"], sig_results["ci_low"],
                        sig_results["ci_high"], sig_results["sharpe_mean"],
                    ), use_container_width=True)

                    # Permutation test for alpha
                    if perm_results:
                        st.markdown("##### Permutation Test for Alpha")
                        perm_class = "stat-sig" if perm_results["is_significant"] else "stat-insig"
                        perm_text = "ALPHA IS SIGNIFICANT" if perm_results["is_significant"] else "ALPHA NOT SIGNIFICANT"
                        perm_color = "#00e88f" if perm_results["is_significant"] else "#ff4757"

                        st.markdown(f'<div class="{perm_class}" style="color:{perm_color};">{perm_text} &nbsp;—&nbsp; p = {perm_results["p_value"]:.4f} &nbsp;|&nbsp; Observed α = {perm_results["observed_alpha"]:+.3f}%</div>', unsafe_allow_html=True)

            with tabs[9]:
                if bt["trades"]:
                    trade_df = pd.DataFrame(bt["trades"])
                    trade_df["date"] = pd.to_datetime(trade_df["date"]).dt.strftime("%Y-%m-%d")

                    sells = [t for t in bt["trades"] if t["type"] == "SELL"]
                    if sells:
                        wins = len([t for t in sells if t.get("pnl_pct", 0) > 0])
                        losses = len([t for t in sells if t.get("pnl_pct", 0) <= 0])
                        avg_win = np.mean([t["pnl_pct"] for t in sells if t.get("pnl_pct", 0) > 0]) if wins else 0
                        avg_loss = np.mean([t["pnl_pct"] for t in sells if t.get("pnl_pct", 0) <= 0]) if losses else 0
                        avg_hold = np.mean([t.get("hold_days", 0) for t in sells])
                        total_cost_trades = sum(t.get("cost", 0) for t in bt["trades"])

                        tc1, tc2, tc3, tc4 = st.columns(4)
                        tc1.metric("Total Round-Trips", len(sells))
                        tc2.metric("Win / Loss", f"{wins} / {losses}")
                        tc3.metric("Avg Win / Avg Loss", f"{avg_win:+.1f}% / {avg_loss:+.1f}%")
                        tc4.metric("Avg Holding Period", f"{avg_hold:.0f} days")

                        # Regime breakdown of trades
                        st.markdown("##### Trade Performance by Regime")
                        regime_trades = {}
                        for t in sells:
                            r = t.get("regime", "unknown")
                            if r not in regime_trades:
                                regime_trades[r] = []
                            regime_trades[r].append(t.get("pnl_pct", 0))

                        regime_cols = st.columns(len(regime_trades))
                        for idx, (regime, pnls) in enumerate(regime_trades.items()):
                            with regime_cols[idx]:
                                avg_pnl = np.mean(pnls)
                                st.metric(f"{regime.title()} Regime",
                                    f"{avg_pnl:+.2f}% avg ({len(pnls)} trades)")

                    st.dataframe(trade_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No trades were executed. Try adjusting the factor weights or MA periods.")

            with tabs[10]:
                st.markdown("""
                <div class="explain-box">
                <strong>🧠 Strategy Logic v2 — What Changed</strong><br><br>

                <strong>1. Graded Factor Scores</strong><br>
                v1 used binary on/off signals. v2 uses continuous scores — momentum gives partial credit,
                mean reversion scales with RSI depth, volume scores grade by z-score magnitude. This creates
                a richer signal that feeds into cross-sectional ranking.<br><br>

                <strong>2. Regime-Aware Execution</strong><br>
                The system classifies each day into trend regimes (bull/bear/neutral) and volatility regimes
                (low/normal/high). In bear markets, signal thresholds are tightened and exit triggers loosened.
                In bull markets, the system gives trades more room to run.<br><br>

                <strong>3. Vol-Targeted Sizing</strong><br>
                Instead of going 100% in or out, position size is calibrated so the position's volatility
                contribution ≈ your target (default 15% annualized). High-vol stocks get smaller positions;
                low-vol stocks get larger positions. This normalizes risk across different names.<br><br>

                <strong>4. Walk-Forward Validation</strong><br>
                The data is split into multiple rolling windows. The strategy is evaluated only on
                out-of-sample (unseen) portions. This prevents the classic mistake of overfitting to
                historical data via parameter tuning.<br><br>

                <strong>5. Statistical Significance</strong><br>
                Bootstrap resampling creates a distribution of possible Sharpe ratios. If the 95% confidence
                interval doesn't include zero, the strategy's edge is statistically significant at the 5% level.
                Permutation tests check whether alpha over buy-and-hold is real or random.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="explain-box">
                <strong>📊 New Metrics Explained</strong><br><br>

                <strong>Information Ratio:</strong> Active return ÷ tracking error. Measures skill per unit of
                active risk taken. Above 0.5 is good, above 1.0 is exceptional.<br>
                <strong>Tracking Error:</strong> Volatility of the difference between strategy and benchmark returns.<br>
                <strong>Tail Ratio:</strong> 95th percentile gain ÷ 5th percentile loss. > 1 means positive skew.<br>
                <strong>Cost Drag:</strong> Total transaction costs as a percentage of starting capital.
                </div>
                """, unsafe_allow_html=True)

            # Current Signal
            st.markdown("---")
            latest = signal_df.iloc[-1]
            sig_text = "BUY" if latest["signal"] == 1 else "HOLD / FLAT"
            sig_class = "signal-buy" if latest["signal"] == 1 else "signal-hold"

            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:16px; margin-top:8px; flex-wrap:wrap;">
                <span style="font-family:'JetBrains Mono'; color:#6b6b80; font-size:0.8rem;">CURRENT SIGNAL →</span>
                <span class="{sig_class}">{sig_text}</span>
                <span style="color:#6b6b80; font-size:0.8rem;">
                    Score: {latest['signal_strength']:.2f} &nbsp;|&nbsp;
                    RSI: {latest['RSI']:.1f} &nbsp;|&nbsp;
                    Vol: {latest['volatility_20d']*100:.1f}% &nbsp;|&nbsp;
                    Regime: {latest['trend_regime'].upper()}
                </span>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # CROSS-SECTIONAL MODE
    # ══════════════════════════════════════════════════════════
    elif mode == "Cross-Sectional":
        if run_btn:
            with st.spinner("Fetching data for entire universe..."):
                universe_data = fetch_multi(list(STOCK_UNIVERSE.keys()), period)

            if not universe_data:
                st.error("Could not fetch data for any stocks.")
                return

            with st.spinner(f"Ranking {len(universe_data)} stocks cross-sectionally..."):
                rank_df = rank_universe_cross_sectional(
                    universe_data, top_n, ma_fast, ma_slow, (wt_mom, wt_mr, wt_vol),
                )

            if rank_df.empty:
                st.error("Could not compute rankings.")
                return

            with st.spinner("Constructing portfolio weights..."):
                weights = construct_portfolio_weights(rank_df, universe_data, port_method, top_n)

            # IC Analysis
            ic_results = None
            if run_ic:
                with st.spinner("Computing rolling Information Coefficient across universe..."):
                    ic_results = compute_rolling_ic(
                        universe_data, ma_fast, ma_slow, (wt_mom, wt_mr, wt_vol),
                    )

            # Long-Short Backtest
            ls_result = None
            if run_long_short:
                with st.spinner(f"Running long-short backtest (top {top_n} vs bottom {bottom_n})..."):
                    ls_result = run_long_short_backtest(
                        universe_data, ma_fast, ma_slow, (wt_mom, wt_mr, wt_vol),
                        top_n, bottom_n, rebal_freq, initial_capital,
                    )

            st.session_state["results_v2"] = {
                "rank_df": rank_df,
                "weights": weights,
                "universe_data": universe_data,
                "mode": "cross_sectional",
                "top_n": top_n,
                "port_method": port_method,
                "ic_results": ic_results,
                "ls_result": ls_result,
            }

        if "results_v2" in st.session_state and st.session_state["results_v2"].get("mode") == "cross_sectional":
            res = st.session_state["results_v2"]
            rank_df = res["rank_df"]
            weights = res["weights"]
            top_n_val = res["top_n"]

            st.markdown('<div class="section-title">📊 Cross-Sectional Universe Ranking</div>', unsafe_allow_html=True)

            st.markdown("""<div class="explain-box">
                <strong>How it works:</strong> Every stock in the universe is scored using the same multi-factor
                model (momentum + mean reversion + volume), with regime and quality adjustments. Stocks are
                ranked by signal strength, and the top N are selected for the portfolio with weights determined
                by your chosen method.
            </div>""", unsafe_allow_html=True)

            # Display ranking table with color coding
            display_df = rank_df.copy()
            display_df = display_df[["rank", "name", "price", "signal_strength", "signal",
                "RSI", "volatility", "regime", "vol_regime", "ma_spread",
                "rolling_sharpe", "return_20d"]].copy()
            display_df.columns = ["Rank", "Name", "Price (₹)", "Signal Score", "Signal",
                "RSI", "Vol (%)", "Trend Regime", "Vol Regime", "MA Spread (%)",
                "Rolling Sharpe", "20d Return (%)"]

            st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

            # Portfolio allocation
            st.markdown('<div class="section-title">💼 Portfolio Allocation</div>', unsafe_allow_html=True)

            if weights:
                weight_df = pd.DataFrame([
                    {"Stock": STOCK_UNIVERSE.get(t, t), "Ticker": t.replace(".NS", ""),
                     "Weight (%)": round(w * 100, 1),
                     "Allocation (₹)": round(w * initial_capital, 0)}
                    for t, w in weights.items()
                ])
                weight_df = weight_df.sort_values("Weight (%)", ascending=False)

                wcol1, wcol2 = st.columns([1, 1])
                with wcol1:
                    st.dataframe(weight_df, use_container_width=True, hide_index=True)
                with wcol2:
                    fig = go.Figure(go.Pie(
                        labels=weight_df["Ticker"], values=weight_df["Weight (%)"],
                        hole=0.55, marker=dict(
                            colors=["#00e88f", "#5b7fff", "#a855f7", "#ffb347", "#ff4757",
                                    "#00b4d8", "#e8e8ef", "#6b6b80", "#f72585", "#4cc9f0"],
                        ),
                        textfont=dict(family="JetBrains Mono", size=11),
                    ))
                    fig.update_layout(**CHART_LAYOUT, height=350, title=f"Portfolio Weights ({res['port_method']})",
                        showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

            # Regime summary across universe
            st.markdown('<div class="section-title">🌡️ Universe Regime Summary</div>', unsafe_allow_html=True)
            regime_summary = rank_df["regime"].value_counts()
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Bull Stocks", int(regime_summary.get("bull", 0)))
            rc2.metric("Neutral Stocks", int(regime_summary.get("neutral", 0)))
            rc3.metric("Bear Stocks", int(regime_summary.get("bear", 0)))
            rc4.metric("Avg Signal Strength", f"{rank_df['signal_strength'].mean():.3f}")

            # ── INFORMATION COEFFICIENT ──
            ic_results = res.get("ic_results")
            if ic_results and "error" not in ic_results:
                st.markdown('<div class="section-title">📐 Information Coefficient Analysis</div>', unsafe_allow_html=True)

                st.markdown("""<div class="explain-box-purple">
                    <strong>What IC Tells You:</strong> The Information Coefficient is the rank correlation between
                    your signal scores today and actual forward returns. IC > 0.05 with t-stat > 2.0 indicates
                    a statistically exploitable signal. IC > 0.10 is exceptional for daily equity signals.
                    This is the single most important metric for evaluating signal quality.
                </div>""", unsafe_allow_html=True)

                # IC summary metrics
                ic_cols = st.columns(len(ic_results))
                for idx, (horizon, data) in enumerate(ic_results.items()):
                    with ic_cols[idx]:
                        sig_icon = "✓" if data["is_significant"] else "✗"
                        cls = "metric-positive" if data["is_significant"] else "metric-negative"
                        st.markdown(render_metric_card(
                            f"{horizon} IC",
                            f"{data['avg_ic']:.4f}",
                            cls,
                        ), unsafe_allow_html=True)
                        st.caption(f"t = {data['t_stat']:.1f} | Hit rate: {data['hit_rate']:.0f}% | {sig_icon}")

                # IC time series chart
                st.plotly_chart(plot_rolling_ic(ic_results), use_container_width=True)

                # Significance summary
                any_sig = any(d["is_significant"] for d in ic_results.values())
                if any_sig:
                    sig_horizons = [k for k, v in ic_results.items() if v["is_significant"]]
                    st.markdown(f"""<div class="stat-sig" style="color:#00e88f;">
                        SIGNAL HAS PREDICTIVE POWER at {', '.join(sig_horizons)} horizons (t > 2.0)
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="stat-insig" style="color:#ff4757;">
                        SIGNAL DOES NOT SHOW SIGNIFICANT PREDICTIVE POWER (all t-stats < 2.0).
                        The cross-sectional signal may not contain exploitable information.
                    </div>""", unsafe_allow_html=True)

            elif ic_results and "error" in ic_results:
                st.markdown('<div class="section-title">📐 Information Coefficient</div>', unsafe_allow_html=True)
                st.warning(f"IC Analysis: {ic_results['error']}")

            # ── LONG-SHORT BACKTEST ──
            ls_result = res.get("ls_result")
            if ls_result and "error" not in ls_result:
                st.markdown('<div class="section-title">⚖️ Long-Short Backtest</div>', unsafe_allow_html=True)

                st.markdown("""<div class="explain-box">
                    <strong>Why Long-Short Matters:</strong> A long-only backtest conflates signal quality with
                    market direction. The long-short test goes long top-ranked stocks and short bottom-ranked stocks,
                    creating a dollar-neutral portfolio. If this generates positive returns, your signal has real
                    cross-sectional predictive power independent of market beta.
                </div>""", unsafe_allow_html=True)

                # L/S metrics
                ls1, ls2, ls3, ls4 = st.columns(4)
                with ls1:
                    cls = "metric-positive" if ls_result["sharpe"] > 0 else "metric-negative"
                    st.markdown(render_metric_card("L/S Sharpe", f"{ls_result['sharpe']:.2f}", cls), unsafe_allow_html=True)
                with ls2:
                    cls = "metric-positive" if ls_result["total_return"] > 0 else "metric-negative"
                    st.markdown(render_metric_card("L/S Total Return", f"{ls_result['total_return']:+.1f}%", cls), unsafe_allow_html=True)
                with ls3:
                    cls = "metric-positive" if ls_result["cagr"] > 0 else "metric-negative"
                    st.markdown(render_metric_card("L/S CAGR", f"{ls_result['cagr']:+.1f}%", cls), unsafe_allow_html=True)
                with ls4:
                    st.markdown(render_metric_card("L/S Max DD", f"{ls_result['max_dd']:.1f}%", "metric-negative"), unsafe_allow_html=True)

                ls5, ls6, ls7, _ = st.columns(4)
                with ls5:
                    st.markdown(render_metric_card("Avg Daily L/S", f"{ls_result['avg_daily_ls_return']:.1f} bps", "metric-neutral"), unsafe_allow_html=True)
                with ls6:
                    st.markdown(render_metric_card("Rebalances", f"{ls_result['n_rebalances']}", "metric-neutral"), unsafe_allow_html=True)

                # Equity curve
                st.plotly_chart(plot_long_short_equity(ls_result), use_container_width=True)
                st.plotly_chart(plot_ls_monthly_returns(ls_result["ls_returns"]), use_container_width=True)

                # Verdict
                if ls_result["sharpe"] > 0.5:
                    st.markdown("""<div class="stat-sig" style="color:#00e88f;">
                        LONG-SHORT PORTFOLIO SHOWS POSITIVE RISK-ADJUSTED RETURNS — signal has cross-sectional predictive power.
                    </div>""", unsafe_allow_html=True)
                elif ls_result["sharpe"] > 0:
                    st.markdown("""<div class="explain-box">
                        Long-short returns are positive but modest. Signal may have weak cross-sectional power
                        or may require cost optimization and better execution to be viable.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="stat-insig" style="color:#ff4757;">
                        LONG-SHORT PORTFOLIO IS NEGATIVE — the signal does not reliably distinguish winners
                        from losers cross-sectionally. Consider alternative features or construction.
                    </div>""", unsafe_allow_html=True)

                # Rebalance log
                with st.expander("📋 Rebalance Log (click to expand)"):
                    rebal_df = pd.DataFrame(ls_result["rebalance_log"])
                    rebal_df["longs"] = rebal_df["longs"].apply(lambda x: ", ".join(x))
                    rebal_df["shorts"] = rebal_df["shorts"].apply(lambda x: ", ".join(x))
                    st.dataframe(rebal_df, use_container_width=True, hide_index=True)

            elif ls_result and "error" in ls_result:
                st.markdown('<div class="section-title">⚖️ Long-Short Backtest</div>', unsafe_allow_html=True)
                st.warning(f"Long-Short: {ls_result['error']}")

    # ── DISCLAIMER ──
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This is a research & educational tool — not financial advice. "
        "Past backtest performance does not guarantee future results. Walk-forward validation "
        "and significance tests reduce but do not eliminate overfitting risk. Always do your own due diligence."
    )


if __name__ == "__main__":
    main()
