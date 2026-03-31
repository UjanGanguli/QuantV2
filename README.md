# ◈ QuantEngine — Multi-Factor Strategy Lab

A production-grade quantitative trading strategy engine built for NSE equities. Combines momentum, mean-reversion, and volume breakout signals with Monte Carlo exit logic and institutional-grade performance analytics.

## Architecture

```
DATA (yfinance)
  → FEATURE ENGINEERING (MA, RSI, Vol Z-Score, MACD, Bollinger)
    → MULTI-FACTOR SIGNAL GENERATION (weighted composite scoring)
      → EVENT-DRIVEN BACKTEST (with transaction costs + MC exits)
        → PERFORMANCE ANALYTICS (Sharpe, Sortino, Alpha, Beta, Drawdown)
```

## Features

- **20 NSE large-cap stocks** pre-loaded (Reliance, TCS, HDFC Bank, etc.)
- **3 orthogonal factors**: Momentum (trend-following), Mean Reversion (RSI oversold), Volume Breakout (Z-score spike)
- **Adjustable factor weights** — tune the strategy in real-time
- **Monte Carlo bootstrapping** — used for both exit decisions and forward price projections
- **Realistic transaction costs** — configurable slippage and commission in basis points
- **Benchmark comparison** — strategy vs buy-and-hold with Alpha/Beta computation
- **Interactive Plotly charts** — candlesticks, equity curves, drawdown, monthly returns, factor heatmaps

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Strategy Logic

| Factor | Condition | Interpretation |
|--------|-----------|----------------|
| Momentum | Close > Fast MA AND Fast MA > Slow MA | Confirmed uptrend |
| Mean Reversion | RSI < 35 | Oversold bounce candidate |
| Volume Breakout | Volume Z-score > 1.5 | Unusual institutional activity |

**Entry**: Weighted composite score ≥ threshold (at least one full-weight factor active)
**Exit**: Monte Carlo 80th percentile < 1.01 OR RSI > 70

## Performance Metrics

- Total Return, CAGR
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown
- Daily Win Rate, Profit Factor
- Beta, Annualized Alpha (vs buy-and-hold benchmark)

## Disclaimer

This is a research and educational tool — not financial advice. Past backtest performance does not guarantee future results.
