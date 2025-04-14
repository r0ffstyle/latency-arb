# Latency Arbitrage Strategy in Crypto

This project implements a latency arbitrage strategy that reacts to large trades on Binance and places trades on HyperLiquid before the price adjusts. It uses order book imbalance and microprice signals for confirmation, with modeling of latency, slippage, and transaction costs.

## Files

| File | Description |
|------|-------------|
| `start.ipynb` | Main notebook for running the strategy and analysis |
| `data_loader.py` | Data ingestion, preprocessing, and trade alignment |
| `utils.py` | Core trading logic, slippage modeling, and statistical analysis |


### Workflow

1. **Data Loading**
   - Binance trades and HyperLiquid order book snapshots

2. **Signal Generation**
   - Detects significant Binance trades using a rolling quantile threshold
   - Aligns signals to HyperLiquid snapshots with a latency offset

3. **Backtesting**
   - Models fill price via order book walking
   - Simulates slippage, latency, and execution risk
   - Applies trailing stops and profit targets

## Features

- Latency and slippage-aware backtest engine
- Conservative or Kelly-based position sizing
- Microprice and imbalance confirmation logic
- Detailed trade statistics, equity tracking, and drawdown calculation
