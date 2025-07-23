"""
latency_arb.config
------------------
Centralised, type-safe configuration for the whole latency-arbitrage stack.

• One singleton (`cfg`) is instantiated at import-time so every module gets the
  exact same object (no accidental divergence).

• Pure data only - no runtime logic, network calls, or heavy imports here.
"""
from __future__     import annotations

import pathlib

from dataclasses    import dataclass, field
from datetime       import timedelta
from typing         import Literal, Tuple

# Global settings
SYMBOL: str = "MEW"

# --------------------------------------------------------------------------- #
# 1.  Exchange-level settings
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class HyperLiquidCfg:
    symbol:              str   = SYMBOL   # case-insensitive “coin” on HL
    testnet:             bool  = False    # Testnet or mainnet
    post_timeout:        float = 10.0     # seconds to await each/post ACK
    max_reconnect_delay: float = 300.0    # exponential backoff cap


@dataclass(slots=True, frozen=True)
class BinanceCfg:
    # Base symbol in uppercase
    base_symbol:         str                             = SYMBOL
    # Derived lowercase symbol and streams
    symbol:              str                             = field(init=False)
    streams:             Tuple[str, ...]                 = field(init=False)
    feed:                Literal["spot", "perp", "both"] = "perp"

    ping_interval:       float           = 20.0             # seconds
    ping_timeout:        float           = 10.0             # seconds
    max_reconnect_delay: float           = 120.0            # seconds

    def __post_init__(self):
        # Derive the trading symbol and websocket streams
        sym = self.base_symbol.lower() + "usdt"
        object.__setattr__(self, "symbol", sym)
        object.__setattr__(self, "streams", (
            f"{sym}@trade",
        ))

@dataclass(slots=True, frozen=True)
class CoinUpCfg:
    symbol:     str     = SYMBOL.lower() + "usdt" # Coinup uses lowercase
    enabled:    bool    = False                    # Feature flag
    weight :    float   = 1.0                     # Signal weight vs Binance

# --------------------------------------------------------------------------- #
# 2.  Strategy / signal parameters
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class StrategyCfg:
    # --- Binance signal thresholds (pre-computed offline) ---
    single_trade_qty_threshold: float = 3_000_000       # mean + xσ over single trade spike
    burst_window_ms:            int   = 100          # ms burst window
    burst_volume_threshold:     float = 54_240_588      # mean + 15σ over burst vols

    # --- HyperLiquid execution ---
    maker_fee:       float = 0.00015
    taker_fee:       float = 0.00045
    guardband_lo:    float = 0.25
    guardband_hi:    float = 1.75
    use_stop_market: bool  = True

    # --- Trailing-stop / TP ---
    use_trailing_stop:       bool  = True
    trailing_activation_bps: float = 2.0
    trailing_distance_bps:   float = 30.0
    break_even_buffer_bps:   float = 2.0
    profit_trailing_bps:     float = 5.0

    # --- Filter configuration ---
    filter_type: Literal["imbalance", "liquidity"] = "liquidity"  # Toggle between filters

    # --- Imbalance threshold ---
    imb_long:   float = +0.0
    imb_short:  float = -0.0

    # --- Liquidity filter ---
    liquidity_buffer_pct: float = 0.20 # Require 20% more liquidity than what we need


# --------------------------------------------------------------------------- #
# 3.  Risk & exposure limits
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class RiskCfg:
    max_gross_position_usd: float = 150.0
    max_net_position_usd:   float = 150.0
    max_daily_loss_usd:     float = 50.0
    max_risk_pct:           float = 0.25   # fraction of account equity per trade

    # --- execution notional limits (USD) ---
    target_trade_notional_usd: float = 15.0     # what we TRY to trade each time
    min_trade_notional_usd:    float = 11.0     # hard floor
    max_trade_notional_usd:    float = 20.0     # hard ceiling


# --------------------------------------------------------------------------- #
# 4.  Latency measuring
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class LatencyCfg:
    # Binance feed latencies
    measure_binance_ws_roundtrip: bool = True     # time between WS ping → pong on Binance
    measure_binance_message_latency: bool = True  # time from Binance trade ts → client receipt

    # Signal detection latency
    measure_signal_detection: bool = True         # time from receipt → marking as “significant”

    # HyperLiquid data latencies
    measure_hl_ws_roundtrip: bool = True          # time between WS ping → pong on HyperLiquid
    measure_hl_data_receive: bool = True          # time from HL trade/book ts → client receipt

    # Execution latencies on HyperLiquid
    measure_signal_to_send: bool = True           # time from signal detect → send_post() call
    measure_order_ack: bool = True                # time from send_post() → POST ACK
    measure_order_fill: bool = True               # time from POST ACK → on_orderUpdates “filled”

    # End-to-end latency
    measure_end_to_end: bool = True               # time from Binance-trade ts → final fill callback

    # Recording behavior
    recording_batch_size: int = 3                 # number of records to batch before flushing to disk


# --------------------------------------------------------------------------- #
# 5.  Persistent logging & accounting
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class RecorderCfg:
    enable_recording: bool    = False
    root: pathlib.Path        = pathlib.Path("./logs")
    trade_log:   pathlib.Path = field(init=False)
    order_log:   pathlib.Path = field(init=False)
    signal_log:  pathlib.Path = field(init=False)
    pnl_log:     pathlib.Path = field(init=False)
    latency_log: pathlib.Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "trade_log",  self.root / "trades.parquet")
        object.__setattr__(self, "order_log",  self.root / "orders.parquet")
        object.__setattr__(self, "signal_log", self.root / "signals.parquet")
        object.__setattr__(self, "pnl_log",    self.root / "pnl.parquet")
        object.__setattr__(self, "latency_log",self.root / "latency.parquet")


# --------------------------------------------------------------------------- #
# 6.  Engine / runtime knobs
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class EngineCfg:
    loop_sleep_ms:           int   = 5         # main async loop granularity
    status_print_interval_s: int   = 30        # heartbeat prints to console
    reconnect_backoff_base:  float = 1.0     # initial back-off (sec)


# --------------------------------------------------------------------------- #
# 7.  Aggregate singleton for easy import
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class _Config:
    hyperliquid: HyperLiquidCfg = HyperLiquidCfg()
    binance:     BinanceCfg     = BinanceCfg()
    coinup:      CoinUpCfg      = CoinUpCfg()
    strategy:    StrategyCfg    = StrategyCfg()
    risk:        RiskCfg        = RiskCfg()
    latency:     LatencyCfg     = LatencyCfg()
    recorder:    RecorderCfg    = RecorderCfg()
    engine:      EngineCfg      = EngineCfg()


# Public instance – import this elsewhere
cfg = _Config()