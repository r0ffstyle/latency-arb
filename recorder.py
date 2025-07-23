import os
import threading
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import cfg
from strategy import Trade, HLBookTop, ExecutionPlan
from risk import Side

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Recorder:
    """
    Collects and incrementally persists all PnL, latency, signal, order
    and trade metrics to Parquet files under cfg.recorder.root.
    Enhanced with aggregation methods for case study reporting.
    """

    def __init__(self) -> None:
        # ensure root dir exists
        self._root: Path = cfg.recorder.root
        self._root.mkdir(parents=True, exist_ok=True)

        # batch size, shared accross trades & latencies
        self._batch: int = cfg.latency.recording_batch_size

        # thread-safe buffers
        self._lock = threading.Lock()
        self._trades: List[Dict[str, Any]] = []
        self._signals: List[Dict[str, Any]] = []
        self._orders: List[Dict[str, Any]] = []
        self._latencies: List[Dict[str, Any]] = []
        self._pnls: List[Dict[str, Any]] = []
        
        # Track signal counts for reporting
        self._signal_stats = defaultdict(int)
        self._rejection_reasons = defaultdict(int)

    # ─── Helpers ──────────────────────────────────────────────────────── #

    def _write_batch(self,
                     records: List[Dict[str, Any]],
                     filename: Path) -> None:
        """
        Write a batch of records to a new parquet file:
          <filename>_<unixms>.parquet
        """
        if not records:
            return
        ts = int(time.time() * 1_000)
        out = filename.with_name(f"{filename.stem}_{ts}{filename.suffix}")
        try:
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, out)
            logger.debug("Flushed %d rows to %s", len(records), out)
        except Exception:
            logger.exception("Failed to write batch to %s", out)

    def _read_all_parquet(self, pattern: str) -> pd.DataFrame:
        """Read all parquet files matching pattern and combine them."""
        files = list(self._root.glob(pattern))
        if not files:
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception:
                logger.exception("Failed to read %s", f)
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # ─── Trade logging ────────────────────────────────────────────────── #

    def record_trade(self, trade: Trade, recv_ts: Optional[float] = None) -> None:
        """
        Log one raw Binance trade tick.
        :param trade: the strategy.BinanceTrade
        :param recv_ts: local receive timestamp (time.time()), default now
        """
        if not cfg.recorder.enable_recording:
            return
        rec = {
            "raw_ts_ms": trade.ts_ms,
            "recv_ts": recv_ts or time.time(),
            "px": trade.px,
            "qty": trade.qty,
            "side": trade.side,
        }
        with self._lock:
            self._trades.append(rec)
            if len(self._trades) >= self._batch:
                self._flush_trades()

    def _flush_trades(self) -> None:
        self._write_batch(self._trades, cfg.recorder.trade_log)
        self._trades.clear()

    # ─── Signal logging ─────────────────────────────────────────────────#

    def record_signal(
        self,
        trade: Trade,
        hl_top: Optional[HLBookTop],
        allowed: bool,
        reason: Union[str, ExecutionPlan],
        detect_ts: Optional[float] = None,
    ) -> None:
        """
        Log one signal attempt (accepted or rejected).
        Includes raw trade, HL top-of-book, and if accepted the ExecutionPlan.
        """
        if not cfg.recorder.enable_recording:
            return
        rec: Dict[str, Any] = {
            "detect_ts": detect_ts or time.time(),
            "raw_ts_ms": trade.ts_ms,
            "trade_px": trade.px,
            "trade_qty": trade.qty,
            "trade_side": trade.side,
            "hl_bid_px": hl_top.bid_px if hl_top else None,
            "hl_bid_sz": hl_top.bid_sz if hl_top else None,
            "hl_ask_px": hl_top.ask_px if hl_top else None,
            "hl_ask_sz": hl_top.ask_sz if hl_top else None,
            "hl_mid_px": (hl_top.bid_px + hl_top.ask_px) / 2 if hl_top else None,
            "allowed": allowed,
        }
        
        # Track stats
        with self._lock:
            self._signal_stats["total"] += 1
            if allowed:
                self._signal_stats["accepted"] += 1
                plan = reason  # type: ignore
                rec.update({
                    "plan_side":          plan.side,
                    "plan_notional_usd":  plan.notional_usd,
                    "plan_guard_lo":      plan.guard_lo,
                    "plan_guard_hi":      plan.guard_hi,
                    "plan_cross_px":      plan.crossing_px,
                    "plan_mid_px":        plan.mid_px,
                    "binance_ts_ms":      plan.binance_ts_ms,
                })
            else:
                self._signal_stats["rejected"] += 1
                self._rejection_reasons[str(reason)] += 1
                rec["reject_reason"] = reason

            self._signals.append(rec)
            if len(self._signals) >= self._batch:
                self._flush_signals()

    def _flush_signals(self) -> None:
        self._write_batch(self._signals, cfg.recorder.signal_log)
        self._signals.clear()

    # ─── Order / fill logging ───────────────────────────────────────────#

    def record_order(
        self,
        plan: ExecutionPlan,
        send_ts: float,
        ack_ts: float,
        statuses: List[Dict[str, Any]],
    ) -> None:
        """
        Log one order submission plus any fills that came back immediately
        in the /post response.
        Calculates ACK latency and slippage vs mid.
        """
        if not cfg.recorder.enable_recording:
            return
        ack_lat = ack_ts - send_ts
        signal_to_send_lat = send_ts - (plan.binance_ts_ms / 1000)
        end_to_end_lat = ack_ts - (plan.binance_ts_ms / 1000)
        
        base: List[Dict[str, Any]] = []
        for st in statuses:
            filled = st.get("filled")
            if not filled:
                continue
            avg_px = float(filled["avgPx"])
            sz     = float(filled["totalSz"])
            fee    = float(st.get("fee", 0.0))
            
            # Calculate slippage against actual mid price at signal time
            if plan.side == "LONG":
                slippage_bps = ((avg_px - plan.mid_px) / plan.mid_px) * 10000
            else:
                slippage_bps = ((plan.mid_px - avg_px) / plan.mid_px) * 10000
                
            rec = {
                "send_ts": send_ts,
                "ack_ts": ack_ts,
                "ack_latency_ms": ack_lat * 1000,
                "signal_to_send_ms": signal_to_send_lat * 1000,
                "end_to_end_ms": end_to_end_lat * 1000,
                "binance_ts_ms": plan.binance_ts_ms,
                "plan_side": plan.side,
                "plan_notional_usd": plan.notional_usd,
                "plan_cross_px": plan.crossing_px,
                "plan_mid_px": plan.mid_px,
                "exec_px": avg_px,
                "filled_sz": sz,
                "fee": fee,
                "slippage_bps": slippage_bps,
                "fill_notional_usd": sz * avg_px,
            }
            base.append(rec)

        with self._lock:
            self._orders.extend(base)
            if len(self._orders) >= self._batch:
                self._flush_orders()

    def _flush_orders(self) -> None:
        self._write_batch(self._orders, cfg.recorder.order_log)
        self._orders.clear()

    # ─── Latency logging ────────────────────────────────────────────────#

    def record_latency(
        self,
        exchange: str,
        stage: str,
        stats: Dict[str, float],
        ts: Optional[float] = None
    ) -> None:
        """
        Record any latency histogram (min/p50/p95) for a given exchange and stage.
        e.g. record_latency("binance", "data_receive", {...})
        """
        if not cfg.recorder.enable_recording:
            return
        
        rec = {
            "ts": ts or time.time(),
            "exchange": exchange,
            "stage": stage,
            "min": stats.get("min"),
            "p50": stats.get("p50"),
            "p95": stats.get("p95"),
        }
        with self._lock:
            self._latencies.append(rec)
            if len(self._latencies) >= self._batch:
                self._flush_latencies()

    def _flush_latencies(self) -> None:
        self._write_batch(self._latencies, cfg.recorder.latency_log)
        self._latencies.clear()

    # ─── PnL logging ────────────────────────────────────────────────────#

    def record_pnl(
        self,
        side: Side,
        notional_usd: float,
        pnl_usd: float,
        ts: Optional[float] = None
    ) -> None:
        """
        Record one realised-PnL event (entry or exit fill).
        """
        if not cfg.recorder.enable_recording:
            return
        rec = {
            "ts": ts or time.time(),
            "side": side,
            "notional_usd": notional_usd,
            "pnl_usd": pnl_usd,
        }
        with self._lock:
            self._pnls.append(rec)
            if len(self._pnls) >= self._batch:
                self._flush_pnls()

    def _flush_pnls(self) -> None:
        self._write_batch(self._pnls, cfg.recorder.pnl_log)
        self._pnls.clear()

    # ─── Final flush ────────────────────────────────────────────────────#

    def flush_all(self) -> None:
        """Force-flush all buffers, regardless of size."""
        with self._lock:
            self._flush_trades()
            self._flush_signals()
            self._flush_orders()
            self._flush_latencies()
            self._flush_pnls()