"""
Engine
==================
Async event-loop that wires together

• Binance Client (trade feeds)
• HyperLiquid Client (execution + HL L2 book)
• Strategy
• RiskManager

No strategy logic lives here - only orchestration, wiring, logging,
and graceful shutdown.
"""
from __future__ import annotations

import asyncio, logging, signal, sys, time
from typing import Any, Dict, Optional

from decimal import getcontext
getcontext().prec = 30

from connectors.binance_connector   import AsyncBinanceClient
from connectors.coinup_connector    import AsyncCoinUpClient
from connectors.hl_connector        import AsyncHyperLiquidClient               # noqa: N812
from strategy                       import LatencyArbStrategy, Trade, HLBookTop, ExecutionPlan, TrailingStopManager, TrailingStopSignal
from risk                           import RiskManager
from config                         import cfg
from recorder                       import Recorder


# ───────────────────────────── logging ──────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("engine")

# ─────────────────────────── helper utils ───────────────────────────
def _binance_to_trade(msg: Dict[str, Any]) -> Trade:  # Return Trade not BinanceTrade
    """Convert raw Binance trade JSON into the strategy's dataclass."""
    px   = float(msg["p"])
    qty  = float(msg["q"])
    ts   = int(msg["T"])
    side = "SELL" if msg["m"] else "BUY"
    feed = msg.get("feed", "binance_spot")  # Include feed type
    return Trade(ts_ms=ts, px=px, qty=qty, side=side, feed=feed)

def _hl_top_from_snapshot(book: Dict[str, Any]) -> Optional[HLBookTop]:
    """
    Extract best bid/ask + sizes from HL L2Book snapshot.
    Returns None until we have the first valid snapshot.
    """
    if not book or "levels" not in book or len(book["levels"]) < 2:
        return None

    bid_lvl, ask_lvl = book["levels"][0][0], book["levels"][1][0]
    return HLBookTop(
        bid_px=float(bid_lvl["px"]),
        bid_sz=float(bid_lvl["sz"]),
        ask_px=float(ask_lvl["px"]),
        ask_sz=float(ask_lvl["sz"]),
    )


# ─────────────────────────────  main  ───────────────────────────────
class Engine:
    def __init__(self) -> None:
        self.binance = AsyncBinanceClient(
            symbol      = cfg.binance.symbol,
            market_type = "perp",
            streams     = list(cfg.binance.streams),
        )

        self.coinup = None
        if cfg.coinup.enabled:
            self.coinup = AsyncCoinUpClient(symbol=cfg.coinup.symbol)

        self.hl = AsyncHyperLiquidClient(
            symbol   = cfg.hyperliquid.symbol,
            testnet  = cfg.hyperliquid.testnet,
            post_timeout = cfg.hyperliquid.post_timeout,
        )

        self.risk = RiskManager()
        self.strategy = LatencyArbStrategy(risk_mgr=self.risk)

        self.recorder = Recorder()

        self.trailing_mgr = TrailingStopManager(cfg.strategy)

        self._hl_top: Optional[HLBookTop] = None
        self._shutting_down = asyncio.Event()

        self._last_trade_by_feed: dict = {}

        self._has_pending_order = False # Hard guard against concurrent order submissions.

        # Exchange metadata, cached once in run()
        self._hl_tick:       float | None = None
        self._hl_price_dec:  int   | None = None
        self._hl_size_dec:   int   | None = None

        # Avoids race conditions in trailing-stop orders
        self._trail_lock = asyncio.Lock()


    # ------------------------------------------------------------------ #
    #  Wiring                                                            #
    # ------------------------------------------------------------------ #
    def _wire_callbacks(self) -> None:
        # ── Binance trades → our handler ──────────────────────────────
        # This is the one and only place we invoke the strategy logic
        self.binance.on_trade(self._handle_trade)

        # ── Binance raw JSON → debug log ─────────────────────────────
        # A separate callback that just prints the raw message for debugging.
        self.binance.on_trade(lambda raw: log.debug("[BN RAW] %s", raw))

        if self.coinup:
            self.coinup.on_trade(self._handle_trade)

        # ── HyperLiquid L2Book → update our local top-of-book ────────
        self.hl.on_order_book(self._handle_hl_l2)

        # ── HyperLiquid *actual fills* → feed into PnL & exposure tracker ────
        self.hl.on_user_fills(self._handle_hl_fill)

        # From now on, everything is fully event-driven

    # ------------------------------------------------------------------ #
    #  Callbacks                                                          #
    # ------------------------------------------------------------------ #
    def _handle_hl_l2(self, data: Dict[str, Any]) -> None:
        # ==================================
        log = logging.getLogger("engine.hl")
        log.debug("raw HL book: %s", data)
        # ==================================
        top = _hl_top_from_snapshot(data)
        if top:
            self._hl_top = top
        if self._hl_top is None:
            log.info("[BN→HL]   dropped: no HL book yet")
            return
        
        # —————————— feed trailing logic ——————————
        # choose the side-specific reference price
        ref_px = top.bid_px if self.trailing_mgr.side == "LONG" else top.ask_px
        signal = self.trailing_mgr.on_price_update(ref_px)
        if signal:
            # schedule the stop order
            asyncio.create_task(self._place_trailing_order(signal))

    def _handle_trade(self, raw: Dict[str, Any]) -> None:
        """
        Unified trade handler for all feeds.
        """
         # ─── 0) receive trade signal ─────────────────────────────────
        recv_ts = time.time()
        if self._hl_top is None:
            return
        
        # Extract source from feed field
        source = raw.get('feed', 'unknown')
        
        # Normalize trade based on source
        if source in ['spot', 'perp']:  # Binance feeds
            trade = _binance_to_trade(raw)
            display_source = 'BINANCE'
        elif source == 'coinup':
            # CoinUp already provides normalized format
            trade = Trade(
                ts_ms=raw['ts_ms'],
                px=raw['px'],
                qty=raw['qty'],
                side=raw['side'],
                feed=source
            )
            display_source = 'COINUP'
        else:
            return
        
        # Store last trade by feed, for status monitor
        self._last_trade_by_feed[source] = raw

        # Log significant trades
        if trade.qty >= cfg.strategy.single_trade_qty_threshold:
            log.info(
                "[%s] trade → time=%d qty=%.6f px=%.6f side=%s",
                display_source, trade.ts_ms, trade.qty, trade.px, trade.side
            )
        
        # Check for pending orders
        if self._has_pending_order:
            if trade.qty >= cfg.strategy.single_trade_qty_threshold:
                log.info("[STRAT] Signal rejected → Order already in flight")
            return
        
        # ─── 2) imbalance decision ────────────────────────────────────
        decision_start = time.time()
        allowed, result = self.strategy.on_trade(trade, self._hl_top)
        decision_end = time.time()
        decision_latency_ms = (decision_end - decision_start) * 1000
        
        if not allowed:
            if trade.qty >= cfg.strategy.single_trade_qty_threshold:
                log.info("[STRAT] Signal rejected → %s (decision time: %.1fms)", 
                        result, decision_latency_ms)
            return
        
        # Measure the latency
        log.info("[LATENCY] L2 → decision logic: %.1fms", decision_latency_ms)

        # Execute trade...
        self._has_pending_order = True

        # record time of send invocation
        send_ts = time.time()
        
        # record signal→order-post latencies
        self.recorder.record_signal(trade, self._hl_top, True, result)
        # decision→send
        send_latency_ms = (send_ts - decision_end) * 1000
        log.info("[LATENCY] decision→send post: %.1fms", send_latency_ms)

        # full breakdown
        log.info(
            "[LATENCY] L2→decision: %.1fms | decision→send: %.1fms",
             decision_latency_ms, send_latency_ms
        )

        # hand off to execution (which will record send→ack separately)
        asyncio.create_task(self._execute_plan(result, recv_ts, decision_latency_ms/1000))

    async def _execute_plan(self, plan: ExecutionPlan, signal_recv_ts: float, decision_time: float) -> None:
        """
        Translate ExecutionPlan → exact contract size & IOC-limit price,
        then submit via hl.place_limit_order().
        """
        try:
            # 1) size (contracts)
            mid_px   = (self._hl_top.bid_px + self._hl_top.ask_px) / 2
            raw_sz   = plan.notional_usd / mid_px           # USD ➜ contracts
            contracts = self.hl.snap_size_to_contracts(raw_sz, self._hl_size_dec)
            size_str  = str(contracts)                      # already precise

            # ----- 2) price (properly rounded string) -----
            price_str = self.hl._round_price(plan.crossing_px, self._hl_tick, self._hl_price_dec)

            # ----- 3) submit IOC-limit (i.e. market) -----
            is_buy = plan.side == "LONG"
            send_ts = time.time()
            
            # Record signal-to-send latency
            self.recorder.record_latency(
                exchange="local",
                stage="decision_to_send",
                stats={"min": None, "p50": (send_ts - signal_recv_ts) * 1000, "p95": None},
                ts=send_ts
            )
            
            reply  = await self.hl.place_limit_order(
                is_buy = is_buy,
                size   = size_str,
                price  = price_str,
                tif    = "Ioc",          # Immediate-or-Cancel
            )
            ack_ts  = time.time()
            # ----- 4) Process any immediate fills from the response -----
            statuses = reply.get("response", {}).get("data", {}).get("statuses", [])
            
            # Process fills that came back immediately in the POST response
            for status in statuses:
                if "filled" in status:
                    filled = status["filled"]
                    # Add the side to the fill data since it's missing
                    filled["side"] = "BUY" if is_buy else "SELL"
                    # Create a fill update in the same format as WebSocket updates
                    fill_update = {"filled": filled}
                    log.info("[EXEC] Processing immediate fill from POST response")
                    self._handle_hl_fill(fill_update)
                    # Mark this OID as processed to skip WebSocket duplicate
                    if "oid" in filled:
                        self.hl._processed_oids.add(filled["oid"])
            
            # ----- 5) Record order details -----
            self.recorder.record_order(plan, send_ts, ack_ts, statuses)
            
            # Record ACK latency
            ack_latency_ms = (ack_ts - send_ts) * 1000
            self.recorder.record_latency(
                exchange="hyperliquid",
                stage="send_to_ack",
                stats={"min": None, "p50": ack_latency_ms, "p95": None},
                ts=ack_ts
            )
            
            # Record full local end-to-end latency
            signal_to_ack_ms = (ack_ts - signal_recv_ts) * 1000
            self.recorder.record_latency(
                exchange="local",
                stage="signal_to_ack",
                stats={"min": None, "p50": signal_to_ack_ms, "p95": None},
                ts=ack_ts
            )
            
            # Log all latencies for this execution
            decision_to_send_ms = (send_ts - signal_recv_ts - decision_time) * 1000
            log.info(
                "[LATENCY] signal→decision: %.1fms | decision→send: %.1fms | send→ack: %.1fms | TOTAL: %.1fms",
                decision_time * 1000,
                decision_to_send_ms,
                ack_latency_ms,
                signal_to_ack_ms
            )

            log.info(
                "[EXEC] %s %s @ %s  ACK=%s",
                "BUY" if is_buy else "SELL",
                size_str,
                price_str,
                reply,
            )

        except Exception as exc:
            log.exception("[EXEC] submission failed: %s", exc)
        finally:
            # Clear the pending flag regardless of success/failure
            self._has_pending_order = False
            log.debug("[EXEC] Cleared pending order flag")

    def _handle_hl_fill(self, upd: dict) -> None:
        """
        Handle every WsUserFills message, iterating through its 'fills' array.
        """
        # 0) ignore the initial snapshot if present
        if upd.get("isSnapshot", False):
            return

        # 1) get the array of fills; bail if empty
        fills = upd.get("fills", [])
        if not fills:
            return

        # 2) process each fill
        for f in fills:
            # normalize side, qty, price
            raw_side = f["side"].upper()
            # HL uses "B" for buy, "S" for sell (and might also send full "BUY"/"SELL")
            if raw_side.startswith("B"):
                side = "LONG"
            else:
                side = "SHORT"
            qty  = float(f["sz"])
            px   = float(f["px"])

            # fees: trust HL fee if present; otherwise use crossed flag
            raw_fee = f.get("fee")
            if raw_fee is not None:
                fee = float(raw_fee)
            else:
                is_taker = bool(f.get("crossed", True))
                rate     = cfg.strategy.taker_fee if is_taker else cfg.strategy.maker_fee
                fee      = qty * px * rate

            # PnL: HL gives you closedPnl on exits, fallback to -fee
            closed_pnl = float(f.get("closedPnl", -fee))

            # record existing size before updating
            prior_size = self.risk.position_size

            # register the fill once (adds/reduces exposure & PnL)
            self.risk.register_fill(
                side       = side,
                qty        = qty,
                price      = px,
                closed_pnl = closed_pnl,
            )

            # get updated size for trailing logic
            current_size = self.risk.position_size

            # activate trailing if we just went from 0 → non-zero
            if abs(prior_size) < 1e-8 and abs(current_size) > 1e-8:
                entry_price  = self.risk.avg_entry_price
                pos_side     = "LONG" if current_size > 0 else "SHORT"
                log.info(
                    "[TRAIL] Activating trailing stop for %s position: size=%.4f entry=%.2f",
                    pos_side, abs(current_size), entry_price
                )
                self.trailing_mgr.start(pos_side, entry_price, abs(current_size))

            # deactivate trailing if we just closed out
            elif abs(prior_size) > 1e-8 and abs(current_size) < 1e-8:
                log.info("[TRAIL] Position closed, deactivating trailing stop")
                self.trailing_mgr.active = False
                self._has_pending_order = False

            # persist to disk
            notional = qty * px
            self.recorder.record_pnl(side, notional, closed_pnl, ts=time.time())

            # final log line
            log.info(
                "\n[@HL_FILL] oid=%s side=%s qty=%.4f px=%.2f fee=%.4f pnl=%.4f | pos: %.4f→%.4f",
                f.get("oid"), side, qty, px, fee, closed_pnl, prior_size, current_size
            )
        
    async def _place_trailing_order(self, signal: TrailingStopSignal) -> None:
        """
        Replace any existing reduce-only stop with a fresh protective order.

        • LONG  →  SELL stop  (limit or market, reduce-only)
        • SHORT →  BUY  stop
        """
        # ── 1. direction & core params ──────────────────────────────────
        is_buy     = signal.side == "SHORT"          # BUY to cover short
        size       = signal.size
        trig_px    = signal.stop_price

        # If we will send a limit order, cross the spread by two ticks
        lim_px     = (trig_px + 2 * self._hl_tick) if is_buy else \
                     (trig_px - 2 * self._hl_tick)

        async with self._trail_lock:
            try:
                # ── 2. cancel older reduce-only stops ───────────────────
                for o in self.hl.get_open_orders():
                    if (o.get("coin") == cfg.hyperliquid.symbol.upper()
                            and o.get("reduceOnly")):
                        await self.hl.cancel_order(o["oid"])
                        log.info("[TRAIL] cancelled stale stop %s", o["oid"])

                # ── 3. place the new protective order ──────────────────
                if cfg.strategy.use_stop_market:
                    await self.hl.place_stop_market(
                        is_buy     = is_buy,
                        size       = size,
                        trigger_px = trig_px,
                        reduce_only= True,
                    )
                    order_type = "stop-market"
                    extra      = ""
                else:
                    await self.hl.place_stop_limit(
                        is_buy      = is_buy,
                        size        = size,
                        trigger_px  = trig_px,
                        limit_px    = lim_px,
                        reduce_only = True,
                    )
                    order_type = "stop-limit"
                    extra      = f"  lim {lim_px:.5f}"

                # ── 4. telemetry ───────────────────────────────────────
                bps = abs(trig_px - self.trailing_mgr.entry_price) \
                      / self.trailing_mgr.entry_price * 10_000
                side_str = "BUY" if is_buy else "SELL"
                log.info(
                    "[TRAIL] posted %s %s %.4f  trig %.5f%s  (%.2f bps)",
                    order_type, side_str, size, trig_px, extra, bps
                )

            except Exception:
                log.exception("[TRAIL] failed to place protective order")


    async def _status_monitor(self):
        """
        Every 30 s, log:
        • HL mid, bid/ask
        • last trades from all feeds (Binance spot/perp, CoinUp)
        """
        while not self._shutting_down.is_set():
            try:
                # HyperLiquid
                mid = self.hl.get_mid_price()
                bid, ask = self.hl.get_best_bid_ask()
                log = logging.getLogger("engine.monitor")
                log.info("HL mid=%.6f  bid=%.6f  ask=%.6f",
                        mid or 0.0,
                        bid or 0.0,
                        ask or 0.0)
                
                # Binance trades (raw format)
                spot = self._last_trade_by_feed.get("spot")
                perp = self._last_trade_by_feed.get("perp")
                if spot:
                    log.info("BN-SPOT last %s %s @ %s",
                            "SELL" if spot["m"] else "BUY",
                            spot["q"], spot["p"])
                if perp:
                    log.info("BN-PERP last %s %s @ %s",
                            "SELL" if perp["m"] else "BUY",
                            perp["q"], perp["p"])
                
                # CoinUp trades (normalized format)
                coinup = self._last_trade_by_feed.get("coinup")
                if coinup:
                    log.info("COINUP last %s %.6f @ %.6f",
                            coinup["side"],
                            coinup["qty"], 
                            coinup["px"])
                            
            except Exception:
                log.exception("Status monitor failure")
            await asyncio.sleep(30)


    # ------------------------------------------------------------------ #
    #  Lifecycle                                                         #
    # ------------------------------------------------------------------ #
    async def _status_printer(self) -> None:
        """Periodic one-liner heartbeat with latency & exposure."""
        while not self._shutting_down.is_set():
            await asyncio.sleep(cfg.engine.status_print_interval_s)

            # get mid‐price for mark‐to‐market
            bid, ask = self.hl.get_best_bid_ask()
            mid = 0.5 * ((bid or 0.0) + (ask or 0.0))

            # realized & unrealized PnL
            r_pnl = self.risk.realized_pnl
            u_pnl = self.risk.unrealized_pnl(mid)

            # Get local processing latency instead of cross-system latency
            # This will show how fast YOUR system processes signals
            log.info(
                "♥ beat  pos=%+.4f@%.2f  mid=%.6f  rPnL=$%+.2f  uPnL=$%+.2f (%.1f%%)  gross=$%.0f  net=$%+.0f",
                self.risk.position_size,
                self.risk.avg_entry_price,
                mid,
                r_pnl,
                u_pnl,
                (u_pnl / (abs(self.risk.position_size) * self.risk.avg_entry_price) * 100) if self.risk.position_size != 0 else 0,
                self.risk.gross_exposure,
                self.risk.net_exposure,
            )

    async def run(self) -> None:
        self._wire_callbacks()

        # Start connectors
        tasks = [
            asyncio.create_task(self.binance.start()),
            asyncio.create_task(self.hl.start()),
            asyncio.create_task(self._status_printer()),
        ]
        # Add CoinUp if enabled
        if self.coinup:
            tasks.append(asyncio.create_task(self.coinup.start()))

        # Subscribe to HL data once WS is up
        await self.hl._connected.wait()
        await self.hl.subscribe_to_l2_book()
        await self.hl.subscribe_to_order_updates()
        await self.hl.subscribe_to_user_fills()

        # fetch both tick & price_dec from HL, and also size_dec via a helper
        tick, price_dec = await self.hl.get_tick_and_decimals(cfg.hyperliquid.symbol)
        size_dec        = self.hl._get_size_decimals(cfg.hyperliquid.symbol)

        self._hl_tick      = tick
        self._hl_price_dec = price_dec
        self._hl_size_dec  = size_dec

        log.info("[INIT] cached HL tick=%.8f, price_dec=%d, size_dec=%d",
                tick, price_dec, size_dec)

        # handle SIGINT / SIGTERM for clean shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutting_down.set)
            except NotImplementedError:
                # Windows: fall back to signal.signal (sync) or ignore
                signal.signal(sig, lambda *_: self._shutting_down.set())

        # start periodic status logging
        self._monitor_task = asyncio.create_task(self._status_monitor())

        # Main wait
        await self._shutting_down.wait()

        log.info("Shutting down …")
        for t in tasks:
            t.cancel()
        await self.binance.close()
        if self.coinup:
            await self.coinup.close()
        await self.hl.close()

    # ------------------------------------------------------------------ #
    #  Entrypoint                                                        #
    # ------------------------------------------------------------------ #
def main() -> None:
    try:
        asyncio.run(Engine().run())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()