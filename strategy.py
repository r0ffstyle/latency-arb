"""
latency_arb.strategy
====================

Pure strategy logic for the Binance-→HyperLiquid latency-arbitrage.

• NO network or asyncio code here - that lives in engine.py
• All tunables are read once from `config.cfg.strategy`
• Side-effects only through the public RiskManager object you inject
"""

from __future__     import annotations

from collections    import deque
from dataclasses    import dataclass
from typing         import Literal, Optional, Any
from config         import cfg
from risk           import RiskManager, Side


# --------------------------------------------------------------------------- #
#   ── Helper dataclasses  ──                                                 #
# --------------------------------------------------------------------------- #
TradeSide = Literal["BUY", "SELL"]


@dataclass(slots=True, frozen=True)
class Trade:
    """
    Generic trade representation for any exchange feed.
    """
    ts_ms: int
    px:    float
    qty:   float
    side:  TradeSide        # BUY == taker was buyer; SELL == taker was seller
    feed: str = "unknown"

@dataclass(slots=True, frozen=True)
class HLBookTop:
    """
    L2 top-of-book data needed for imbalance & reference price.
    """
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float

@dataclass(slots=True, frozen=True)
class ExecutionPlan:
    """
    Returned by the strategy when a signal is approved.
    The engine (not the strategy) is responsible for turning this into a
    properly-rounded IOC-limit and submitting it to HyperLiquid.
    """
    side:          Side          # LONG / SHORT         (HL semantics)
    notional_usd:  float         # USD value we want to trade
    crossing_px:   float         # *raw* guard-band price (float)
    guard_lo:      float         # informational / logging only
    guard_hi:      float
    mid_px:        float         # actual market mid at signal time
    binance_ts_ms: int           # original Binance trade timestamp for e2e latency

@dataclass
class TrailingStopSignal:
    """
    Represents a request to place a stop order (either fee-adjusted break-even or trailing-stop).
    """
    stop_price: float
    side: str                    # "LONG" or "SHORT"
    size: float

class TrailingStopManager:
    """
    Immediate stop loss with progressive trailing logic:
    1. Initial stop: X bps below entry (immediate protection)
    2. Move to break-even when price reaches break-even
    3. Trail stop up maintaining fixed distance from high water mark
    """
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.active: bool = False
        self.side: Optional[str] = None
        self.entry_price: float = 0.0
        self.size: float = 0.0
        self.high_water: float = 0.0
        self.break_even_price: Optional[float] = None
        self.last_stop_price: Optional[float] = None
        self.stop_reached_breakeven: bool = False
        
    def start(self, side: str, entry_price: float, size: float) -> None:
        """Activate trailing-stop logic when a position is opened."""
        if not cfg.strategy.use_trailing_stop:
            return
            
        self.active = True
        self.side = side
        self.entry_price = entry_price
        self.size = size
        self.high_water = entry_price
        self.last_stop_price = None
        self.stop_reached_breakeven = False

        # Compute fee-adjusted break-even - Taker on both legs
        taker_fee = cfg.strategy.taker_fee
        if side == "LONG":
            self.break_even_price = entry_price * (1 + taker_fee) / (1 - taker_fee)
        else:
            self.break_even_price = entry_price * (1 - taker_fee) / (1 + taker_fee)

    def on_price_update(self, ref_price: float) -> Optional[TrailingStopSignal]:
        """
        Progressive stop management:
        1. Start with stop at -X bps from entry
        2. Move to break-even when profitable enough
        3. Trail stop maintaining distance from high water
        """
        if not self.active or self.side is None:
            return None
        
        # Update high-water mark
        if self.side == "LONG" and ref_price > self.high_water:
            self.high_water = ref_price
        elif self.side == "SHORT" and ref_price < self.high_water:
            self.high_water = ref_price

        # Calculate current stop price based on strategy phase
        if self.side == "LONG":
            # Phase 1: Initial stop loss (X bps below entry)
            initial_stop = self.entry_price * (1 - cfg.strategy.trailing_distance_bps / 10_000)
            
            # Phase 2: Break-even stop (if price has reached break-even + buffer)
            breakeven_trigger = self.break_even_price * (1 + cfg.strategy.break_even_buffer_bps / 10_000)
            if ref_price >= breakeven_trigger:
                self.stop_reached_breakeven = True
            
            # Phase 3: Trailing stop (if profitable)
            profit_bps = (self.high_water - self.entry_price) / self.entry_price * 10_000
            if profit_bps >= cfg.strategy.trailing_activation_bps:
                trailing_stop = self.high_water * (1 - cfg.strategy.profit_trailing_bps / 10_000)
            else:
                trailing_stop = 0
            
            # Choose the highest stop level
            if trailing_stop > 0:
                new_stop_px = max(trailing_stop, self.break_even_price)
            elif self.stop_reached_breakeven:
                new_stop_px = self.break_even_price
            else:
                new_stop_px = initial_stop
                
        else:  # SHORT
            # Phase 1: Initial stop loss (X bps above entry)
            initial_stop = self.entry_price * (1 + cfg.strategy.trailing_distance_bps / 10_000)
            
            # Phase 2: Break-even stop
            breakeven_trigger = self.break_even_price * (1 - cfg.strategy.break_even_buffer_bps / 10_000)
            if ref_price <= breakeven_trigger:
                self.stop_reached_breakeven = True
            
            # Phase 3: Trailing stop
            profit_bps = (self.entry_price - self.high_water) / self.entry_price * 10_000
            if profit_bps >= cfg.strategy.trailing_activation_bps:
                trailing_stop = self.high_water * (1 + cfg.strategy.profit_trailing_bps / 10_000)
            else:
                trailing_stop = float('inf')
            
            # Choose the lowest stop level
            if trailing_stop < float('inf'):
                new_stop_px = min(trailing_stop, self.break_even_price)
            elif self.stop_reached_breakeven:
                new_stop_px = self.break_even_price
            else:
                new_stop_px = initial_stop

        # ── SAFETY CLAMP — keep stop on correct side and never widen risk ──
        step = cfg.strategy.trailing_distance_bps / 10_000   # distance as fraction

        if self.side == "LONG":
            # stop must be STRICTLY below price
            min_allowed = ref_price * (1 - step)              # one cushion below
            floor       = self.break_even_price if self.stop_reached_breakeven else min_allowed
            if new_stop_px >= ref_price or new_stop_px < floor:
                new_stop_px = max(floor, min_allowed)

        else:  # SHORT
            # stop must be STRICTLY above price
            max_allowed = ref_price * (1 + step)              # one cushion above
            ceil        = self.break_even_price if self.stop_reached_breakeven else max_allowed
            if new_stop_px <= ref_price or new_stop_px > ceil:
                new_stop_px = min(ceil, max_allowed)
        # ────────────────────────────────────────────────────────────────────

            
        # Send stop order if it's the first one or has moved favorably
        if self.last_stop_price is None:
            self.last_stop_price = new_stop_px
            return TrailingStopSignal(
                stop_price=new_stop_px,
                side=self.side,
                size=self.size
            )
        elif self.side == "LONG" and new_stop_px > self.last_stop_price * 1.0001:  # 1 bp minimum move
            self.last_stop_price = new_stop_px
            return TrailingStopSignal(
                stop_price=new_stop_px,
                side=self.side,
                size=self.size
            )
        elif self.side == "SHORT" and new_stop_px < self.last_stop_price * 0.9999:
            self.last_stop_price = new_stop_px
            return TrailingStopSignal(
                stop_price=new_stop_px,
                side=self.side,
                size=self.size
            )
            
        return None
    
# --------------------------------------------------------------------------- #
#                             ──  Strategy  ──                                #
# --------------------------------------------------------------------------- #
class LatencyArbStrategy:
    """
    Stateless *interface* + tiny amount of rolling state that belongs
    logically to the strategy (threshold estimators & burst window).
    """

    def __init__(self, risk_mgr: RiskManager) -> None:
        self.cfg   = cfg.strategy
        self.risk  = risk_mgr

        # Burst detector: keep (ts_ms, qty) for trades within last N seconds
        self._burst_window: deque[tuple[int, float]] = deque()

    # ---------------------------------------------------------------------- #
    #  PUBLIC API                                                            #
    # ---------------------------------------------------------------------- #
    def on_trade(
        self,
        trade: Trade,
        hl_top: Optional[HLBookTop]
    ) -> tuple[bool, str | ExecutionPlan]:
        """
        Process *one* Binance trade and decide whether to execute on HL.

        Returns (True, ExecutionPlan)  if a validated signal should be executed
                (False, reason)        otherwise
        """

        # ----- 1. Is trade "significant"? (single-spike) ------------------- #
        is_single_spike = trade.qty >= self.cfg.single_trade_qty_threshold

        # ----- 2. Burst detector ------------------------------------------- #
        now_ms = trade.ts_ms
        self._burst_window.append((now_ms, trade.qty))
        cutoff = now_ms - self.cfg.burst_window_ms
        while self._burst_window and self._burst_window[0][0] < cutoff:
            self._burst_window.popleft()

        burst_vol = sum(q for _t, q in self._burst_window)
        is_burst  = burst_vol >= self.cfg.burst_volume_threshold

        if not (is_single_spike or is_burst):
            return False, "Below static significance thresholds"

        # ------------------------------------------------------------------ #
        #  3. Check book data availability                                   #
        # ------------------------------------------------------------------ #
        if hl_top is None:
            return False, "No HL order-book data yet"

        # Pre-calculate commonly used values for latency optimization
        mid_px = (hl_top.bid_px + hl_top.ask_px) / 2
        hl_side: Side = "LONG" if trade.side == "BUY" else "SHORT"

        # ------------------------------------------------------------------ #
        #  4. Apply chosen filter (imbalance or liquidity)                   #
        # ------------------------------------------------------------------ #
        if self.cfg.filter_type == "imbalance":
            # Legacy imbalance filter
            allowed, reason = self._check_imbalance_filter(trade, hl_top)
            if not allowed:
                return False, reason
        else:  # liquidity filter
            # New liquidity-based filter
            allowed, reason = self._check_liquidity_filter(
                trade.side, 
                hl_top, 
                cfg.risk.target_trade_notional_usd,
                mid_px
            )
            if not allowed:
                return False, reason

        # ------------------------------------------------------------------ #
        #  5. Notional sizing & risk checks                                  #
        # ------------------------------------------------------------------ #
        notional_usd = cfg.risk.target_trade_notional_usd

        # respect *only* the lower bound so we never submit dust orders
        if notional_usd < cfg.risk.min_trade_notional_usd:
            notional_usd = cfg.risk.min_trade_notional_usd
            
        allowed, reason = self.risk.pre_trade_check(hl_side, notional_usd)
        if not allowed:
            return False, f"Risk block: {reason}"

        # ------------------------------------------------------------------ #
        #  6. Guard-band price (raw float)                                   #
        # ------------------------------------------------------------------ #
        ref_px   = (hl_top.bid_px + hl_top.ask_px) / 2
        guard_lo = ref_px * self.cfg.guardband_lo
        guard_hi = ref_px * self.cfg.guardband_hi
        crossing_px = guard_hi if hl_side == "LONG" else guard_lo

        plan = ExecutionPlan(
            side          = hl_side,
            notional_usd  = notional_usd,
            crossing_px   = crossing_px,
            guard_lo      = guard_lo,
            guard_hi      = guard_hi,
            mid_px        = ref_px,
            binance_ts_ms = trade.ts_ms,
        )
        return True, plan

    # ------------------------------------------------------------------ #
    #  FILTER IMPLEMENTATIONS                                            #
    # ------------------------------------------------------------------ #
    def _check_imbalance_filter(
        self, 
        trade: Trade, 
        hl_top: HLBookTop
    ) -> tuple[bool, str]:
        """
        Legacy imbalance-based directional filter.
        """
        imbalance = self._calc_imbalance(hl_top)
        
        if trade.side == "BUY":
            # only go long if ask‐side is weak (imbalance very negative)
            if imbalance >= cfg.strategy.imb_short:
                return False, f"Ask side too strong for BUY (imb={imbalance:.2f})"
        elif trade.side == "SELL":
            # only go short if bid‐side is weak (imbalance very positive)
            if imbalance <= cfg.strategy.imb_long:
                return False, f"Bid side too strong for SELL (imb={imbalance:.2f})"
                
        return True, "Imbalance check passed"

    def _check_liquidity_filter(
        self, 
        trade_side: TradeSide,
        hl_top: HLBookTop,
        notional_usd: float,
        mid_px: float
    ) -> tuple[bool, str]:
        """
        Liquidity-based filter: ensure sufficient depth on the side we're taking.
        
        For latency arbitrage, we need to ensure there's enough liquidity
        to fill our order PLUS a buffer for other takers.
        """
        # Calculate contracts needed (already have mid_px passed in)
        contracts_needed = notional_usd / mid_px
        
        # Apply buffer (e.g., 20% more than we need)
        required_liquidity = contracts_needed * (1 + self.cfg.liquidity_buffer_pct)
        
        if trade_side == "BUY":
            # We're buying, so we need ask liquidity
            available = hl_top.ask_sz
            if available < required_liquidity:
                return False, f"Insufficient ask liquidity: {available:.2f} < {required_liquidity:.2f} needed"
        else:  # SELL
            # We're selling, so we need bid liquidity
            available = hl_top.bid_sz
            if available < required_liquidity:
                return False, f"Insufficient bid liquidity: {available:.2f} < {required_liquidity:.2f} needed"
                
        return True, f"Liquidity check passed ({available:.2f} available)"

    # ------------------------------------------------------------------ #
    #  INTERNAL HELPERS                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calc_imbalance(top: HLBookTop) -> float:
        bid_depth = top.bid_sz
        ask_depth = top.ask_sz
        tot_sz = bid_depth + ask_depth
        if tot_sz == 0:
            return 0.0
        return (ask_depth - bid_depth) / tot_sz