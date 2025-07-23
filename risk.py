"""
latency_arb.risk
================

Lightweight run-time risk-management helper.

 * All tunables are read from `config.cfg.risk`
 * Only plain-Python; no connector logic inside           -> easy unit-testing
 * Immutability for parameters, mutability for live PnL / exposure state
"""
from __future__     import annotations

import enum, datetime as _dt

from dataclasses    import dataclass, field
from typing         import Literal, Optional

from config import cfg


Side = Literal["LONG", "SHORT"]          # convenience alias


# --------------------------------------------------------------------------- #
#   ── Dataclasses that hold *mutable* run-time state ──                      #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Exposure:
    """Keeps running totals in *USD notional*."""
    long_usd:  float = 0.0
    short_usd: float = 0.0

    @property
    def gross(self) -> float:                     # total absolute exposure
        return self.long_usd + self.short_usd

    @property
    def net(self) -> float:                       # directional exposure
        return self.long_usd - self.short_usd

    # ---- helpers ---------------------------------------------------------- #
    def add(self, side: Side, notional: float) -> None:
        if side == "LONG":
            self.long_usd += notional
        else:
            self.short_usd += notional

    def reduce(self, side: Side, notional: float) -> None:
        if side == "LONG":
            self.long_usd = max(0.0, self.long_usd - notional)
        else:
            self.short_usd = max(0.0, self.short_usd - notional)


class ResetMode(str, enum.Enum):
    ROLLING = "rolling"   # manual / session-based
    DAILY   = "daily"     # auto reset at each UTC day boundary


@dataclass(slots=True)
class PnLTracker:
    reset_mode: ResetMode = ResetMode.ROLLING
    realised_usd: float   = 0.0
    _epoch: _dt.datetime  = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
    )

    # ── public api ──────────────────────────────────────────────
    def add(self, pnl_usd: float) -> None:
        """Record realised P&L (signed)."""
        self._check_reset()
        self.realised_usd += pnl_usd

    def value(self) -> float:
        """Return current session (or day) P&L."""
        self._check_reset()
        return self.realised_usd

    def age(self) -> _dt.timedelta:
        """How long has the current session been running?"""
        return _dt.datetime.now(_dt.timezone.utc) - self._epoch

    def reset(self) -> None:
        """Manually resets the tracker (for rolling mode)."""
        self._do_reset()

    # ── internal helpers ────────────────────────────────────────
    def _check_reset(self) -> None:
        """Auto-reset if reset_mode == DAILY and UTC date rolled over."""
        if self.reset_mode == ResetMode.DAILY:
            today = _dt.datetime.now(_dt.timezone.utc).date()
            if today != self._epoch.date():
                self._do_reset()

    def _do_reset(self) -> None:
        """Zero the counters and start a new epoch."""
        self.realised_usd = 0.0
        self._epoch = _dt.datetime.now(_dt.timezone.utc)

    # ------------------------------------------------------------------
    #  Back-compat for legacy callers
    # ------------------------------------------------------------------
    def reset_if_new_day(self) -> None:
        """
        DEPRECATED - kept only so existing code that still calls
        `tracker.reset_if_new_day()` doesn't break.  
        Internally delegates to the new `_check_reset()` method.
        """
        self._check_reset()

class PositionTracker:
    def __init__(self):
        self.size:         float = 0.0   # +long, –short
        self.avg_price:    float = 0.0
        self.realized_pnl: float = 0.0
        self.last_action_was_open: bool = False

    @property
    def notional(self) -> float:
        return abs(self.size) * self.avg_price

    def unrealized_pnl(self, mark_price: float) -> float:
        """Calculate unrealized PnL in USD, including entry fees."""
        if abs(self.size) < 0.0001:
            return 0.0
        
        # Base PnL in USD
        if self.size > 0:  # LONG
            pnl_usd = (mark_price - self.avg_price) * self.size
        else:              # SHORT
            pnl_usd = (self.avg_price - mark_price) * abs(self.size)
                
        # Subtract entry fees (already paid)
        entry_notional = abs(self.size) * self.avg_price
        entry_fee = entry_notional * 0.00045  # taker fee
        
        return pnl_usd - entry_fee

    def on_fill(
        self,
        side: Side,
        qty: float,
        price: float,
        closed_pnl: float = 0.0
    ) -> None:
        """
        Update position size, average price, and realized PnL.
        Sets last_action_was_open accordingly.
        """
        signed = qty if side == "LONG" else -qty
        # OPEN or build: same sign
        if self.size == 0 or self.size * signed > 0:
            # new weighted average
            new_size = self.size + signed
            self.avg_price = (
                abs(self.size) * self.avg_price
                + abs(signed) * price
            ) / abs(new_size)
            self.size = new_size
            self.last_action_was_open = True

        # REDUCE or CLOSE or REVERSE: opposite sign
        else:
            # how many contracts are we closing?
            close_qty = min(abs(signed), abs(self.size))
            self.realized_pnl += closed_pnl
            self.size += signed  # shrink or flip

            # if we flipped beyond zero, set new avg_price on the overshoot
            if abs(signed) > close_qty:
                self.avg_price = price

            self.last_action_was_open = False

# --------------------------------------------------------------------------- #
#                    ──  Main Risk Manager  ──                                #
# --------------------------------------------------------------------------- #
class RiskManager:
    """
    Stateless interface + stateful internals:

        >>> risk = RiskManager()
        >>> allowed, reason = risk.pre_trade_check("LONG", 8_000)
        >>> if allowed: … execute …

    Exposed methods
    ---------------
    • pre_trade_check(side, notional) -> tuple[bool, str]
    • register_fill(side, notional, pnl_change)
    • should_kill() -> bool
    """

    def __init__(self) -> None:
        self._exposure = Exposure()
        self._pnl      = PnLTracker()
        self._pos      = PositionTracker()

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #
    def pre_trade_check(self, side: Side, notional_usd: float) -> tuple[bool, str]:
        """Validate *before* submitting an order."""
        self._pnl.reset_if_new_day()

        # Check if we already have an open position
        if abs(self._pos.size) > 0.0001:  # small epsilon for float comparison
            return False, f"Already have an open position (size={self._pos.size:.4f})"

        # Size sanity
        if not (cfg.risk.min_trade_notional_usd <= notional_usd <= cfg.risk.max_trade_notional_usd):
            return False, (
                f"Trade notional outside [{cfg.risk.min_trade_notional_usd}, "
                f"{cfg.risk.max_trade_notional_usd}] USD window"
            )

        # Exposure limits
        new_exp = Exposure(
            long_usd  = self._exposure.long_usd  + (notional_usd if side == "LONG"  else 0.0),
            short_usd = self._exposure.short_usd + (notional_usd if side == "SHORT" else 0.0),
        )

        if new_exp.gross > cfg.risk.max_gross_position_usd:
            return False, f"Gross exposure cap exceeded ({new_exp.gross:.0f} > {cfg.risk.max_gross_position_usd})"

        if abs(new_exp.net) > cfg.risk.max_net_position_usd:
            return False, f"Net exposure cap exceeded ({new_exp.net:+.0f} > {cfg.risk.max_net_position_usd})"

        # Per-trade risk vs. equity
        est_risk = notional_usd * cfg.risk.max_risk_pct
        if est_risk > cfg.risk.max_daily_loss_usd:
            return False, "Single-trade risk exceeds daily loss limit"

        # Daily stop-loss
        if self._pnl.realised_usd - est_risk < -cfg.risk.max_daily_loss_usd:
            return False, "Would breach daily loss limit"

        return True, "OK"
    
    def register_fill(
        self,
        side:       Side,
        qty:        float,
        price:      float,
        closed_pnl: float = 0.0,
    ) -> None:
        """
        Unified fill handler:
         1) Update PositionTracker (size, avg_price, realized_pnl)
         2) For exits/reversals: reduce old side by the closed qty
         3) For entries/reversals:   add new side by any opened qty
         4) Record realized PnL via PnLTracker.add()
        """
        # 0) Snapshot our prior net contract size
        prior_size = self._pos.size  # + for LONG, – for SHORT

        # 1) Position math (updates self._pos.size and sets last_action_was_open)
        self._pos.on_fill(side, qty, price, closed_pnl)

        # 2) Figure out how many contracts were closed vs opened
        closed_qty = min(abs(qty), abs(prior_size))
        # Opened: any remaining beyond that
        opened_qty = abs(qty) - closed_qty

        # 3) Reduce the USD-notional of whatever side we just *closed*
        if closed_qty > 0:
            # the fill.side tells us *which way* we traded,
            # but the *opposite* side is what we were reducing
            closed_side = "LONG" if side == "SHORT" else "SHORT"
            self._exposure.reduce(closed_side, closed_qty * price)

        # 4) Add the USD-notional for any *new* position opened
        if opened_qty > 0:
            self._exposure.add(side, opened_qty * price)

        # 5) Record the realized PnL (entry→exit delta minus fees)
        self._pnl.add(closed_pnl)


    def should_kill(self) -> bool:
        """
        Return True if the strategy has hit its daily loss cap.
        """
        self._pnl.reset_if_new_day()
        return self._pnl.realised_usd <= -cfg.risk.max_daily_loss_usd

    # ------------------------------------------------------------------ #
    #  Introspection helpers for logging / monitoring                    #
    # ------------------------------------------------------------------ #
    @property
    def gross_exposure(self) -> float:
        return self._exposure.gross

    @property
    def net_exposure(self) -> float:
        return self._exposure.net

    @property
    def daily_pnl(self) -> float:
        self._pnl.reset_if_new_day()
        return self._pnl.realised_usd
    
    @property
    def position_size(self) -> float:
        return self._pos.size

    @property
    def avg_entry_price(self) -> float:
        return self._pos.avg_price

    def unrealized_pnl(self, mark_price: float) -> float:
        """Mark-to-market PnL in USD, matching HL's calculation"""
        return self._pos.unrealized_pnl(mark_price)

    @property
    def realized_pnl(self) -> float:
        return self._pos.realized_pnl