import asyncio, aiohttp, websockets, json, time, logging
import itertools, os, numpy as np

from collections                import deque
from typing                     import Any, Callable, Dict, List, Optional, Tuple
from websockets.legacy.client   import WebSocketClientProtocol
from eth_account                import Account
from hyperliquid.utils.signing  import sign_l1_action, get_timestamp_ms

# Price rounding
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
getcontext().prec = 40                      # plenty

# Constants
PING_INTERVAL = 30.0
PING_TIMEOUT = 10.0
HEALTH_CHECK_INTERVAL = 5.0
MAX_IDLE_TIME = 60.0          # seconds before forcing reconnect
MAX_PONG_WAIT = 90.0          # seconds before forcing reconnect
BASE_RECONNECT_DELAY = 1.0    # seconds
MAX_RECONNECT_DELAY = 300.0   # seconds

GUARDBAND_LOW   = 0.25
GUARDBAND_HIGH  = 1.75

DEFAULT_POST_TIMEOUT = 20.0

TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"
MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
MAINNET_API_URL = "https://api.hyperliquid.xyz"
TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"

logger = logging.getLogger("AsyncHyperLiquidClient")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


class AsyncHyperLiquidClient:
    """
    Asyncio-based client for HyperLiquid.

    Manages a single WS connection with automatic reconnection,
    heartbeat, and health checks, plus non-blocking HTTP via aiohttp.
    """

    def __init__(
        self,
        symbol: str,
        *,
        post_timeout: float = DEFAULT_POST_TIMEOUT,
        is_spot: bool = False,
        perp_dex: str = "",     # "" -> canonical pool
        perp_dex_idx: int = 0,  # 0  -> canonical pool
        testnet: bool = True,
        api_key: Optional[str] = None,
        eth_private_key: Optional[str] = None
    ):
        # Authentication
        if eth_private_key is not None:
            raw_key = eth_private_key.strip()
        else:
            raw_key = os.getenv("MetaMask_secret", "").strip()

        if not raw_key:
            raise RuntimeError(
                "No Ethereum private key supplied. "
                "Either pass eth_private_key='0x…' or set the MetaMask_secret env-var."
            )
        
        self.eth_private_key = raw_key
        # Canonical account objects
        self.wallet   = Account.from_key(self.eth_private_key)
        self._account = self.wallet            # convenience alias
        self.vault    = self.wallet.address.lower()

        # Params
        self.symbol       = symbol.upper()
        self.is_spot      = is_spot
        self.perp_dex     = perp_dex
        self.perp_dex_idx = perp_dex_idx
        self.testnet      = testnet
        self.ws_url       = TESTNET_WS_URL if testnet else MAINNET_WS_URL
        self.api_url      = TESTNET_API_URL if testnet else MAINNET_API_URL
        self.api_key      = api_key
        self.post_timeout = post_timeout

        # Async resources
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[WebSocketClientProtocol]         = None

        # Connection control
        self._connected        = asyncio.Event()
        self._should_reconnect = True
        self._reconnect_delay  = BASE_RECONNECT_DELAY

        # State tracking
        self.user_fills: deque          = deque(maxlen=500)
        self.last_message_time: float   = 0.0
        self.last_ping_time: float      = 0.0
        self.last_pong_time: float      = 0.0
        self._vol_calc                  = RollingVolatility(window_size=100)
        self._last_vol: Optional[float] = None

        # Latency tracking
        self._data_lat: deque = deque(maxlen=2_000)
        self._order_lat: deque = deque(maxlen=1_000)
        self._pending_ts: Dict[int, float] = {}
        self._printed_l2_latency = False

        # Data caches
        self.order_book: Optional[Dict[str, Any]] = None
        self.open_orders: List[Dict[str, Any]]    = []
        self.trade_cache                          = deque(maxlen=1000)
        self._processed_oids: set                 = set()  # Track OIDs already processed from POST

        # Callbacks by channel
        self._callbacks: Dict[str, List[Callable[[Any], None]]] = {
            "order_book":    [],
            "trade":         [],
            "order_updates": [],
            "order_fill":    [],
            "connection":    [],
            "volatility":    [],
            "userFills":     [],
        }

        # Internal lock
        self._lock = asyncio.Lock()

        # Tasks
        self._listener_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task]   = None

        # Subscriptions
        self._subscriptions: List[str] = []

        # Post request tracking
        self._pending: Dict[int, asyncio.Future] = {}
        self._post_seq = itertools.count(1)

        # in‐memory cache for HL metadata (tick/decimals/asset_id)
        self._meta_cache: dict[str, tuple[int,int,int]] = {}
        # one‐time caches so we only ever hit /info once
        self._cached_tick_decimals: tuple[int,int] | None = None
        self._cached_asset_id:     int               | None = None

    async def start(self) -> None:
        """Start the connection loop; handles reconnects automatically."""
        if not self.http_session:
            self.http_session = aiohttp.ClientSession()

        while self._should_reconnect:
            try:
                logger.info("Connecting to WebSocket %s", self.ws_url)
                self.ws = await websockets.connect(
                    self.ws_url,
                    ping_interval=PING_INTERVAL,
                    ping_timeout=PING_TIMEOUT
                )
                self._connected.set()
                for cb in self._callbacks["connection"]:
                    cb("connected")
                logger.info("WebSocket connected")
                # reset backoff
                self._reconnect_delay = BASE_RECONNECT_DELAY

                self._listener_task = asyncio.create_task(self._listen())

                # after successful connect:
                await self._resubscribe_all()
                
                # launch listener and health-check tasks
                self._health_task = asyncio.create_task(self._health_monitor())

                # wait until listener ends (connection closed)
                await self._listener_task

            except Exception as exc:
                logger.error("Connection error: %s", exc)
            finally:
                self._connected.clear()
                for cb in self._callbacks["connection"]:
                    cb("disconnected")
                if self.ws:
                    await self.ws.close()
                if self._health_task:
                    self._health_task.cancel()

                logger.info("Reconnecting in %.1f seconds", self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(MAX_RECONNECT_DELAY, self._reconnect_delay * 2)

    async def _listen(self) -> None:
        """Receive messages and dispatch to handlers."""
        try:
            async for raw in self.ws:
                self.last_message_time = time.time()
                await self._handle_message(raw)
        except websockets.ConnectionClosed as e:
            logger.warning("WebSocket closed: %s", e)
        except Exception as exc:
            logger.error("Listener error: %s", exc)

    async def _health_monitor(self) -> None:
        """Periodic health checks and send heartbeat pings."""
        try:
            while True:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                await self._check_idle_timeout()
                await self._send_ping_if_due()
        except asyncio.CancelledError:
            pass

    async def _check_idle_timeout(self) -> None:
        """Force reconnect if no messages or pongs in allowed window."""
        now = time.time()
        if now - self.last_message_time > MAX_IDLE_TIME:
            logger.warning("No messages for %.0f seconds", MAX_IDLE_TIME)
            await self._reconnect()
        elif self.last_ping_time and (now - self.last_pong_time > MAX_PONG_WAIT):
            logger.warning("No heartbeat response for %.0f seconds", MAX_PONG_WAIT)
            await self._reconnect()

    async def _reconnect(self) -> None:
        """Close current WS to trigger reconnect logic in start()."""
        if self.ws:
            await self.ws.close()

    async def _send_ping_if_due(self) -> None:
        """Send a ping message if interval has passed."""
        now = time.time()
        if now - self.last_ping_time >= PING_INTERVAL:
            await self.send_ping()

    async def send_ping(self) -> None:
        """Send a WebSocket ping for heartbeat."""
        if not self._connected.is_set():
            return
        try:
            await self.ws.ping()                        # RFC 6455 frame
            await self.ws.send('{"method":"ping"}')     # HL heartbeat
            self.last_ping_time = time.time()
            logger.info("Sent ping + HL heartbeat")
        except Exception as exc:
            logger.error("Ping error: %s", exc)

    async def _handle_message(self, raw: str) -> None:
        """Parse and dispatch a single WebSocket frame."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("WS-frame was not JSON: %s", raw)
            return

        # swallow plain-string heart-beats
        if not isinstance(msg, dict):
            return

        # first see if it belongs to the post/RPC/error handler
        if await self._route_response(msg):
            return                    # already handled

        # the rest need a 'channel'
        channel = msg.get("channel")
        if channel in ("pong", "heartbeat"):
            self.last_pong_time = time.time()
            return
        if channel == "subscriptionResponse":
            return

        handler = getattr(self, f"_on_{channel}", None)
        if handler:
            await handler(msg.get("data", {}))
        else:
            logger.error("Unhandled WS channel %s", channel)


    def register_callback(self, channel: str, fn: Callable[[Any], None]) -> None:
        """
        Register a callback for any channel name.
        If we haven't seen the key yet, create the list on the fly.
        """
        self._callbacks.setdefault(channel, []).append(fn)

    async def _on_order_book(self, data: Any) -> None:
        async with self._lock:
            self.order_book = data
        for fn in self._callbacks["order_book"]:
            fn(data)

    async def _on_trade(self, data: Any) -> None:
        async with self._lock:
            self.trade_cache.append(data)
        for fn in self._callbacks["trade"]:
            fn(data)

    async def close(self) -> None:
        """Clean up tasks and HTTP session."""
        self._should_reconnect = False
        if self._listener_task:
            self._listener_task.cancel()
        if self._health_task:
            self._health_task.cancel()
        if self.ws:
            await self.ws.close()
        if self.http_session:
            await self.http_session.close()

    async def _ensure_ws(self) -> None:
        """Wait until WebSocket is connected before sending."""
        if not self._connected.is_set():
            await asyncio.wait_for(self._connected.wait(), timeout=5)

    # ==== Subscriptions ====

    async def subscribe(self, topic: str, **params: Any) -> None:
        """
        Subscribe to a WebSocket channel.
        Records the subscription so we can re-subscribe after reconnect.
        """
        sub = {"type": topic, **params}
        payload = json.dumps({"method": "subscribe", "subscription": sub})
        key = json.dumps(sub, sort_keys=True)

        # record for auto–resubscribe
        if key not in self._subscriptions:
            self._subscriptions.append(key)

        await self._ensure_ws()
        await self.ws.send(payload)
        logger.info("Subscribed to %s %s", topic, params)

    async def unsubscribe(self, topic: str, **params: Any) -> None:
        """
        Unsubscribe from a WebSocket channel and remove from our list.
        """
        sub = {"type": topic, **params}
        payload = json.dumps({"method": "unsubscribe", "subscription": sub})
        key = json.dumps(sub, sort_keys=True)

        if key in self._subscriptions:
            self._subscriptions.remove(key)

        await self._ensure_ws()
        await self.ws.send(payload)
        logger.info("Unsubscribed from %s %s", topic, params)

    async def _resubscribe_all(self) -> None:
        """
        After reconnect, re-send every subscription we've recorded.
        """
        for key in self._subscriptions:
            sub = json.loads(key)
            payload = json.dumps({"method": "subscribe", "subscription": sub})
            await self.ws.send(payload)
            logger.info("Re-subscribed to %s", sub)

        # user-specific streams
        if self._account:
            await self.subscribe("orderUpdates", user=self._account.address)
            await self.subscribe("userFills",   user=self._account.address)

    # ==== Convenience subscriptions ====

    async def subscribe_to_l2_book(self) -> None:
        """Subscribe to live L2 order book updates for our symbol."""
        await self.subscribe("l2Book", coin=self.symbol)

    async def unsubscribe_from_l2_book(self) -> None:
        """Unsubscribe from L2 order book updates."""
        await self.unsubscribe("l2Book", coin=self.symbol)

    async def subscribe_to_trades(self) -> None:
        """Subscribe to live trade ticks for our symbol."""
        await self.subscribe("trades", coin=self.symbol)

    async def unsubscribe_from_trades(self) -> None:
        """Unsubscribe from trade ticks."""
        await self.unsubscribe("trades", coin=self.symbol)

    async def subscribe_to_order_updates(self) -> None:
        """
        Subscribe to order status updates (requires authentication).
        Fires our _on_orderUpdates handler as orders fill/cancel.
        """
        if not self._account:
            raise RuntimeError("No Ethereum account—cannot subscribe to order updates")
        await self.subscribe("orderUpdates", user=self._account.address)

    async def unsubscribe_from_order_updates(self) -> None:
        """Unsubscribe from order status updates."""
        if not self._account:
            return
        await self.unsubscribe("orderUpdates", user=self._account.address)

    async def subscribe_to_user_fills(self) -> None:
        """Subscribe to post-trade fills (requires authentication)."""
        if not self._account:
            raise RuntimeError("No Ethereum account—cannot subscribe to user fills")
        await self.subscribe("userFills", user=self._account.address)

    async def unsubscribe_from_user_fills(self) -> None:
        """Unsubscribe from post-trade fills."""
        if not self._account:
            return
        await self.unsubscribe("userFills", user=self._account.address)

    async def _route_response(self, msg: dict) -> bool:
        """
        Handle /post ACKs, RPC replies and error frames.
        Returns True when the frame has been fully handled, else False.
        """
        # 1) /post ACK ---------------------------------------------------
        if msg.get("channel") == "post":
            rid  = msg["data"]["id"]
            resp = msg["data"]["response"]
            fut  = self._pending.pop(rid, None)
            t0   = self._pending_ts.pop(rid, None)
            if fut and not fut.done():
                if resp.get("type") == "error":
                    fut.set_exception(RuntimeError(resp.get("payload")))
                else:
                    fut.set_result(resp)
            if t0 is not None:
                self._order_lat.append(time.time() - t0)
            return True

        # 2) generic RPC reply ------------------------------------------
        if "id" in msg:
            rid = msg["id"]
            fut = self._pending.pop(rid, None)
            t0  = self._pending_ts.pop(rid, None)
            if fut and not fut.done():
                fut.set_result(msg.get("result", msg))
            if t0 is not None:
                self._order_lat.append(time.time() - t0)
            return True

        # 3) error frame -------------------------------------------------
        if msg.get("channel") == "error":
            payload = msg.get("data")
            if isinstance(payload, str) and "Already subscribed" in payload:
                return True     # benign
            logger.error("WS ERROR: %s", payload)

            if isinstance(payload, dict) and "id" in payload:
                rid = payload["id"]
                fut = self._pending.pop(rid, None)
                if fut and not fut.done():
                    fut.set_exception(RuntimeError(payload.get("payload", payload)))
            if isinstance(payload, str):
                return # Already logged - nothing to wake
            return True

        # nothing caught
        return False



    # ==== Orders ====
    async def send_post(
        self,
        request: Dict[str, Any],            # already signed
        timeout: float | None = None,
    ) -> Dict[str, Any]:
        """
        Wrap *request* in the standard
            {"method":"post","id":…,"request":…}
        envelope, send it, and wait for the ACK.

        *request* must already contain the correct
        vaultAddress + signature → place_limit_order() et al. handle that.
        """
        await self._ensure_ws()

        timeout = timeout or self.post_timeout
        req_id = next(self._post_seq)

        ws_msg = {
            "method": "post",
            "id":     req_id,
            "request": request,
        }

        fut = asyncio.get_running_loop().create_future()
        self._pending[req_id]    = fut
        self._pending_ts[req_id] = time.time()

        await self.ws.send(json.dumps(ws_msg, separators=(",", ":")))
        logger.info("POST id=%s sent - awaiting reply …", req_id)

        try:
            reply = await asyncio.wait_for(fut, timeout)
            if reply["type"] == "error":
                raise RuntimeError(reply["payload"])
            return reply["payload"]
        finally:
            self._pending.pop(req_id, None)
            self._pending_ts.pop(req_id, None)

    # --------------------------------------------------------------------
    #  Meta helpers  – PERPS ONLY
    # --------------------------------------------------------------------
    _meta_cache: dict[str, tuple[float, int, int, int]] = {}

    async def _ensure_meta(self, coin: str, *, dex: str = "") -> None:
        """
        Cache  (tick, price_dec, size_dec, asset_id)  for *coin*.

        1. Use the explicit tick in /info.meta if present
           (fields: minTick, priceIncrement, perpPriceIncrement).
        2. Otherwise derive tick from szDecimals as per
              price_dec = MAX_DEC - szDecimals
              tick      = 10 ** (-price_dec)
           where MAX_DEC is 6 for perps, 8 for spot
           - see HL 'Tick and lot size' docs.
        """
        coin = coin.upper()
        if coin in self._meta_cache:
            return

        await self._ensure_http()
        url  = f"{self.api_url}/info"
        body = {"type": "meta"}
        if dex:
            body["dex"] = dex

        async with self.http_session.post(url, json=body) as r:
            r.raise_for_status()
            meta = await r.json()

        for idx, entry in enumerate(meta["universe"]):
            if entry["name"].upper() != coin:
                continue

            # ---- size precision (always present) ----------------------
            size_dec = int(entry["szDecimals"])

            # ---- preferred: tick supplied by the exchange -------------
            tick_raw = (
                entry.get("minTick")
                or entry.get("priceIncrement")
                or entry.get("perpPriceIncrement")
            )
            if tick_raw is not None:
                tick = float(tick_raw)
                price_dec = (
                    len(str(tick_raw).split(".")[1])
                    if "." in str(tick_raw) else 0
                )
            else:
                # ---- fallback: derive tick from szDecimals ------------
                MAX_DEC = 8 if self.is_spot else 6
                price_dec = max(0, MAX_DEC - size_dec)
                tick = 10 ** (-price_dec) if price_dec > 0 else 1

            # ---- asset-ID (unchanged) --------------------------------
            asset_id = (
                idx if not dex
                else 100000 + self.perp_dex_idx * 10000 + idx
            )

            self._meta_cache[coin] = (tick, price_dec, size_dec, asset_id)
            return

        raise ValueError(f"{coin} not in /info meta (dex='{dex}')")


    async def get_tick_and_decimals(self, coin: str) -> tuple[int,int]:
        """Fetch tick size & decimal precision once, then re-use."""
        if self._cached_tick_decimals is not None:
            return self._cached_tick_decimals

        await self._ensure_meta(coin, dex=self.perp_dex or "")
        tick, price_dec, size_dec, _ = self._meta_cache[coin]
        td = (tick, price_dec)
        self._cached_tick_decimals = td
        return td

    async def _get_asset_id(self, coin: str) -> int:
        """Return the asset ID (once fetched via /info)."""
        if self._cached_asset_id is not None:
            return self._cached_asset_id

        # ensure the metadata call has populated our cache
        await self._ensure_meta(coin, dex=self.perp_dex or "")

        # unpack (tick, price_decimals, size_decimals, asset_id)
        _, _, _, asset_id = self._meta_cache[coin.upper()]
        self._cached_asset_id = asset_id
        return asset_id

    def _get_size_decimals(self, coin: str) -> int:
        """
        Retrieve the size-precision that was cached by _ensure_meta().
        """
        _, _, size_dec, _ = self._meta_cache[coin.upper()]
        return size_dec

    # ---------------------------------------------------------------
    #  Price / size rounding helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _round_price(
        px: float | str,
        tick: float,
        price_dec: int,
        *,
        is_buy: bool | None = None,          # NEW kw-arg
    ) -> str:
        """
        Return *px* as a wire-safe price string that

        • is snapped to *tick*
            - ROUND_UP for buys (ceil)  → order never posts below desired price
            - ROUND_DOWN for sells (floor)
        • has ≤ 5 significant figures    (HL hard limit)
        • has ≤ price_dec decimal places (tick rule)
        • never uses scientific notation
        """
        d_px   = Decimal(str(px))
        d_tick = Decimal(str(tick))

        # snap to tick with direction-aware rounding
        rounding = ROUND_UP if is_buy else ROUND_DOWN
        snapped  = (d_px / d_tick).to_integral_value(rounding=rounding) * d_tick

        def render(num: Decimal, dec: int) -> str:
            s = f"{num:.{dec}f}".rstrip("0").rstrip(".") or "0"
            return s

        s = render(snapped, price_dec)

        # trim to ≤ 5 significant figures
        sigfigs = len(s.replace(".", "").lstrip("0"))
        if sigfigs > 5:
            excess   = sigfigs - 5
            new_dec  = max(0, price_dec - excess)
            snapped  = snapped.quantize(Decimal(10) ** -new_dec, ROUND_DOWN)
            s        = render(snapped, new_dec)

        return s


    @staticmethod
    def _round_size(sz: float, sz_dec: int) -> str:
        """
        Snap *sz* to the exchange precision and return a canonical string:

        • sz_dec == 0 → returns an **integer string** like "415"
        • sz_dec >  0 → returns exactly sz_dec decimal places without
                        trailing zeros being stripped off unexpectedly
        """
        from decimal import Decimal, ROUND_DOWN, getcontext
        getcontext().prec = 30

        q   = Decimal(10) ** (-sz_dec)   # quantisation step
        d   = Decimal(str(sz)).quantize(q, rounding=ROUND_DOWN)

        if sz_dec == 0:
            return str(int(d))           # "415"
        fmt = f"{{0:.{sz_dec}f}}"
        return fmt.format(d)             # e.g. "0.0061"
    
    @staticmethod
    def snap_size_to_contracts(raw_sz: float, sz_dec: int) -> float:
        """
        Return *raw_sz* rounded **down** to the nearest 10⁻ˢᶻ_dec,
        as a float (handy for risk maths).  Uses the same quantisation
        logic as _round_size(), so there is exactly one place that
        knows the exchange precision.
        """
        q = Decimal(10) ** (-sz_dec)               #   1, 0.1, 0.01, …
        return float(Decimal(str(raw_sz)).quantize(q, ROUND_DOWN))

    
    # ---------------
    # Order placement
    # ---------------
    # ────────────────────────────────────────────────────────────────────
    # MARKET ORDER  (IOC-limit within ±80% guard-band)
    # ────────────────────────────────────────────────────────────────────
    async def place_market_order(
        self,
        is_buy: bool,
        size: float,
        *,
        reduce_only: bool = False,
        grouping: str = "na",
    ) -> dict:
        """
        Fire-and-forget market execution implemented as an IOC limit
        whose price is inside the 80 % guard-band enforced by HL:

            guardLo  = reference_px x 0.25
            guardHi  = reference_px x 1.75

        • buy  → post at  guardHi   (well above ask)
        • sell → post at  guardLo   (well below bid)

        reference_px  = best-mid if we have it, otherwise 1_000.
        """
        # ── 1. reference price & guard-band ─────────────────────────────
        mid = self.get_mid_price()
        ref = mid if mid and mid > 0 else 1_000.0            # fallback

        guard_lo = ref * GUARDBAND_LOW         # −75 % above the -80% floor
        guard_hi = ref * GUARDBAND_HIGH         # +75 % below the +80% cap

        # ----------------------------------------------------------------
        # Use *cached* meta:  tick, price_dec, size_dec  (already fetched
        # once by Engine on start-up, so this is just a dict lookup)
        # ----------------------------------------------------------------
        tick, price_dec, size_dec, _ = self._meta_cache[self.symbol]

        crossing_px = guard_hi if is_buy else guard_lo
        price_str   = self._round_price(crossing_px, tick, price_dec)

        # 2) snap size
        size_str = self._round_size(size, size_dec)

        # ── 3. build IOC limit order ───────────────────────────────────
        asset_id = await self._get_asset_id(self.symbol)
        order = {
            "a": asset_id,
            "b": bool(is_buy),
            "p": price_str,
            "s": size_str,
            "r": bool(reduce_only),
            "t": {"limit": {"tif": "Ioc"}},      # Immediate-or-Cancel
        }
        action = {"type": "order", "orders": [order], "grouping": grouping}

        # ── 4. sign + POST ─────────────────────────────────────────────
        nonce     = get_timestamp_ms()
        signature = sign_l1_action(
            wallet        = self.wallet,
            action        = action,
            active_pool   = None, # self.vault
            nonce         = nonce,
            expires_after = None,
            is_mainnet    = not self.testnet,
        )
        payload = {"action": action, "nonce": nonce, "signature": signature}
        return await self.send_post({"type": "action", "payload": payload})

    # ────────────────────────────────────────────────────────────────────
    # LIMIT ORDER
    # ────────────────────────────────────────────────────────────────────
    async def place_limit_order(
        self,
        is_buy: bool,
        size: float,
        price: float,
        *,
        tif: str = "Gtc",
        reduce_only: bool = False,
        grouping: str = "na",
    ) -> dict:
        
        # snap to exchange precision
        tick, price_dec, size_dec, _ = self._meta_cache[self.symbol]
        price_str = self._round_price(price, tick, price_dec)
        size_str  = self._round_size(size,  size_dec)

        # ── build action ──
        asset_id = await self._get_asset_id(self.symbol)
        order = {
            "a": asset_id,
            "b": is_buy,
            "p": price_str,
            "s": size_str,
            "r": reduce_only,
            "t": {"limit": {"tif": tif}},
        }
        action = {"type": "order", "orders": [order], "grouping": grouping}

        # ── sign with NO pool / vault ──
        nonce     = get_timestamp_ms()
        signature = sign_l1_action(
            wallet        = self.wallet,     # eth_account.Account
            action        = action,
            active_pool   = None, # self.vault
            nonce         = nonce,
            expires_after = None,
            is_mainnet    = not self.testnet,
        )

        payload = {
            "action":    action,
            "nonce":     nonce,
            "signature": signature,
        }

        return await self.send_post({"type": "action", "payload": payload})
    
    # ──────────────────────────────────────────────────────────────────
    #  STOP-LIMIT  (trigger order – remains off-book until fired)
    # ──────────────────────────────────────────────────────────────────
    async def place_stop_limit(
        self,
        *,
        is_buy: bool,
        size: float,
        trigger_px: float,
        limit_px: float,
        reduce_only: bool = True,
        tif: str = "Gtc",
        grouping: str = "na",
    ) -> dict:
        # snap inputs to exchange precision
        tick, price_dec, size_dec, asset_id = self._meta_cache[self.symbol]
        trigger_str = self._round_price(trigger_px, tick, price_dec, is_buy=is_buy)
        limit_str   = self._round_price(limit_px,   tick, price_dec, is_buy=is_buy)
        size_str    = self._round_size(size, size_dec)

        order = {
            "a": asset_id,           # asset
            "b": is_buy,             # True = buy
            "p": limit_str,          # limitPx used *after* trigger
            "s": size_str,           # size
            "r": reduce_only,        # reduce-only
            "t": {                   # ---- NESTED trigger object ----
                "trigger": {
                    "isMarket":  False,
                    "triggerPx": trigger_str,
                    "tpsl":      "sl"          # tell HL this is a stop-loss
                }
            },
        }
        action = {"type": "order", "orders": [order], "grouping": grouping}

        nonce     = get_timestamp_ms()
        signature = sign_l1_action(
            wallet        = self.wallet,
            action        = action,
            active_pool   = None,
            nonce         = nonce,
            expires_after = None,
            is_mainnet    = not self.testnet,
        )
        payload = {"action": action, "nonce": nonce, "signature": signature}
        return await self.send_post({"type": "action", "payload": payload})
    
    # --------------------------------------------------------------------
    #  STOP-IOC helper: behaves like a “stop-market” without isMarket=true
    # --------------------------------------------------------------------
    async def place_stop_market(          # ← keep the old name for callers
        self,
        *,
        is_buy: bool,
        size: float,
        trigger_px: float,
        reduce_only: bool = True,
        grouping: str = "na",
    ) -> dict:
        """
        Protective stop that stays invisible until `triggerPx` is touched,
        then fires an IOC limit deep inside HL's ±80 % guard-band.

        For longs (SELL stop)  → limitPx = 25 % *reference   (far below bid)
        For shorts (BUY stop)  → limitPx = 175 % *reference  (far above ask)
        """
        # ── 1. snap trigger & size to exchange precision ──────────────────
        tick, price_dec, size_dec, asset_id = self._meta_cache[self.symbol]
        trig_str = self._round_price(trigger_px, tick, price_dec, is_buy=is_buy)
        size_str = self._round_size(size, size_dec)

        # ── 2. pick an aggressive limitPx for IOC fill ────────────────────
        # Use current mid if we have it; fall back to trigger price.
        mid   = self.get_mid_price() or trigger_px
        guard_lo = mid * GUARDBAND_LOW          # –75 % (inside HL’s 80 % floor)
        guard_hi = mid * GUARDBAND_HIGH          # +75 % (inside HL’s 80 % cap)
        limit_px = guard_hi if is_buy else guard_lo
        limit_str = self._round_price(limit_px, tick, price_dec)

        # ── 3. build the order dict ───────────────────────────────────────
        order = {
            "a": asset_id,
            "b": is_buy,
            "p": limit_str,          # aggressive price, crosses book
            "s": size_str,
            "r": reduce_only,
            "t": {
                "trigger": {
                    "isMarket": False,
                    "triggerPx": trig_str,
                    "tpsl": "sl"
                }
            },
        }
        action = {"type": "order", "orders": [order], "grouping": grouping}

        # ── 4. sign and POST ──────────────────────────────────────────────
        nonce     = get_timestamp_ms()
        signature = sign_l1_action(
            wallet        = self.wallet,
            action        = action,
            active_pool   = None,
            nonce         = nonce,
            expires_after = None,
            is_mainnet    = not self.testnet,
        )
        payload = {"action": action, "nonce": nonce, "signature": signature}
        return await self.send_post({"type": "action", "payload": payload})



    # ────────────────────────────────────────────────────────────────────
    # CANCEL ORDER
    # ────────────────────────────────────────────────────────────────────
    async def cancel_order(self, oid: int | str) -> dict:
        # (1)  look-up asset_id locally – _ensure_meta ran earlier
        _, _, _, asset_id = self._meta_cache[self.symbol]

        action   = {"type": "cancel", "cancels": [{"a": asset_id, "o": int(oid)}]}

        nonce     = get_timestamp_ms()
        signature = sign_l1_action(
            wallet=self.wallet,
            action=action,
            active_pool=None,        # keep NONE – works without vault
            nonce=nonce,
            expires_after=None,
            is_mainnet=not self.testnet,
        )
        payload = {"action": action, "nonce": nonce, "signature": signature}

        return await self.send_post({"type": "action", "payload": payload})


    # Channel handlers
    async def _on_l2Book(self, data: Dict[str, Any]) -> None:
        recv_ms   = int(time.time() * 1_000)
        server_ms = int(data.get("time", 0))
        if server_ms and not self._printed_l2_latency:
            lat_ms = recv_ms - server_ms
            print(f"[HL L2] one-off latency: {lat_ms} ms")
            self._printed_l2_latency = True

        async with self._lock:
            self.order_book = data
        for cb in self._callbacks["order_book"]:
            cb(data)

    async def _on_trades(self, data: List[Dict[str, Any]]) -> None:
        """Handle incoming trade updates."""
        for trade in data:
            px = float(trade['px'])

            # Latency capture
            ex_ts_ms = trade.get("ts")
            if ex_ts_ms is not None:
                recv_latency = time.time() - (ex_ts_ms / 1_000)
                self._data_lat.append(recv_latency)

            vol = self._vol_calc.update_price(px)
            if vol is not None:
                # Store it
                self._last_vol = vol

                # Fire callbacks
                for cb in self._callbacks['volatility']:
                    cb(vol)

            self.trade_cache.append(trade)
        for cb in self._callbacks["trade"]:
            cb(data)

    async def _on_orderUpdates(self, data: Any) -> None:
        """Handle order / fill updates safely."""
        logger.info("ORDER UPDATE RECEIVED: %s", data)  # TODO delete
        try:
            # raw rejects come through as a plain string
            if isinstance(data, str):
                logger.error("Order-reject: %s", data)
                for cb in self._callbacks.get("orderUpdates", []):
                    cb(data)
                return

            if not isinstance(data, list):
                logger.error("orderUpdates unexpected payload: %s", data)
                return

            async with self._lock:
                for upd in data:
                    if "order" in upd:                     # open / cancel / etc.
                        order  = upd["order"]
                        oid    = order.get("oid")
                        status = upd.get("status", "open")
                    elif "filled" in upd:                  # shorthand fill
                        filled = upd["filled"]
                        oid    = filled.get("oid")
                        status = "filled"
                        order  = None
                    else:
                        # unknown message shape → just forward & skip bookkeeping
                        logger.debug("Ignoring unknown orderUpdates packet: %s", upd)
                        for cb in self._callbacks.get("orderUpdates", []):
                            cb(upd)
                        continue

                    # --- mutate self.open_orders --------------------------------
                    if status == "filled":
                        self.open_orders = [o for o in self.open_orders if o.get("oid") != oid]
                        
                        # Skip if we already processed this fill from POST response
                        if oid in self._processed_oids:
                            logger.info("Skipping duplicate fill for oid=%s (already processed from POST)", oid)
                            continue
                        
                        # Create a proper fill structure for callbacks
                        if order:
                            fill_data = {
                                "filled": {
                                    "oid": oid,
                                    "side": "BUY" if order.get("side") == "B" else "SELL",
                                    "sz": order.get("origSz", "0"),
                                    "avgPx": order.get("limitPx"),  # For limit orders
                                    "totalSz": order.get("origSz", "0"),
                                }
                            }
                            for cb in self._callbacks.get("order_fill", []):
                                cb(fill_data)

                    elif status in ("canceled", "rejected", "marginCanceled"):
                        self.open_orders = [o for o in self.open_orders
                                            if o.get("oid") != oid]
                    else:                                   # open / replaced
                        if order:
                            found = next((o for o in self.open_orders
                                        if o.get("oid") == oid), None)
                            if found:
                                found.update(order)
                            else:
                                self.open_orders.append(order)

            # fan-out full batch
            for cb in self._callbacks.get("orderUpdates", []):
                cb(data)

        except Exception as exc:
            # Never crash the listener – just log and carry on
            logger.exception("orderUpdates handler error: %s", exc)


    async def _on_userFills(self, data: Dict[str, Any]) -> None:
        """
        Handle user fills (post-trade reports).
        HL sends a dict: { isSnapshot?: bool, user: str, fills: [WsFill, …] }
        """
        # 1) cache each fill in our local deque (for get_user_fills())
        for fill in data.get("fills", []):
            self.user_fills.append(fill)

        # 2) dispatch the *entire* payload to callbacks,
        #    so your engine will see data["isSnapshot"] and data["fills"]
        for cb in self._callbacks.get("userFills", []):
            cb(data)

    # ==== Public getters ====
    def get_l2_book(self) -> Optional[Dict[str, Any]]:
        """Return last L2Book snapshot (or None)."""
        return self.order_book

    def get_trades(self, n: int = 100) -> List[Dict[str, Any]]:
        """Return the last n trades."""
        return list(self.trade_cache)[-n:]

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Return current open orders."""
        return list(self.open_orders)

    def get_user_fills(self, n: int = 100) -> List[Dict[str, Any]]:
        """Return the last n fills."""
        return list(self.user_fills)[-n:]
    
    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Return the top-of-book bid and ask as floats, or (None, None).
        """
        book = self.get_l2_book()
        levels = (book or {}).get("levels")
        if not levels or len(levels) < 2:
            return None, None

        try:
            bid_px = float(levels[0][0]["px"])
            ask_px = float(levels[1][0]["px"])
        except Exception:
            return None, None

        return bid_px, ask_px

    def get_mid_price(self) -> Optional[float]:
        """
        Return (bid+ask)/2, or None if unavailable.
        """
        bid, ask = self.get_best_bid_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2
    
    def get_realized_vol(self) -> Optional[float]:
        """Return the most recent rolling-volatility (log-returns)."""
        return self._last_vol
    
    def get_data_latency_stats(self) -> Dict[str, float]:
        """
        Return min / p50 / p95 market-data latency (seconds) over the deque.
        """
        if not self._data_lat:
            return {}
        arr = np.fromiter(self._data_lat, dtype=float)
        return {
            "min":  float(arr.min()),
            "p50":  float(np.percentile(arr, 50)),
            "p95":  float(np.percentile(arr, 95)),
        }
    
    def get_order_latency_stats(self) -> Dict[str, float]:
        """
        Return min / p50 / p95 post-ack latency (seconds) over the deque.
        """
        if not self._order_lat:
            return {}
        arr = np.fromiter(self._order_lat, dtype=float)
        return {
            "min":  float(arr.min()),
            "p50":  float(np.percentile(arr, 50)),
            "p95":  float(np.percentile(arr, 95)),
    }

    async def _ensure_http(self):
        if not self.http_session or self.http_session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self.http_session = aiohttp.ClientSession(timeout=timeout)

    async def get_account_state(self) -> Dict[str, Any]:
        """
        Fetch the authenticated user's account state from the /info endpoint.
        Returns a dict or {"error": "..."} if not authenticated.
        """
        if not self._account:
            return {"error": "Authentication required"}

        await self._ensure_http()
        payload = {"type": "userState", "user": self._account.address}
        url = f"{self.api_url}/info"
        async with self.http_session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ==== Callback registration ====
    def on_order_book(self, fn: Callable[[Any], None]) -> None:
        self._callbacks["order_book"].append(fn)

    def on_trade(self, fn: Callable[[Any], None]) -> None:
        self._callbacks["trade"].append(fn)

    def on_order_fill(self, fn: Callable[[Any], None]) -> None:
        self._callbacks["order_fill"].append(fn)

    def on_connection(self, fn: Callable[[Any], None]) -> None:
        self._callbacks["connection"].append(fn)

    def on_volatility(self, fn: Callable[[float], None]) -> None:
        """Register a callback to receive each new rolling volatility."""
        self._callbacks['volatility'].append(fn)

    def on_user_fills(self, fn: Callable[[Any], None]) -> None:
        """Register a callback for each WsFill."""
        self._callbacks.setdefault("userFills", []).append(fn)
        
    
class RollingVolatility:
    """
    Maintains a rolling estimate of volatility (annualized or raw)
    using log-returns and a fixed window, in O(1) time per update.
    """
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.sq_buf = deque(maxlen=window_size)
        self.sum_squares = 0.0
        self.prev_price: float = None

    def update_price(self, price: float) -> float:
        """
        Call this on each new price. Returns the current volatility
        (stddev of log-returns over the window), or None if not enough data.
        """
        if self.prev_price is None:
            self.prev_price = price
            return None

        # Compute log‐return
        r = np.log(price / self.prev_price)
        self.prev_price = price

        # Update rolling sum of squares
        r2 = r * r
        if len(self.sq_buf) == self.window_size:
            # Remove oldest squared‐return
            old = self.sq_buf.popleft()
            self.sum_squares -= old

        self.sq_buf.append(r2)
        self.sum_squares += r2

        # Only compute if we have at least one return
        n = len(self.sq_buf)
        if n < 2:
            return None

        # Sample standard deviation of returns
        var = self.sum_squares / (n - 1)
        return np.sqrt(var)