from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import logging
import random
import ssl
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np

log = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
MAX_CANDLES = 5000
CANDLE_CHUNK_MINUTES = 4000

CANDLE_DTYPE = np.dtype(
    [
        ("ts", "i8"),
        ("bid_close_dc", "i4"),
        ("ask_close_dc", "i4"),
        ("bid_low_dc", "i4"),
        ("bid_high_dc", "i4"),
        ("ask_low_dc", "i4"),
        ("ask_high_dc", "i4"),
        ("last_dc", "i4"),
        ("volume", "f8"),
        ("open_interest", "f8"),
    ],
    align=True,
)

PRINT_DTYPE = np.dtype(
    [("ts_ns", "i8"), ("price_dc", "i4"), ("qty", "f8"), ("taker_side", "i1")],
    align=True,
)


class KalshiError(RuntimeError):
    pass


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _field(d: dict, name: str):

    # live endpoints use close_dollars / volume_fp, /historical uses close / volume
    for key in (f"{name}_dollars", f"{name}_fp", name):
        if d.get(key) is not None:
            return d[key]
    return None


def parse_ts(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value * 1e9)
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1e9)
    except ValueError:
        return 0


def market_outcome(market: dict) -> int | None:
    result = str(market.get("result") or "").strip().lower()
    return {"yes": 1, "no": 0}.get(result)


@dataclass
class KalshiClient:

    base: str = BASE
    min_interval_s: float = 0.15
    max_retries: int = 8
    timeout_s: float = 45.0
    _ctx: ssl.SSLContext = field(default_factory=ssl.create_default_context, repr=False)
    _last_call: float = field(default=0.0, repr=False)
    _conn: http.client.HTTPSConnection | None = field(default=None, repr=False)

    # one keep-alive connection: new TLS handshakes get reset under load
    def _connection(self) -> http.client.HTTPSConnection:
        if self._conn is None:
            host = urllib.parse.urlsplit(self.base).netloc
            self._conn = http.client.HTTPSConnection(host, timeout=self.timeout_s, context=self._ctx)
        return self._conn

    def _drop(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def get(self, path: str, **params: Any) -> dict:
        query = {k: v for k, v in params.items() if v is not None}
        url = self.base + path + ("?" + urllib.parse.urlencode(query) if query else "")
        parts = urllib.parse.urlsplit(url)
        target = parts.path + (f"?{parts.query}" if parts.query else "")
        last: Exception | None = None
        for attempt in range(self.max_retries):
            gap = time.monotonic() - self._last_call
            if gap < self.min_interval_s:
                time.sleep(self.min_interval_s - gap)
            try:
                conn = self._connection()
                conn.request("GET", target, headers={"Accept": "application/json",
                                                     "Connection": "keep-alive"})
                resp = conn.getresponse()
                body = resp.read()
                self._last_call = time.monotonic()
                if resp.status == 200:
                    return json.loads(body.decode("utf-8"))
                if resp.getheader("Connection", "").lower() == "close":
                    self._drop()
                if resp.status != 429 and 400 <= resp.status < 500:
                    raise KalshiError(f"HTTP {resp.status} for {url}: "
                                      f"{body[:400].decode('utf-8', 'replace')}")
                last = KalshiError(f"HTTP {resp.status}")
            except (http.client.HTTPException, ConnectionError, TimeoutError,
                    ssl.SSLError, OSError) as exc:
                self._last_call = time.monotonic()
                self._drop()
                last = exc
            time.sleep(min(8.0, 0.4 * 2**attempt) * (0.7 + 0.6 * random.random()))
        raise KalshiError(f"GET {url} failed after {self.max_retries} attempts: {last}")


@dataclass
class HistoryStore:

    root: Path = Path("data/history")
    client: KalshiClient = field(default_factory=KalshiClient)
    cache: bool = True

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.cache:
            (self.root / "raw").mkdir(parents=True, exist_ok=True)

    def _cache_path(self, path: str, params: dict) -> Path:
        key = json.dumps([path, sorted(params.items())], sort_keys=True)
        digest = hashlib.sha1(key.encode()).hexdigest()[:20]
        stem = path.strip("/").replace("/", "_")[:60]
        return self.root / "raw" / f"{stem}__{digest}.json.gz"

    def get(self, path: str, **params: Any) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        if not self.cache:
            return self.client.get(path, **params)
        f = self._cache_path(path, params)
        if f.exists():
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        payload = self.client.get(path, **params)
        tmp = f.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(f)
        return payload

    def series_list(self, max_pages: int = 200) -> list[dict]:
        out, cursor = [], None
        for _ in range(max_pages):
            j = self.get("/series", limit=200, cursor=cursor)
            rows = j.get("series") or []
            out.extend(rows)
            cursor = j.get("cursor")
            if not cursor or not rows:
                break
        return out

    def historical_markets(self, series_ticker: str, max_pages: int) -> tuple[list[dict], bool]:
        out, cursor = [], None
        for _ in range(max_pages):
            j = self.get("/historical/markets", limit=1000, series_ticker=series_ticker, cursor=cursor)
            rows = j.get("markets") or []
            out.extend(rows)
            cursor = j.get("cursor")
            if not cursor or not rows:
                return out, True
        return out, False

    def _chunks(self, start: int, end: int, interval: int) -> Iterator[tuple[int, int]]:
        span = CANDLE_CHUNK_MINUTES * interval * 60
        t = start
        while t < end:
            yield t, min(t + span, end)
            t += span

    def candlesticks(self, ticker: str, start_ts: int, end_ts: int, interval: int = 1) -> np.ndarray:
        rows = []
        tk = urllib.parse.quote(ticker, safe="")
        for lo, hi in self._chunks(int(start_ts), int(end_ts), interval):
            j = self.get(f"/historical/markets/{tk}/candlesticks",
                         start_ts=lo, end_ts=hi, period_interval=interval)
            for cs in j.get("candlesticks") or []:
                bid, ask, price = cs.get("yes_bid") or {}, cs.get("yes_ask") or {}, cs.get("price") or {}
                last = _field(price, "close") or _field(price, "previous")
                rows.append((
                    int(cs.get("end_period_ts") or 0),
                    round(_f(_field(bid, "close")) * 1000), round(_f(_field(ask, "close")) * 1000),
                    round(_f(_field(bid, "low")) * 1000), round(_f(_field(bid, "high")) * 1000),
                    round(_f(_field(ask, "low")) * 1000), round(_f(_field(ask, "high")) * 1000),
                    round(_f(last) * 1000),
                    _f(_field(cs, "volume")), _f(_field(cs, "open_interest")),
                ))
        if not rows:
            return np.empty(0, dtype=CANDLE_DTYPE)
        arr = np.array(rows, dtype=CANDLE_DTYPE)
        arr = arr[np.argsort(arr["ts"], kind="stable")]
        _, keep = np.unique(arr["ts"], return_index=True)
        return arr[np.sort(keep)]

    def market_candlesticks(self, market: dict, days_before_close: float = 7.0) -> np.ndarray:
        open_ns, close_ns = parse_ts(market.get("open_time")), parse_ts(market.get("close_time"))
        if not open_ns or not close_ns or close_ns <= open_ns:
            return np.empty(0, dtype=CANDLE_DTYPE)
        start_ns = max(open_ns, close_ns - int(days_before_close * 86400 * 1e9))
        return self.candlesticks(market["ticker"], start_ns // 10**9, close_ns // 10**9)

    def trades(self, ticker: str, max_pages: int = 200) -> np.ndarray:
        rows, cursor = [], None
        for _ in range(max_pages):
            j = self.get("/historical/trades", ticker=ticker, limit=1000, cursor=cursor)
            batch = j.get("trades") or []
            for t in batch:
                qty = _f(t.get("count_fp"))
                price_dc = round(_f(t.get("yes_price_dollars")) * 1000)
                if qty <= 0 or not 1 <= price_dc <= 999:
                    continue
                side = {"yes": 1, "no": -1}.get(str(t.get("taker_side", "")).lower(), 0)
                rows.append((parse_ts(t.get("created_time")), price_dc, qty, side))
            cursor = j.get("cursor")
            if not cursor or not batch:
                break
        else:
            raise KalshiError(f"{ticker}: more than {max_pages} trade pages")
        if not rows:
            return np.empty(0, dtype=PRINT_DTYPE)
        arr = np.array(rows, dtype=PRINT_DTYPE)
        return arr[np.argsort(arr["ts_ns"], kind="stable")]
