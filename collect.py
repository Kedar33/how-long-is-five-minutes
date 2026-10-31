from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


import numpy as np

from kalshi import HistoryStore, KalshiError, _f, market_outcome, parse_ts

ROOT = Path("data/hist")
CENSUS = ROOT / "census"
SAMPLE = ROOT / "sample.json"
FAILED = ROOT / "failed.jsonl"
MKT = ROOT / "mkt"
FIELDS = ("ticker", "event_ticker", "market_type", "result", "open_time", "close_time",
          "volume_fp", "price_level_structure", "status")
# some old tickers have characters windows won't take in file names
_BAD = str.maketrans({c: "_" for c in '<>:"/\\|?*'})

log = logging.getLogger("collect")


def market_file(ticker: str, suffix: str = ".npz", root: Path | None = None) -> Path:

    return (root or MKT) / f"{ticker.translate(_BAD)}{suffix}"


def is_combinatorial(series: str) -> bool:
    t = series.upper()
    return t.startswith("KXMVE") or "MVE" in t or "CROSSCATEGORY" in t


def census(args) -> int:
    CENSUS.mkdir(parents=True, exist_ok=True)
    series = HistoryStore().series_list()
    meta = {s["ticker"]: {k: s.get(k) for k in ("category", "frequency", "title", "fee_type")}
            for s in series}
    (ROOT / "series.json").write_text(json.dumps(meta))
    todo = sorted(t for t in meta if not is_combinatorial(t) and not (CENSUS / f"{t}.json.gz").exists())
    print(f"{len(series):,} series, {len(todo):,} to census")
    local = threading.local()

    def one(tk: str) -> None:
        if not hasattr(local, "store"):
            local.store = HistoryStore()
        for attempt in range(6):
            try:
                rows, complete = local.store.historical_markets(tk, args.max_pages)
                break
            except KalshiError as exc:
                log.warning("%s attempt %d: %s", tk, attempt + 1, exc)
                time.sleep(20 * (attempt + 1))
        else:
            return
        keep = [{k: m.get(k) for k in FIELDS} | ({} if complete else {"census_window": "recent"})
                for m in rows]
        out = CENSUS / f"{tk}.json.gz"
        tmp = out.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(keep, fh)
        tmp.replace(out)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    return 0


def _eligible(m: dict, min_volume: float) -> bool:
    return (m.get("market_type") in (None, "binary") and market_outcome(m) is not None
            and _f(m.get("volume_fp")) >= min_volume)


def sample(args) -> int:
    series_meta = json.loads((ROOT / "series.json").read_text())
    by_event: dict[str, dict] = {}
    n_all = 0
    for f in sorted(CENSUS.glob("*.json.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            rows = json.load(fh)
        n_all += len(rows)
        for m in rows:
            if not _eligible(m, args.min_volume):
                continue
            ev = m.get("event_ticker") or m["ticker"]
            if ev not in by_event or _f(m["volume_fp"]) > _f(by_event[ev]["volume_fp"]):
                by_event[ev] = dict(m, series=f.name.split(".json")[0],
                                    census_window=m.get("census_window", "full"))
    rng = random.Random(args.seed)
    per_series: dict[str, list[dict]] = {}
    for m in by_event.values():
        per_series.setdefault(m["series"], []).append(m)
    picks = []
    for ser in sorted(per_series):
        evs = sorted(per_series[ser], key=lambda m: m["ticker"])
        if len(evs) > args.max_per_series:
            evs = rng.sample(evs, args.max_per_series)
        picks.extend(evs)
    rng.shuffle(picks)
    for m in picks:
        sm = series_meta.get(m["series"], {})
        m["category"], m["frequency"] = sm.get("category"), sm.get("frequency")
    SAMPLE.write_text(json.dumps({"markets_in_census": n_all, "events_eligible": len(by_event),
                                  "markets": picks}))
    print(f"{n_all:,} archived markets, {len(by_event):,} eligible events, {len(picks):,} sampled")
    return 0


def fetch(args) -> int:
    root = Path(args.out_dir) if args.out_dir else MKT
    root.mkdir(parents=True, exist_ok=True)
    picks = json.loads(Path(args.sample_file or SAMPLE).read_text())["markets"]
    if args.only is not None:
        picks = [m for m in picks if m["ticker"] in args.only]
    todo = [m for m in picks if not market_file(m["ticker"], root=root).exists()]
    print(f"{len(todo):,} of {len(picks):,} to fetch")
    local = threading.local()
    lock = threading.Lock()

    def one(m: dict) -> None:
        if not hasattr(local, "store"):
            local.store = HistoryStore(cache=False)
        tk = m["ticker"]
        try:
            trades = local.store.trades(tk, max_pages=args.trade_pages)
            candles = local.store.market_candlesticks(m)
            tmp = market_file(tk, ".tmp.npz", root)
            np.savez_compressed(tmp, candles=candles, trades=trades,
                                outcome=np.int8(market_outcome(m)),
                                close_ns=np.int64(parse_ts(m["close_time"])),
                                meta=np.array(json.dumps(m)))
            tmp.replace(market_file(tk, root=root))
        except Exception as exc:
            with lock, FAILED.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ticker": tk, "err": str(exc)[:300]}) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    return 0


def strikes(args) -> int:
    series_meta = json.loads((ROOT / "series.json").read_text())
    by_ev = {m["event_ticker"]: m for m in json.loads(SAMPLE.read_text())["markets"]}
    rng = random.Random(args.seed)
    chosen = set(rng.sample(sorted(by_ev), args.events))
    found: dict[str, list[dict]] = {}
    for f in sorted(CENSUS.glob("*.json.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            rows = json.load(fh)
        ser = f.name.split(".json")[0]
        sm = series_meta.get(ser, {})
        for m in rows:
            if m.get("event_ticker") in chosen and _eligible(m, args.min_volume):
                found.setdefault(m["event_ticker"], []).append(
                    dict(m, series=ser, category=sm.get("category"),
                         census_window=m.get("census_window", "full")))
    picks = []
    for ev in sorted(found):
        ms = sorted(found[ev], key=lambda m: m["ticker"])
        if len(ms) > args.max_strikes:
            main = [m for m in ms if m["ticker"] == by_ev[ev]["ticker"]]
            rest = [m for m in ms if m["ticker"] != by_ev[ev]["ticker"]]
            ms = main + rng.sample(rest, args.max_strikes - len(main))
        picks.extend(ms)
    (ROOT / "sample_strikes.json").write_text(json.dumps({"events": len(found), "markets": picks}))
    print(f"{len(found):,} events, {len(picks):,} strikes")
    return 0


def retry(args) -> int:
    if not FAILED.exists():
        return 0
    root = Path(args.out_dir) if args.out_dir else MKT
    failed = {json.loads(line)["ticker"] for line in FAILED.open(encoding="utf-8")}
    args.only = {t for t in failed if not market_file(t, root=root).exists()}
    return fetch(args)


def _keep_awake() -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    _keep_awake()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census")
    c.add_argument("--max-pages", type=int, default=150)
    c.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("sample")
    s.add_argument("--min-volume", type=float, default=200.0)
    s.add_argument("--max-per-series", type=int, default=2000)
    s.add_argument("--seed", type=int, default=20261022)
    for name in ("fetch", "retry"):
        f = sub.add_parser(name)
        f.add_argument("--workers", type=int, default=6)
        f.add_argument("--trade-pages", type=int, default=200 if name == "fetch" else 2000)
        f.add_argument("--sample-file", default=None)
        f.add_argument("--out-dir", default=None)
        f.set_defaults(only=None)
    k = sub.add_parser("strikes")
    k.add_argument("--events", type=int, default=2500)
    k.add_argument("--max-strikes", type=int, default=30)
    k.add_argument("--min-volume", type=float, default=200.0)
    k.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    return {"census": census, "sample": sample, "fetch": fetch, "strikes": strikes,
            "retry": retry}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
