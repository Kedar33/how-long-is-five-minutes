from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CENTS = 100.0
HORIZONS_S = (60, 300, 900, 1800, 3600)

OBS_BASE = [
    ("ts_ns", "i8"),
    ("price", "f8"),
    ("direction", "i1"),
    ("qty", "f8"),
    ("mid", "f8"),
    ("spread", "f8"),
    ("outcome", "i1"),
    ("secs_to_close", "f8"),
]


def valid_quote_mask(candles: np.ndarray) -> np.ndarray:

    bid, ask = candles["bid_close_dc"], candles["ask_close_dc"]
    # a missing side comes back as 0 or 1000
    return (bid > 0) & (ask > 0) & (ask < 1000) & (ask > bid)


def build_observations(candles, trades, outcome: int, close_ns: int, horizons_s=HORIZONS_S):
    dtype = np.dtype(OBS_BASE + [(f"mid_h{h}", "f8") for h in horizons_s], align=True)
    if candles.size == 0 or trades.size == 0:
        return np.empty(0, dtype=dtype)
    quotes = candles[valid_quote_mask(candles)]
    if quotes.size == 0:
        return np.empty(0, dtype=dtype)
    q_ts = quotes["ts"].astype(np.int64) * 1_000_000_000
    q_bid = quotes["bid_close_dc"] / 1000.0
    q_ask = quotes["ask_close_dc"] / 1000.0
    q_mid = (q_bid + q_ask) / 2.0

    use = trades[(trades["taker_side"] != 0) & (trades["qty"] > 0)]
    idx = np.searchsorted(q_ts, use["ts_ns"], side="right") - 1
    use, idx = use[idx >= 0], idx[idx >= 0]
    if use.size == 0:
        return np.empty(0, dtype=dtype)

    out = np.empty(use.size, dtype=dtype)
    out["ts_ns"] = use["ts_ns"]
    out["price"] = use["price_dc"] / 1000.0
    out["direction"] = use["taker_side"]
    out["qty"] = use["qty"]
    out["mid"] = q_mid[idx]
    out["spread"] = (q_ask - q_bid)[idx]
    out["outcome"] = outcome
    out["secs_to_close"] = (close_ns - use["ts_ns"]) / 1e9
    for h in horizons_s:
        target = use["ts_ns"] + int(h * 1e9)
        j = np.searchsorted(q_ts, target, side="right") - 1
        fut = np.full(use.size, np.nan)

        # past the last quote revision the horizon is unobserved
        inside = (j >= 0) & (target <= q_ts[-1])
        fut[inside] = q_mid[j[inside]]
        out[f"mid_h{h}"] = fut
    return out


def terminal_quote(candles: np.ndarray, close_ns: int) -> tuple[float, int]:
    if candles.size == 0:
        return float("nan"), 0
    ok = valid_quote_mask(candles) & (candles["ts"] <= close_ns // 1_000_000_000)
    if not ok.any():
        return float("nan"), 0
    last = candles[ok][-1]
    return (float(last["bid_close_dc"]) + float(last["ask_close_dc"])) / 2000.0, int(last["ts"])


def _cr1_var(values, weights, clusters) -> float:
    w = weights / weights.sum()
    score = w * (values - np.sum(w * values))
    _, inv = np.unique(clusters, return_inverse=True)
    g = int(inv.max()) + 1
    if g < 2:
        return float("nan")
    s = np.bincount(inv, weights=score, minlength=g)
    return float(g / (g - 1) * np.sum(s**2))


def clustered_var(values, weights, clusters, clusters2=None) -> float:
    w = np.ones_like(values) if weights is None else weights.astype(np.float64)
    v1 = _cr1_var(values, w, clusters)
    if clusters2 is None:
        return v1
    v2 = _cr1_var(values, w, clusters2)
    both = np.char.add(np.char.add(clusters.astype(str), "|"), clusters2.astype(str))
    v = v1 + v2 - _cr1_var(values, w, both)
    return v if v > 0 else max(v1, v2)


@dataclass
class Estimate:
    name: str
    value: float
    se: float
    n_obs: int
    n_clusters: int

    @property
    def t(self) -> float:
        return self.value / self.se if self.se > 0 else float("nan")


def _mean(name, values, weights, clusters, clusters2=None) -> Estimate:
    w = np.ones_like(values) if weights is None else weights
    return Estimate(name, float(np.sum(w * values) / np.sum(w)),
                    float(np.sqrt(clustered_var(values, w, clusters, clusters2))),
                    int(values.size), int(np.unique(clusters).size))


@dataclass
class ThreeTerm:
    horizon_s: int
    A: Estimate
    B: Estimate
    C: Estimate
    total: Estimate
    share: Estimate


def three_term(obs, mid_T, horizon_s, clusters, clusters2=None, weights=None) -> ThreeTerm:
    col = f"mid_h{horizon_s}"
    keep = np.isfinite(obs[col]) & np.isfinite(obs["mid"]) & np.isfinite(mid_T)
    o, mT, cl = obs[keep], mid_T[keep], clusters[keep]
    cl2 = clusters2[keep] if clusters2 is not None else None
    w = weights[keep].astype(np.float64) if weights is not None else np.ones(o.size)
    d = o["direction"].astype(np.float64)
    y = o["outcome"].astype(np.float64)
    a = d * (o[col] - o["mid"]) * CENTS
    b = d * (mT - o[col]) * CENTS
    c = d * (y - mT) * CENTS
    tot = a + b + c
    A, B, C = (_mean(n, v, w, cl, cl2) for n, v in (("A", a), ("B", b), ("C", c)))
    T = _mean("total", tot, w, cl, cl2)
    if T.value:
        share = A.value / T.value
        se = float(np.sqrt(clustered_var((a - share * tot) / T.value, w, cl, cl2)))
    else:
        share, se = float("nan"), float("nan")
    return ThreeTerm(horizon_s, A, B, C, T, Estimate("share", share, se, int(o.size), A.n_clusters))


def precision_weighted_C(obs, mid_T, clusters, floor: float = 0.01) -> Estimate:
    keep = np.isfinite(mid_T) & np.isfinite(obs["mid"])
    o, mT, cl = obs[keep], mid_T[keep], clusters[keep]
    c = o["direction"] * (o["outcome"] - mT) * CENTS
    var = np.maximum(mT * (1 - mT), floor * (1 - floor))
    return _mean("C weighted", c.astype(np.float64), 1.0 / var, cl)


def crossfit_calibration(mid, outcome, groups, folds: int = 5, seed: int = 0, strata=None):
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    fold = rng.integers(0, folds, size=ug.size)[inv]
    w = 1.0 / np.bincount(inv)[inv]
    strata = np.zeros(mid.size, dtype=int) if strata is None else strata
    out = np.full(mid.size, np.nan)
    for s in np.unique(strata):
        in_s = strata == s
        for k in range(folds):
            train, test = in_s & (fold != k), in_s & (fold == k)
            if not test.any() or np.unique(inv[train]).size < 50:
                continue
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(mid[train], outcome[train].astype(float), sample_weight=w[train])
            out[test] = iso.predict(mid[test])
    return out
