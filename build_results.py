from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


import numpy as np

from decomposition import build_observations, terminal_quote, valid_quote_mask
from kalshi import parse_ts
from sums import cr1_from_sums, ratio_from_sums

MKT = Path("data/hist/mkt")
OUT = Path("out")
H_MIN = (1, 2, 5, 10, 15, 30, 60, 120, 240, 480)
H = tuple(60 * m for m in H_MIN)
J5 = H_MIN.index(5)
TTC_EDGES = (3600, 86400)
CAL_N = 50
FOLDS = 5
WINDOW_S = 7 * 86400
PRICE_UNDERLYING = ("KXINX", "KXNASDAQ", "INX", "NASDAQ", "KXBTC", "KXETH", "BTC", "ETH",
                    "KXSOL", "KXDOGE", "KXXRP", "KXUSDJPY", "KXEURUSD", "KXWTI", "KXGOLD")


SINGLE_NAME_RE = re.compile(
    r"EARNING|MENTION|SAY|SPEECH|PRESS ?CONF|KPI|\bIPO|IPO\b|CEO|SPOTIFY|NETFLIX|APP ?STORE|"
    r"SPACEX|STARSHIP|TESLA|MUSK|OPENAI|TIKTOK|SBGUEST|SUPER BOWL GUEST|ATTEND|"
    r"INAUGURATION|ACQUIRE|ACQ|LAYOFF|DELIVER|SUBSCRIBER|BILLBOARD.*SPOTIFY", re.I)
BROAD_RE = re.compile(
    r"\bINX|INXU|INXD|NASDAQ|S&P|SPX|\bCPI|INFLATION|\bFED\b|FEDDECISION|FOMC|RATE CUT|"
    r"FED FUNDS|GDP|PAYROLL|JOBS|UNEMPLOY|\bPCE\b|RECESSION|GAS PRICE|USDJPY|EURUSD|GBPUSD|"
    r"WTI|\bOIL\b|GOLD|SILVER|TREASUR|10Y|2Y|YIELD|BTC|BITCOIN|ETH|ETHEREUM|SOL|DOGE|XRP|"
    r"CRYPTO", re.I)


SPORTS_MENTION_RE = re.compile(
    r"NBA|MLB|NCAA|NFL|NHL|WNBA|WC|WCMENTION|FIGHT|UFC|BOXING|F1|NASCAR|PGA|GOLF|SOCCER|"
    r"EPL|ANNOUNCER|SUPERBOWLAD|SBMENTION|OLYMPIC|TENNIS|MARCH ?MADNESS|COLLEGE", re.I)


def classify(series: str, meta: dict) -> str:
    cat = (meta.get("category") or "").strip()
    text = f"{series} {meta.get('title') or ''}"
    if cat in ("Sports", "Climate and Weather", "Politics", "Elections"):
        return "unclassified"
    if cat == "Mentions" and SPORTS_MENTION_RE.search(text):
        return "unclassified"
    if cat in ("Mentions", "Companies"):
        return "single-name"
    if cat in ("Crypto", "Commodities"):
        return "broad-based"
    if cat in ("Financials", "Economics", "Science and Technology", "Entertainment",
               "World", "Social", "Transportation", "Health"):
        if SINGLE_NAME_RE.search(text):
            return "single-name"
        if cat in ("Financials", "Economics") and BROAD_RE.search(text):
            return "broad-based"
    return "unclassified"


MIN_KEEP = 0.70
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e1"


def _key(ticker: str) -> int:
    return int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)


_SERIES = None


def _series_meta():
    global _SERIES
    if _SERIES is None:
        _SERIES = json.loads(Path("data/hist/series.json").read_text())
    return _SERIES


def _load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    close_ns = int(z["close_ns"])
    obs = build_observations(z["candles"], z["trades"], int(z["outcome"]), close_ns, horizons_s=H)
    return z, meta, close_ns, obs


def _deepest(obs):
    seen = np.stack([np.isfinite(obs[f"mid_h{h}"]) for h in H], axis=1)
    return np.where(seen.any(1), seen.sum(1) - 1, -1).astype(np.int8)


def pass1(path: str):
    z, meta, close_ns, obs = _load(path)
    if obs.size == 0:
        return None
    mT, mT_ts = terminal_quote(z["candles"], close_ns)
    if not np.isfinite(mT):
        return None
    deep = _deepest(obs)
    keep = deep >= 0
    obs, deep = obs[keep], deep[keep]
    if obs.size == 0:
        return None
    d = obs["direction"].astype(float)
    y = float(obs["outcome"][0])
    mid = obs["mid"]
    nh, a, b, c = (np.zeros(len(H)) for _ in range(4))
    inc_n, inc_s = np.zeros(len(H) - 1), np.zeros(len(H) - 1)
    for j, h in enumerate(H):
        m = deep >= j
        f = obs[f"mid_h{h}"][m]
        nh[j] = m.sum()
        a[j] = np.sum(d[m] * (f - mid[m])) * 100
        b[j] = np.sum(d[m] * (mT - f)) * 100
        c[j] = np.sum(d[m] * (y - mT)) * 100
        if j:
            f0 = obs[f"mid_h{H[j - 1]}"][m]
            inc_n[j - 1], inc_s[j - 1] = m.sum(), np.sum(d[m] * (f - f0)) * 100


    m5 = deep >= J5
    f5 = obs[f"mid_h{H[J5]}"]
    stale = float(np.mean(f5[m5] == mid[m5])) if m5.any() else np.nan
    f1 = obs[f"mid_h{H[0]}"]
    i15 = (float(m5.sum()), float(np.sum(d[m5] * (f5[m5] - f1[m5])) * 100))
    qty = obs["qty"]
    qsum = (float(qty[m5].sum()),
            float(np.sum(qty[m5] * d[m5] * (f5[m5] - mid[m5])) * 100),
            float(np.sum(qty[m5] * d[m5] * (mT - f5[m5])) * 100),
            float(np.sum(qty[m5] * d[m5] * (y - mT)) * 100))
    q = z["candles"][valid_quote_mask(z["candles"])]
    q_ts = q["ts"].astype(np.int64) * 1_000_000_000
    q_mid = (q["bid_close_dc"] + q["ask_close_dc"]) / 2000.0
    jj = np.searchsorted(q_ts, obs["ts_ns"] - H[J5] * 1_000_000_000, side="right") - 1
    past = np.where(jj >= 0, mid - q_mid[np.maximum(jj, 0)], np.nan)
    agree = np.sign(past) * d
    grp = np.where(~np.isfinite(past), -1, np.where(agree > 0, 0, np.where(agree < 0, 1, 2)))
    mom = [(float((m5 & (grp == g)).sum()),
            float(np.sum(d[m5 & (grp == g)] * (f5[m5 & (grp == g)] - mid[m5 & (grp == g)])) * 100),
            float(np.sum(d[m5 & (grp == g)] * (mT - f5[m5 & (grp == g)])) * 100),
            float(np.sum(d[m5 & (grp == g)] * (y - mT)) * 100)) for g in range(3)]

    open_ns = parse_ts(meta.get("open_time")) or (close_ns - WINDOW_S * 10**9)
    hours = max(1 / 60, (close_ns - max(open_ns, close_ns - WINDOW_S * 10**9)) / 3.6e12)
    rng = np.random.default_rng(_key(meta["ticker"]))
    take = rng.choice(obs.size, CAL_N, replace=False) if obs.size > CAL_N else np.arange(obs.size)
    series = meta.get("series") or meta["ticker"].split("-")[0]
    return dict(
        path=path, ticker=meta["ticker"], series=series,
        event=meta.get("event_ticker") or meta["ticker"],
        cat=meta.get("category") or "?", day=meta["close_time"][:10],
        window=meta.get("census_window", "full"),
        cls=classify(series, _series_meta().get(series, {})),
        pu=series.upper().startswith(PRICE_UNDERLYING),
        volume=float(meta.get("volume_fp") or 0), y=y, mT=mT,
        gap_s=float(close_ns // 1_000_000_000 - mT_ts),
        n_trades=int(obs.size), rate=float(obs.size / hours), stale=stale,
        spread=float(np.mean(obs["spread"]) * 100), price=float(np.mean(mid) * 100),
        nh=nh, a=a, b=b, c=c, inc_n=inc_n, inc_s=inc_s, qsum=qsum, mom=mom, i15=i15,
        cal_mid=mid[take].astype(np.float32),
        cal_ttc=np.digitize(obs["secs_to_close"][take], TTC_EDGES).astype(np.int8),
    )


_CURVES = None
_CAT_ID = None


def _init_pass2(curves, cat_id):
    global _CURVES, _CAT_ID
    _CURVES, _CAT_ID = curves, cat_id


def fit_curves(rows):
    from sklearn.isotonic import IsotonicRegression

    cat_id = {c: i for i, c in enumerate(sorted({r["cat"] for r in rows}))}
    x, yv, w, s, f = [], [], [], [], []
    for r in rows:
        k = r["cal_mid"].size
        x.append(r["cal_mid"]); yv.append(np.full(k, r["y"])); w.append(np.full(k, 1.0 / k))
        s.append(cat_id[r["cat"]] * 3 + r["cal_ttc"]); f.append(np.full(k, _key(r["ticker"]) % FOLDS))
    x, yv, w, s, f = map(np.concatenate, (x, yv, w, s, f))
    curves = {}
    for st in np.unique(s):
        for fo in range(FOLDS):
            m = (s == st) & (f != fo)
            if m.sum() >= 500:
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                iso.fit(x[m], yv[m], sample_weight=w[m])
                curves[(int(st), fo)] = (iso.X_thresholds_, iso.y_thresholds_)
    return curves, cat_id


def pass2(args):
    path, cat, ticker = args
    z, meta, close_ns, obs = _load(path)
    deep = _deepest(obs)
    obs, deep = obs[deep >= 0], deep[deep >= 0]
    fold = _key(ticker) % FOLDS
    strata = _CAT_ID[cat] * 3 + np.digitize(obs["secs_to_close"], TTC_EDGES)
    e = np.full(obs.size, np.nan)
    for st in np.unique(strata):
        cv = _CURVES.get((int(st), fold))
        if cv is not None:
            k = strata == st
            e[k] = np.interp(obs["mid"][k], cv[0], cv[1])
    d = obs["direction"].astype(float)
    y = float(obs["outcome"][0])
    ok = np.isfinite(e)
    info, mis, cnt = np.zeros(len(H)), np.zeros(len(H)), np.zeros(len(H))
    for j in range(len(H)):
        m = ok & (deep >= j)
        cnt[j] = m.sum()
        info[j] = np.sum(d[m] * (y - e[m])) * 100
        mis[j] = np.sum(d[m] * (e[m] - obs["mid"][m])) * 100
    return info, mis, cnt


def ev_mean(S, N, groups):
    ok = N > 0
    return cr1_from_sums(S[ok] / N[ok], np.ones(ok.sum()), groups[ok])


def ev_ratio(SA, NA, SB, NB, groups):
    ok = (NA > 0) & (NB > 0)
    return ratio_from_sums(SA[ok] / NA[ok], SB[ok] / NB[ok], np.ones(ok.sum()), groups[ok])


def three(R, idx, j, event=True):
    nh, a, b, c = R["nh"][idx, j], R["a"][idx, j], R["b"][idx, j], R["c"][idx, j]
    g = R["dc"][idx]
    ok = nh > 0
    nh, a, b, c, g = nh[ok], a[ok], b[ok], c[ok], g[ok]
    if event:
        A, B, C, W = a / nh, b / nh, c / nh, np.ones(nh.size)
    else:
        A, B, C, W = a, b, c, nh
    T = A + B + C
    return dict(n=int(ok.sum()), A=cr1_from_sums(A, W, g), B=cr1_from_sums(B, W, g),
                C=cr1_from_sums(C, W, g), T=cr1_from_sums(T, W, g), share=ratio_from_sums(A, T, W, g))


def quintiles(x, mask):
    return np.quantile(x[mask], [0, .2, .4, .6, .8, 1])


def in_q(x, q, i):
    hi = x <= q[i + 1] if i == 4 else x < q[i + 1]
    return (x >= q[i]) & hi


class Macros:
    def __init__(self):
        self.lines = []

    def num(self, name, v, fmt="{:.2f}"):
        s = fmt.format(v)
        if s.startswith("-"):
            s = r"\ensuremath{-}" + s[1:]
        self.lines.append(f"\\newcommand{{\\{name}}}{{{s}}}")

    def int(self, name, v):
        self.lines.append(f"\\newcommand{{\\{name}}}{{{v:,}}}".replace(",", "{,}"))

    def raw(self, name, s):
        self.lines.append(f"\\newcommand{{\\{name}}}{{{s}}}")


def fmt_range(lo, hi) -> str:
    f = lambda x: f"{x:,.1f}" if x < 10 else f"{x:,.0f}"
    return f"{f(lo)}--{f(hi)}"


def fmt_ratio(share, info) -> str:

    if not np.isfinite(info[1]) or info[1] == 0 or abs(info[0] / info[1]) < 2:
        return "n.d."
    return fmt_se(*share)


def fmt_se(m, se, f="{:.2f}"):
    s = f.format(m)
    s = s.replace("-", r"\ensuremath{-}") if s.startswith("-") else s
    return f"{s} ({f.format(se)})"


def _num(x: float) -> str:
    return f"{x:,.1f}" if x < 10 else f"{x:,.0f}"


def _range(q, qi: int) -> str:
    if qi == 0:
        return f"< {_num(q[1])}"
    if qi == 4:
        return f"> {_num(q[4])}"
    return f"{_num(q[qi])}–{_num(q[qi + 1])}"


def plot(table, q):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(1.0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(1.02, 1.0, "settlement\n(truth)", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8.5, color=INK2)
    ax.axvline(5, color=GRID, lw=6, zorder=0)
    ax.text(5, 0.04, "5 min\n(convention;\nRule 605 max)", ha="center", va="bottom",
            fontsize=7.5, color=INK2, transform=ax.get_xaxis_transform())
    labels = ["Q1 least active", "Q2", "Q3", "Q4", "Q5 most active"]
    for qi in range(5):
        pts = [r for r in table if r[0] == qi + 1]

        n0 = pts[0][4]
        pts = [r for r in pts if r[4] >= MIN_KEEP * n0]
        x = np.array([r[3] for r in pts]); y = np.array([r[5] for r in pts])
        se = np.array([r[6] for r in pts])
        lab = f"{labels[qi]} ({_range(q, qi)} trades/h)"
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=RAMP[qi], alpha=0.13, lw=0, zorder=2)
        ax.plot(x, y, color=RAMP[qi], lw=2, marker=MARKERS[qi], ms=4.5,
                markeredgecolor="white", markeredgewidth=0.8, label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(H_MIN)

    ax.set_xticklabels(["1m", "2m", "5m", "10m", "", "30m", "1h", "2h", "4h", "8h"])
    ax.set_xlim(0.85, 560)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("Horizon after the trade")
    ax.set_ylabel("Share of the trade's information\nreflected in the midquote")
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right", ncol=1,
              title="Non-sports markets, by trades per hour", title_fontsize=7.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_impact_curve.pdf")
    fig.savefig(OUT / "fig_impact_curve.png", dpi=200)


BLUE, ORANGE = "#2a78d6", "#eb6834"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    return plt


def _finish(ax, fig, name):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")


def fig_calibration(rows, sports):
    plt = _style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    edges = np.linspace(0, 1, 21)
    out = {}
    for lab, want, col, mk in (("Non-sports", False, BLUE, "o"), ("Sports", True, ORANGE, "s")):
        idx = [i for i, r in enumerate(rows) if bool(sports[i]) == want]
        x = np.concatenate([rows[i]["cal_mid"] for i in idx]).astype(float)
        y = np.concatenate([np.full(rows[i]["cal_mid"].size, rows[i]["y"]) for i in idx])
        w = np.concatenate([np.full(rows[i]["cal_mid"].size, 1.0 / rows[i]["cal_mid"].size) for i in idx])
        g = np.concatenate([np.full(rows[i]["cal_mid"].size, i) for i in idx])
        b = np.clip(np.digitize(x, edges) - 1, 0, 19)
        px, py, pe = [], [], []
        for k in range(20):
            m = b == k
            if w[m].sum() < 50:
                continue
            mean, se = cr1_from_sums(w[m] * y[m], w[m], g[m])
            px.append(np.sum(w[m] * x[m]) / w[m].sum()); py.append(mean); pe.append(se)
        px, py, pe = map(np.array, (px, py, pe))
        ax.errorbar(px, py, yerr=1.96 * pe, fmt=mk, ms=4.5, color=col, mec="white", mew=0.7,
                    elinewidth=1, capsize=0, label=lab, zorder=3)
        out[lab] = (px, py)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Midquote at the trade")
    ax.set_ylabel("Share of markets settling YES")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    _finish(ax, fig, "fig_calibration")
    return out


def fig_model_fit(R, ns, n_bins=20):
    plt = _style()
    q = np.quantile(R["rate"][ns], np.linspace(0, 1, n_bins + 1))
    pts = []
    for i in range(n_bins):
        hi = R["rate"] <= q[i + 1] if i == n_bins - 1 else R["rate"] < q[i + 1]
        idx = ns & (R["rate"] >= q[i]) & hi
        sh = ev_ratio(R["a"][idx, J5], R["nh"][idx, J5], R["info"][idx, J5], R["cnt"][idx, J5], R["dc"][idx])
        pts.append((float(np.nanmean(R["stale"][idx])), sh[0], sh[1], i))
    pts = np.array(pts)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    xmax = float(np.ceil((np.nanmax(pts[:, 0]) + 0.03) * 20) / 20)
    sgrid = np.linspace(0.001, xmax, 200)
    ax.fill_between(sgrid, 1 - np.sqrt(sgrid), 1 - sgrid, color="#86b6ef", alpha=0.25, lw=0, zorder=1,
                    label=r"Model bounds, $\theta\in[\frac{1}{2},1]$")
    ax.plot(sgrid, 1 - sgrid, color="#2a78d6", lw=1.2, zorder=2, label=r"$\theta=1$: $1-s$")
    ax.plot(sgrid, 1 - np.sqrt(sgrid), color="#2a78d6", lw=1.2, ls=(0, (4, 2)), zorder=2,
            label=r"$\theta=\frac{1}{2}$: $1-\sqrt{s}$")
    ax.axhline(1.0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(pts[:, 0], pts[:, 1], yerr=1.96 * pts[:, 2], fmt="o", ms=4.5, color="#104281",
                mec="white", mew=0.7, elinewidth=1, capsize=0, zorder=3,
                label="Non-sports markets, 20 activity bins")
    thin = pts[np.nanargmax(pts[:, 0])]
    ax.text(thin[0], thin[1] + 0.07, "least active", fontsize=8, color=INK2, ha="center", zorder=6)
    ax.text(0.012, 1.5, "most active", fontsize=8, color=INK2, ha="left", va="center", zorder=6)
    ax.set_xlim(0, xmax); ax.set_ylim(0.2, 1.6)
    ax.set_xlabel("Share of trades with the midquote unchanged five minutes later, $s$")
    ax.set_ylabel("Share of information captured at five minutes")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    _finish(ax, fig, "fig_model_fit")
    return pts


def fig_reversal(R, cls):
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.axvline(5, color=GRID, lw=6, zorder=0)
    out = {}
    for lab, key, col, mk in (("Single-name", "single-name", ORANGE, "s"),
                              ("Broad-based", "broad-based", BLUE, "o")):
        base = cls == key
        n0 = None
        xs, A_, I_ = [], [], []
        for j, hm in enumerate(H_MIN):
            idx = base & (R["nh"][:, j] > 0) & (R["cnt"][:, j] > 0)
            if n0 is None:
                n0 = idx.sum()
            if idx.sum() < MIN_KEEP * n0:
                break
            xs.append(hm)
            A_.append(ev_mean(R["a"][idx, j], R["nh"][idx, j], R["dc"][idx]))
            I_.append(ev_mean(R["info"][idx, j], R["cnt"][idx, j], R["dc"][idx]))
        xs = np.array(xs)
        a_m, a_s = np.array([v[0] for v in A_]), np.array([v[1] for v in A_])
        i_m, i_s = np.array([v[0] for v in I_]), np.array([v[1] for v in I_])
        ax.fill_between(xs, a_m - 1.96 * a_s, a_m + 1.96 * a_s, color=col, alpha=0.15, lw=0)
        ax.plot(xs, a_m, color=col, lw=2, marker=mk, ms=4.5, mec="white", mew=0.8,
                label=f"{lab}: midquote move")
        ax.fill_between(xs, i_m - 1.96 * i_s, i_m + 1.96 * i_s, color=col, alpha=0.08, lw=0)
        ax.plot(xs, i_m, color=col, lw=1.4, ls=(0, (4, 2)),
                label=f"{lab}: information at settlement")
        out[lab] = (xs, a_m, i_m)
    ax.set_xscale("log")
    ax.set_xticks(H_MIN)
    ax.set_xticklabels(["1m", "2m", "5m", "10m", "", "30m", "1h", "2h", "4h", "8h"])
    ax.set_xlim(0.85, 560)
    ax.set_xlabel("Horizon after the trade")
    ax.set_ylabel("Cents per contract")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7.8)
    _finish(ax, fig, "fig_reversal")
    return out


def fig_categories(R, cat, min_markets=1000):
    plt = _style()
    rows_ = []
    for c in sorted(set(cat)):
        idx = cat == c
        if idx.sum() < min_markets or c == "Sports":
            continue
        sh = ev_ratio(R["a"][idx, J5], R["nh"][idx, J5], R["info"][idx, J5], R["cnt"][idx, J5], R["dc"][idx])
        rows_.append((c, float(np.median(R["rate"][idx])), sh[0], sh[1], int(idx.sum())))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.axhline(1.0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for c, x, y, e, nm in rows_:
        col = ORANGE if c == "Mentions" else "#104281"
        ax.errorbar([x], [y], yerr=[1.96 * e], fmt="o", ms=5, color=col, mec="white", mew=0.7,
                    elinewidth=1, capsize=0, zorder=3)
        off = {"Elections": (-4, 9), "Politics": (6, 6), "Economics": (-6, -13),
               "Entertainment": (4, -13), "Financials": (6, 3)}.get(c, (6, 3))
        ha = "right" if off[0] < 0 else "left"
        ax.annotate(c.replace("Climate and Weather", "Weather"), (x, y), xytext=off,
                    textcoords="offset points", fontsize=8, color=INK, ha=ha)
    ax.set_xscale("log")
    ax.set_xlabel("Median trades per hour (log scale)")
    ax.set_ylabel("Share of information captured at five minutes")
    ax.grid(axis="y", color=GRID, lw=0.6)
    _finish(ax, fig, "fig_categories")
    return rows_


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-markets", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--from-cache", action="store_true")
    args = ap.parse_args(argv)
    cache = Path("data/hist/results_cache.pkl")

    files = sorted(str(f) for f in MKT.glob("*.npz") if not f.name.endswith(".tmp.npz"))
    if args.stride > 1:
        files = files[:: args.stride]
    if args.max_markets:
        files = files[: args.max_markets]
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    if args.from_cache:
        with open(cache, "rb") as fh:
            rows, p2 = pickle.load(fh)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [r for r in pool.map(pass1, files, chunksize=64) if r is not None]
        curves, cat_id = fit_curves(rows)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_pass2,
                                 initargs=(curves, cat_id)) as pool:
            p2 = list(pool.map(pass2, [(r["path"], r["cat"], r["ticker"]) for r in rows], chunksize=64))
        with open(cache, "wb") as fh:
            pickle.dump((rows, p2), fh, protocol=pickle.HIGHEST_PROTOCOL)

    n = len(rows)
    R = {k: np.array([r[k] for r in rows]) for k in
         ("nh", "a", "b", "c", "inc_n", "inc_s", "rate", "stale", "gap_s", "n_trades",
          "spread", "price", "y", "volume", "pu")}
    R["info"] = np.array([p[0] for p in p2])
    R["mis"] = np.array([p[1] for p in p2])
    R["cnt"] = np.array([p[2] for p in p2])
    R["qsum"] = np.array([r["qsum"] for r in rows])
    R["mom"] = np.array([r["mom"] for r in rows])
    cat = np.array([r["cat"] for r in rows])
    cls = np.array([r["cls"] for r in rows])
    window = np.array([r["window"] for r in rows])
    R["dc"] = np.array([f"{r['day']}|{r['cat']}" for r in rows])
    sports = cat == "Sports"
    allm = np.ones(n, bool)
    ns = ~sports
    res = {"n_markets": n, "n_files": len(files)}
    M = Macros()
    log = []

    def P(s):
        log.append(s)


    sample = json.loads(Path("data/hist/sample.json").read_text())
    M.int("NArchived", sample["markets_in_census"])
    M.int("NEligible", sample["events_eligible"])
    M.int("NSampled", len(sample["markets"]))
    M.int("NSeriesCap", 2000)
    M.int("NMarkets", n)
    M.raw("NTrades", f"{R['n_trades'].sum() / 1e6:.1f} million")
    M.int("NClusters", int(np.unique(R["dc"]).size))
    M.int("NPriceU", int(R["pu"].sum()))


    # sports kept separate: most traded market is picked on in-game volume
    rows_h = []
    for grp, mask in (("ns", ns), ("sp", sports)):
        for j, hm in enumerate(H_MIN):
            rows_h.append((grp, hm, three(R, np.where(mask)[0], j, event=True)))
    ev = {hm: r for (g_, hm, r) in rows_h if g_ == "ns"}
    tw = {hm: three(R, np.where(ns)[0], j, event=False) for j, hm in enumerate(H_MIN)}
    pooled = three(R, np.where(allm)[0], J5)
    sp5 = three(R, np.where(sports)[0], J5)
    M.num("ShareFiveAll", 100 * pooled["share"][0], "{:.1f}")
    M.num("ShareFiveSports", 100 * sp5["share"][0], "{:.1f}")
    M.num("ShareOne", 100 * ev[1]["share"][0], "{:.1f}"); M.num("ShareOneSE", 100 * ev[1]["share"][1], "{:.1f}")
    M.num("ShareFive", 100 * ev[5]["share"][0], "{:.1f}"); M.num("ShareFiveSE", 100 * ev[5]["share"][1], "{:.1f}")
    M.num("ShareHour", 100 * ev[60]["share"][0], "{:.1f}"); M.num("ShareHourSE", 100 * ev[60]["share"][1], "{:.1f}")
    M.num("AFive", ev[5]["A"][0]); M.num("BFive", ev[5]["B"][0]); M.num("CFive", ev[5]["C"][0])
    M.num("TotFive", ev[5]["T"][0])
    M.num("ShareFiveTrade", 100 * tw[5]["share"][0], "{:.0f}")
    M.num("ShareFiveTradeSE", 100 * tw[5]["share"][1], "{:.0f}")
    with open(OUT / "tab_horizons.tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{rrcccccc}\n\\toprule\n"
                 "Horizon & Markets & $A_h$ & $B_h$ & $C$ & $A_h+B_h+C$ & Share captured \\\\\n\\midrule\n")
        for grp, title in (("ns", "Panel A: non-sports"), ("sp", "Panel B: sports")):
            fh.write(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\\n")
            for (g_, hm, r) in rows_h:
                if g_ != grp:
                    continue
                lab = f"{hm} min" if hm < 60 else f"{hm // 60} h"
                fh.write(f"{lab} & {r['n']:,} & {fmt_se(*r['A'])} & {fmt_se(*r['B'])} & "
                         f"{fmt_se(*r['C'])} & {fmt_se(*r['T'])} & {fmt_se(*r['share'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


    incs = []
    for j in range(len(H) - 1):
        m = ns & (R["inc_n"][:, j] > 0)
        incs.append(cr1_from_sums(R["inc_s"][m, j] / R["inc_n"][m, j], np.ones(m.sum()), R["dc"][m]))
        P(f"increment {H_MIN[j]}->{H_MIN[j + 1]}m: {incs[-1][0]:+.3f} ({incs[-1][1]:.3f})")

    i15 = np.array([r["i15"] for r in rows])
    k = ns & (i15[:, 0] > 0)
    inc15 = cr1_from_sums(i15[k, 1] / i15[k, 0], np.ones(k.sum()), R["dc"][k])
    M.num("IncOneFive", inc15[0]); M.num("IncOneFiveT", inc15[0] / inc15[1], "{:.1f}")
    last = incs[H_MIN.index(240)]
    M.num("IncFourHour", last[0]); M.num("IncFourHourT", last[0] / last[1], "{:.1f}")
    incs_t = [x[0] / x[1] for x in incs]
    M.num("IncTMin", min(incs_t + [inc15[0] / inc15[1]]), "{:.1f}")
    M.num("IncTMax", max(incs_t + [inc15[0] / inc15[1]]), "{:.1f}")

    n5 = R["nh"][:, J5]
    ok5 = (n5 > 0) & np.isfinite(R["stale"])
    M.num("StaleTW", 100 * np.sum(R["stale"][ok5] * n5[ok5]) / n5[ok5].sum(), "{:.1f}")
    ok5ns = ok5 & ns
    M.num("StaleTWNS", 100 * np.sum(R["stale"][ok5ns] * n5[ok5ns]) / n5[ok5ns].sum(), "{:.1f}")
    res["increments"] = incs


    pu = three(R, np.where(R["pu"])[0], J5)
    M.num("SharePU", 100 * pu["share"][0], "{:.1f}"); M.num("SharePUSE", 100 * pu["share"][1], "{:.1f}")


    def gradient(mask):
        q = quintiles(R["rate"], mask)
        out = []
        for i in range(5):
            idx = mask & in_q(R["rate"], q, i)
            sh = ev_ratio(R["a"][idx, J5], R["nh"][idx, J5], R["info"][idx, J5], R["cnt"][idx, J5], R["dc"][idx])
            out.append(dict(lo=q[i], hi=q[i + 1], n=int(idx.sum()), share=sh,
                            stale=float(np.nanmean(R["stale"][idx])),
                            mis=ev_mean(R["mis"][idx, J5], R["cnt"][idx, J5], R["dc"][idx]),
                            info=ev_mean(R["info"][idx, J5], R["cnt"][idx, J5], R["dc"][idx])))
        return out
    g_ns, g_sp, g_pu = gradient(ns), gradient(sports), gradient(R["pu"].astype(bool))
    names = ["One", "Two", "Three", "Four", "Five"]
    for i, gq in enumerate(g_ns):
        M.num(f"G{names[i]}", gq["share"][0]); M.num(f"G{names[i]}SE", gq["share"][1])
        M.num(f"Stale{names[i]}", 100 * gq["stale"], "{:.1f}")
        P(f"NS Q{i + 1} {gq['lo']:.2f}-{gq['hi']:.2f}/h n={gq['n']:,} stale={gq['stale']:.3f} "
          f"share={gq['share'][0]:.3f}({gq['share'][1]:.3f}) mis={gq['mis'][0]:+.3f}")
    M.num("MisMaxNS", max(abs(gq["mis"][0]) for gq in g_ns))
    M.num("InfoMinNS", min(gq["info"][0] for gq in g_ns))


    thetas = []
    for i, gq in enumerate(g_ns):
        sh, st = gq["share"][0], gq["stale"]
        th = np.log(1 - sh) / np.log(st) if (0 < sh < 1 and 0 < st < 1) else np.nan
        thetas.append(th)
        P(f"theta Q{i + 1}: {th:.2f}")
        if np.isfinite(th):
            M.num(f"Theta{names[i]}", th)
        else:
            M.raw(f"Theta{names[i]}", "---")
    res["theta"] = thetas


    inside = 0
    for i, gq in enumerate(g_ns):
        lo, hi = 1 - np.sqrt(gq["stale"]), 1 - gq["stale"]
        ok_ = lo <= gq["share"][0] <= hi
        inside += ok_
        P(f"bounds Q{i + 1}: [{lo:.2f}, {hi:.2f}] share={gq['share'][0]:.2f} {'inside' if ok_ else 'OUTSIDE'}")
        M.num(f"BoundLo{names[i]}", lo); M.num(f"BoundHi{names[i]}", hi)
    M.raw("BoundsInside", ["none", "one", "two", "three", "four", "all five"][inside])
    for i in range(4):
        M.num(f"S{names[i]}", g_sp[i]["share"][0])
    mis_sp = [gq["mis"][0] for gq in g_sp]
    M.num("MisSportsLo", min(mis_sp), "{:.1f}"); M.num("MisSportsHi", max(mis_sp), "{:.1f}")
    with open(OUT / "tab_gradient.tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{lrccc}\n\\toprule\n"
                 "Trades per hour & Markets & Midquote unchanged at 5 min & Information captured & Mispricing (cents) \\\\\n\\midrule\n")
        for title, gg in (("Panel A: non-sports", g_ns), ("Panel B: sports", g_sp),
                          ("Panel C: price-underlying (indices, crypto)", g_pu)):
            fh.write(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\\n")
            for i, gq in enumerate(gg):
                fh.write(f"Q{i + 1}: {fmt_range(gq['lo'], gq['hi'])} & {gq['n']:,} & {100 * gq['stale']:.1f}\\% & "
                         f"{fmt_ratio(gq['share'], gq['info'])} & {fmt_se(*gq['mis'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


    info_ns = ev_mean(R["info"][ns, J5], R["cnt"][ns, J5], R["dc"][ns])
    mis_ns = ev_mean(R["mis"][ns, J5], R["cnt"][ns, J5], R["dc"][ns])
    M.num("InfoNS", info_ns[0]); M.num("InfoNSSE", info_ns[1])
    M.num("MisNS", mis_ns[0]); M.num("MisNSSE", mis_ns[1])


    for g, nm in ((1, "BContra"), (2, "BFlat")):
        mm = R["mom"][:, g, :]
        k = ns & (mm[:, 0] > 0)
        est = cr1_from_sums(mm[k, 2] / mm[k, 0], np.ones(k.sum()), R["dc"][k])
        M.num(nm, est[0]); M.num(nm + "SE", est[1])


    gaps = [(0, 300), (300, 3600), (3600, 86400), (86400, 1e12)]
    gshare = []
    for lo, hi in gaps:
        idx = np.where(ns & (R["gap_s"] >= lo) & (R["gap_s"] < hi))[0]
        gshare.append((idx.size, three(R, idx, J5)["share"]) if idx.size >= 100 else (idx.size, (np.nan, np.nan)))
    within_day = [s[0] for nn, s in gshare[:3] if np.isfinite(s[0])]
    M.num("GapLo", min(within_day)); M.num("GapHi", max(within_day))
    M.num("GapLong", gshare[3][1][0]); M.int("GapLongN", gshare[3][0])
    M.num("GapMedianNS", float(np.median(R["gap_s"][ns]) / 60), "{:.0f}")


    base = three(R, np.where(ns)[0], J5)
    excl = three(R, np.where(ns & (window == "full"))[0], J5)
    M.num("WinBase", 100 * base["share"][0], "{:.1f}"); M.num("WinExcl", 100 * excl["share"][0], "{:.1f}")
    rob = [("Non-sports (baseline)", base),
           ("Excluding window-censused series", excl),
           ("Volume $\\geq$ 1,000 contracts", three(R, np.where(ns & (R["volume"] >= 1000))[0], J5))]


    person = (cat == "Mentions") | (cls == "single-name")
    settle = three(R, np.where(ns & ~person)[0], J5)
    rob.append(("Excluding person-settled markets", settle))
    g_set = gradient(ns & ~person)
    M.int("SetN", int((ns & person).sum()))
    M.num("SetShare", 100 * settle["share"][0], "{:.1f}")
    M.num("SetShareSE", 100 * settle["share"][1], "{:.1f}")
    M.num("SetGOne", g_set[0]["share"][0]); M.num("SetGFive", g_set[4]["share"][0])
    M.num("SetGFiveSE", g_set[4]["share"][1])
    P(f"settlement robustness: dropped {int(person.sum()):,}; share={settle['share'][0]:.3f}; "
      f"NS gradient " + " ".join(f"{gq['share'][0]:.2f}" for gq in g_set))
    qs = R["qsum"]
    k = ns & (qs[:, 0] > 0)
    A, B, C = qs[k, 1], qs[k, 2], qs[k, 3]
    cw = ratio_from_sums(A, A + B + C, qs[k, 0], R["dc"][k])
    tot = cr1_from_sums(A + B + C, qs[k, 0], R["dc"][k])
    rob.append(("Contract-weighted", dict(n=int(k.sum()), share=cw, info=tot)))
    rob.append(("All markets, including sports", pooled))
    rob.append(("Sports only", sp5))
    with open(OUT / "tab_robustness.tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{lrc}\n\\toprule\nSpecification & Markets & Share captured at 5 min \\\\\n\\midrule\n")
        for lab, r in rob:
            cell = fmt_ratio(r["share"], r["info"]) if "info" in r else fmt_se(*r["share"])
            fh.write(f"{lab} & {r['n']:,} & {cell} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


    bo = {}
    for c_ in ("single-name", "broad-based"):
        idx = cls == c_
        bo[c_] = dict(
            n=int(idx.sum()), rate=float(np.median(R["rate"][idx])), stale=float(np.nanmean(R["stale"][idx])),
            A=ev_mean(R["a"][idx, J5], R["nh"][idx, J5], R["dc"][idx]),
            info=ev_mean(R["info"][idx, J5], R["cnt"][idx, J5], R["dc"][idx]),
            mis=ev_mean(R["mis"][idx, J5], R["cnt"][idx, J5], R["dc"][idx]),
            tot=ev_mean(R["a"][idx, J5] + R["b"][idx, J5] + R["c"][idx, J5], R["nh"][idx, J5], R["dc"][idx]),
            share=ev_ratio(R["a"][idx, J5], R["nh"][idx, J5], R["info"][idx, J5], R["cnt"][idx, J5], R["dc"][idx]))
    for c_, s_ in (("single-name", "sn"), ("broad-based", "bb")):
        b_ = bo[c_]
        M.int(f"BON{s_}", b_["n"])
        M.num(f"BORate{s_}", b_["rate"], "{:.1f}" if b_["rate"] < 10 else "{:.0f}")
        M.num(f"BOStale{s_}", 100 * b_["stale"], "{:.1f}")
        for key, nm in (("A", "A"), ("info", "I"), ("mis", "M"), ("share", "Share")):
            M.num(f"BO{nm}{s_}", b_[key][0]); M.num(f"BO{nm}{s_}SE", b_[key][1])

    def gap(key):
        (x1, s1), (x2, s2) = bo["single-name"][key], bo["broad-based"][key]
        return x1 - x2, (x1 - x2) / float(np.hypot(s1, s2))
    for key, nm in (("A", "A"), ("info", "I"), ("tot", "Tot")):
        gv, gt = gap(key)
        M.num(f"BOGap{nm}", gv); M.num(f"BOGap{nm}T", gt, "{:.1f}")
        P(f"BO gap {key}: {gv:+.3f} t={gt:.2f}")
    both = cls != "unclassified"
    qb = quintiles(R["rate"], both)
    with open(OUT / "tab_bo_activity.tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{lcc}\n\\toprule\nTrades per hour & Single-name & Broad-based \\\\\n\\midrule\n")
        for i in range(5):
            cells = []
            for c_ in ("single-name", "broad-based"):
                idx = both & (cls == c_) & in_q(R["rate"], qb, i)
                if idx.sum() < 30:
                    cells.append(f"--- ({int(idx.sum())} markets)")
                else:
                    sh = ev_ratio(R["a"][idx, J5], R["nh"][idx, J5], R["info"][idx, J5], R["cnt"][idx, J5], R["dc"][idx])
                    inf = ev_mean(R["info"][idx, J5], R["cnt"][idx, J5], R["dc"][idx])
                    cells.append(f"{fmt_ratio(sh, inf)} [{int(idx.sum()):,}]")
            fh.write(f"Q{i + 1}: {fmt_range(qb[i], qb[i + 1])} & {cells[0]} & {cells[1]} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


    with open(OUT / "tab_desc.tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{lrrrrr}\n\\toprule\nCategory & Events & Trades (000s) & Median trades/h & "
                 "Mean quoted spread (\\textcent) & Settled YES \\\\\n\\midrule\n")
        cats = sorted(np.unique(cat), key=lambda c_: -(cat == c_).sum())
        for c_ in cats:
            idx = cat == c_
            if idx.sum() < 100:
                continue
            fh.write(f"{c_} & {int(idx.sum()):,} & {R['n_trades'][idx].sum() / 1e3:,.0f} & "
                     f"{np.median(R['rate'][idx]):,.1f} & {np.mean(R['spread'][idx]):.1f} & "
                     f"{100 * np.mean(R['y'][idx]):.0f}\\% \\\\\n")
        fh.write(f"\\midrule\nAll & {n:,} & {R['n_trades'].sum() / 1e3:,.0f} & {np.median(R['rate']):,.1f} & "
                 f"{np.mean(R['spread']):.1f} & {100 * np.mean(R['y']):.0f}\\% \\\\\n\\bottomrule\n\\end{{tabular}}\n")


    q = quintiles(R["rate"], ns)
    table = []
    for qi in range(5):
        inq = ns & in_q(R["rate"], q, qi)
        for j, hm in enumerate(H_MIN):
            idx = inq & (R["nh"][:, j] > 0) & (R["cnt"][:, j] > 0)
            if idx.sum() < 100:
                continue
            sh = ev_ratio(R["a"][idx, j], R["nh"][idx, j], R["info"][idx, j], R["cnt"][idx, j], R["dc"][idx])
            table.append((qi + 1, q[qi], q[qi + 1], hm, int(idx.sum()), sh[0], sh[1]))
    with open(OUT / "fig_impact_curve.csv", "w") as fh:
        fh.write("quintile,rate_lo,rate_hi,horizon_min,markets,share,se\n")
        for row in table:
            fh.write(",".join(f"{v:.6g}" for v in row) + "\n")
    plot(table, q)
    pick = {(r[0], r[3]): r[5] for r in table}
    M.num("FigNQOneFive", pick[(1, 5)]); M.num("FigNQOneFour", pick.get((1, 240), np.nan))
    M.num("FigNQTwoFive", pick[(2, 5)]); M.num("FigNQTwoFour", pick.get((2, 240), np.nan))


    cal = fig_calibration(rows, sports)
    px, py = cal["Sports"]
    k = np.argmin(np.abs(px - 0.10))
    M.num("CalSportsTen", 100 * py[k], "{:.0f}")
    px, py = cal["Non-sports"]
    k = np.argmin(np.abs(px - 0.10))
    M.num("CalNSTen", 100 * py[k], "{:.0f}")
    mf = fig_model_fit(R, ns)
    inside_bins = int(np.sum((mf[:, 1] >= 1 - np.sqrt(mf[:, 0])) & (mf[:, 1] <= 1 - mf[:, 0])))
    M.int("FitBinsInside", inside_bins)
    rv = fig_reversal(R, cls)
    xs, a_m, i_m = rv["Single-name"]
    M.raw("RevSNAbove", "every" if np.all(a_m > i_m) else "most")
    M.num("RevSNLast", xs[-1] / 60 if xs[-1] >= 60 else xs[-1], "{:.0f}")
    M.raw("RevSNLastUnit", "hours" if xs[-1] >= 120 else ("hour" if xs[-1] >= 60 else "minutes"))
    cats = fig_categories(R, cat)
    P("categories: " + "; ".join(f"{c} {x:.1f}/h {y:.2f}" for c, x, y, _, _ in cats))
    if len(cats) > 2:
        lx = np.log([r[1] for r in cats if r[0] != "Sports"])
        ly = np.array([r[2] for r in cats if r[0] != "Sports"])
        M.num("CatCorr", float(np.corrcoef(lx, ly)[0, 1]))
        M.int("CatN", len(cats))


    header = ("% generated by build_results.py\n"
              f"% {n:,} markets; built {time.strftime('%Y-%m-%d %H:%M')}.\n")
    (OUT / "numbers.tex").write_text(header + "\n".join(M.lines) + "\n", encoding="utf-8")
    res["log"] = log
    res["increments_t"] = incs_t
    Path("data/hist/results.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote numbers.tex ({len(M.lines)} macros), tables and figure in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
