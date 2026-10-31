from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _collapse(groups, *arrays):
    if groups is None:
        return arrays
    _, inv = np.unique(groups, return_inverse=True)
    return tuple(np.bincount(inv, weights=a) for a in arrays)


def cr1_from_sums(S, W, groups=None) -> tuple[float, float]:
    S, W = _collapse(groups, S, W)
    keep = W > 0
    S, W = S[keep], W[keep]
    g = S.size
    if g < 2:
        return float("nan"), float("nan")
    mean = S.sum() / W.sum()
    score = (S - mean * W) / W.sum()
    return float(mean), float(np.sqrt(g / (g - 1) * np.sum(score**2)))


def ratio_from_sums(SA, ST, W, groups=None) -> tuple[float, float]:
    SA, ST, W = _collapse(groups, SA, ST, W)
    keep = W > 0
    SA, ST, W = SA[keep], ST[keep], W[keep]
    g = SA.size
    mt = ST.sum() / W.sum()
    if g < 2 or mt == 0:
        return float("nan"), float("nan")
    r = SA.sum() / ST.sum()
    psi = (SA - r * ST) / (W.sum() * mt)
    return float(r), float(np.sqrt(g / (g - 1) * np.sum(psi**2)))


@dataclass
class MarketSums:
    n: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    @classmethod
    def empty(cls, k: int) -> "MarketSums":
        return cls(np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k))

    def three_term(self, horizon_s: int, event_weight: bool = False, groups=None) -> dict:
        n = self.n
        if event_weight:
            with np.errstate(invalid="ignore", divide="ignore"):
                f = np.where(n > 0, 1.0 / n, 0.0)
            W = (n > 0).astype(float)
        else:
            f, W = np.ones_like(n), n
        A, B, C = self.a * f, self.b * f, self.c * f
        T = A + B + C
        out = {k: cr1_from_sums(S, W, groups) for k, S in (("A", A), ("B", B), ("C", C), ("total", T))}
        out["share"] = ratio_from_sums(A, T, W, groups)
        out["n_markets"] = int((n > 0).sum())
        out["n_trades"] = int(n.sum())
        out["horizon_s"] = horizon_s
        return out
