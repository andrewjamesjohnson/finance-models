#!/usr/bin/env python3
"""
mv_engine.py - Markowitz mean-variance engine.

Every mean variance computation in closed form, and
adds what Excel's Solver does by hand: constrained optimisation (no
shorting, bounds, excluded names), a market model (betas / alphas / CAPM),
risk contributions, annualisation, and a Yahoo price fetcher.

Conventions (the usual textbook ones unless you override them):
  * returns are SIMPLE per-period returns          r_t = P_t / P_{t-1} - 1
  * moments are POPULATION moments (divide by N)   Cov = (R-mu)'(R-mu) / N
  * rf is a PER-PERIOD rate                         0.001 = 0.10% per month
  * Sharpe = (mu_p - rf) / sd_p
  * shorting is allowed (Solver's "Make Unconstrained Variables Non-Negative" is
    UNCHECKED in the plain model); use no_short=True / bounds to forbid it.

A Universe may carry mu=None (covariance-only inputs): then everything that does
not need means (st devs, correlations, MinVar, risk contributions) still works and
full_report says what is missing.  Every silent normalisation lands in U.notes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

try:  # pandas only for file IO
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
                7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven",
                12: "Twelve"}


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class Portfolio:
    name: str
    weights: np.ndarray
    mean: float
    var: float
    sd: float
    sharpe: float
    note: str = ""

    def to_dict(self, names: Sequence[str]) -> dict:
        return {
            "name": self.name,
            "weights": {n: float(w) for n, w in zip(names, self.weights)},
            "weights_list": [float(w) for w in self.weights],
            "sum_weights": float(self.weights.sum()),
            "mean": None if _isnan(self.mean) else float(self.mean),
            "sd": float(self.sd),
            "var": float(self.var),
            "sharpe": None if _isnan(self.sharpe) else float(self.sharpe),
            "note": self.note,
        }


def _isnan(x) -> bool:
    try:
        return x is None or math.isnan(x)
    except TypeError:
        return False


@dataclass
class Universe:
    names: list[str]
    mu: Optional[np.ndarray]            # per-period arithmetic means (N,) or None
    cov: np.ndarray                     # covariance matrix (N, N)
    rf: float = 0.0                     # per-period risk-free rate
    ddof: int = 0                       # 0 = population, 1 = sample
    returns: Optional[np.ndarray] = None        # (T, N) per-period returns
    prices: Optional[np.ndarray] = None         # (T+1, N) prices
    dates: Optional[list[str]] = None           # labels for price rows
    market_name: Optional[str] = None
    market_returns: Optional[np.ndarray] = None  # (T,)
    market_prices: Optional[np.ndarray] = None   # (T+1,)
    periods_per_year: Optional[int] = None
    return_type: str = "simple"
    source: str = ""
    notes: list[str] = field(default_factory=list)

    # derived -------------------------------------------------------------- #
    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def has_means(self) -> bool:
        return self.mu is not None

    @property
    def sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def var(self) -> np.ndarray:
        return np.diag(self.cov).copy()

    @property
    def corr(self) -> np.ndarray:
        s = self.sd
        return self.cov / np.outer(s, s)

    @property
    def T(self) -> Optional[int]:
        return None if self.returns is None else int(self.returns.shape[0])

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"'{name}' is not in the universe {self.names}")

    def require_means(self, what: str = "this") -> None:
        if self.mu is None:
            raise ValueError(f"{what} needs expected returns (means); only a covariance/correlation matrix was given. "
                             "Add a means table, a --means file, or price/return data.")

    def subset(self, keep: Sequence[str]) -> "Universe":
        idx = [self.index(k) for k in keep]
        return Universe(
            names=[self.names[i] for i in idx],
            mu=None if self.mu is None else self.mu[idx].copy(),
            cov=self.cov[np.ix_(idx, idx)].copy(),
            rf=self.rf, ddof=self.ddof,
            returns=None if self.returns is None else self.returns[:, idx].copy(),
            prices=None if self.prices is None else self.prices[:, idx].copy(),
            dates=self.dates, market_name=self.market_name,
            market_returns=self.market_returns, market_prices=self.market_prices,
            periods_per_year=self.periods_per_year, return_type=self.return_type,
            source=self.source + f" (subset {list(keep)})", notes=list(self.notes),
        )


# --------------------------------------------------------------------------- #
# Returns and moments  (Week-2 material: prices -> returns -> mean / sd / cov)
# --------------------------------------------------------------------------- #
def simple_returns(prices: np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    return p[1:] / p[:-1] - 1.0


def log_returns(prices: np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    return np.log(p[1:] / p[:-1])


def moments(returns: np.ndarray, ddof: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector and covariance matrix the textbook way:
    {=MMULT(TRANSPOSE(R - mu), R - mu) / COUNT(R)}   (ddof=0, population).
    ddof=1 divides by N-1 instead (Excel COVARIANCE.S / VAR.S)."""
    R = np.asarray(returns, dtype=float)
    T = R.shape[0]
    mu = R.mean(axis=0)
    D = R - mu
    cov = D.T @ D / (T - ddof)
    return mu, cov


def cov_from_corr(sd: Sequence[float], corr: np.ndarray) -> np.ndarray:
    s = np.asarray(sd, dtype=float)
    return np.asarray(corr, dtype=float) * np.outer(s, s)


def constant_corr(n: int, rho: float) -> np.ndarray:
    return np.full((n, n), float(rho)) + (1.0 - float(rho)) * np.eye(n)


def annualize(mean: Optional[float], sd: float, periods_per_year: int) -> dict:
    """Arithmetic annualisation (mean x p, sd x sqrt p) plus the geometric mean."""
    return {
        "mean_arith": None if _isnan(mean) else mean * periods_per_year,
        "mean_geom": None if _isnan(mean) else (1.0 + mean) ** periods_per_year - 1.0,
        "sd": sd * math.sqrt(periods_per_year),
    }


def percent_scale(kind: str, values: np.ndarray) -> tuple[float, str]:
    """Decide whether numbers arrived in percent units (1.23 meaning 1.23%).
    Returns (divisor, note).  kind: 'returns' | 'mean' | 'sd' | 'var' | 'cov'."""
    v = np.asarray(values, dtype=float)
    if kind == "cov":
        v = np.diag(v) if v.ndim == 2 else v
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0, ""
    med = float(np.median(np.abs(v)))
    if kind == "sd" and med > 1.0:
        return 100.0, f"st devs look like percents (median {med:.2f}); divided by 100"
    if kind == "returns" and med > 0.5:
        return 100.0, f"returns look like percents (median |r| {med:.2f}); divided by 100"
    if kind == "mean" and med > 0.5:
        return 100.0, f"means look like percents (median |mean| {med:.2f}); divided by 100"
    if kind == "var" and med > 1.0:
        return 1e4, "variances look like percent-squared; divided by 10,000"
    if kind == "cov" and med > 1.0:
        return 1e4, "covariance diagonal looks like percent-squared; divided by 10,000"
    return 1.0, ""


MARKET_ALIASES = {
    "SPY": ["SPX", "^GSPC", "GSPC", "SP500", "S&P500", "S&P 500", "SPTR", "^SPX", "SP"],
    "DIA": ["DJI", "^DJI", "DOW", "DJIA", "INDU"],
    "QQQ": ["NDX", "^NDX", "NASDAQ", "^IXIC", "IXIC", "COMP"],
    "IWM": ["RUT", "^RUT", "RTY"],
}


def resolve_market(market: Optional[str], cols: Sequence[str], notes: list[str]) -> Optional[str]:
    """Return the column that plays the market.  Accepts the exact header, a case-insensitive match,
    or a known alias (SPY for an SPX column, DIA for DJI, ...).  Raises with the candidates otherwise."""
    if not market:
        return None
    cols = list(cols)
    if market in cols:
        return market
    low = {c.lower(): c for c in cols}
    if market.lower() in low:
        notes.append(f"market column '{market}' matched '{low[market.lower()]}' (case-insensitive)")
        return low[market.lower()]
    fam = None
    for k, al in MARKET_ALIASES.items():
        if market.upper() == k or market.upper() in (a.upper() for a in al):
            fam = [k] + al
            break
    if fam:
        for c in cols:
            if c.upper() in (f.upper() for f in fam):
                notes.append(f"market '{market}' is the '{c}' column in this file (same index, different symbol)")
                return c
    import difflib
    hint = difflib.get_close_matches(market, cols, n=3, cutoff=0.5)
    raise KeyError(f"market column '{market}' is not among {cols}"
                   + (f"; did you mean {', '.join(hint)}?" if hint else "") + ". Use the header exactly as it appears in the file.")


def _end_of(end: str):
    """'2025' -> Dec 31 2025, '2025-12' -> Dec 31 2025, '2025-12-01' -> that day.  All inclusive."""
    return pd.Period(str(end).strip()).end_time


def normalise_series(dates, M: np.ndarray, names: Sequence[str], kind: str, notes: list[str],
                     resample: Optional[str] = None, percent: Optional[bool] = None,
                     start: Optional[str] = None, end: Optional[str] = None, last: Optional[int] = None):
    """Sort ascending by date, drop rows with holes, optionally resample, apply a date window, fix percent units.
    kind: 'prices' | 'returns'.  Returns (dates as ISO strings or None, matrix).
    start / end: inclusive bounds ('2019-12', '2025-12-31', '2025'); a month or year means the whole month or year.
    last: keep only the most recent N RETURNS (N+1 prices when kind == 'prices')."""
    if pd is None:
        raise RuntimeError("pandas required")
    M = np.asarray(M, dtype=float)
    idx = None
    ds = list(dates) if dates is not None else None
    if ds and len(ds) == M.shape[0] and all(d is not None for d in ds):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(pd.Series(ds), errors="coerce")
        if parsed.notna().all():
            idx = parsed
            order = np.argsort(idx.values, kind="stable")
            if not np.all(order == np.arange(len(order))):
                M = M[order]
                idx = idx.iloc[order].reset_index(drop=True)
                notes.append("dates were not in ascending order; sorted oldest -> newest before taking returns")
    good = np.isfinite(M).all(axis=1)
    if not good.all():
        dropped = int((~good).sum())
        M = M[good]
        if idx is not None:
            idx = idx[good].reset_index(drop=True)
        notes.append(f"dropped {dropped} row(s) with missing values (returns bridge the gap)")
    if resample and idx is not None:
        rule = {"M": "ME", "ME": "ME", "W": "W-FRI", "Q": "QE", "A": "YE", "Y": "YE"}.get(resample.upper(), resample)
        df = pd.DataFrame(M, index=pd.DatetimeIndex(idx), columns=list(names))
        if kind == "prices":
            df = df.resample(rule).last().dropna()
        else:
            df = (1.0 + df).resample(rule).prod().dropna() - 1.0
        M = df.to_numpy()
        idx = pd.Series(df.index)
        notes.append(f"resampled to {resample} ({len(M)} rows)")
    elif resample and idx is None:
        notes.append("WARNING --resample ignored: no parseable dates")
    if (start or end) and idx is not None:
        keep = np.ones(len(M), dtype=bool)
        if start:
            keep &= np.asarray(pd.DatetimeIndex(idx) >= pd.Timestamp(str(start).strip()))
        if end:
            keep &= np.asarray(pd.DatetimeIndex(idx) <= _end_of(end))
        dropped = int((~keep).sum())
        M = M[keep]
        idx = idx[keep].reset_index(drop=True)
        if len(M) == 0:
            raise ValueError(f"the window {start or '...'} .. {end or '...'} leaves no rows")
        notes.append(f"date window {start or 'first'} .. {end or 'last'}: kept {len(M)} rows "
                     f"({str(idx.iloc[0].date())} .. {str(idx.iloc[-1].date())}), dropped {dropped}"
                     + (f" -> {len(M) - 1} returns" if kind == "prices" else ""))
    elif (start or end) and idx is None:
        notes.append("WARNING --start/--end ignored: no parseable dates")
    if last is not None and int(last) > 0:
        n_keep = int(last) + 1 if kind == "prices" else int(last)
        if len(M) > n_keep:
            M = M[-n_keep:]
            if idx is not None:
                idx = idx[-n_keep:].reset_index(drop=True)
            notes.append(f"kept the last {int(last)} returns" + (f" ({n_keep} prices)" if kind == "prices" else ""))
    if kind == "returns":
        div, note = percent_scale("returns", M)
        if percent is True:
            div, note = 100.0, "returns divided by 100 (--percent-units)"
        if percent is False:
            div, note = 1.0, ""
        if div != 1.0:
            M = M / div
            notes.append(note)
    dates_out = [str(d.date()) for d in idx] if idx is not None else (list(map(str, ds)) if ds else None)
    return dates_out, M


def histogram(x: Sequence[float], step: float = 0.01, lo: Optional[float] = None, hi: Optional[float] = None) -> dict:
    """Excel FREQUENCY() semantics.  bins are UPPER bounds: bin k counts values in (bin[k-1], bin[k]],
    the first bin counts values <= bin[0], and 'overflow' counts values above the last bin (the extra cell
    FREQUENCY spills).  Default bins: floor(min/step)*step .. ceil(max/step)*step, so the first bin is empty
    unless a value sits exactly on it and the overflow cell is always 0."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("histogram: no data")
    step = float(step)
    lo_b = math.floor(v.min() / step + 1e-9) * step if lo is None else float(lo)
    hi_b = math.ceil(v.max() / step - 1e-9) * step if hi is None else float(hi)
    nb = int(round((hi_b - lo_b) / step)) + 1
    bins = [round(lo_b + k * step, 10) for k in range(nb)]
    counts = []
    prev = -np.inf
    for b in bins:
        counts.append(int(np.sum((v > prev + 1e-12) & (v <= b + 1e-12))))
        prev = b
    overflow = int(np.sum(v > bins[-1] + 1e-12))
    dec = 0 if step >= 0.01 - 1e-12 else 1
    labels = [f"{(b - step) * 100:.{dec}f}% to {b * 100:.{dec}f}%" for b in bins]
    n = int(v.size)
    return {"step": step, "bins": bins, "labels": labels, "counts": counts, "overflow": overflow,
            "rel_freq": [c / n for c in counts], "n": n, "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "sd": float(v.std()), "modal_bin": labels[int(np.argmax(counts))],
            "note": "bins are upper bounds; bucket 'a to b' holds returns in (a, b]; built like =FREQUENCY(returns, bins)"}


# --------------------------------------------------------------------------- #
# Portfolio arithmetic  ("PF Stats using Matrix")
# --------------------------------------------------------------------------- #
def portfolio_stats(U: Universe, w: Sequence[float], name: str = "PF", note: str = "") -> Portfolio:
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.shape[0] != U.n:
        raise ValueError(f"{name}: {w.shape[0]} weights for {U.n} assets")
    mean = float(w @ U.mu) if U.mu is not None else float("nan")   # {=MMULT(TRANSPOSE(w), mu)}
    var = float(w @ U.cov @ w)                                       # {=MMULT(MMULT(TRANSPOSE(w), Cov), w)}
    sd = math.sqrt(max(var, 0.0))                                    # =SQRT(var)
    sharpe = (mean - U.rf) / sd if (sd > 0 and not math.isnan(mean)) else float("nan")   # =(mean - rF)/sd
    return Portfolio(name, w, mean, var, sd, sharpe, note)


def equal_weight(U: Universe) -> Portfolio:
    return portfolio_stats(U, np.full(U.n, 1.0 / U.n), "EW PF", "equal weights (the Week-2 'PF' row)")


def cov_between(U: Universe, wA: Sequence[float], wB: Sequence[float]) -> float:
    """Covariance of two portfolios: {=MMULT(MMULT(TRANSPOSE(wA), Cov), wB)}."""
    return float(np.asarray(wA) @ U.cov @ np.asarray(wB))


def risk_contributions(U: Universe, w: Sequence[float]) -> dict:
    """Marginal contribution to risk (Cov w)_i / sd_p, share of variance w_i (Cov w)_i / var_p
    (sums to 1), and contribution to sd = share x sd_p.  Excel: {=w * MMULT(Cov, w) / var}."""
    w = np.asarray(w, dtype=float)
    cw = U.cov @ w
    var = float(w @ cw)
    sd = math.sqrt(max(var, 0.0))
    if var <= 0:
        z = [0.0] * U.n
        return {"marginal": z, "share_of_var": z, "contrib_sd": z}
    share = w * cw / var
    return {"marginal": (cw / sd).tolist(), "share_of_var": share.tolist(), "contrib_sd": (share * sd).tolist()}


def portfolio_dict(U: Universe, p: Portfolio) -> dict:
    d = p.to_dict(U.names)
    d["risk"] = risk_contributions(U, p.weights)
    return d


# --------------------------------------------------------------------------- #
# Closed-form frontier  (what Solver finds numerically)
# --------------------------------------------------------------------------- #
@dataclass
class FrontierCoefficients:
    A: float
    B: float
    C: float
    D: float
    g: np.ndarray
    h: np.ndarray
    cov_inv: np.ndarray

    def weights_for_mean(self, m: float) -> np.ndarray:
        return self.g + self.h * m

    def var_for_mean(self, m: float) -> float:
        return (self.C * m * m - 2.0 * self.A * m + self.B) / self.D

    def mean_for_sd(self, sd: float, upper: bool = True) -> float:
        disc = self.A * self.A - self.C * (self.B - self.D * sd * sd)
        if disc < 0:
            raise ValueError(
                f"target sd {sd:.6f} is below the minimum-variance sd {math.sqrt(1.0 / self.C):.6f}; "
                "no frontier portfolio has that little risk")
        root = math.sqrt(disc)
        return (self.A + root) / self.C if upper else (self.A - root) / self.C


def _inverse(U: Universe) -> np.ndarray:
    try:
        return np.linalg.inv(U.cov)
    except np.linalg.LinAlgError as e:
        raise ValueError("covariance matrix is singular (duplicate assets, a constant series, or more assets than "
                         "observations); MINVERSE would fail in Excel too") from e


def frontier_coefficients(U: Universe) -> FrontierCoefficients:
    U.require_means("the frontier")
    Si = _inverse(U)
    one = np.ones(U.n)
    A = float(one @ Si @ U.mu)
    B = float(U.mu @ Si @ U.mu)
    C = float(one @ Si @ one)
    D = B * C - A * A
    if D <= 0:
        raise ValueError("degenerate frontier (all means equal?)")
    g = (B * (Si @ one) - A * (Si @ U.mu)) / D
    h = (C * (Si @ U.mu) - A * (Si @ one)) / D
    return FrontierCoefficients(A, B, C, D, g, h, Si)


def gmv(U: Universe) -> Portfolio:
    """Global minimum-variance portfolio: w = Cov^-1 1 / (1' Cov^-1 1).  Needs no means.
    Class route: Solver, Min Var (F7) subject to sum of weights = 1."""
    Si = _inverse(U)
    one = np.ones(U.n)
    w = Si @ one / float(one @ Si @ one)
    return portfolio_stats(U, w, "MinVar", "global minimum variance (Solver: min Var s.t. sum w = 1)")


def tangency(U: Universe) -> Portfolio:
    """Tangency / max-Sharpe portfolio: w = Cov^-1 (mu - rf) / (1' Cov^-1 (mu - rf)).
    Class route: Solver, Max Sharpe (F9) subject to sum of weights = 1."""
    U.require_means("the tangency portfolio")
    Si = _inverse(U)
    ex = U.mu - U.rf
    denom = float(np.ones(U.n) @ Si @ ex)
    if abs(denom) < 1e-15:
        raise ValueError("tangency undefined: 1' Cov^-1 (mu - rf) = 0")
    w = Si @ ex / denom
    p = portfolio_stats(U, w, "TG", "tangency / max Sharpe (Solver: max Sharpe s.t. sum w = 1)")
    if denom < 0:
        p.note += "  WARNING: rf is above the min-variance mean, so the 'tangency' sits on the LOWER branch (negative Sharpe)"
    return p


def pct_label(x: float) -> str:
    """0.065 -> '6.5%', 0.08 -> '8%', 0.015 -> '1.5%', 0.06125 -> '6.125%'."""
    t = f"{x * 100:.4f}".rstrip("0").rstrip(".")
    return t + "%"


def frontier_at_sd(U: Universe, target_sd: float, upper: bool = True) -> Portfolio:
    """Efficient portfolio with a given sd.
    Class route: Solver, Max Sharpe s.t. sum w = 1 AND StD (F6) = target
    (with sd pinned, max Sharpe == max mean == the upper branch of the frontier)."""
    fc = frontier_coefficients(U)
    m = fc.mean_for_sd(target_sd, upper)
    w = fc.weights_for_mean(m)
    label = f"sd={pct_label(target_sd)}"
    return portfolio_stats(U, w, label, f"frontier at sd = {target_sd:.4%} (Solver: max Sharpe s.t. sum w = 1, StD = {target_sd:.4%})")


def frontier_at_mean(U: Universe, target_mean: float) -> Portfolio:
    """Minimum-variance portfolio with a given mean.
    Class route: Solver, Min Var s.t. sum w = 1 AND Mean (F5) = target."""
    fc = frontier_coefficients(U)
    w = fc.weights_for_mean(target_mean)
    label = f"mean={pct_label(target_mean)}"
    return portfolio_stats(U, w, label, f"frontier at mean = {target_mean:.4%} (Solver: min Var s.t. sum w = 1, Mean = {target_mean:.4%})")


def frontier_share_of_tangency(U: Universe, tg: Portfolio, g: Portfolio, target_sd: float) -> float:
    """The live-Excel route to a target-sd frontier portfolio: w = a*TG + (1-a)*GMV with
    a = sqrt((sd_t^2 - sd_g^2) / (sd_tg^2 - sd_g^2)).  Uses Cov(GMV, any frontier pf) = var_GMV."""
    num = target_sd ** 2 - g.var
    den = tg.var - g.var
    if num < 0:
        raise ValueError(f"target sd {target_sd:.6f} below GMV sd {g.sd:.6f}")
    return math.sqrt(num / den)


def frontier_curve(U: Universe, n: int = 121, mean_lo: Optional[float] = None,
                   mean_hi: Optional[float] = None) -> list[dict]:
    """Analytic frontier (both branches) as (sd, mean) points, for plotting."""
    fc = frontier_coefficients(U)
    g = gmv(U)
    span = max(U.mu.max() - U.mu.min(), 1e-6)
    lo = g.mean - 1.5 * span if mean_lo is None else mean_lo
    hi = U.mu.max() + 1.5 * span if mean_hi is None else mean_hi
    out = []
    for m in np.linspace(lo, hi, n):
        out.append({"mean": float(m), "sd": math.sqrt(max(fc.var_for_mean(float(m)), 0.0))})
    return out


# --------------------------------------------------------------------------- #
# Two risky funds  ("Risky-Risky" block: the data table that traces the frontier)
# --------------------------------------------------------------------------- #
def two_fund_mix(U: Universe, pA: Portfolio, pB: Portfolio, a: float) -> dict:
    """Mix a of fund A with (1-a) of fund B.  The usual table computes
    mean = a*meanA + (1-a)*meanB and sd = sqrt(a^2 sdA^2 + (1-a)^2 sdB^2 + 2 a (1-a) Cov(A,B));
    the implied asset weights are a*wA + (1-a)*wB."""
    b = 1.0 - a
    c = cov_between(U, pA.weights, pB.weights)
    mean = a * pA.mean + b * pB.mean
    var = a * a * pA.var + b * b * pB.var + 2.0 * a * b * c
    w = a * pA.weights + b * pB.weights
    check = portfolio_stats(U, w)
    return {"a": a, "wA": a, "wB": b, "mean": mean, "sd": math.sqrt(max(var, 0.0)),
            "weights": [float(x) for x in w], "sd_from_weights": check.sd, "mean_from_weights": check.mean}


def two_fund_table(U: Universe, pA: Portfolio, pB: Portfolio,
                   grid: Iterable[float] = np.arange(-8.0, 8.01, 0.5)) -> dict:
    rows = [two_fund_mix(U, pA, pB, float(a)) for a in grid]
    return {"fund_a": pA.name, "fund_b": pB.name, "cov_ab": cov_between(U, pA.weights, pB.weights),
            "rows": rows}


# --------------------------------------------------------------------------- #
# Risky + riskless  ("Risky-Riskless Superportfolio" block: the CAL)
# --------------------------------------------------------------------------- #
def cal_point(U: Universe, tg: Portfolio, w_tg: float) -> dict:
    """wTG in the tangency portfolio, 1-wTG in the riskless asset."""
    w_rf = 1.0 - w_tg
    return {"wTG": w_tg, "wRF": w_rf,
            "mean": w_tg * tg.mean + w_rf * U.rf,
            "sd": w_tg * tg.sd,
            "asset_weights": [float(w_tg * x) for x in tg.weights]}


def cal_table(U: Universe, tg: Portfolio, grid: Iterable[float] = np.arange(0.0, 2.001, 0.1)) -> dict:
    rows = [cal_point(U, tg, float(k)) for k in grid]
    return {"slope_sharpe": tg.sharpe, "intercept_rf": U.rf, "rows": rows}


def cal_for_target(U: Universe, tg: Portfolio, target_sd: Optional[float] = None,
                   target_mean: Optional[float] = None) -> dict:
    if target_sd is not None:
        k = target_sd / tg.sd
    elif target_mean is not None:
        k = (target_mean - U.rf) / (tg.mean - U.rf)
    else:
        raise ValueError("give target_sd or target_mean")
    p = cal_point(U, tg, k)
    p["target_sd"] = target_sd
    p["target_mean"] = target_mean
    p["asset_weights_named"] = {n: w for n, w in zip(U.names, p["asset_weights"])}
    return p


# --------------------------------------------------------------------------- #
# Numeric "Solver" (SLSQP) - for constraints the closed forms cannot take
# --------------------------------------------------------------------------- #
def solve_numeric(U: Universe, objective: str = "max_sharpe", target_sd: Optional[float] = None,
                  target_mean: Optional[float] = None, no_short: bool = False,
                  bounds: Optional[Sequence[tuple]] = None, fixed: Optional[dict] = None,
                  start: Optional[Sequence[float]] = None, name: Optional[str] = None) -> tuple[Portfolio, object]:
    """Same problem Excel Solver (GRG Nonlinear) solves, run with scipy SLSQP.

    objective: 'max_sharpe' | 'min_var' | 'max_return'
    target_sd / target_mean: optional equality constraints (a second constraint)
    no_short: bounds every weight at >= 0 (Solver's "Make Unconstrained Variables Non-Negative")
    bounds:  per-asset (lo, hi) tuples; None for unbounded
    fixed:   {name_or_index: value} pins a weight (0 to take an asset out)
    start:   starting weights (default: equal weights)
    """
    from scipy.optimize import minimize

    if objective != "min_var" or target_mean is not None:
        U.require_means(f"objective {objective}")
    n, mu, S, rf = U.n, U.mu, U.cov, U.rf

    def mean(w):
        return float(w @ mu)

    def var(w):
        return float(w @ S @ w)

    if objective == "max_sharpe":
        def f(w):
            return -(mean(w) - rf) / math.sqrt(max(var(w), 1e-18))
    elif objective == "min_var":
        f = var
    elif objective == "max_return":
        def f(w):
            return -mean(w)
    else:
        raise ValueError(objective)

    cons = [{"type": "eq", "fun": lambda w: float(w.sum()) - 1.0}]
    if target_sd is not None:
        cons.append({"type": "eq", "fun": lambda w: math.sqrt(max(var(w), 0.0)) - target_sd})
    if target_mean is not None:
        cons.append({"type": "eq", "fun": lambda w: mean(w) - target_mean})

    b = [(0.0, None) if no_short else (None, None) for _ in range(n)]
    if bounds is not None:
        b = [tuple(x) for x in bounds]
    if fixed:
        for k, v in fixed.items():
            i = U.index(k) if isinstance(k, str) else int(k)
            b[i] = (float(v), float(v))

    x0 = np.full(n, 1.0 / n) if start is None else np.asarray(start, dtype=float)
    res = minimize(f, x0, method="SLSQP", bounds=b, constraints=cons,
                   options={"ftol": 1e-14, "maxiter": 5000})
    label = name or f"solver:{objective}"
    p = portfolio_stats(U, res.x, label, f"numeric SLSQP ({objective}; success={res.success}; {res.message})")
    return p, res


# --------------------------------------------------------------------------- #
# Market model  (betas, alphas, CAPM) - needs a benchmark return series
# --------------------------------------------------------------------------- #
def market_model(U: Universe) -> Optional[dict]:
    if U.returns is None or U.market_returns is None or U.mu is None:
        return None
    R = U.returns
    rm = np.asarray(U.market_returns, dtype=float)
    T = R.shape[0]
    d = T - U.ddof
    mm = rm.mean()
    vm = float(((rm - mm) ** 2).sum() / d)
    sm = math.sqrt(vm)
    cov_im = ((R - R.mean(axis=0)) * (rm - mm)[:, None]).sum(axis=0) / d
    beta = cov_im / vm
    var_i = np.diag(U.cov)
    alpha_j = U.mu - U.rf - beta * (mm - U.rf)          # Jensen's alpha
    intercept = U.mu - beta * mm                         # Excel INTERCEPT()
    r2 = beta ** 2 * vm / var_i
    resid_var = var_i - beta ** 2 * vm
    with np.errstate(divide="ignore", invalid="ignore"):
        treynor = np.where(np.abs(beta) > 1e-12, (U.mu - U.rf) / beta, np.nan)
    capm = U.rf + beta * (mm - U.rf)
    assets = []
    for i, nme in enumerate(U.names):
        assets.append({"name": nme, "beta": float(beta[i]), "alpha_jensen": float(alpha_j[i]),
                       "intercept": float(intercept[i]), "r2": float(r2[i]),
                       "resid_sd": float(math.sqrt(max(resid_var[i], 0.0))),
                       "systematic_sd": float(abs(beta[i]) * sm),
                       "treynor": float(treynor[i]), "capm_expected": float(capm[i]),
                       "cov_with_market": float(cov_im[i])})
    return {"market": U.market_name, "market_mean": float(mm), "market_sd": sm, "market_var": vm,
            "market_sharpe": float((mm - U.rf) / sm) if sm > 0 else None, "assets": assets}


def portfolio_market_stats(U: Universe, mm: dict, w: Sequence[float]) -> dict:
    """Portfolio beta/alpha and the single-index variance decomposition."""
    w = np.asarray(w, dtype=float)
    beta = np.array([a["beta"] for a in mm["assets"]])
    alpha = np.array([a["alpha_jensen"] for a in mm["assets"]])
    resid = np.array([a["resid_sd"] for a in mm["assets"]]) ** 2
    bp = float(w @ beta)
    ap = float(w @ alpha)
    sys_var = bp * bp * mm["market_var"]
    resid_var = float((w * w) @ resid)           # uncorrelated residuals (single-index model)
    total = float(w @ U.cov @ w)
    return {"beta": bp, "alpha_jensen": ap, "systematic_var": sys_var, "systematic_sd": math.sqrt(sys_var),
            "resid_var_sim": resid_var, "resid_sd_sim": math.sqrt(max(resid_var, 0.0)),
            "total_var_actual": total, "total_var_sim": sys_var + resid_var,
            "capm_expected": U.rf + bp * (mm["market_mean"] - U.rf)}


def growth_of_one(U: Universe, weights_by_name: dict) -> Optional[dict]:
    """In-sample growth of $1 per asset and per constant-weight (rebalanced) portfolio."""
    if U.returns is None:
        return None
    R = U.returns
    out = {"dates": U.dates[1:] if U.dates and len(U.dates) == R.shape[0] + 1 else U.dates,
           "assets": {n: np.cumprod(1.0 + R[:, i]).tolist() for i, n in enumerate(U.names)},
           "portfolios": {}}
    for k, w in weights_by_name.items():
        rp = R @ np.asarray(w, dtype=float)
        out["portfolios"][k] = np.cumprod(1.0 + rp).tolist()
    if U.market_returns is not None:
        out["market"] = {U.market_name: np.cumprod(1.0 + np.asarray(U.market_returns)).tolist()}
    return out


# --------------------------------------------------------------------------- #
# Loading a universe from plain frames / files
# --------------------------------------------------------------------------- #
def _read_table(path: str | Path, sheet: Optional[str] = None):
    if pd is None:
        raise RuntimeError("pandas is required to read files")
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(p, sheet_name=sheet or 0)
    return pd.read_csv(p, sep=None, engine="python")


def _looks_like_dates(values, frac: float = 0.6) -> bool:
    vals = [v for v in values if v is not None and str(v).strip() != ""]
    if len(vals) < 3:
        return False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(pd.Series([str(v) for v in vals]), errors="coerce")
    return float(parsed.notna().mean()) >= frac


def _split_date_column(df, notes: list[str]):
    """Return (dates or None, numeric frame).  Handles a dates-across (transposed) layout."""
    if pd is None:
        raise RuntimeError("pandas is required")
    d = df.copy()
    # transposed: headers are dates, first column holds the names
    if len(d.columns) > 3 and _looks_like_dates(list(d.columns)[1:]) and d.iloc[:, 0].dtype == object:
        first = d.columns[0]
        d = d.set_index(first).T.reset_index().rename(columns={"index": "Date"})
        d.columns = [str(c) for c in d.columns]
        notes.append("input was transposed (dates across the top); flipped to dates down the rows")
    dates = None
    if isinstance(d.index, pd.DatetimeIndex):
        dates = [str(x.date()) for x in d.index]
    else:
        first = d.columns[0]
        col = d[first]
        if str(first).strip().lower() in ("date", "dates", "month", "period", "time", "day", "week") or col.dtype == object:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(col, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    dates = [str(x.date()) if pd.notna(x) else None for x in parsed]
                else:
                    dates = [str(x) for x in col]
            except Exception:
                dates = [str(x) for x in col]
            d = d.drop(columns=[first])
    d = d.apply(pd.to_numeric, errors="coerce")
    d = d.dropna(axis=1, how="all")
    return dates, d


def universe_from_frame(df, kind: str = "prices", rf: float = 0.0, ddof: int = 0,
                        market: Optional[str] = None, exclude: Sequence[str] = (),
                        include: Optional[Sequence[str]] = None, return_type: str = "simple",
                        periods_per_year: Optional[int] = None, source: str = "",
                        resample: Optional[str] = None, percent: Optional[bool] = None,
                        start: Optional[str] = None, end: Optional[str] = None, last: Optional[int] = None) -> Universe:
    notes: list[str] = []
    dates, d = _split_date_column(df, notes)
    cols = [str(c).strip() for c in d.columns]
    d.columns = cols
    M = d.to_numpy(dtype=float)
    dates, M = normalise_series(dates, M, cols, "prices" if kind == "prices" else "returns", notes, resample, percent,
                                start=start, end=end, last=last)
    mkt = None
    keep = list(range(len(cols)))
    market = resolve_market(market, cols, notes)
    if market:
        mi = cols.index(market)
        mkt = M[:, mi]
        keep = [i for i in keep if i != mi]
    if include:
        missing = [c for c in include if c not in cols]
        if missing:
            raise KeyError(f"include names not found: {missing}")
        keep = [cols.index(c) for c in include]
    keep = [i for i in keep if cols[i] not in set(exclude)]
    X = M[:, keep]
    names = [cols[i] for i in keep]
    if kind == "prices":
        R = log_returns(X) if return_type == "log" else simple_returns(X)
        Rm = None if mkt is None else (log_returns(mkt) if return_type == "log" else simple_returns(mkt))
        prices, mprices = X, mkt
    elif kind == "returns":
        R, Rm, prices, mprices = X, mkt, None, None
    else:
        raise ValueError(kind)
    if R.shape[0] < 2:
        raise ValueError("fewer than 2 return observations after cleaning")
    mu, cov = moments(R, ddof)
    return Universe(names=names, mu=mu, cov=cov, rf=rf, ddof=ddof, returns=R, prices=prices,
                    dates=dates, market_name=market, market_returns=Rm, market_prices=mprices,
                    periods_per_year=periods_per_year, return_type=return_type, source=source or kind, notes=notes)


def universe_from_moments(names: Sequence[str], mu: Optional[Sequence[float]] = None, cov: Optional[np.ndarray] = None,
                          sd: Optional[Sequence[float]] = None, corr: Optional[np.ndarray] = None,
                          assume_corr: Optional[float] = None, rf: float = 0.0, ddof: int = 0,
                          periods_per_year: Optional[int] = None, source: str = "moments") -> Universe:
    names = [str(x).strip() for x in names]
    n = len(names)
    notes: list[str] = []
    mu_arr = None if mu is None else np.asarray(mu, dtype=float).reshape(-1)
    if cov is None:
        if sd is None:
            raise ValueError("give cov, or sd + corr (or sd + assume_corr)")
        if corr is None and assume_corr is None:
            raise ValueError("st devs without correlations: pass corr, or assume_corr=rho for a constant-correlation matrix")
        if corr is None:
            corr = constant_corr(n, assume_corr)
            notes.append(f"ASSUMPTION: no correlations given; every pair set to {assume_corr:g}")
        cov = cov_from_corr(sd, np.asarray(corr, dtype=float))
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (n, n) or (mu_arr is not None and mu_arr.shape[0] != n):
        raise ValueError(f"shape mismatch: {n} names, mu {None if mu_arr is None else mu_arr.shape}, cov {cov.shape}")
    return Universe(names=names, mu=mu_arr, cov=cov, rf=rf, ddof=ddof, periods_per_year=periods_per_year, source=source, notes=notes)


def read_matrix_file(path: str | Path) -> tuple[list[str], np.ndarray]:
    """CSV/XLSX with a header row of names and (optionally) a first column of names."""
    df = _read_table(path)
    first = df.columns[0]
    if df[first].dtype == object:
        df = df.set_index(first)
    names = [str(c).strip() for c in df.columns]
    return names, df.to_numpy(dtype=float)


def read_vector_file(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Two-column (name, value) or one-row-with-headers file."""
    df = _read_table(path)
    if df.shape[1] == 2 and df.iloc[:, 0].dtype == object:
        return [str(x).strip() for x in df.iloc[:, 0]], df.iloc[:, 1].to_numpy(dtype=float)
    return [str(c).strip() for c in df.columns], df.iloc[0].to_numpy(dtype=float)


def universe_from_spec(spec: dict) -> Universe:
    """One JSON document: {"names":[..], "mu":[..], "cov":[[..]] | "sd"+"corr" | "sd"+"assume_corr",
    "rf": .., "ddof": 0, "periods_per_year": 12}  or  {"prices": {"dates":[..], "AAPL":[..], ..},
    "market": "SPY", ...}  or {"returns": {...}}."""
    rf = float(spec.get("rf", 0.0))
    ddof = int(spec.get("ddof", 0))
    ppy = spec.get("periods_per_year")
    if "prices" in spec or "returns" in spec:
        if pd is None:
            raise RuntimeError("pandas required")
        kind = "prices" if "prices" in spec else "returns"
        block = spec[kind]
        df = pd.DataFrame(block)
        return universe_from_frame(df, kind=kind, rf=rf, ddof=ddof, market=spec.get("market"),
                                   exclude=spec.get("exclude", ()), include=spec.get("include"),
                                   return_type=spec.get("return_type", "simple"), periods_per_year=ppy,
                                   source=spec.get("source", "spec"), resample=spec.get("resample"))
    return universe_from_moments(spec["names"], spec.get("mu"), cov=spec.get("cov"), sd=spec.get("sd"),
                                 corr=spec.get("corr"), assume_corr=spec.get("assume_corr"), rf=rf, ddof=ddof,
                                 periods_per_year=ppy, source=spec.get("source", "spec"))


# --------------------------------------------------------------------------- #
# Yahoo Finance fetch (no key; unofficial endpoint, best effort)
# --------------------------------------------------------------------------- #
def fetch_yahoo(tickers: Sequence[str], years: int = 5, interval: str = "1mo",
                drop_partial: bool = True, timeout: int = 30):
    """Adjusted closes for the tickers, aligned on date.  Returns a pandas DataFrame
    indexed by date.  interval: 1mo | 1wk | 1d."""
    import datetime as _dt
    import urllib.parse
    import requests
    if pd is None:
        raise RuntimeError("pandas required")
    frames = []
    for t in tickers:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(t)}?range={years}y&interval={interval}"
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            j = r.json()
            res = j["chart"]["result"]
            if not res:
                raise ValueError(j["chart"].get("error") or "empty result")
            res = res[0]
        except Exception as e:
            raise ValueError(f"could not fetch '{t}' from Yahoo ({e}); check the ticker symbol (BRK-B, ^GSPC, EURUSD=X style) "
                             "or download a CSV and use --prices") from e
        ts = res["timestamp"]
        ind = res["indicators"]
        closes = ind.get("adjclose", [{}])[0].get("adjclose") or ind["quote"][0]["close"]
        s = pd.Series(closes, index=pd.to_datetime(ts, unit="s").normalize(), name=t, dtype=float)
        s = s[~s.index.duplicated(keep="last")]
        frames.append(s)
    df = pd.concat(frames, axis=1).dropna()
    if drop_partial and len(df) > 1:
        today = _dt.date.today()
        last = df.index[-1].date()
        same_period = (interval == "1mo" and last.year == today.year and last.month == today.month) or \
                      (interval == "1wk" and (today - last).days < 7) or \
                      (interval == "1d" and last == today)
        if same_period:
            df = df.iloc[:-1]
    df.index.name = "Date"
    return df


# --------------------------------------------------------------------------- #
# The full report (everything, plus extras)
# --------------------------------------------------------------------------- #
def parse_weights(spec, names: Sequence[str]) -> Optional[np.ndarray]:
    """'0.25,0.25,0.25,0.25' or 'AAPL=0.5,AMGN=0.5' or list/dict -> array aligned to names."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        w = np.zeros(len(names))
        lut = {str(n).strip().lower(): i for i, n in enumerate(names)}
        for k, v in spec.items():
            key = str(k).strip().lower()
            if key not in lut:
                raise KeyError(f"weight for '{k}' but the universe is {list(names)}")
            w[lut[key]] = _pct(v)
        return w
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if all("=" in p for p in parts):
            return parse_weights({p.split("=")[0].strip(): p.split("=")[1] for p in parts}, names)
        w = np.array([_pct(p) for p in parts], dtype=float)
        if w.shape[0] != len(names):
            raise ValueError(f"{w.shape[0]} weights for {len(names)} assets {list(names)}; use NAME=weight pairs to be explicit")
        return w
    return np.asarray(spec, dtype=float)


def _pct(s) -> float:
    """'6.5%' -> 0.065 ; '0.065' -> 0.065 ; 6.5 -> 6.5 (numbers pass through)."""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)


def full_report(U: Universe, targets_sd: Sequence[float] = (), targets_mean: Sequence[float] = (),
                user_weights: Optional[Sequence[float]] = None,
                cal_grid: Iterable[float] = np.arange(0.0, 2.001, 0.1),
                two_fund_grid: Iterable[float] = np.arange(-8.0, 8.01, 0.5),
                funds: str = "auto", numeric_check: bool = True, no_short: bool = False,
                cal_targets: Sequence[tuple] = (), zero_names: Sequence[str] = (),
                histogram_of: Optional[str] = None, hist_step: float = 0.01) -> dict:
    """Compute every block of the report for this universe.

    funds: which two portfolios the 'Risky-Risky' data table mixes.
           'auto'  -> the first two sd targets if there are two, else TG + GMV
           'tg_gmv' -> always TG + GMV
    no_short: adds numeric no-short versions (closed forms always allow shorting).
              With a StD pinned, max Sharpe = max mean, so these ARE the 'max return, no shorts, StD = s' answers.
    cal_targets: [("sd", 0.08), ("mean", 0.02), ("wtg", 0.8)] -> CAL allocations that hit those targets
                 (or sit at that share in the tangency portfolio).
    zero_names: names pinned at 0 (an investor who will not hold them): closed forms on the remaining
                names, plus the shortfall against the unconstrained frontier at each StD target.
    histogram_of: an asset name, the market name or 'market': FREQUENCY-style histogram of its returns.
    """
    names = U.names
    g = gmv(U)
    ew = equal_weight(U)
    user = portfolio_stats(U, user_weights, "User PF", "the weights you typed") if user_weights is not None else None
    rep = {
        "inputs": {
            "names": names, "n_assets": U.n, "rf": U.rf, "ddof": U.ddof,
            "moments": "population (divide by N)" if U.ddof == 0 else "sample (divide by N-1)",
            "T": U.T, "return_type": U.return_type, "periods_per_year": U.periods_per_year,
            "market": U.market_name, "source": U.source, "has_means": U.has_means,
            "notes": list(U.notes),
        },
        "asset_stats": [{"name": n, "mean": (float(U.mu[i]) if U.has_means else None), "sd": float(U.sd[i]),
                         "var": float(U.var[i]),
                         "sharpe": (float((U.mu[i] - U.rf) / U.sd[i]) if U.has_means else None)} for i, n in enumerate(names)],
        "cov": U.cov.tolist(),
        "corr": U.corr.tolist(),
        "equal_weight": portfolio_dict(U, ew),
        "user_portfolio": portfolio_dict(U, user) if user else None,
        "gmv": portfolio_dict(U, g),
    }
    # the 'which stock is furthest right / highest' questions
    sd_ = U.sd
    ext = {"max_sd": {"name": names[int(np.argmax(sd_))], "sd": float(sd_.max())},
           "min_sd": {"name": names[int(np.argmin(sd_))], "sd": float(sd_.min())}}
    if U.has_means:
        sh_ = (U.mu - U.rf) / sd_
        ext.update({"max_mean": {"name": names[int(np.argmax(U.mu))], "mean": float(U.mu.max())},
                    "min_mean": {"name": names[int(np.argmin(U.mu))], "mean": float(U.mu.min())},
                    "max_sharpe": {"name": names[int(np.argmax(sh_))], "sharpe": float(sh_.max())}})
    rep["extremes"] = ext
    # histogram of one return series (FREQUENCY semantics), when returns exist
    if histogram_of:
        series, label = None, histogram_of
        if U.market_returns is not None and (histogram_of.lower() == "market" or histogram_of == U.market_name):
            series, label = U.market_returns, U.market_name
        elif U.returns is not None and histogram_of in names:
            series = U.returns[:, names.index(histogram_of)]
        elif U.returns is not None and histogram_of.lower() in ("ew", "equal", "pf"):
            series, label = U.returns.mean(axis=1), "PF (equal wt)"
        if series is None:
            rep["histogram"] = {"of": histogram_of, "error": "no return series by that name (needs prices/returns; use an asset name, the market name or 'market')"}
        else:
            rep["histogram"] = {"of": label, **histogram(series, hist_step)}
    if not U.has_means:
        rep["missing"] = ("no expected returns: tangency, frontier targets, the CAL and the two-fund table need means. "
                          "Everything above (st devs, correlations, MinVar, risk contributions) is complete.")
        rep["tangency"] = None
        rep["frontier_sd_targets"], rep["frontier_mean_targets"] = [], []
        rep["two_fund"], rep["cal"], rep["market_model"] = None, None, None
        return rep

    tg = tangency(U)
    fc = frontier_coefficients(U)
    sd_pfs = [frontier_at_sd(U, float(t)) for t in targets_sd]
    mean_pfs = [frontier_at_mean(U, float(t)) for t in targets_mean]
    if funds == "auto" and len(sd_pfs) >= 2:
        pA, pB = sd_pfs[0], sd_pfs[1]
    elif funds == "auto" and len(sd_pfs) == 1:
        pA, pB = sd_pfs[0], tg
    else:
        pA, pB = tg, g
    tf = two_fund_table(U, pA, pB, list(two_fund_grid))
    cal = cal_table(U, tg, list(cal_grid))
    cal["targets"] = []
    for kind, val in cal_targets:
        if kind == "wtg":
            pt = cal_point(U, tg, float(val))
            cal["targets"].append({"target_sd": None, "target_mean": None, "target_wtg": float(val),
                                   "wTG": float(val), "wRF": 1.0 - float(val), "sd": pt["sd"], "mean": pt["mean"],
                                   "asset_weights_named": {n: float(val) * float(w) for n, w in zip(names, tg.weights)},
                                   "note": f"{1 - float(val):.0%} at rF and {float(val):.0%} in the tangency portfolio"})
        else:
            cal["targets"].append(cal_for_target(U, tg, target_sd=val if kind == "sd" else None,
                                                 target_mean=val if kind == "mean" else None))
    # is the equal-weight portfolio efficient?  Compare it with the frontier portfolio at the SAME StD.
    try:
        few = frontier_at_sd(U, ew.sd)
        rep["ew_efficiency"] = {"ew_sd": ew.sd, "ew_mean": ew.mean, "frontier_mean_at_ew_sd": few.mean,
                                "gap": few.mean - ew.mean, "frontier_portfolio": portfolio_dict(U, few),
                                "verdict": ("inefficient: the frontier portfolio with the same StD earns "
                                            f"{(few.mean - ew.mean):.4%} more per period") if few.mean - ew.mean > 1e-10 else "on the frontier"}
    except Exception as e:  # pragma: no cover
        rep["ew_efficiency"] = {"error": str(e)}
    # names pinned at zero: closed forms on the remaining names, shortfall vs the unconstrained frontier
    zn = [n for n in zero_names if n in names]
    if zn:
        keep = [n for n in names if n not in zn]
        Uz = U.subset(keep)

        def lift(pz: Portfolio, label: str, note: str) -> dict:
            w = np.zeros(U.n)
            for n_, v in zip(keep, pz.weights):
                w[names.index(n_)] = v
            return portfolio_dict(U, portfolio_stats(U, w, label, note))

        zw = {"zero": zn, "kept": keep, "sd_targets": [],
              "tangency": lift(tangency(Uz), f"TG (no {', '.join(zn)})", "max Sharpe with those names pinned at 0"),
              "gmv": lift(gmv(Uz), f"MinVar (no {', '.join(zn)})", "min variance with those names pinned at 0")}
        for t, p in zip(targets_sd, sd_pfs):
            try:
                pz = frontier_at_sd(Uz, float(t))
            except ValueError as e:
                zw["sd_targets"].append({"target_sd": float(t), "error": str(e)})
                continue
            d = lift(pz, f"sd = {pct_label(float(t))} (no {', '.join(zn)})",
                     "max mean at this StD with those names pinned at 0 (closed form on the remaining names; Solver: max mean s.t. sum = 1, StD = target, those weights = 0)")
            d["unconstrained_mean"] = p.mean
            d["shortfall_vs_unconstrained"] = p.mean - d["mean"]
            d["target_sd"] = float(t)
            zw["sd_targets"].append(d)
        rep["zero_weight"] = zw
    rep.update({
        "frontier_coefficients": {"A": fc.A, "B": fc.B, "C": fc.C, "D": fc.D,
                                  "gmv_mean": fc.A / fc.C, "gmv_var": 1.0 / fc.C},
        "tangency": portfolio_dict(U, tg),
        "frontier_sd_targets": [portfolio_dict(U, p) for p in sd_pfs],
        "frontier_mean_targets": [portfolio_dict(U, p) for p in mean_pfs],
        "two_fund": tf,
        "cal": cal,
        "market_model": market_model(U),
    })
    for p in sd_pfs:  # the live-Excel 'a' share for each sd target
        p_dict = next(d for d in rep["frontier_sd_targets"] if d["name"] == p.name)
        p_dict["a_share_of_tangency"] = frontier_share_of_tangency(U, tg, g, p.sd)
    if rep["market_model"]:
        mm = rep["market_model"]
        rep["market_model"]["portfolios"] = {
            "tangency": portfolio_market_stats(U, mm, tg.weights),
            "gmv": portfolio_market_stats(U, mm, g.weights),
            "equal_weight": portfolio_market_stats(U, mm, ew.weights),
            "user": portfolio_market_stats(U, mm, user.weights) if user else None,
        }
    if U.returns is not None:
        wbn = {"TG": tg.weights, "MinVar": g.weights, "EW": ew.weights}
        if user:
            wbn["User"] = user.weights
        rep["growth"] = growth_of_one(U, wbn)
    if U.periods_per_year:
        p = U.periods_per_year
        rep["annualized"] = {
            "periods_per_year": p,
            "rf": U.rf * p,
            "assets": {n: annualize(float(U.mu[i]), float(U.sd[i]), p) for i, n in enumerate(names)},
            "tangency": annualize(tg.mean, tg.sd, p),
            "gmv": annualize(g.mean, g.sd, p),
            "equal_weight": annualize(ew.mean, ew.sd, p),
            "user": annualize(user.mean, user.sd, p) if user else None,
        }
    if numeric_check:
        chk = {}
        try:
            pt, _ = solve_numeric(U, "max_sharpe", name="TG (SLSQP)")
            pg, _ = solve_numeric(U, "min_var", name="MinVar (SLSQP)")
            chk["tangency_numeric"] = pt.to_dict(names)
            chk["gmv_numeric"] = pg.to_dict(names)
            chk["max_abs_weight_diff_tg"] = float(np.abs(pt.weights - tg.weights).max())
            chk["max_abs_weight_diff_gmv"] = float(np.abs(pg.weights - g.weights).max())
            for t, p in zip(targets_sd, sd_pfs):
                ps, _ = solve_numeric(U, "max_sharpe", target_sd=float(t), name=f"{p.name} (SLSQP)")
                chk[f"sd_target_{t}"] = {"numeric": ps.to_dict(names), "max_abs_weight_diff": float(np.abs(ps.weights - p.weights).max())}
        except Exception as e:  # scipy missing etc.
            chk["error"] = str(e)
        rep["numeric_check"] = chk
    if no_short:
        ns = {}
        try:
            pt, _ = solve_numeric(U, "max_sharpe", no_short=True, name="TG (no short)")
            pg, _ = solve_numeric(U, "min_var", no_short=True, name="MinVar (no short)")
            ns["tangency"] = portfolio_dict(U, pt)
            ns["gmv"] = portfolio_dict(U, pg)
            for t, p in zip(targets_sd, sd_pfs):
                try:
                    ps, _ = solve_numeric(U, "max_sharpe", target_sd=float(t), no_short=True, name=f"{p.name} (no short)")
                    ns[p.name] = portfolio_dict(U, ps)
                except Exception as e:
                    ns[p.name] = {"error": str(e)}
        except Exception as e:
            ns["error"] = str(e)
        rep["no_short"] = ns
    return rep


# --------------------------------------------------------------------------- #
# Console rendering
# --------------------------------------------------------------------------- #
def _fmt_pct(x, d=4):
    return "   n/a" if _isnan(x) else f"{x * 100:.{d}f}%"


def _fmt(x, d=4):
    return "n/a" if _isnan(x) else f"{x:.{d}f}"


def render_report(rep: dict) -> str:
    names = rep["inputs"]["names"]
    out = []
    inp = rep["inputs"]
    out.append(f"Universe: {', '.join(names)}   rf/period = {_fmt_pct(inp['rf'])}   moments = {inp['moments']}"
               + (f"   T = {inp['T']}" if inp["T"] else "") + (f"   market = {inp['market']}" if inp["market"] else ""))
    if inp.get("notes"):
        out.append("NOTES")
        for n in inp["notes"]:
            out.append(f"  - {n}")
    if rep.get("missing"):
        out.append(f"PARTIAL: {rep['missing']}")
    out.append("")
    out.append("ASSET STATS                mean        st dev      var          sharpe")
    for a in rep["asset_stats"]:
        out.append(f"  {a['name']:<18} {_fmt_pct(a['mean']):>10}  {_fmt_pct(a['sd']):>10}  {a['var']:.6f}   {_fmt(a['sharpe'])}")
    out.append("")
    out.append("COVARIANCE (matrix algebra, /N)" if inp["ddof"] == 0 else "COVARIANCE (sample, /N-1)")
    hdr = "  " + " " * 10 + "".join(f"{n[:11]:>12}" for n in names)
    out.append(hdr)
    for n, row in zip(names, rep["cov"]):
        out.append(f"  {n[:10]:<10}" + "".join(f"{v:12.7f}" for v in row))
    out.append("CORRELATION")
    out.append(hdr)
    for n, row in zip(names, rep["corr"]):
        out.append(f"  {n[:10]:<10}" + "".join(f"{v:12.4f}" for v in row))
    out.append("")

    def block(p):
        if not p:
            return
        out.append(f"{p['name']:<18} mean {_fmt_pct(p['mean'], 4):>9}  sd {_fmt_pct(p['sd'], 4):>9}  var {p['var']:.6f}  sharpe {_fmt(p['sharpe'])}   sum w {p['sum_weights']:.4f}")
        out.append("   weights: " + "  ".join(f"{n} {w * 100:6.2f}%" for n, w in p["weights"].items()))
        if p.get("risk"):
            out.append("   risk share: " + "  ".join(f"{n} {s * 100:6.1f}%" for n, s in zip(p["weights"].keys(), p["risk"]["share_of_var"])))
        if p.get("note"):
            out.append(f"   {p['note']}")

    out.append("PORTFOLIOS")
    block(rep["user_portfolio"])
    block(rep["equal_weight"])
    block(rep["gmv"])
    if not rep.get("tangency"):
        return "\n".join(out)
    block(rep["tangency"])
    for p in rep["frontier_sd_targets"]:
        block(p)
        out.append(f"   live-Excel share of TG: a = {p['a_share_of_tangency']:.6f}  (w = a*TG + (1-a)*MinVar)")
    for p in rep["frontier_mean_targets"]:
        block(p)
    out.append("")
    tf = rep["two_fund"]
    out.append(f"RISKY-RISKY  (funds: {tf['fund_a']} and {tf['fund_b']}; Cov(A,B) = {tf['cov_ab']:.10f})")
    out.append("     a(=wA)      st dev       mean")
    for r in tf["rows"]:
        out.append(f"  {r['a']:7.2f}   {_fmt_pct(r['sd'], 2):>9}  {_fmt_pct(r['mean'], 2):>9}")
    out.append("")
    cal = rep["cal"]
    out.append(f"RISKY-RISKLESS (CAL)  slope = Sharpe(TG) = {cal['slope_sharpe']:.4f}, intercept = rf = {_fmt_pct(cal['intercept_rf'], 2)}")
    out.append("     wTG     wRF     st dev       mean")
    for r in cal["rows"]:
        out.append(f"  {r['wTG']:6.2f}  {r['wRF']:6.2f}   {_fmt_pct(r['sd'], 2):>9}  {_fmt_pct(r['mean'], 2):>9}")
    for t in cal.get("targets", []):
        if t.get("target_sd") is not None:
            what = f"sd = {_fmt_pct(t['target_sd'], 2)}"
        elif t.get("target_mean") is not None:
            what = f"mean = {_fmt_pct(t['target_mean'], 2)}"
        else:
            what = f"wTG = {t['target_wtg']:.2f}"
        out.append(f"  CAL target {what}: wTG {t['wTG']:.4f}, wRF {t['wRF']:.4f} -> sd {_fmt_pct(t['sd'], 4)}, mean {_fmt_pct(t['mean'], 4)}; dollar weights "
                   + "  ".join(f"{n} {w * 100:6.2f}%" for n, w in t["asset_weights_named"].items()))
    if rep.get("ew_efficiency") and "error" not in rep["ew_efficiency"]:
        e = rep["ew_efficiency"]
        out.append("")
        out.append(f"EQUAL-WEIGHT EFFICIENCY: EW has sd {_fmt_pct(e['ew_sd'], 4)} and mean {_fmt_pct(e['ew_mean'], 4)}; the frontier portfolio at the SAME sd "
                   f"has mean {_fmt_pct(e['frontier_mean_at_ew_sd'], 4)} -> gap {_fmt_pct(e['gap'], 4)} per period ({e['verdict']})")
        block(e["frontier_portfolio"])
    if rep.get("zero_weight"):
        z = rep["zero_weight"]
        out.append("")
        out.append(f"NAMES PINNED AT ZERO: {', '.join(z['zero'])} (closed forms on the remaining {len(z['kept'])} names)")
        block(z["tangency"])
        block(z["gmv"])
        for d in z["sd_targets"]:
            if "error" in d:
                out.append(f"  sd = {_fmt_pct(d['target_sd'], 2)}: {d['error']}")
                continue
            block(d)
            out.append(f"   unconstrained frontier mean at this sd {_fmt_pct(d['unconstrained_mean'], 4)} -> shortfall {_fmt_pct(d['shortfall_vs_unconstrained'], 4)} per period (inefficient by that much)")
    if rep.get("extremes"):
        ex = rep["extremes"]
        out.append("")
        line = f"EXTREMES: furthest right (max sd) {ex['max_sd']['name']} {_fmt_pct(ex['max_sd']['sd'], 3)}; lowest sd {ex['min_sd']['name']} {_fmt_pct(ex['min_sd']['sd'], 3)}"
        if ex.get("max_mean"):
            line += f"; highest mean {ex['max_mean']['name']} {_fmt_pct(ex['max_mean']['mean'], 3)}; lowest mean {ex['min_mean']['name']} {_fmt_pct(ex['min_mean']['mean'], 3)}; best Sharpe {ex['max_sharpe']['name']} {ex['max_sharpe']['sharpe']:.4f}"
        out.append(line)
    if rep.get("histogram"):
        h = rep["histogram"]
        out.append("")
        if "error" in h:
            out.append(f"HISTOGRAM of {h['of']}: {h['error']}")
        else:
            out.append(f"HISTOGRAM of {h['of']} returns, {h['step'] * 100:g}% buckets (FREQUENCY semantics: bin = upper bound), n = {h['n']}, "
                       f"min {_fmt_pct(h['min'], 2)}, max {_fmt_pct(h['max'], 2)}, mean {_fmt_pct(h['mean'], 3)}, sd {_fmt_pct(h['sd'], 3)}, modal bucket {h['modal_bin']}")
            for lab, c, rf_ in zip(h["labels"], h["counts"], h["rel_freq"]):
                out.append(f"  {lab:>16}  {c:3d}  {'#' * c}  ({rf_:.1%})")
            out.append(f"  {'> last bin':>16}  {h['overflow']:3d}")
    if rep.get("market_model"):
        mm = rep["market_model"]
        out.append("")
        out.append(f"MARKET MODEL vs {mm['market']}  (market mean {_fmt_pct(mm['market_mean'])}, sd {_fmt_pct(mm['market_sd'])})")
        out.append("  name        beta     alpha(J)    R^2     resid sd   CAPM E[r]   Treynor")
        for a in mm["assets"]:
            out.append(f"  {a['name'][:10]:<10} {a['beta']:6.3f}  {_fmt_pct(a['alpha_jensen']):>9}  {a['r2']:6.3f}  {_fmt_pct(a['resid_sd']):>9}  {_fmt_pct(a['capm_expected']):>9}   {_fmt(a['treynor'])}")
        for k, v in mm["portfolios"].items():
            if v:
                out.append(f"  PF {k:<12} beta {v['beta']:.3f}  alpha {_fmt_pct(v['alpha_jensen'])}  sys sd {_fmt_pct(v['systematic_sd'])}  resid sd(SIM) {_fmt_pct(v['resid_sd_sim'])}")
    if rep.get("growth"):
        gr = rep["growth"]
        out.append("")
        out.append("GROWTH OF $1 (in-sample, constant weights, rebalanced every period)")
        out.append("  " + "  ".join(f"{k} {v[-1]:.3f}" for k, v in gr["portfolios"].items()) + "   assets: " + "  ".join(f"{k} {v[-1]:.3f}" for k, v in gr["assets"].items()))
    if rep.get("annualized"):
        an = rep["annualized"]
        out.append("")
        out.append(f"ANNUALIZED (x{an['periods_per_year']} mean, x sqrt{an['periods_per_year']} sd; rf {_fmt_pct(an['rf'], 2)})")
        for k in ("tangency", "gmv", "equal_weight", "user"):
            if an.get(k):
                out.append(f"  {k:<13} mean {_fmt_pct(an[k]['mean_arith'], 2):>8} (geom {_fmt_pct(an[k]['mean_geom'], 2)})  sd {_fmt_pct(an[k]['sd'], 2)}")
    if rep.get("numeric_check") and "error" not in rep["numeric_check"]:
        nc = rep["numeric_check"]
        out.append("")
        out.append(f"SOLVER CHECK (SLSQP vs closed form): max |dw| TG {nc['max_abs_weight_diff_tg']:.2e}, MinVar {nc['max_abs_weight_diff_gmv']:.2e}")
    if rep.get("no_short"):
        out.append("")
        out.append("NO-SHORT versions (numeric):")
        for k, v in rep["no_short"].items():
            if isinstance(v, dict) and "weights" in v:
                block(v)
    return "\n".join(out)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_jsonable(obj), indent=2))
