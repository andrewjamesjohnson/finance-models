#!/usr/bin/env python3
"""
riskstats.py - the risk numbers an advisor reports, and an honest out-of-sample test.

  series_stats   mean, sd, Sortino, max drawdown (with when), worst / best period, historical and
                 parametric VaR / CVaR (95%), skew, kurtosis, hit rate, tracking error and information
                 ratio versus a benchmark, beta
  walk_forward   re-estimate on a trailing window, hold the next period, roll: realized growth,
                 Sharpe, drawdown and turnover for TG / MinVar / EW / user (and long-only variants)
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

import mv_engine as mv


def drawdown(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.cumprod(1.0 + np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(g)
    return g / peak - 1.0, g


def series_stats(r: np.ndarray, rf: float = 0.0, ppy: Optional[int] = None, bench: Optional[np.ndarray] = None,
                 dates: Optional[Sequence[str]] = None) -> dict:
    r = np.asarray(r, dtype=float)
    T = r.shape[0]
    mean, sd = float(r.mean()), float(r.std(ddof=0))
    down = np.minimum(r - rf, 0.0)
    dd_dev = float(math.sqrt((down ** 2).mean()))
    dd, g = drawdown(r)
    i_trough = int(np.argmin(dd))
    i_peak = int(np.argmax(g[:i_trough + 1])) if i_trough > 0 else 0
    srt = np.sort(r)
    k = max(int(math.floor(0.05 * T)), 1)
    var95 = float(np.percentile(r, 5))
    cvar95 = float(srt[:k].mean())
    z = 1.6448536269514722
    out = {
        "T": T, "mean": mean, "sd": sd, "sharpe": (mean - rf) / sd if sd > 0 else None,
        "sortino": (mean - rf) / dd_dev if dd_dev > 0 else None, "downside_dev": dd_dev,
        "max_drawdown": float(dd.min()), "drawdown_peak": (dates[i_peak] if dates and i_peak < len(dates) else i_peak),
        "drawdown_trough": (dates[i_trough] if dates and i_trough < len(dates) else i_trough),
        "worst": float(r.min()), "best": float(r.max()), "hit_rate": float((r > 0).mean()),
        "var95_hist": var95, "cvar95_hist": cvar95, "var95_param": mean - z * sd,
        "skew": float(((r - mean) ** 3).mean() / sd ** 3) if sd > 0 else None,
        "kurtosis_excess": float(((r - mean) ** 4).mean() / sd ** 4 - 3.0) if sd > 0 else None,
        "growth_of_1": float(g[-1]),
    }
    if ppy:
        out["annual"] = {"mean": mean * ppy, "sd": sd * math.sqrt(ppy), "geometric": float(g[-1] ** (ppy / T) - 1.0),
                         "sharpe": ((mean - rf) * ppy) / (sd * math.sqrt(ppy)) if sd > 0 else None}
    if bench is not None:
        b = np.asarray(bench, dtype=float)
        active = r - b
        te = float(active.std(ddof=0))
        cov = float(((r - mean) * (b - b.mean())).mean())
        out["tracking_error"] = te
        out["information_ratio"] = float(active.mean() / te) if te > 0 else None
        out["beta"] = cov / float(b.var()) if b.var() > 0 else None
        out["active_return"] = float(active.mean())
    return out


def walk_forward(U: mv.Universe, window: int = 36, long_only: bool = False, user_weights: Optional[Sequence[float]] = None) -> dict:
    """Roll through the history: estimate on R[t-window:t], hold for period t."""
    if U.returns is None:
        raise ValueError("walk-forward needs a return history")
    R = U.returns
    T, N = R.shape
    if T < window + 6:
        raise ValueError(f"walk-forward needs at least {window + 6} periods (have {T}); lower --walk-window")
    import constrained as cn
    strategies = {"TG": None, "MinVar": None, "EW": None}
    if user_weights is not None:
        strategies["User (fixed)"] = None
    if long_only:
        strategies["TG long-only"] = None
        strategies["MinVar long-only"] = None
    realized = {k: [] for k in strategies}
    turnover = {k: [] for k in strategies}
    prev = {k: None for k in strategies}
    dates = U.dates[window:] if U.dates and len(U.dates) == T else None
    for t in range(window, T):
        Rw = R[t - window:t]
        mu, cov = mv.moments(Rw, U.ddof)
        Uw = mv.Universe(U.names, mu, cov, rf=U.rf, ddof=U.ddof)
        ws = {}
        try:
            ws["TG"] = mv.tangency(Uw).weights
        except Exception:
            ws["TG"] = np.full(N, 1.0 / N)
        try:
            ws["MinVar"] = mv.gmv(Uw).weights
        except Exception:
            ws["MinVar"] = np.full(N, 1.0 / N)
        ws["EW"] = np.full(N, 1.0 / N)
        if user_weights is not None:
            ws["User (fixed)"] = np.asarray(user_weights, dtype=float)
        if long_only:
            c = cn.Constraints(long_only=True)
            try:
                ws["TG long-only"] = cn.tangency(Uw, c).weights
            except Exception:
                ws["TG long-only"] = np.full(N, 1.0 / N)
            try:
                ws["MinVar long-only"] = cn.gmv(Uw, c).weights
            except Exception:
                ws["MinVar long-only"] = np.full(N, 1.0 / N)
        for k, w in ws.items():
            realized[k].append(float(w @ R[t]))
            if prev[k] is not None:
                turnover[k].append(float(np.abs(w - prev[k]).sum()))
            prev[k] = w
    out = {"window": window, "periods": T - window, "dates": dates, "strategies": {}}
    bench = U.market_returns[window:] if U.market_returns is not None else None
    for k in strategies:
        r = np.array(realized[k])
        st = series_stats(r, U.rf, U.periods_per_year, bench, dates)
        st["avg_turnover"] = float(np.mean(turnover[k])) if turnover[k] else 0.0
        st["realized_returns"] = r.tolist()
        st["growth"] = np.cumprod(1.0 + r).tolist()
        out["strategies"][k] = st
    if bench is not None:
        st = series_stats(np.asarray(bench), U.rf, U.periods_per_year, None, dates)
        st["growth"] = np.cumprod(1.0 + np.asarray(bench)).tolist()
        out["strategies"][U.market_name or "market"] = st
    return out


def section(U: mv.Universe, rep: dict, window: int = 36, long_only: bool = False, user_weights=None) -> dict:
    if U.returns is None:
        return {"error": "risk statistics need a return history (prices or returns)"}
    R = U.returns
    ppy = U.periods_per_year
    bench = U.market_returns
    sec = {"assets": {}, "portfolios": {}, "in_sample_note": "portfolio stats below use the FULL-sample optimal weights (in-sample); the walk-forward block is the out-of-sample test"}
    for i, n in enumerate(U.names):
        sec["assets"][n] = series_stats(R[:, i], U.rf, ppy, bench, U.dates[1:] if U.dates and len(U.dates) == R.shape[0] + 1 else U.dates)
    pfs = {"TG": rep["tangency"]["weights_list"], "MinVar": rep["gmv"]["weights_list"], "EW": rep["equal_weight"]["weights_list"]}
    if rep.get("user_portfolio"):
        pfs["User"] = rep["user_portfolio"]["weights_list"]
    if rep.get("constrained"):
        pfs["TG (constrained)"] = rep["constrained"]["tangency"]["weights_list"]
    for k, w in pfs.items():
        sec["portfolios"][k] = series_stats(R @ np.asarray(w), U.rf, ppy, bench, U.dates[1:] if U.dates and len(U.dates) == R.shape[0] + 1 else U.dates)
    if bench is not None:
        sec["market"] = series_stats(np.asarray(bench), U.rf, ppy, None)
    try:
        sec["walk_forward"] = walk_forward(U, window, long_only, user_weights)
    except ValueError as e:
        sec["walk_forward"] = {"error": str(e)}
    return sec


def _f(x, pct=True, d=2):
    if x is None:
        return "  n/a"
    return f"{x * 100:.{d}f}%" if pct else f"{x:.{d}f}"


def render(sec: dict, names: Sequence[str]) -> str:
    if "error" in sec:
        return "RISK: " + sec["error"]
    out = ["RISK STATISTICS (per period; drawdown and VaR from the history)"]
    out.append(f"  {'series':<18}{'mean':>8}{'sd':>8}{'Sortino':>9}{'maxDD':>9}{'worst':>8}{'VaR95':>8}{'CVaR95':>8}{'TE':>8}{'IR':>7}{'beta':>7}")
    for grp in ("assets", "portfolios"):
        for k, s in sec[grp].items():
            out.append(f"  {k:<18}{_f(s['mean']):>8}{_f(s['sd']):>8}{_f(s['sortino'], False):>9}{_f(s['max_drawdown']):>9}{_f(s['worst']):>8}{_f(s['var95_hist']):>8}{_f(s['cvar95_hist']):>8}"
                       f"{_f(s.get('tracking_error')):>8}{_f(s.get('information_ratio'), False):>7}{_f(s.get('beta'), False):>7}")
    if sec.get("market"):
        s = sec["market"]
        out.append(f"  {'market':<18}{_f(s['mean']):>8}{_f(s['sd']):>8}{_f(s['sortino'], False):>9}{_f(s['max_drawdown']):>9}{_f(s['worst']):>8}{_f(s['var95_hist']):>8}{_f(s['cvar95_hist']):>8}")
    out.append(f"  ({sec['in_sample_note']})")
    wf = sec.get("walk_forward", {})
    if "error" in wf:
        out.append("  walk-forward: " + wf["error"])
    else:
        out.append(f"WALK-FORWARD (estimate on the trailing {wf['window']} periods, hold the next one; {wf['periods']} out-of-sample periods)")
        out.append(f"  {'strategy':<18}{'mean':>8}{'sd':>8}{'Sharpe':>8}{'maxDD':>9}{'growth':>8}{'turnover':>10}{'TE':>8}{'IR':>7}")
        for k, s in wf["strategies"].items():
            out.append(f"  {k:<18}{_f(s['mean']):>8}{_f(s['sd']):>8}{_f(s['sharpe'], False, 3):>8}{_f(s['max_drawdown']):>9}{s['growth_of_1']:>8.3f}{_f(s.get('avg_turnover')):>10}{_f(s.get('tracking_error')):>8}{_f(s.get('information_ratio'), False):>7}")
    return "\n".join(out)
