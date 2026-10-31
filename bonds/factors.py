"""Factor model regressions: CAPM and Fama French.

Excess return = simple monthly return - RF; regressors = the factor columns (Mkt-RF, SMB, HML, RMW, CMA, ...).
b = (X'X)^-1 X'y with a column of ones is what the sheet's MMULT(MINVERSE(...)) block returns; LINEST returns the
same numbers in reverse order plus its statistics, all reproduced here. Out of sample: predicted = intercept +
loadings x realized factors + RF, alpha = actual raw return - predicted, then the average.
"""
import datetime as dt

import numpy as np


def ym(x):
    """date / datetime / 202604 / '2026-04' / 'Apr-26' -> 'YYYYMM'."""
    if isinstance(x, (dt.date, dt.datetime)):
        return f"{x.year}{x.month:02d}"
    s = str(x).strip()
    if s.isdigit() and len(s) == 6:
        return s
    if len(s) >= 7 and s[4] in "-/" and s[:4].isdigit():
        return s[:4] + s[5:7]
    raise ValueError(f"not a month: {x!r}")


def month_add(m, k):
    y, mo = int(m[:4]), int(m[4:]) + k
    y += (mo - 1) // 12
    mo = (mo - 1) % 12 + 1
    return f"{y}{mo:02d}"


def months_between(start, end):
    out, m = [], ym(start)
    end = ym(end)
    while m <= end:
        out.append(m)
        m = month_add(m, 1)
    return out


def returns(price_by_month, log=False):
    """{YYYYMM: price vector} -> {YYYYMM: return vector} (needs the month before)."""
    out = {}
    for m, p in price_by_month.items():
        prev = month_add(m, -1)
        if prev in price_by_month:
            p0 = np.asarray(price_by_month[prev], float)
            p1 = np.asarray(p, float)
            out[m] = np.log(p1 / p0) if log else p1 / p0 - 1
    return out


def regress(X, Y):
    """X: T x k factors (no constant), Y: T x n. OLS with intercept, LINEST's statistics."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    t = X.shape[0]
    xc = np.column_stack([np.ones(t), X])
    inv = np.linalg.inv(xc.T @ xc)
    b = inv @ xc.T @ Y
    resid = Y - xc @ b
    df = t - xc.shape[1]
    sse = (resid ** 2).sum(0)
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    sey = np.sqrt(sse / df)
    se = np.sqrt(np.outer(np.diag(inv), sey ** 2))
    r2 = 1 - sse / sst
    fstat = ((sst - sse) / (xc.shape[1] - 1)) / (sse / df)
    return dict(b=b, se=se, t=b / se, r2=r2, sey=sey, df=df, sse=sse, ssreg=sst - sse, F=fstat)


def run(price_by_month, factor_by_month, tickers, factor_names, start, end, oos=None, pct=None, log=False):
    """price_by_month: {YYYYMM: [p per ticker]}; factor_by_month: {YYYYMM: [f1..fk, RF]} (RF last).
    pct=None detects percent units (|Mkt-RF| > 1 anywhere). Returns a dict with every block of FF1-2."""
    rets = returns(price_by_month, log)
    win = months_between(start, end)
    missing = [m for m in win if m not in rets or m not in factor_by_month]
    if missing:
        raise ValueError(f"months missing in prices or factors: {missing[:6]}")
    if pct is None:
        pct = max(abs(v[0]) for v in factor_by_month.values()) > 1.0
    scale = 0.01 if pct else 1.0
    F = np.array([factor_by_month[m] for m in win], float) * scale
    X, rf = F[:, :-1], F[:, -1]
    R = np.array([rets[m] for m in win])
    Y = R - rf[:, None]
    reg = regress(X, Y)
    if oos is None:
        oos = sorted(m for m in rets if m > ym(end) and m in factor_by_month)
    out = dict(window=win, oos=list(oos), pct=pct, tickers=list(tickers), factors=list(factor_names), reg=reg,
               b=reg["b"], X=X, rf=rf, Y=Y)
    if oos:
        Fo = np.array([factor_by_month[m] for m in oos], float) * scale
        Xo, rfo = Fo[:, :-1], Fo[:, -1]
        actual = np.array([rets[m] for m in oos])
        pred = np.column_stack([np.ones(len(oos)), Xo]) @ reg["b"] + rfo[:, None]
        alpha = actual - pred
        out.update(Xo=Xo, rfo=rfo, actual=actual, actual_excess=actual - rfo[:, None], pred=pred, alpha=alpha,
                   avg_alpha=alpha.mean(0))
    return out
