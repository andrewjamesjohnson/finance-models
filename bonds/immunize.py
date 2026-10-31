"""Immunization: match the liabilities' present value and duration.

Liabilities: t = (date - settle) / 365, PV = FV / (1 + own yield)^t (annual compounding), PV weights.
Liability Macaulay duration = sum(w t); modified = sum(w t / (1 + r)). Assets: portfolio duration = sum(w D_bond)
with D = MDURATION (the default, the assets' MODIFIED duration after matching) or
DURATION (Macaulay). The weights solve sum = 100% and duration = liability duration, inside the bounds
(for example 0% to 25%). With more bonds than equations there are many answers: the one given is the minimum-norm
solution (closest to equal weights), which Solver can land near but not exactly on.
Rate scenarios: every bond re-priced at its YTM + shift, every liability at its rate + shift; excess = bonds - liability.
"""
import numpy as np
from scipy.optimize import minimize

from xlbond import as_date, price


def liabilities(settle, rows, year_days=365):
    """rows: [(rate, date, face)] -> years, PV, weights, total PV, Macaulay and modified duration."""
    settle = as_date(settle)
    r = np.array([float(x[0]) for x in rows])
    t = np.array([(as_date(x[1]) - settle).days / year_days for x in rows])
    fv = np.array([float(x[2]) for x in rows])
    pv = fv / (1 + r) ** t
    w = pv / pv.sum()
    return dict(rates=r, years=t, face=fv, pv=pv, w=w, total=float(pv.sum()),
                mac=float((w * t).sum()), mod=float((w * t / (1 + r)).sum()),
                conv=float((w * t * (t + 1) / (1 + r) ** 2).sum()))


def duration_range(d, lo, hi):
    """The lowest and highest portfolio duration reachable with sum = 1 and lo <= w <= hi (greedy LP)."""
    d = np.asarray(d, float)

    def extreme(order):
        w = np.full(len(d), lo)
        left = 1 - w.sum()
        for i in order:
            add = min(hi - lo, left)
            w[i] += add
            left -= add
        return float(w @ d) if abs(left) < 1e-12 else None

    return extreme(np.argsort(d)), extreme(np.argsort(-d))


def weights(d, target, lo=0.0, hi=1.0, cvx=None, ctarget=None):
    """Weights with sum 1 and w . d = target (and w . cvx = ctarget when given) inside [lo, hi], closest to equal
    weights. -> (w, method)."""
    d = np.asarray(d, float)
    n = len(d)
    rows, rhs = [np.ones(n), d], [1.0, target]
    if cvx is not None:
        rows.append(np.asarray(cvx, float))
        rhs.append(ctarget)
    a, b = np.vstack(rows), np.array(rhs)
    w = a.T @ np.linalg.solve(a @ a.T, b)
    if (w >= lo - 1e-12).all() and (w <= hi + 1e-12).all():
        return w, "minimum-norm (closest to equal weights)"
    if cvx is None:
        dmin, dmax = duration_range(d, lo, hi)
        if dmin is None or not (dmin - 1e-12 <= target <= dmax + 1e-12):
            raise ValueError(f"target duration {target:.4f} is outside what the bonds reach under the bounds "
                             f"[{lo:.0%}, {hi:.0%}]: {dmin} to {dmax}")
    res = minimize(lambda x: ((x - 1 / n) ** 2).sum(), np.clip(w, lo, hi), jac=lambda x: 2 * (x - 1 / n),
                   method="SLSQP", bounds=[(lo, hi)] * n,
                   constraints=[{"type": "eq", "fun": lambda x: a @ x - b, "jac": lambda x: a}],
                   options=dict(ftol=1e-16, maxiter=1000))
    x = res.x
    if abs(a @ x - b).max() > 1e-9 or (x < lo - 1e-9).any() or (x > hi + 1e-9).any():
        what = "the duration and the convexity" if cvx is not None else "the duration"
        raise ValueError(f"no weights inside [{lo:.0%}, {hi:.0%}] match {what}")
    return x, "closest to equal weights inside the bounds (bounded least squares)"


def scenario_table(settle, bonds, invest, liab, shifts, freq=2):
    """bonds: [(coupon, maturity, ytm)] decimals; invest: $ per bond at shift 0. Face = invest / price(ytm) x 100.
    -> rows of (shift, [bond prices], bond value, liability value, excess)."""
    settle = as_date(settle)
    p0 = [price(settle, m, c, y, 100, freq) for c, m, y in bonds]
    face = [i / p * 100 for i, p in zip(invest, p0)]
    rows = []
    for s in shifts:
        ps = [price(settle, m, c, y + s, 100, freq) for c, m, y in bonds]
        bv = sum(f * p / 100 for f, p in zip(face, ps))
        lv = float((liab["face"] / (1 + liab["rates"] + s) ** liab["years"]).sum())
        rows.append((s, ps, bv, lv, bv - lv))
    return face, rows
