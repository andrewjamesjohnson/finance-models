#!/usr/bin/env python3
"""
constrained.py - long-only and constrained mean-variance: what advisors actually run.

The closed forms in mv_engine assume shorting.  Real mandates say long-only, caps per name,
caps per group ("tech <= 40%"), a turnover limit from the current book.  This module solves
those with SLSQP (the same problem Excel Solver solves with the extra constraints) and traces
the CONSTRAINED efficient frontier, which is a different curve from the unconstrained hyperbola.

Everything here is additive: nothing in mv_engine changes.  When no constraint binds, the
answers equal the closed forms (the self-test checks that).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

import mv_engine as mv


@dataclass
class Constraints:
    long_only: bool = False
    lower: dict = field(default_factory=dict)          # name -> lower bound
    upper: dict = field(default_factory=dict)          # name -> upper bound
    default_bounds: tuple = (None, None)               # applied to every asset unless overridden
    groups: list = field(default_factory=list)         # [(label, [names], lo, hi)]
    fixed: dict = field(default_factory=dict)          # name -> pinned weight (0 removes a name)
    current: Optional[np.ndarray] = None               # current weights, for the turnover cap
    turnover_cap: Optional[float] = None               # sum |w - current| <= cap

    def describe(self) -> list[str]:
        out = []
        if self.long_only:
            out.append("long only (every weight >= 0)")
        lo, hi = self.default_bounds
        if lo is not None or hi is not None:
            out.append(f"every weight in [{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}]")
        for k, v in self.lower.items():
            out.append(f"{k} >= {v:.2%}")
        for k, v in self.upper.items():
            out.append(f"{k} <= {v:.2%}")
        for label, names, glo, ghi in self.groups:
            out.append(f"group {label} ({', '.join(names)}) in [{glo:.0%}, {ghi:.0%}]")
        for k, v in self.fixed.items():
            out.append(f"{k} pinned at {v:.2%}")
        if self.turnover_cap is not None:
            out.append(f"turnover from current holdings <= {self.turnover_cap:.0%}")
        return out or ["none (only sum of weights = 1)"]

    def is_trivial(self) -> bool:
        return not (self.long_only or self.lower or self.upper or self.groups or self.fixed
                    or self.turnover_cap is not None or self.default_bounds != (None, None))

    def bounds_for(self, U: mv.Universe) -> list[tuple]:
        lo, hi = self.default_bounds
        if self.long_only and lo is None:
            lo = 0.0
        b = [(lo, hi) for _ in range(U.n)]
        for k, v in self.lower.items():
            i = U.index(k)
            b[i] = (v, b[i][1])
        for k, v in self.upper.items():
            i = U.index(k)
            b[i] = (b[i][0], v)
        for k, v in self.fixed.items():
            i = U.index(k)
            b[i] = (v, v)
        return b


def _feasible_start(U: mv.Universe, cons: Constraints) -> np.ndarray:
    b = cons.bounds_for(U)
    w = np.full(U.n, 1.0 / U.n)
    lo = np.array([x[0] if x[0] is not None else -np.inf for x in b])
    hi = np.array([x[1] if x[1] is not None else np.inf for x in b])
    w = np.clip(w, lo, hi)
    s = w.sum()
    if abs(s) > 1e-12 and np.all(np.isfinite(lo)) and np.all(lo == 0):
        w = w / s
    return w


def solve(U: mv.Universe, cons: Constraints, objective: str = "max_sharpe", target_sd: Optional[float] = None,
          target_mean: Optional[float] = None, start: Optional[np.ndarray] = None, name: str = "") -> mv.Portfolio:
    """objective: max_sharpe | min_var | max_return.  target_sd is an UPPER bound on sd (the efficient
    point sits on it when it is above the constrained MinVar sd); target_mean is an equality."""
    from scipy.optimize import minimize
    if objective != "min_var" or target_mean is not None:
        U.require_means(f"objective {objective}")
    mu, S, rf, n = U.mu, U.cov, U.rf, U.n
    idx = {nme: i for i, nme in enumerate(U.names)}

    def mean(w):
        return float(w @ mu)

    def var(w):
        return float(w @ S @ w)

    if objective == "max_sharpe":
        f = lambda w: -(mean(w) - rf) / math.sqrt(max(var(w), 1e-18))   # noqa: E731
    elif objective == "min_var":
        f = var
    elif objective == "max_return":
        f = lambda w: -mean(w)   # noqa: E731
    else:
        raise ValueError(objective)
    cons_list = [{"type": "eq", "fun": lambda w: float(w.sum()) - 1.0}]
    if target_mean is not None:
        cons_list.append({"type": "eq", "fun": lambda w: mean(w) - target_mean})
    if target_sd is not None:
        cons_list.append({"type": "ineq", "fun": lambda w: target_sd ** 2 - var(w)})
    for label, names, glo, ghi in cons.groups:
        ids = [idx[nme] for nme in names]
        cons_list.append({"type": "ineq", "fun": lambda w, ids=ids, glo=glo: float(w[ids].sum()) - glo})
        cons_list.append({"type": "ineq", "fun": lambda w, ids=ids, ghi=ghi: ghi - float(w[ids].sum())})
    if cons.turnover_cap is not None and cons.current is not None:
        w0 = np.asarray(cons.current, dtype=float)
        cons_list.append({"type": "ineq", "fun": lambda w: cons.turnover_cap - float(np.sqrt((w - w0) ** 2 + 1e-14).sum())})
    x0 = _feasible_start(U, cons) if start is None else np.asarray(start, dtype=float)
    best = None
    for attempt, s0 in enumerate([x0, _feasible_start(U, cons), np.full(n, 1.0 / n)]):
        res = minimize(f, s0, method="SLSQP", bounds=cons.bounds_for(U), constraints=cons_list,
                       options={"ftol": 1e-13, "maxiter": 3000})
        if res.success and (best is None or res.fun < best.fun - 1e-15):
            best = res
        if best is not None and attempt >= 1:
            break
    if best is None:
        raise ValueError(f"no feasible solution for {objective} under: " + "; ".join(cons.describe())
                         + (f"; target sd {target_sd:.4%}" if target_sd else "") + (f"; target mean {target_mean:.4%}" if target_mean else ""))
    w = best.x
    w[np.abs(w) < 1e-10] = 0.0
    label = name or f"{objective}"
    p = mv.portfolio_stats(U, w, label, "constrained (SLSQP): " + "; ".join(cons.describe()))
    return p


def tangency(U, cons):
    return solve(U, cons, "max_sharpe", name="TG (constrained)")


def gmv(U, cons):
    return solve(U, cons, "min_var", name="MinVar (constrained)")


def max_return(U, cons):
    return solve(U, cons, "max_return", name="MaxReturn (constrained)")


def at_sd(U, cons, target):
    p = solve(U, cons, "max_return", target_sd=float(target), name=f"sd<={mv.pct_label(target)} (constrained)")
    if p.sd < float(target) - 1e-6:
        p.note += f"  NOTE: the risk budget does not bind; under these constraints no portfolio reaches sd {float(target):.2%}, so this is the max-return portfolio (sd {p.sd:.2%})"
    return p


def at_mean(U, cons, target):
    return solve(U, cons, "min_var", target_mean=float(target), name=f"mean={mv.pct_label(target)} (constrained)")


def frontier(U: mv.Universe, cons: Constraints, n: int = 21) -> list[dict]:
    """Constrained efficient frontier: min variance at each target mean between the constrained
    MinVar mean and the constrained maximum-return mean."""
    g = gmv(U, cons)
    mx = max_return(U, cons)
    out = []
    prev = g.weights
    for m in np.linspace(g.mean, mx.mean, n):
        try:
            p = solve(U, cons, "min_var", target_mean=float(m), start=prev)
            prev = p.weights
            out.append({"mean": p.mean, "sd": p.sd, "sharpe": p.sharpe, "weights": [float(x) for x in p.weights]})
        except ValueError:
            continue
    return out


def section(U: mv.Universe, cons: Constraints, targets_sd: Sequence[float] = (), targets_mean: Sequence[float] = (),
            n_frontier: int = 21) -> dict:
    tg = tangency(U, cons)
    g = gmv(U, cons)
    mx = max_return(U, cons)
    sec = {"constraints": cons.describe(), "trivial": cons.is_trivial(),
           "tangency": mv.portfolio_dict(U, tg), "gmv": mv.portfolio_dict(U, g), "max_return": mv.portfolio_dict(U, mx),
           "sd_targets": [], "mean_targets": [], "frontier": frontier(U, cons, n_frontier)}
    for t in targets_sd:
        try:
            sec["sd_targets"].append(mv.portfolio_dict(U, at_sd(U, cons, t)))
        except ValueError as e:
            sec["sd_targets"].append({"name": f"sd<={mv.pct_label(t)}", "error": str(e)})
    for t in targets_mean:
        try:
            sec["mean_targets"].append(mv.portfolio_dict(U, at_mean(U, cons, t)))
        except ValueError as e:
            sec["mean_targets"].append({"name": f"mean={mv.pct_label(t)}", "error": str(e)})
    try:
        u_tg = mv.tangency(U)
        sec["cost_of_constraints"] = {"unconstrained_sharpe": u_tg.sharpe, "constrained_sharpe": tg.sharpe,
                                      "sharpe_given_up": u_tg.sharpe - tg.sharpe}
    except Exception:
        pass
    if cons.current is not None:
        sec["turnover_to_tangency"] = float(np.abs(tg.weights - cons.current).sum())
    return sec


def render(sec: dict, names: Sequence[str]) -> str:
    out = ["CONSTRAINED (long-only / caps / groups / turnover)"]
    out.append("  constraints: " + "; ".join(sec["constraints"]))

    def block(p):
        if "error" in p:
            out.append(f"  {p['name']:<24} infeasible: {p['error']}")
            return
        out.append(f"  {p['name']:<24} mean {p['mean']:.4%}  sd {p['sd']:.4%}  sharpe {p['sharpe']:.4f}")
        out.append("     weights: " + "  ".join(f"{n} {w * 100:6.2f}%" for n, w in zip(names, p["weights_list"])))
        if "NOTE" in p.get("note", ""):
            out.append("     " + p["note"].split("NOTE: ")[1])
    for k in ("tangency", "gmv", "max_return"):
        block(sec[k])
    for p in sec["sd_targets"] + sec["mean_targets"]:
        block(p)
    if sec.get("cost_of_constraints"):
        c = sec["cost_of_constraints"]
        out.append(f"  Sharpe given up to the constraints: {c['sharpe_given_up']:.4f}  (unconstrained {c['unconstrained_sharpe']:.4f} -> constrained {c['constrained_sharpe']:.4f})")
    if "turnover_to_tangency" in sec:
        out.append(f"  turnover from current holdings to the constrained tangency: {sec['turnover_to_tangency']:.1%}")
    fr = sec["frontier"]
    if fr:
        out.append(f"  constrained frontier: {len(fr)} points from mean {fr[0]['mean']:.4%} (sd {fr[0]['sd']:.4%}) to mean {fr[-1]['mean']:.4%} (sd {fr[-1]['sd']:.4%})")
    return "\n".join(out)


def parse_groups(specs: Sequence[str], names: Sequence[str]) -> list:
    """'Tech:AAPL,AMZN:0:0.4' -> ('Tech', ['AAPL','AMZN'], 0.0, 0.4)."""
    out = []
    for s in specs or []:
        parts = s.split(":")
        if len(parts) != 4:
            raise ValueError(f"group '{s}' must be LABEL:NAME1,NAME2:LO:HI")
        label, members, lo, hi = parts
        mem = [m.strip() for m in members.split(",") if m.strip()]
        for m in mem:
            if m not in names:
                raise KeyError(f"group {label}: '{m}' is not in {list(names)}")
        out.append((label.strip(), mem, mv._pct(lo), mv._pct(hi)))
    return out
