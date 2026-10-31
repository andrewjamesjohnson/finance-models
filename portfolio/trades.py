#!/usr/bin/env python3
"""
trades.py - from weights to an order ticket: shares, dollars, cash left, buys and sells from the
current holdings, turnover.  A client has dollars and shares, not weights.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

import mv_engine as mv


def trade_list(names: Sequence[str], target_w: Sequence[float], value: float, prices: Sequence[float],
               current_shares: Optional[Sequence[float]] = None, cash_buffer: float = 0.0,
               fractional: bool = False, min_trade_value: float = 0.0) -> dict:
    names = list(names)
    w = np.asarray(target_w, dtype=float)
    px = np.asarray(prices, dtype=float)
    if np.any(px <= 0):
        raise ValueError("every price must be positive")
    cur = np.zeros(len(names)) if current_shares is None else np.asarray(current_shares, dtype=float)
    cur_value = float(cur @ px)
    investable = value * (1.0 - cash_buffer)
    target_dollars = w * investable
    raw_shares = target_dollars / px
    shares = raw_shares if fractional else np.round(raw_shares)
    # keep negative share counts only where the target weight is negative (a short); otherwise floor at 0
    shares = np.where(w < 0, shares, np.maximum(shares, 0.0))
    trade = shares - cur
    trade_value = trade * px
    if min_trade_value > 0:
        small = np.abs(trade_value) < min_trade_value
        trade = np.where(small, 0.0, trade)
        shares = cur + trade
        trade_value = trade * px
    final_value = float(shares @ px)
    cash_left = value - final_value if current_shares is None else (value - final_value)
    rows = []
    for i, n in enumerate(names):
        rows.append({"name": n, "price": float(px[i]), "target_weight": float(w[i]), "target_dollars": float(target_dollars[i]),
                     "target_shares": float(shares[i]), "current_shares": float(cur[i]), "trade_shares": float(trade[i]),
                     "trade_dollars": float(trade_value[i]), "side": "BUY" if trade[i] > 0 else ("SELL" if trade[i] < 0 else "HOLD"),
                     "final_weight": float(shares[i] * px[i] / value) if value else 0.0,
                     "current_weight": float(cur[i] * px[i] / value) if value else 0.0})
    turnover = float(np.abs(trade_value).sum() / value) if value else 0.0
    return {"value": value, "cash_buffer": cash_buffer, "investable": investable, "fractional": fractional,
            "current_value": cur_value, "rows": rows, "invested": final_value, "cash_left": cash_left,
            "buys": float(trade_value[trade_value > 0].sum()), "sells": float(-trade_value[trade_value < 0].sum()),
            "turnover": turnover, "rounding_drift": [float(r["final_weight"] - r["target_weight"]) for r in rows]}


def section(U: mv.Universe, rep: dict, which: str, value: Optional[float], prices: Optional[Sequence[float]],
            price_date: Optional[str], holdings: Optional[dict], cash_buffer: float = 0.0, fractional: bool = False) -> dict:
    """which: 'TG' | 'MinVar' | 'EW' | 'User' | a target name | 'TG (constrained)' ... resolved from rep."""
    names = U.names
    target = _resolve(rep, which)
    if target is None:
        raise KeyError(f"no portfolio called '{which}' in the report")
    if prices is None:
        if U.prices is None:
            raise ValueError("give --price-now NAME=price,... (no price history to take the last prices from)")
        prices = U.prices[-1]
        price_date = price_date or (U.dates[-1] if U.dates else "last row of the price file")
    cur = None
    if holdings:
        cur = np.array([float(holdings.get(n, 0.0)) for n in names])
        if value is None:
            value = float(cur @ np.asarray(prices))
    if value is None:
        raise ValueError("give --value (portfolio value) or --holdings")
    tl = trade_list(names, target["weights_list"], value, prices, cur, cash_buffer, fractional)
    tl["target_portfolio"] = target["name"]
    tl["price_date"] = price_date
    tl["names"] = names
    return tl


def _resolve(rep: dict, which: str):
    key = which.strip().lower()
    cands = {"tg": rep.get("tangency"), "tangency": rep.get("tangency"), "minvar": rep.get("gmv"), "gmv": rep.get("gmv"),
             "ew": rep.get("equal_weight"), "equal": rep.get("equal_weight"), "user": rep.get("user_portfolio")}
    if key in cands and cands[key]:
        return cands[key]
    for p in rep.get("frontier_sd_targets", []) + rep.get("frontier_mean_targets", []):
        if p["name"].lower() == key:
            return p
    c = rep.get("constrained")
    if c:
        cc = {"tg (constrained)": c["tangency"], "tg-lo": c["tangency"], "minvar (constrained)": c["gmv"], "minvar-lo": c["gmv"]}
        if key in cc:
            return cc[key]
        for p in c["sd_targets"] + c["mean_targets"]:
            if p.get("name", "").lower() == key:
                return p
    ns = rep.get("no_short") or {}
    if key in ("tg (no short)", "tg-ns") and ns.get("tangency"):
        return ns["tangency"]
    return None


def render(tl: dict) -> str:
    out = [f"TRADES to implement {tl['target_portfolio']}  (value ${tl['value']:,.0f}, prices as of {tl['price_date']}, cash buffer {tl['cash_buffer']:.0%}"
           + (", fractional shares" if tl["fractional"] else ", whole shares") + ")"]
    out.append(f"  {'name':<10}{'price':>10}{'target w':>10}{'target $':>14}{'shares':>10}{'current':>10}{'trade':>10}{'trade $':>14}  side   final w")
    for r in tl["rows"]:
        out.append(f"  {r['name']:<10}{r['price']:>10.2f}{r['target_weight']:>10.2%}{r['target_dollars']:>14,.0f}{r['target_shares']:>10,.0f}{r['current_shares']:>10,.0f}{r['trade_shares']:>+10,.0f}{r['trade_dollars']:>+14,.0f}  {r['side']:<5} {r['final_weight']:>7.2%}")
    out.append(f"  invested ${tl['invested']:,.0f}, cash left ${tl['cash_left']:,.0f}; buys ${tl['buys']:,.0f}, sells ${tl['sells']:,.0f}; turnover {tl['turnover']:.1%} of value; "
               f"largest rounding drift {max(abs(x) for x in tl['rounding_drift']):.2%}")
    return "\n".join(out)
