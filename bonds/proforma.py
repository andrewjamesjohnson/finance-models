"""Percent of sales pro formas.

One year = the I/S, the B/S and the plug (Additional Funds Needed = total assets - total liabilities and net worth),
driven by the percent column and these rules:
  sales = prior x (1 + growth); beginning inventory = prior ending; ending inventory = inv% x sales;
  CofGS = cogs% x sales; purchases = CofGS + ending - beginning; purchase discount = disc% x purchases;
  operating expense = opex% x sales; EBIT = gross profit + discount - opex;
  bank interest = bank rate x the year's bank balance; fixed interest = term rate x (current portion + term loan)
  ("we assume debt was paid off on Jan 1"; interest_on="begin" or "average" for the other readings); tax = the case's schedule 0.15x50 + 0.25x25 + 0.34x25 + 0.39x(IBT - 100);
  cash, receivables, property, accrued = their % x sales; payables = purchases x AP days / 365;
  current portion stays, the term loan falls by it; Holtz and trade notes stay; net worth = prior + net income.
Goal seek: any one input (cogs%, the bank balance, growth, opex%, inv%) for AFN = 0, the way an analyst runs
Goal Seek / Solver on the sheet. Cash flow statement: indirect, CFO + CFI + CFF = the change in cash.
"""
from scipy.optimize import brentq

ITEMS = ["sales", "beg", "purch", "avail", "end", "cogs", "gp", "disc", "opex", "ebit", "int_bank", "int_fixed",
         "ibt", "tax", "ni", "cash", "ar", "inv", "ca", "prop", "ta", "bank", "holtz", "trade", "ap", "accr",
         "ltd_cur", "cl", "term", "tl", "nw", "tlnw", "afn"]


def case_tax(ibt):
    """The template's own formula (valid for income before taxes between 100 and 335, in $000)."""
    return 0.15 * 50 + 0.25 * 25 + 0.34 * 25 + 0.39 * (ibt - 100)


DEFAULTS = dict(rate_bank=0.11, rate_fixed=0.10, ap_days=10, year_days=365, interest_on="end")


def year(prev, a):
    """prev: the prior year's dict (needs sales, end, bank, holtz, trade, ltd_cur, term, nw);
    a: assumptions (growth, inv, cogs, disc, opex, cash, ar, prop, accr, bank, rates, ap_days, year_days)."""
    a = {**DEFAULTS, **a}
    y = {}
    y["sales"] = prev["sales"] * (1 + a["growth"])
    y["beg"] = prev["end"]
    y["end"] = a["inv"] * y["sales"]
    y["cogs"] = a["cogs"] * y["sales"]
    y["purch"] = y["cogs"] + y["end"] - y["beg"]
    y["avail"] = y["beg"] + y["purch"]
    y["gp"] = y["sales"] - y["cogs"]
    y["disc"] = a["disc"] * y["purch"]
    y["opex"] = a["opex"] * y["sales"]
    y["ebit"] = y["gp"] + y["disc"] - y["opex"]
    y["ltd_cur"] = prev["ltd_cur"]
    y["term"] = prev["term"] - y["ltd_cur"]
    y["bank"] = a["bank"] if a.get("bank") is not None else prev["bank"]
    y["holtz"] = prev.get("holtz", 0.0)
    y["trade"] = prev.get("trade", 0.0)
    pb, pf = prev["bank"], prev["ltd_cur"] + prev["term"]
    yb, yf = y["bank"], y["ltd_cur"] + y["term"]
    bal = dict(end=(yb, yf), begin=(pb, pf), average=((pb + yb) / 2, (pf + yf) / 2))[a["interest_on"]]
    y["int_bank"] = a["rate_bank"] * bal[0]
    y["int_fixed"] = a["rate_fixed"] * bal[1]
    y["ibt"] = y["ebit"] - y["int_bank"] - y["int_fixed"]
    y["tax"] = case_tax(y["ibt"])
    y["ni"] = y["ibt"] - y["tax"]
    y["cash"] = a["cash"] * y["sales"]
    y["ar"] = a["ar"] * y["sales"]
    y["inv"] = y["end"]
    y["ca"] = y["cash"] + y["ar"] + y["inv"]
    y["prop"] = a["prop"] * y["sales"]
    y["ta"] = y["ca"] + y["prop"]
    y["ap"] = y["purch"] * a["ap_days"] / a["year_days"]
    y["accr"] = a["accr"] * y["sales"]
    y["cl"] = y["bank"] + y["holtz"] + y["trade"] + y["ap"] + y["accr"] + y["ltd_cur"]
    y["tl"] = y["cl"] + y["term"]
    y["nw"] = prev["nw"] + y["ni"]
    y["tlnw"] = y["tl"] + y["nw"]
    y["afn"] = y["ta"] - y["tlnw"]
    return y


def solve(prev, a, var, lo, hi, target=0.0):
    """Goal seek: the value of assumption `var` that makes AFN = target. -> (value, year dict)."""
    f = lambda v: year(prev, {**a, var: v})["afn"] - target
    v = brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500)
    return v, year(prev, {**a, var: v})


SEARCH = dict(cogs=(0.3, 0.99), bank=(-1e6, 1e6), growth=(-0.9, 5.0), opex=(0.0, 0.9), inv=(0.0, 1.0))


def cash_flow(prev, y):
    """Indirect cash flow statement for year y against the prior balance sheet `prev`
    (prev may carry holtz_lt, the long-term Holtz note of the historical sheet)."""
    o = {"ni": y["ni"]}
    o["ar"] = prev["ar"] - y["ar"]
    o["inv"] = prev["inv"] - y["inv"]
    o["ap"] = y["ap"] - prev["ap"]
    o["accr"] = y["accr"] - prev["accr"]
    o["cfo"] = o["ni"] + o["ar"] + o["inv"] + o["ap"] + o["accr"]
    o["ppe"] = prev["prop"] - y["prop"]
    o["cfi"] = o["ppe"]
    o["notes"] = (y["bank"] + y["holtz"] + y["trade"]) - (prev["bank"] + prev["holtz"] + prev["trade"]
                                                         + prev.get("holtz_lt", 0.0))
    o["termloan"] = (y["ltd_cur"] + y["term"]) - (prev["ltd_cur"] + prev["term"])
    o["cff"] = o["notes"] + o["termloan"]
    o["net"] = o["cfo"] + o["cfi"] + o["cff"]
    o["bs_cash"] = y["cash"] - prev["cash"]
    return o
