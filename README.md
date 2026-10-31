# Finance Models

Two toolkits in plain Python: a portfolio optimizer, and bonds, factors and pro formas. Every result has a closed form or a check behind it.

## Portfolio Optimizer (`portfolio/`)

| File | What it does |
|---|---|
| `mv_engine.py` | Mean variance in closed form: tangency, minimum variance, the efficient frontier, the capital allocation line, two fund mixes, risk contributions, market model betas |
| `constrained.py` | The same problems with limits: long only, bounds per name, group limits, pinned names, a turnover cap |
| `estimation.py` | Steadier inputs: shrinkage and other robust estimates of means and covariances |
| `riskstats.py` | Risk statistics: drawdowns, value at risk, tracking |
| `trades.py` | From current weights to the trades that get you there |
| `ingest.py` | Reads prices, returns, or means and covariances from almost any table |

## Bonds, Factors and Pro Forma (`bonds/`)

| File | What it does |
|---|---|
| `xlbond.py` | Bond price, yield, duration and modified duration, matching Excel's PRICE, YIELD, DURATION and MDURATION |
| `immunize.py` | Immunization: pick bond weights that match the liabilities' present value and duration, then test rate shifts |
| `factors.py` | Factor model regressions: CAPM and Fama French, in sample and out of sample |
| `proforma.py` | Percent of sales pro formas: income statement, balance sheet, the funding plug, and a goal seek on any input |

## Run it

```bash
pip install numpy scipy pandas
python3 -m pytest -q
```

The tests use made-up numbers: the closed form tangency must match a numeric solver, a par bond must price at 100, and yield must round trip.

---

All rights reserved. Shared to show my work.
