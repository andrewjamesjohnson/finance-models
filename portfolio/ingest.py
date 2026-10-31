#!/usr/bin/env python3
"""
ingest.py - find the numbers in whatever you were handed, and validate them.

    sniff(path)          -> every block worth knowing about in a workbook / CSV / TSV / pasted text:
                            covariance or correlation matrices, mean / st dev / var tables, weights
                            tables, a risk-free cell, price or return time series (any orientation)
    assemble(blocks,...) -> a mv_engine.Universe built from the best combination of those blocks,
                            with a list of notes saying exactly which cells were used
    validate(U, ...)     -> symmetry, positive-definiteness, duplicate / zero-variance assets,
                            unit sanity (percent vs decimal), rF sanity, singular covariance

Conventions: population moments, simple returns, per-period rates.
Nothing here guesses silently: every normalisation ends up in Universe.notes.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import mv_engine as mv

# --------------------------------------------------------------------------- #
# cell helpers
# --------------------------------------------------------------------------- #
_STAT_PATTERNS = [
    ("mean", re.compile(r"^(mean|means|average|avg|expected( return)?|mu|e\[?r\]?|exp\.? ?ret\.?|return|ret)$", re.I)),
    ("sd", re.compile(r"^(st\.? ?devs?|stdev|std|sd|sigma|volatility|vol|standard deviation|st dev|risk)$", re.I)),
    ("var", re.compile(r"^(var|vars|variance|variances)$", re.I)),
    ("sharpe", re.compile(r"^sharpe( ratio)?$", re.I)),
    ("beta", re.compile(r"^betas?$", re.I)),
]
_RF_PATTERN = re.compile(r"^(r\s*f|rf|r_f|risk[\s-]*free( rate)?|riskless( rate)?|t[\s-]*bills?( rate)?)$", re.I)
_WEIGHT_PATTERN = re.compile(r"weight", re.I)
_TOTAL_PATTERN = re.compile(r"^(total|sum|sum w|100%)$", re.I)


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and not (isinstance(v, float) and math.isnan(v))


def _is_str(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _as_date(v) -> Optional[dt.date]:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s or len(s) < 6 or not re.search(r"\d", s):
            return None
        try:
            import warnings
            import pandas as pd
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = pd.to_datetime(s, errors="coerce")
            if t is pd.NaT or t is None:
                return None
            return t.date()
        except Exception:
            return None
    return None


def _to_num(v):
    """'1,234.5' / '12.3%' / '(0.5)' -> float; else the value unchanged."""
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        neg = s.startswith("(") and s.endswith(")")
        if neg:
            s = s[1:-1]
        pct = s.endswith("%")
        if pct:
            s = s[:-1]
        try:
            f = float(s)
        except ValueError:
            return v
        if pct:
            f /= 100.0
        return -f if neg else f
    return v


# --------------------------------------------------------------------------- #
# grids
# --------------------------------------------------------------------------- #
def load_grids(path: str | Path, text: Optional[str] = None) -> dict[str, list[list]]:
    """{sheet name: 2-D list of cell values}.  xlsx via openpyxl (cached values), anything else
    as delimited text (comma / tab / semicolon sniffed).  `text` bypasses the file."""
    if text is not None:
        return {"paste": _parse_text(text)}
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        from openpyxl import load_workbook
        wb = load_workbook(p, data_only=True, read_only=True)
        grids = {}
        for ws in wb.worksheets:
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            if any(any(c is not None for c in r) for r in rows):
                grids[ws.title] = rows
        return grids
    if p.suffix.lower() == ".xls":
        import pandas as pd
        return {name: df.values.tolist() for name, df in pd.read_excel(p, sheet_name=None, header=None).items()}
    return {p.name: _parse_text(p.read_text(encoding="utf-8-sig", errors="replace"))}


def _parse_text(text: str) -> list[list]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    sample = "\n".join(lines[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        delim = dialect.delimiter
    except csv.Error:
        delim = "\t" if "\t" in sample else ("," if "," in sample else None)
    grid = []
    for ln in lines:
        cells = ln.split(delim) if delim else re.split(r"\s{2,}|\t", ln)
        grid.append([_to_num(c.strip()) if c.strip() != "" else None for c in cells])
    width = max(len(r) for r in grid)
    for r in grid:
        r.extend([None] * (width - len(r)))
    return grid


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    kind: str                      # matrix_cov | matrix_corr | stats | weights | rf | series_prices | series_returns
    sheet: str
    anchor: str                    # A1 address of the block's top-left data cell
    names: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    note: str = ""

    def __str__(self):
        extra = ""
        if self.kind.startswith("matrix"):
            extra = f"{len(self.names)}x{len(self.names)}"
        elif self.kind == "stats":
            extra = "columns " + ", ".join(self.data.keys())
        elif self.kind == "rf":
            extra = f"= {self.data['value']}"
        elif self.kind.startswith("series"):
            extra = f"{self.data['T']} rows"
        elif self.kind == "weights":
            extra = ", ".join(f"{n} {w:.2%}" for n, w in zip(self.names, self.data["values"]))
        return f"{self.kind:<14} {self.sheet}!{self.anchor:<8} {extra}  {('[' + ', '.join(self.names[:8]) + (', …' if len(self.names) > 8 else '') + ']') if self.names and self.kind != 'weights' else ''}  {self.note}".rstrip()


def _a1(r: int, c: int) -> str:
    from openpyxl.utils import get_column_letter
    return f"{get_column_letter(c + 1)}{r + 1}"


def _cell(grid, r, c):
    if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[r]):
        return None
    return grid[r][c]


def _find_matrices(grid, sheet) -> list[Block]:
    out = []
    R = len(grid)
    for r in range(R):
        row = grid[r]
        c = 0
        while c < len(row):
            if _is_str(row[c]) and not _is_num(_to_num(row[c])):
                c2 = c
                while c2 + 1 < len(row) and _is_str(row[c2 + 1]) and not _is_num(_to_num(row[c2 + 1])):
                    c2 += 1
                labels = [_norm(x) for x in row[c:c2 + 1]]
                k = len(labels)
                if k >= 2 and len(set(labels)) == k:
                    # column labels must repeat down the column just left of the block
                    lc = c - 1
                    if lc >= 0:
                        col_labels = [_norm(_cell(grid, r + 1 + i, lc)) if _is_str(_cell(grid, r + 1 + i, lc)) else None for i in range(k)]
                        if col_labels == labels:
                            vals = [[_to_num(_cell(grid, r + 1 + i, c + j)) for j in range(k)] for i in range(k)]
                            if all(_is_num(v) for rw in vals for v in rw):
                                M = np.array(vals, dtype=float)
                                diag_one = np.allclose(np.diag(M), 1.0, atol=1e-9)
                                sym = np.allclose(M, M.T, atol=1e-9 * max(1.0, np.abs(M).max()))
                                kind = "matrix_corr" if diag_one else "matrix_cov"
                                out.append(Block(kind, sheet, _a1(r + 1, c), [str(row[c + j]).strip() for j in range(k)],
                                                 {"matrix": M, "symmetric": sym}, "" if sym else "NOT symmetric"))
                c = c2 + 1
            else:
                c += 1
    return out


def _stat_key(v) -> Optional[str]:
    if not _is_str(v):
        return None
    s = _norm(v)
    for key, pat in _STAT_PATTERNS:
        if pat.match(s):
            return key
    return None


def _find_stats(grid, sheet) -> list[Block]:
    out = []
    seen = set()
    for r in range(len(grid)):
        row = grid[r]
        for c in range(len(row)):
            key = _stat_key(row[c])
            if not key or (r, c) in seen:
                continue
            # header run: contiguous stat headers to the right
            c0 = c
            while c0 - 1 >= 0 and _stat_key(row[c0 - 1]):
                c0 -= 1
            c1 = c
            while c1 + 1 < len(row) and _stat_key(row[c1 + 1]):
                c1 += 1
            for cc in range(c0, c1 + 1):
                seen.add((r, cc))
            keys = [_stat_key(row[cc]) for cc in range(c0, c1 + 1)]
            # label column: first column left of c0 holding a string in row r+1
            lab = None
            for lc in range(c0 - 1, max(-1, c0 - 4), -1):
                if _is_str(_cell(grid, r + 1, lc)) and not _is_num(_to_num(_cell(grid, r + 1, lc))):
                    lab = lc
                    break
            if lab is None:
                continue
            names, cols = [], {k: [] for k in keys}
            rr = r + 1
            while _is_str(_cell(grid, rr, lab)) and not _is_num(_to_num(_cell(grid, rr, lab))):
                vals = [_to_num(_cell(grid, rr, cc)) for cc in range(c0, c1 + 1)]
                if not any(_is_num(v) for v in vals):
                    break
                if _TOTAL_PATTERN.match(_norm(_cell(grid, rr, lab))):
                    break
                names.append(str(_cell(grid, rr, lab)).strip())
                for k, v in zip(keys, vals):
                    cols[k].append(float(v) if _is_num(v) else np.nan)
                rr += 1
            if len(names) >= 2:
                out.append(Block("stats", sheet, _a1(r + 1, c0), names, {k: np.array(v) for k, v in cols.items()}))
    return out


def _find_weights(grid, sheet) -> list[Block]:
    out = []
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            v = grid[r][c]
            if _is_str(v) and _WEIGHT_PATTERN.search(v) and not _stat_key(v):
                names, vals = [], []
                rr = r + 1
                while _is_str(_cell(grid, rr, c)) and not _is_num(_to_num(_cell(grid, rr, c))):
                    if _TOTAL_PATTERN.match(_norm(_cell(grid, rr, c))):
                        break
                    w = _to_num(_cell(grid, rr, c + 1))
                    if not _is_num(w):
                        break
                    names.append(str(_cell(grid, rr, c)).strip())
                    vals.append(float(w))
                    rr += 1
                if len(names) >= 2:
                    out.append(Block("weights", sheet, _a1(r + 1, c + 1), names, {"values": np.array(vals)},
                                     f"sum {sum(vals):.4f}"))
    return out


def _find_rf(grid, sheet) -> list[Block]:
    out = []
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            v = grid[r][c]
            if _is_str(v) and _RF_PATTERN.match(_norm(v)):
                for (rr, cc) in ((r, c + 1), (r + 1, c), (r, c + 2)):
                    x = _to_num(_cell(grid, rr, cc))
                    if _is_num(x):
                        out.append(Block("rf", sheet, _a1(rr, cc), [], {"value": float(x)}))
                        break
    return out


def _find_series(grid, sheet) -> list[Block]:
    """Date column (or date row) with >= 8 numeric columns/rows alongside."""
    out = []
    R = len(grid)
    C = max((len(r) for r in grid), default=0)
    MIN = 8
    # vertical: dates down a column
    for c in range(C):
        r = 0
        while r < R:
            if _as_date(_cell(grid, r, c)) is not None or (_is_num(_cell(grid, r, c)) and 20000 < _cell(grid, r, c) < 80000):
                r0 = r
                while r + 1 < R and (_as_date(_cell(grid, r + 1, c)) is not None or (_is_num(_cell(grid, r + 1, c)) and 20000 < _cell(grid, r + 1, c) < 80000)):
                    r += 1
                if r - r0 + 1 >= MIN:
                    blk = _series_from_vertical(grid, sheet, c, r0, r)
                    if blk:
                        out.append(blk)
                r += 1
            else:
                r += 1
    # horizontal: dates across a row
    for r in range(R):
        row = grid[r]
        c = 0
        while c < len(row):
            if _as_date(row[c]) is not None:
                c0 = c
                while c + 1 < len(row) and _as_date(row[c + 1]) is not None:
                    c += 1
                if c - c0 + 1 >= MIN:
                    blk = _series_from_horizontal(grid, sheet, r, c0, c)
                    if blk:
                        out.append(blk)
                c += 1
            else:
                c += 1
    return out


def _dates_from(cells):
    ds = []
    for v in cells:
        d = _as_date(v)
        if d is None and _is_num(v):
            from openpyxl.utils.datetime import from_excel
            d = from_excel(v).date()
        ds.append(d)
    return ds


def _series_kind(M: np.ndarray) -> str:
    finite = M[np.isfinite(M)]
    if finite.size == 0:
        return "series_returns"
    if finite.min() > 0 and np.median(finite) > 1.5:
        return "series_prices"
    return "series_returns"


def _series_from_vertical(grid, sheet, c, r0, r1):
    dates = _dates_from([_cell(grid, r, c) for r in range(r0, r1 + 1)])
    cols, names = [], []
    C = max(len(grid[r]) for r in range(r0, r1 + 1))
    for cc in range(C):
        if cc == c:
            continue
        vals = [_to_num(_cell(grid, r, cc)) for r in range(r0, r1 + 1)]
        n_num = sum(1 for v in vals if _is_num(v))
        if n_num >= 0.9 * len(vals) and n_num >= MIN_ROWS(len(vals)):
            hdr = _cell(grid, r0 - 1, cc)
            if not _is_str(hdr):
                hdr = _cell(grid, r0 - 2, cc)
            names.append(str(hdr).strip() if _is_str(hdr) else f"col{_a1(r0, cc)}")
            cols.append([float(v) if _is_num(v) else np.nan for v in vals])
    if len(cols) < 1:
        return None
    M = np.array(cols, dtype=float).T
    kind = _series_kind(M)
    return Block(kind, sheet, _a1(r0, c), names, {"dates": dates, "matrix": M, "T": M.shape[0]})


def _series_from_horizontal(grid, sheet, r, c0, c1):
    dates = _dates_from([grid[r][c] for c in range(c0, c1 + 1)])
    rows, names = [], []
    for rr in range(r + 1, len(grid)):
        vals = [_to_num(_cell(grid, rr, c)) for c in range(c0, c1 + 1)]
        n_num = sum(1 for v in vals if _is_num(v))
        if n_num >= 0.9 * len(vals):
            lab = None
            for lc in range(c0 - 1, max(-1, c0 - 4), -1):
                if _is_str(_cell(grid, rr, lc)):
                    lab = str(_cell(grid, rr, lc)).strip()
                    break
            names.append(lab or f"row{rr + 1}")
            rows.append([float(v) if _is_num(v) else np.nan for v in vals])
        elif rows:
            break
    if not rows:
        return None
    M = np.array(rows, dtype=float).T
    kind = _series_kind(M)
    return Block(kind, sheet, _a1(r, c0), names, {"dates": dates, "matrix": M, "T": M.shape[0]}, "transposed (dates across)")


def MIN_ROWS(n):
    return max(4, int(0.9 * n))


def sniff(path: str | Path = None, text: Optional[str] = None, sheet: Optional[str] = None) -> list[Block]:
    grids = load_grids(path, text)
    if sheet:
        if sheet not in grids:
            raise KeyError(f"sheet '{sheet}' not in {list(grids)}")
        grids = {sheet: grids[sheet]}
    blocks: list[Block] = []
    for name, grid in grids.items():
        if not grid:
            continue
        blocks += _find_series(grid, name)
        blocks += _find_matrices(grid, name)
        blocks += _find_stats(grid, name)
        blocks += _find_weights(grid, name)
        blocks += _find_rf(grid, name)
    return blocks


# --------------------------------------------------------------------------- #
# units and normalisation
# --------------------------------------------------------------------------- #
def _percent_scale(kind: str, values: np.ndarray) -> tuple[float, str]:
    """Decide whether numbers are in percent (1.23 meaning 1.23%).  Returns (divisor, note)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0, ""
    if kind in ("sd", "mean", "returns"):
        med = float(np.median(np.abs(v)))
        if kind == "sd" and med > 1.0:
            return 100.0, f"st devs look like percents (median {med:.2f}); divided by 100"
        if kind == "returns" and med > 0.5:
            return 100.0, f"returns look like percents (median |r| {med:.2f}); divided by 100"
        if kind == "mean" and med > 0.5:
            return 100.0, f"means look like percents (median |mean| {med:.2f}); divided by 100"
    if kind == "var" and float(np.median(v)) > 1.0:
        return 1e4, "variances look like percent-squared; divided by 10,000"
    if kind == "cov" and float(np.median(np.diag(v.reshape(int(math.sqrt(v.size)), -1)) if v.ndim == 1 else np.diag(v))) > 1.0:
        return 1e4, "covariance diagonal looks like percent-squared; divided by 10,000"
    return 1.0, ""


def normalise_series(dates: Sequence, M: np.ndarray, names: Sequence[str], kind: str, notes: list[str],
                     resample: Optional[str] = None, percent: Optional[bool] = None,
                     start: Optional[str] = None, end: Optional[str] = None, last: Optional[int] = None):
    """Sort ascending by date, drop rows with holes, optionally resample, apply the date window, fix percent
    units.  One implementation for every input path: this delegates to mv_engine.normalise_series."""
    ds = None
    if dates is not None:
        ds = [d if d is None else (d.isoformat() if hasattr(d, "isoformat") else str(d)) for d in dates]
    return mv.normalise_series(ds, M, names, "prices" if kind == "series_prices" else "returns", notes,
                               resample=resample, percent=percent, start=start, end=end, last=last)


# --------------------------------------------------------------------------- #
# assemble a universe
# --------------------------------------------------------------------------- #
def assemble(blocks: list[Block], rf: Optional[float] = None, market: Optional[str] = None,
             ddof: int = 0, exclude: Sequence[str] = (), include: Optional[Sequence[str]] = None,
             prefer: str = "auto", resample: Optional[str] = None, percent: Optional[bool] = None,
             return_type: str = "simple", periods_per_year: Optional[int] = None,
             weights_block: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None,
             last: Optional[int] = None) -> tuple[mv.Universe, Optional[np.ndarray]]:
    """Pick the best blocks and build a Universe.  Returns (U, weights or None).

    prefer: 'auto' (time series if present, else moments), 'series', 'moments'.
    weights_block: 'first' or an anchor like 'Four!C5' to take a weights table as the user portfolio.
    start / end / last: date window applied to a price or return series (see mv_engine.normalise_series).
    """
    notes: list[str] = []
    series = [b for b in blocks if b.kind.startswith("series")]
    covs = [b for b in blocks if b.kind == "matrix_cov"]
    corrs = [b for b in blocks if b.kind == "matrix_corr"]
    stats = [b for b in blocks if b.kind == "stats"]
    rfs = [b for b in blocks if b.kind == "rf"]
    wts = [b for b in blocks if b.kind == "weights"]

    U = None
    if series and prefer in ("auto", "series"):
        prices = [b for b in series if b.kind == "series_prices"]
        ser = max(prices, key=lambda b: b.data["T"] * len(b.names)) if prices else max(series, key=lambda b: b.data["T"] * len(b.names))
        names = [str(n) for n in ser.names]
        dates, M = normalise_series(ser.data["dates"], ser.data["matrix"], names, ser.kind, notes, resample, percent,
                                    start=start, end=end, last=last)
        cols = list(range(len(names)))
        mkt = None
        market = mv.resolve_market(market, names, notes)
        if market:
            mi = names.index(market)
            mkt = M[:, mi]
            cols = [i for i in cols if i != mi]
        if include:
            cols = [names.index(n) for n in include]
        cols = [i for i in cols if names[i] not in set(exclude)]
        X = M[:, cols]
        use_names = [names[i] for i in cols]
        if ser.kind == "series_prices":
            R = mv.log_returns(X) if return_type == "log" else mv.simple_returns(X)
            Rm = None if mkt is None else (mv.log_returns(mkt) if return_type == "log" else mv.simple_returns(mkt))
            P, Pm = X, mkt
        else:
            R, Rm, P, Pm = X, mkt, None, None
        mu, cov = mv.moments(R, ddof)
        U = mv.Universe(use_names, mu, cov, rf=0.0, ddof=ddof, returns=R, prices=P, dates=dates, market_name=market,
                        market_returns=Rm, market_prices=Pm, periods_per_year=periods_per_year, return_type=return_type,
                        source=f"{ser.sheet}!{ser.anchor} ({ser.kind.split('_')[1]}, {R.shape[0]} returns)")
        notes.insert(0, f"{ser.kind.replace('series_', '')} block at {ser.sheet}!{ser.anchor}: {', '.join(names)}")
    elif covs or corrs or stats:
        matrix = None
        if covs:
            matrix = max(covs, key=lambda b: len(b.names))
            names = list(matrix.names)
            M = matrix.data["matrix"].copy()
            div, note = _percent_scale("cov", M)
            if percent is True:
                div, note = 1e4, "covariance divided by 10,000 (--percent-units)"
            if div != 1.0 and percent is not False:
                M = M / div
                notes.append(note)
            cov = M
            notes.append(f"covariance from {matrix.sheet}!{matrix.anchor} ({len(names)}x{len(names)})")
        elif corrs:
            matrix = max(corrs, key=lambda b: len(b.names))
            names = list(matrix.names)
            cov = None
            notes.append(f"correlation from {matrix.sheet}!{matrix.anchor}")
        else:
            names = list(stats[0].names)
            cov = None
        st = _best_stats(stats, names)
        mu = sd = None
        if st is not None:
            lut = {_norm(n): i for i, n in enumerate(st.names)}
            miss = [n for n in names if _norm(n) not in lut]
            if miss and stats and not covs and not corrs:
                pass
            if not miss:
                if "mean" in st.data:
                    mu = np.array([st.data["mean"][lut[_norm(n)]] for n in names])
                    div, note = _percent_scale("mean", mu)
                    if percent is True:
                        div, note = 100.0, "means divided by 100 (--percent-units)"
                    if div != 1.0 and percent is not False:
                        mu = mu / div
                        notes.append(note)
                    notes.append(f"means from {st.sheet}!{st.anchor}")
                if "sd" in st.data:
                    sd = np.array([st.data["sd"][lut[_norm(n)]] for n in names])
                    div, note = _percent_scale("sd", sd)
                    if percent is True:
                        div, note = 100.0, "st devs divided by 100 (--percent-units)"
                    if div != 1.0 and percent is not False:
                        sd = sd / div
                        notes.append(note)
                    notes.append(f"st devs from {st.sheet}!{st.anchor}")
                elif "var" in st.data:
                    var = np.array([st.data["var"][lut[_norm(n)]] for n in names])
                    div, note = _percent_scale("var", var)
                    if div != 1.0 and percent is not False:
                        var = var / div
                        notes.append(note)
                    sd = np.sqrt(var)
                    notes.append(f"st devs = sqrt(variances) from {st.sheet}!{st.anchor}")
                extra = [n for n in st.names if _norm(n) not in {_norm(x) for x in names}]
                if extra:
                    notes.append(f"ignored rows in the stats table that are not assets: {', '.join(extra)}")
            else:
                notes.append(f"stats table at {st.sheet}!{st.anchor} lacks {miss}; not used")
        if cov is None and corrs:
            if sd is None:
                raise ValueError("a correlation matrix needs st devs (a 'St Dev' column) to become a covariance matrix")
            cov = mv.cov_from_corr(sd, matrix.data["matrix"])
            notes.append("covariance = corr * sd_i * sd_j")
        if cov is None:
            raise ValueError("found means/st devs but no covariance or correlation matrix; pass --assume-corr X to proceed with a constant correlation")
        if sd is not None and cov is not None and not np.allclose(np.sqrt(np.diag(cov)), sd, rtol=2e-2):
            notes.append("WARNING: the St Dev column does not match sqrt(diag(cov)); the matrix wins")
        keep = [n for n in names if n not in set(exclude)]
        if include:
            keep = list(include)
        idx = [names.index(n) for n in keep]
        U = mv.Universe(keep, (mu[idx] if mu is not None else None), cov[np.ix_(idx, idx)], rf=0.0, ddof=ddof,
                        periods_per_year=periods_per_year, source=f"{matrix.sheet}!{matrix.anchor}" if matrix else "stats")
    else:
        raise ValueError("no usable block found: need a price/return series, or a covariance/correlation matrix with a means table")

    # rF
    if rf is not None:
        U.rf = float(rf)
    elif rfs:
        U.rf = rfs[0].data["value"]
        notes.append(f"rF from {rfs[0].sheet}!{rfs[0].anchor} = {U.rf}")
    else:
        notes.append("no rF found; using 0 (pass --rf)")
    # weights
    w = None
    if weights_block and wts:
        blk = wts[0] if weights_block == "first" else next((b for b in wts if f"{b.sheet}!{b.anchor}".lower().startswith(weights_block.lower())), None)
        if blk is None:
            raise KeyError(f"weights block '{weights_block}' not found; have " + ", ".join(f"{b.sheet}!{b.anchor}" for b in wts))
        lut = {_norm(n): v for n, v in zip(blk.names, blk.data["values"])}
        try:
            w = np.array([lut[_norm(n)] for n in U.names])
            notes.append(f"user weights from {blk.sheet}!{blk.anchor}")
        except KeyError as e:
            raise KeyError(f"weights block {blk.sheet}!{blk.anchor} has no entry for {e}")
    elif wts:
        notes.append("weights tables found (" + "; ".join(f"{b.sheet}!{b.anchor}" for b in wts) + "); pass --use-weights first to use one")
    U.notes = notes
    return U, w


def _best_stats(stats: list[Block], names: Sequence[str]) -> Optional[Block]:
    if not stats:
        return None
    want = {_norm(n) for n in names}

    def score(b):
        have = {_norm(n) for n in b.names}
        return (len(want & have), ("mean" in b.data) + ("sd" in b.data) + ("var" in b.data))
    return max(stats, key=score)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate(U: mv.Universe, fix_psd: bool = False, strict: bool = True) -> list[str]:
    """Checks that catch hand-typed and mis-scaled inputs.  Appends to U.notes; raises on fatal
    problems unless strict=False.  Returns the list of messages added."""
    msgs: list[str] = []
    n = U.n
    # names
    clean = [str(x).strip() for x in U.names]
    if any(c == "" for c in clean):
        raise ValueError("an asset has an empty name")
    if len(set(_norm(c) for c in clean)) != n:
        raise ValueError(f"duplicate asset names: {clean}")
    U.names = [("'" + c) if c[:1] in "=+-@" else c for c in clean]
    # symmetry
    C = np.asarray(U.cov, dtype=float)
    if C.shape != (n, n):
        raise ValueError(f"covariance is {C.shape}, expected {(n, n)}")
    asym = np.abs(C - C.T).max()
    scale = max(np.abs(C).max(), 1e-18)
    if asym > 1e-6 * scale:
        i, j = np.unravel_index(np.argmax(np.abs(C - C.T)), C.shape)
        msg = f"covariance is not symmetric: [{U.names[i]},{U.names[j]}] = {C[i, j]:.8g} but [{U.names[j]},{U.names[i]}] = {C[j, i]:.8g}"
        if strict:
            raise ValueError(msg + "  (fix the typo; a covariance matrix must be symmetric)")
        msgs.append("WARNING " + msg + "; symmetrised")
    if asym > 0:
        C = (C + C.T) / 2.0
    # variances
    d = np.diag(C)
    zero = [U.names[i] for i in range(n) if d[i] <= 0]
    if zero:
        raise ValueError(f"zero or negative variance for {zero}: a constant series or a typo")
    # correlations in range, duplicates
    sd = np.sqrt(d)
    corr = C / np.outer(sd, sd)
    off = corr - np.eye(n)
    if np.abs(off).max() > 1.0 + 1e-9:
        i, j = np.unravel_index(np.argmax(np.abs(off)), corr.shape)
        raise ValueError(f"implied correlation between {U.names[i]} and {U.names[j]} is {corr[i, j]:.4f}, outside [-1, 1]: a typo in the matrix")
    dup = [(U.names[i], U.names[j]) for i in range(n) for j in range(i + 1, n) if abs(corr[i, j]) > 0.9999]
    if dup:
        msgs.append("WARNING near-perfect correlation (duplicate series?): " + ", ".join(f"{a}/{b}" for a, b in dup) + "; the covariance is singular, drop one")
    # positive definiteness
    eig = np.linalg.eigvalsh(C)
    if eig.min() < -1e-10 * scale:
        if fix_psd:
            w, V = np.linalg.eigh(C)
            w = np.clip(w, 1e-12 * scale, None)
            C = V @ np.diag(w) @ V.T
            C = (C + C.T) / 2.0
            msgs.append(f"covariance was not positive semi-definite (min eigenvalue {eig.min():.3g}); clipped to the nearest PSD matrix (--fix-psd)")
        else:
            raise ValueError(f"covariance is not positive semi-definite (min eigenvalue {eig.min():.3g}): a hand-typed matrix with inconsistent entries. Re-check it, or pass --fix-psd to clip it to the nearest valid matrix")
    cond = float(eig.max() / max(eig.min(), 1e-300))
    if cond > 1e12:
        msgs.append(f"WARNING covariance is ill-conditioned (condition number {cond:.2g}); closed forms may be unstable")
    U.cov = C
    if U.T is not None and U.T < n + 1:
        msgs.append(f"WARNING only {U.T} observations for {n} assets; the covariance is singular or nearly so")
    # units and rates
    if U.mu is not None and np.abs(U.mu).max() > 0.5:
        msgs.append(f"WARNING a mean of {np.abs(U.mu).max():.3g} per period looks like a percent or an annual figure")
    if sd.max() > 1.0:
        msgs.append(f"WARNING a st dev of {sd.max():.3g} per period looks like a percent")
    if abs(U.rf) > 0.02:
        msgs.append(f"WARNING rF = {U.rf:.4g} per period is high; an annual rate typed as monthly? (0.001 = 0.10%/month)")
    if U.mu is not None:
        try:
            g = mv.gmv(U)
            if U.rf >= g.mean:
                msgs.append(f"WARNING rF ({U.rf:.4%}) is at or above the MinVar mean ({g.mean:.4%}); the tangency flips to the lower branch")
        except Exception:
            pass
    U.notes = list(getattr(U, "notes", []) or []) + msgs
    return msgs


def describe(blocks: list[Block]) -> str:
    if not blocks:
        return "nothing recognisable found"
    return "\n".join(str(b) for b in blocks)
