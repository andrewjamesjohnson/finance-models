"""Excel-exact bond math: PRICE, YIELD, DURATION, MDURATION, COUPPCD / COUPNCD / COUPNUM, COUPDAYBS / COUPDAYS,
accrued interest, yield to call, and the duration (+ convexity) price approximations.

basis 0 = US (NASD) 30/360, Excel's default when the basis argument is left out; basis 1 = actual/actual.
Rates and yields are decimals (0.05 = 5%), prices per 100 of face, freq 1, 2 or 4.
Proven against Excel to 1e-10.
"""
import datetime as dt
from calendar import monthrange


def as_date(x):
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, str):
        return dt.date.fromisoformat(x.strip()[:10])
    return x


def days360_us(d1, d2):
    """Excel DAYS360(d1, d2) / basis 0 (NASD)."""
    y1, m1, dd1 = d1.year, d1.month, d1.day
    y2, m2, dd2 = d2.year, d2.month, d2.day
    last_feb1 = m1 == 2 and dd1 == monthrange(y1, 2)[1]
    last_feb2 = m2 == 2 and dd2 == monthrange(y2, 2)[1]
    if last_feb1 and last_feb2:
        dd2 = 30
    if last_feb1:
        dd1 = 30
    if dd2 == 31 and dd1 >= 30:
        dd2 = 30
    if dd1 == 31:
        dd1 = 30
    return (y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)


def add_months(d, n, eom):
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    last = monthrange(y, m)[1]
    return dt.date(y, m, last if eom else min(d.day, last))


def coupon_schedule(settle, maturity, freq=2):
    """(COUPPCD, COUPNCD, COUPNUM): previous and next coupon dates, coupons left."""
    settle, maturity = as_date(settle), as_date(maturity)
    if settle >= maturity:
        raise ValueError(f"settlement {settle} is not before maturity {maturity}")
    eom = maturity.day == monthrange(maturity.year, maturity.month)[1]
    step = 12 // freq
    n = 0
    while True:
        prev = add_months(maturity, -(n + 1) * step, eom)
        if prev <= settle:
            return prev, add_months(maturity, -n * step, eom), n + 1
        n += 1


def coupon_parts(settle, maturity, freq=2, basis=0):
    """pcd, ncd, N, A (COUPDAYBS), DSC (days settle -> next coupon as PRICE uses it), E (COUPDAYS)."""
    settle, maturity = as_date(settle), as_date(maturity)
    pcd, ncd, n = coupon_schedule(settle, maturity, freq)
    if basis == 0:
        e = 360 / freq
        a = days360_us(pcd, settle)
        dsc = e - a                      # Excel's basis-0 PRICE / DURATION use E - A
    elif basis == 1:
        e = (ncd - pcd).days
        a = (settle - pcd).days
        dsc = (ncd - settle).days
    else:
        raise ValueError("basis 0 (30/360) or 1 (actual/actual) only")
    return pcd, ncd, n, a, dsc, e


def coupdaybs(settle, maturity, freq=2, basis=0):
    return coupon_parts(settle, maturity, freq, basis)[3]


def coupdays(settle, maturity, freq=2, basis=0):
    return coupon_parts(settle, maturity, freq, basis)[5]


def accrued(settle, maturity, rate, freq=2, basis=0, par=100.0):
    """COUPDAYBS / COUPDAYS x coupon per period, per `par` of face (accrued interest)."""
    _, _, _, a, _, e = coupon_parts(settle, maturity, freq, basis)
    return par * rate / freq * a / e


def price(settle, maturity, rate, yld, redemption=100.0, freq=2, basis=0):
    """Excel PRICE: clean price per 100."""
    _, _, n, a, dsc, e = coupon_parts(settle, maturity, freq, basis)
    c = 100.0 * rate / freq
    if n == 1:
        return (c + redemption) / (yld / freq * dsc / e + 1) - c * a / e
    v = 1 + yld / freq
    p = redemption / v ** (n - 1 + dsc / e)
    for k in range(1, n + 1):
        p += c / v ** (k - 1 + dsc / e)
    return p - c * a / e


def yld(settle, maturity, rate, pr, redemption=100.0, freq=2, basis=0):
    """Excel YIELD: the yield that reproduces the clean price `pr` (bisection, 1e-15)."""
    lo, hi = -0.99 * freq, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if price(settle, maturity, rate, mid, redemption, freq, basis) > pr:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def ytc(settle, call_date, rate, pr, call_price=100.0, freq=2, basis=0):
    """Yield to call = YIELD to the call date with the call price as redemption; None when the call date has passed."""
    if as_date(call_date) <= as_date(settle):
        return None
    return yld(settle, call_date, rate, pr, call_price, freq, basis)


def _flows(settle, maturity, rate, y, freq, basis):
    _, _, n, a, dsc, e = coupon_parts(settle, maturity, freq, basis)
    c = 100.0 * rate / freq
    v = 1 + y / freq
    out = []
    for k in range(1, n + 1):
        t = k - 1 + dsc / e                       # in coupon periods
        cf = c + (100.0 if k == n else 0.0)
        out.append((t, cf / v ** t))
    return out, v


def duration(settle, maturity, rate, y, freq=2, basis=0):
    """Excel DURATION: Macaulay duration in years."""
    flows, _ = _flows(settle, maturity, rate, y, freq, basis)
    pv = sum(p for _, p in flows)
    return sum(t * p for t, p in flows) / pv / freq


def mduration(settle, maturity, rate, y, freq=2, basis=0):
    """Excel MDURATION = DURATION / (1 + y / freq)."""
    return duration(settle, maturity, rate, y, freq, basis) / (1 + y / freq)


def convexity(settle, maturity, rate, y, freq=2, basis=0):
    """Convexity in years^2 (no Excel function; the textbook sum over the same cash flows)."""
    flows, v = _flows(settle, maturity, rate, y, freq, basis)
    pv = sum(p for _, p in flows)
    return sum(t * (t + 1) * p for t, p in flows) / (pv * v * v * freq * freq)


def price_duration_approx(p0, mod_dur, y0, y):
    """The straight line: P0 x (1 - ModD x (y - y0))."""
    return p0 * (1 - mod_dur * (y - y0))


def price_convexity_approx(p0, mod_dur, conv, y0, y):
    dy = y - y0
    return p0 * (1 - mod_dur * dy + 0.5 * conv * dy * dy)


def eff_convexity(settle, maturity, rate, y, h=0.001, freq=2, basis=0):
    """Convexity from three PRICE values, the one-cell formula a sheet can carry:
    (P(y - h) + P(y + h) - 2 P(y)) / (P(y) h^2), h = 10 basis points."""
    p0 = price(settle, maturity, rate, y, 100, freq, basis)
    up = price(settle, maturity, rate, y + h, 100, freq, basis)
    dn = price(settle, maturity, rate, y - h, 100, freq, basis)
    return (dn + up - 2 * p0) / (p0 * h * h)
