#!/usr/bin/env python3
"""
estimation.py - estimation error, the thing that makes textbook Markowitz unusable as-is.

  ledoit_wolf_cc      Ledoit-Wolf (2004) shrinkage of the sample covariance toward the constant-
                      correlation target ("Honey, I shrunk the sample covariance matrix")
  bayes_stein         Jorion (1986) shrinkage of the means toward the minimum-variance mean
  bootstrap           resampled tangency / MinVar weights: percentiles instead of one number
  implied_returns     reverse optimisation: the expected returns that make a given portfolio optimal
  black_litterman     market-implied prior + views -> posterior means and covariance (He-Litterman)
  section / render    the report block comparing TG under every estimator

Additive: mv_engine is untouched; every result is a new Universe or a plain dict.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

import mv_engine as mv


# --------------------------------------------------------------------------- #
def ledoit_wolf_cc(R: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Returns (shrunk covariance, delta, target F).  Population sample covariance as the base."""
    R = np.asarray(R, dtype=float)
    T, N = R.shape
    X = R - R.mean(axis=0)
    S = X.T @ X / T
    sd = np.sqrt(np.diag(S))
    corr = S / np.outer(sd, sd)
    rbar = (corr.sum() - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, np.diag(S))
    # pi: asymptotic variances of the sample covariance entries
    Y = X ** 2
    pi_mat = (Y.T @ Y) / T - S ** 2
    pi_hat = float(pi_mat.sum())
    # rho: sum of asymptotic covariances between shrinkage target and sample entries
    term1 = ((X ** 3).T @ X) / T
    help_ = term1 - np.outer(np.diag(S), np.ones(N)) * S
    np.fill_diagonal(help_, 0.0)
    rho_diag = float(np.diag(pi_mat).sum())
    theta = help_
    rho_off = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rho_off += 0.5 * rbar * (math.sqrt(S[j, j] / S[i, i]) * theta[i, j] + math.sqrt(S[i, i] / S[j, j]) * theta[j, i])
    rho_hat = rho_diag + rho_off
    gamma = float(((F - S) ** 2).sum())
    kappa = (pi_hat - rho_hat) / gamma if gamma > 0 else 0.0
    delta = max(0.0, min(1.0, kappa / T))
    Sigma = delta * F + (1.0 - delta) * S
    return Sigma, float(delta), F


def bayes_stein(mu: np.ndarray, cov: np.ndarray, T: int) -> tuple[np.ndarray, float, float]:
    """Jorion's Bayes-Stein means: shrink toward the minimum-variance portfolio's mean.
    Returns (mu_bs, nu, mu_gmv)."""
    mu = np.asarray(mu, dtype=float)
    Si = np.linalg.inv(cov)
    one = np.ones(len(mu))
    mu_g = float(one @ Si @ mu / (one @ Si @ one))
    d = mu - mu_g
    N = len(mu)
    nu = (N + 2) / ((N + 2) + T * float(d @ Si @ d))
    nu = max(0.0, min(1.0, nu))
    return (1.0 - nu) * mu + nu * mu_g, float(nu), mu_g


def bootstrap(U: mv.Universe, n: int = 500, seed: int = 0, T: Optional[int] = None,
              long_only: bool = False) -> dict:
    """Resample the returns (rows with replacement) or, without returns, draw from N(mu, cov)
    with T observations.  Reports percentiles of the tangency and MinVar weights and Sharpes."""
    rng = np.random.default_rng(seed)
    N = U.n
    if U.returns is not None:
        R = U.returns
        Tn = R.shape[0]
        draw = lambda: R[rng.integers(0, Tn, Tn)]  # noqa: E731
        kind = f"nonparametric, {Tn} rows resampled with replacement"
    else:
        Tn = T or 60
        draw = lambda: rng.multivariate_normal(U.mu, U.cov, size=Tn)  # noqa: E731
        kind = f"parametric, {Tn} draws from N(mu, cov)"
    tg_w, g_w, tg_s = [], [], []
    for _ in range(n):
        Rb = draw()
        mu_b, cov_b = mv.moments(Rb, U.ddof)
        Ub = mv.Universe(U.names, mu_b, cov_b, rf=U.rf, ddof=U.ddof)
        try:
            if long_only:
                import constrained as cn
                c = cn.Constraints(long_only=True)
                t = cn.tangency(Ub, c)
                g = cn.gmv(Ub, c)
            else:
                t = mv.tangency(Ub)
                g = mv.gmv(Ub)
        except Exception:
            continue
        tg_w.append(t.weights)
        g_w.append(g.weights)
        tg_s.append(t.sharpe)
    tg_w, g_w, tg_s = np.array(tg_w), np.array(g_w), np.array(tg_s)

    def pct(a):
        return {"p5": np.percentile(a, 5, axis=0).tolist(), "p25": np.percentile(a, 25, axis=0).tolist(),
                "p50": np.percentile(a, 50, axis=0).tolist(), "p75": np.percentile(a, 75, axis=0).tolist(),
                "p95": np.percentile(a, 95, axis=0).tolist(), "mean": a.mean(axis=0).tolist()}
    return {"kind": kind, "n": int(len(tg_s)), "long_only": long_only,
            "tangency": pct(tg_w), "gmv": pct(g_w),
            "tangency_sharpe": {"p5": float(np.percentile(tg_s, 5)), "p50": float(np.percentile(tg_s, 50)), "p95": float(np.percentile(tg_s, 95))},
            "tangency_weight_sd": tg_w.std(axis=0).tolist()}


def implied_returns(U: mv.Universe, w: Sequence[float], risk_aversion: Optional[float] = None) -> dict:
    """Reverse optimisation: pi = lambda * Cov w are the excess returns that make w the tangency
    portfolio.  lambda defaults to the value that makes w's implied mean equal its sample mean."""
    w = np.asarray(w, dtype=float)
    var = float(w @ U.cov @ w)
    if risk_aversion is None:
        if U.mu is None:
            raise ValueError("give risk_aversion when there are no sample means")
        lam = (float(w @ U.mu) - U.rf) / var
    else:
        lam = float(risk_aversion)
    pi = lam * (U.cov @ w)
    return {"lambda": lam, "implied_excess": pi.tolist(), "implied_means": (U.rf + pi).tolist(),
            "sample_means": (U.mu.tolist() if U.mu is not None else None),
            "portfolio_mean": float(U.rf + w @ pi), "portfolio_sd": math.sqrt(var)}


def parse_view(spec: str, names: Sequence[str]) -> tuple[dict, float, Optional[float]]:
    """'AAPL=2%' | 'AAPL-AMZN=1%' | 'AAPL=2%@0.8' (confidence) -> (coefficients, value, confidence)."""
    conf = None
    if "@" in spec:
        spec, c = spec.rsplit("@", 1)
        conf = float(c)
    lhs, rhs = spec.split("=")
    val = mv._pct(rhs.strip())
    coefs = {}
    lhs = lhs.replace(" ", "")
    tokens = lhs.replace("-", "+-").split("+")
    for tok in tokens:
        if not tok:
            continue
        sign = -1.0 if tok.startswith("-") else 1.0
        nm = tok.lstrip("-").lstrip("+")
        if nm not in names:
            raise KeyError(f"view '{spec}': '{nm}' is not in {list(names)}")
        coefs[nm] = coefs.get(nm, 0.0) + sign
    return coefs, val, conf


def black_litterman(U: mv.Universe, w_mkt: Sequence[float], views: Sequence[tuple], tau: float = 0.05,
                    risk_aversion: Optional[float] = None, default_confidence: float = 0.5) -> dict:
    """views: [(coefs {name: +-1}, value, confidence or None)].  Absolute views are PER-PERIOD
    total returns (the prior pi is in excess terms; the view is converted).  Confidence c in (0, 1]
    scales the view variance: omega_k = (1-c)/c * (P tau Sigma P')_k  (c = 1 -> near certainty)."""
    S = U.cov
    N = U.n
    w_mkt = np.asarray(w_mkt, dtype=float)
    if risk_aversion is None:
        if U.mu is None:
            raise ValueError("give risk_aversion when there are no sample means")
        lam = (float(w_mkt @ U.mu) - U.rf) / float(w_mkt @ S @ w_mkt)
    else:
        lam = float(risk_aversion)
    pi = lam * (S @ w_mkt)                                    # prior excess returns
    if not views:
        mu_post = U.rf + pi
        return {"lambda": lam, "tau": tau, "prior_excess": pi.tolist(), "posterior_means": mu_post.tolist(),
                "posterior_cov": S.tolist(), "views": [], "note": "no views: posterior = market-implied prior"}
    P = np.zeros((len(views), N))
    Q = np.zeros(len(views))
    omega = np.zeros(len(views))
    tS = tau * S
    desc = []
    for k, (coefs, val, conf) in enumerate(views):
        for nm, cval in coefs.items():
            P[k, U.index(nm)] = cval
        absolute = abs(sum(coefs.values())) > 1e-12
        Q[k] = (val - U.rf) if absolute else val                # absolute views: total -> excess
        c = conf if conf is not None else default_confidence
        c = min(max(c, 1e-3), 1.0)
        base = float(P[k] @ tS @ P[k])
        omega[k] = base * (1.0 - c) / c if c < 1.0 else base * 1e-6
        desc.append(f"{' '.join(f'{v:+.0f}x{n}' for n, v in coefs.items())} = {val:.4%} ({'absolute' if absolute else 'relative'}, confidence {c:.0%})")
    Om = np.diag(omega)
    tS_inv = np.linalg.inv(tS)
    M = np.linalg.inv(tS_inv + P.T @ np.linalg.inv(Om) @ P)
    mu_excess = M @ (tS_inv @ pi + P.T @ np.linalg.inv(Om) @ Q)
    return {"lambda": lam, "tau": tau, "prior_excess": pi.tolist(), "posterior_means": (U.rf + mu_excess).tolist(),
            "posterior_cov": (S + M).tolist(), "views": desc, "P": P.tolist(), "Q": Q.tolist(), "omega": omega.tolist()}


# --------------------------------------------------------------------------- #
def section(U: mv.Universe, long_only: bool = False, n_boot: int = 500, seed: int = 0,
            market_weights: Optional[Sequence[float]] = None, views: Sequence[tuple] = (), tau: float = 0.05,
            view_confidence: float = 0.5, user_weights: Optional[Sequence[float]] = None) -> dict:
    """The 'Robust' block: TG and MinVar under sample / LW / Bayes-Stein / LW+BS / BL estimators,
    bootstrap ranges, implied returns for the user's portfolio."""
    import constrained as cn
    names = U.names
    sec = {"estimators": {}, "notes": []}
    T = U.T or 60
    if U.T is None:
        sec["notes"].append(f"no return history: Ledoit-Wolf needs returns; Bayes-Stein and the bootstrap assume T = {T} observations")

    def solve_pair(Ux, label):
        if long_only:
            c = cn.Constraints(long_only=True)
            t, g = cn.tangency(Ux, c), cn.gmv(Ux, c)
        else:
            t, g = mv.tangency(Ux), mv.gmv(Ux)
        # evaluate BOTH under the SAMPLE moments too, so estimators are compared on one yardstick
        t_s = mv.portfolio_stats(U, t.weights)
        return {"label": label, "tangency": mv.portfolio_dict(Ux, t), "gmv": mv.portfolio_dict(Ux, g),
                "tangency_under_sample_moments": {"mean": t_s.mean, "sd": t_s.sd, "sharpe": t_s.sharpe}}

    sec["estimators"]["sample"] = solve_pair(U, "sample means + sample covariance")
    # Ledoit-Wolf
    if U.returns is not None:
        Slw, delta, F = ledoit_wolf_cc(U.returns)
        Ulw = mv.Universe(names, U.mu, Slw, rf=U.rf, ddof=U.ddof, returns=U.returns)
        sec["ledoit_wolf"] = {"delta": delta, "cov": Slw.tolist(), "target_constant_corr": float((F / np.outer(np.sqrt(np.diag(F)), np.sqrt(np.diag(F))))[0, 1]) if U.n > 1 else None}
        sec["estimators"]["ledoit_wolf"] = solve_pair(Ulw, f"sample means + Ledoit-Wolf covariance (delta = {delta:.3f})")
    else:
        Ulw = None
    # Bayes-Stein means
    mu_bs, nu, mu_g = bayes_stein(U.mu, U.cov, T)
    Ubs = mv.Universe(names, mu_bs, U.cov, rf=U.rf, ddof=U.ddof)
    sec["bayes_stein"] = {"nu": nu, "shrink_target_mean": mu_g, "means": mu_bs.tolist()}
    sec["estimators"]["bayes_stein"] = solve_pair(Ubs, f"Bayes-Stein means (nu = {nu:.3f}) + sample covariance")
    if Ulw is not None:
        Uboth = mv.Universe(names, mu_bs, Ulw.cov, rf=U.rf, ddof=U.ddof)
        sec["estimators"]["bs_plus_lw"] = solve_pair(Uboth, "Bayes-Stein means + Ledoit-Wolf covariance")
    # Black-Litterman
    if market_weights is not None or views:
        wm = np.asarray(market_weights, dtype=float) if market_weights is not None else np.full(U.n, 1.0 / U.n)
        if market_weights is None:
            sec["notes"].append("Black-Litterman prior uses EQUAL weights as the market portfolio (pass --market-weights for real caps)")
        bl = black_litterman(U, wm, views, tau=tau, default_confidence=view_confidence)
        Ubl = mv.Universe(names, np.array(bl["posterior_means"]), np.array(bl["posterior_cov"]), rf=U.rf, ddof=U.ddof)
        sec["black_litterman"] = bl
        sec["estimators"]["black_litterman"] = solve_pair(Ubl, "Black-Litterman posterior (" + (", ".join(bl["views"]) or "no views") + ")")
    # bootstrap
    sec["bootstrap"] = bootstrap(U, n=n_boot, seed=seed, T=T, long_only=long_only)
    # implied returns
    if user_weights is not None:
        sec["implied_user"] = implied_returns(U, user_weights)
    sec["implied_equal_weight"] = implied_returns(U, np.full(U.n, 1.0 / U.n))
    return sec


def render(sec: dict, names: Sequence[str]) -> str:
    out = ["ROBUST ESTIMATION (how much the answer depends on the inputs)"]
    for n in sec.get("notes", []):
        out.append(f"  note: {n}")
    out.append("  tangency weights by estimator (Sharpe measured on the SAMPLE moments so estimators are comparable):")
    out.append("    " + f"{'estimator':<58}" + "".join(f"{n[:9]:>10}" for n in names) + "   sharpe(sample)")
    for key, e in sec["estimators"].items():
        w = e["tangency"]["weights_list"]
        out.append("    " + f"{e['label'][:58]:<58}" + "".join(f"{x * 100:9.1f}%" for x in w) + f"   {e['tangency_under_sample_moments']['sharpe']:.4f}")
    if "ledoit_wolf" in sec:
        out.append(f"  Ledoit-Wolf shrinkage intensity delta = {sec['ledoit_wolf']['delta']:.3f} (0 = sample, 1 = constant-correlation target)")
    bs = sec["bayes_stein"]
    out.append(f"  Bayes-Stein: nu = {bs['nu']:.3f}; means pulled toward the MinVar mean {bs['shrink_target_mean']:.4%}: " + "  ".join(f"{n} {m:.3%}" for n, m in zip(names, bs["means"])))
    if "black_litterman" in sec:
        bl = sec["black_litterman"]
        out.append(f"  Black-Litterman: lambda {bl['lambda']:.2f}, tau {bl['tau']}; views: " + ("; ".join(bl["views"]) or "none"))
        out.append("    posterior means: " + "  ".join(f"{n} {m:.3%}" for n, m in zip(names, bl["posterior_means"])))
    b = sec["bootstrap"]
    out.append(f"  bootstrap ({b['kind']}, {b['n']} draws{', long-only' if b['long_only'] else ''}): tangency weight 5th..95th percentile")
    for i, n in enumerate(names):
        out.append(f"    {n:<10} {b['tangency']['p5'][i] * 100:7.1f}% .. {b['tangency']['p95'][i] * 100:7.1f}%   (median {b['tangency']['p50'][i] * 100:6.1f}%, sd {b['tangency_weight_sd'][i] * 100:5.1f} pts)")
    s = b["tangency_sharpe"]
    out.append(f"    tangency Sharpe 5th..95th: {s['p5']:.3f} .. {s['p95']:.3f} (median {s['p50']:.3f})")
    if "implied_user" in sec:
        im = sec["implied_user"]
        out.append(f"  implied returns that make YOUR weights optimal (lambda {im['lambda']:.2f}): " + "  ".join(f"{n} {m:.3%}" for n, m in zip(names, im["implied_means"])))
    im = sec["implied_equal_weight"]
    out.append(f"  implied returns for equal weights (lambda {im['lambda']:.2f}): " + "  ".join(f"{n} {m:.3%}" for n, m in zip(names, im["implied_means"])))
    return "\n".join(out)
