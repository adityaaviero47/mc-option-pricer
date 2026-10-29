"""Vectorised Monte Carlo pricers under GBM with variance reduction."""
import numpy as np
from .analytic import bs_price, geometric_asian_price

def _result(x, disc):
    v = disc * x
    return dict(price=v.mean(), se=v.std(ddof=1) / np.sqrt(len(v)))

def european_mc(S, K, T, r, sigma, n_paths, kind="call", method="plain", seed=0):
    """method: plain | antithetic | control (control variate = discounted terminal stock)."""
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    drift, vol = (r - 0.5 * sigma**2) * T, sigma * np.sqrt(T)
    pay = (lambda s: np.maximum(s - K, 0)) if kind == "call" else (lambda s: np.maximum(K - s, 0))
    if method == "antithetic":
        z = rng.standard_normal(n_paths // 2)
        ST1, ST2 = S * np.exp(drift + vol * z), S * np.exp(drift - vol * z)
        return _result(0.5 * (pay(ST1) + pay(ST2)), disc)
    z = rng.standard_normal(n_paths)
    ST = S * np.exp(drift + vol * z)
    p = pay(ST)
    if method == "control":
        c = ST - S * np.exp(r * T)               # E[c] = 0 under Q
        b = np.cov(p, c)[0, 1] / c.var(ddof=1)
        p = p - b * c
    return _result(p, disc)

def _paths(S, T, r, sigma, n_paths, n_steps, rng, antithetic=False):
    dt = T / n_steps
    z = rng.standard_normal((n_paths // 2 if antithetic else n_paths, n_steps))
    if antithetic:
        z = np.vstack([z, -z])
    inc = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    return S * np.exp(np.cumsum(inc, axis=1))       # fixings at t_1..t_n

def asian_mc(S, K, T, r, sigma, n_paths, n_steps, kind="call", control=True, seed=0):
    """Arithmetic-average Asian; control variate = geometric Asian with known price."""
    rng = np.random.default_rng(seed)
    P = _paths(S, T, r, sigma, n_paths, n_steps, rng)
    disc = np.exp(-r * T)
    A, G = P.mean(axis=1), np.exp(np.log(P).mean(axis=1))
    if kind == "call":
        pa, pg = np.maximum(A - K, 0), np.maximum(G - K, 0)
    else:
        pa, pg = np.maximum(K - A, 0), np.maximum(K - G, 0)
    if not control:
        return _result(pa, disc)
    geo_exact = geometric_asian_price(S, K, T, r, sigma, n_steps, kind) / disc
    b = np.cov(pa, pg)[0, 1] / pg.var(ddof=1)
    return _result(pa - b * (pg - geo_exact), disc)

def barrier_mc(S, K, B, T, r, sigma, n_paths, n_steps, bridge=True, seed=0):
    """Up-and-out call. bridge=True applies the Brownian-bridge crossing probability between
    monitoring dates, approximating continuous monitoring."""
    rng = np.random.default_rng(seed)
    P = _paths(S, T, r, sigma, n_paths, n_steps, rng)
    P = np.hstack([np.full((n_paths, 1), S), P])
    alive = np.ones(n_paths)
    if bridge:
        dt = T / n_steps
        lo, hi = np.log(B / P[:, :-1]), np.log(B / P[:, 1:])
        p_cross = np.where((lo > 0) & (hi > 0), np.exp(-2 * lo * hi / (sigma**2 * dt)), 1.0)
        alive = np.prod(1 - p_cross, axis=1)
    else:
        alive = (P.max(axis=1) < B).astype(float)
    return _result(alive * np.maximum(P[:, -1] - K, 0), np.exp(-r * T))

def up_and_out_call_exact(S, K, B, T, r, sigma):
    """Continuous-monitoring up-and-out call (K < B), Reiner-Rubinstein."""
    from scipy.stats import norm
    if S >= B: return 0.0
    sq = sigma * np.sqrt(T)
    lam = (r + 0.5 * sigma**2) / sigma**2
    x1 = np.log(S / B) / sq + lam * sq
    y = np.log(B**2 / (S * K)) / sq + lam * sq
    y1 = np.log(B / S) / sq + lam * sq
    c = bs_price(S, K, T, r, sigma)
    cui = (S * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - sq)
           - S * (B / S)**(2 * lam) * (norm.cdf(-y) - norm.cdf(-y1))
           + K * np.exp(-r * T) * (B / S)**(2 * lam - 2) * (norm.cdf(-y + sq) - norm.cdf(-y1 + sq)))
    return c - cui

def mc_greeks(S, K, T, r, sigma, n_paths, seed=0):
    """Delta & vega by pathwise derivative; gamma by likelihood-ratio x pathwise mix
    is noisy, so gamma uses central finite difference with common random numbers."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    disc = np.exp(-r * T)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    itm = ST > K
    delta = disc * (itm * ST / S)
    vega = disc * (itm * ST * (np.sqrt(T) * z - sigma * T))
    h = 0.01 * S
    f = lambda s0: disc * np.maximum(s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z) - K, 0)
    gamma = (f(S + h) - 2 * f(S) + f(S - h)) / h**2
    out = {}
    for k, v in dict(delta=delta, vega=vega, gamma=gamma).items():
        out[k] = (v.mean(), v.std(ddof=1) / np.sqrt(n_paths))
    return out
