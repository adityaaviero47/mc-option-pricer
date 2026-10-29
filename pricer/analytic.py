"""Closed-form benchmarks: Black-Scholes and geometric-average Asian (Kemna-Vorst)."""
import numpy as np
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, kind="call", q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, kind="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    if kind == "call":
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

def geometric_asian_price(S, K, T, r, sigma, n_steps, kind="call"):
    """Discrete geometric-average Asian option (fixings at t_i = i*T/n), closed form."""
    n = n_steps
    sig_g = sigma * np.sqrt((n + 1) * (2 * n + 1) / (6 * n**2))
    mu_g = (r - 0.5 * sigma**2) * (n + 1) / (2 * n) + 0.5 * sig_g**2
    d1 = (np.log(S / K) + (mu_g + 0.5 * sig_g**2) * T) / (sig_g * np.sqrt(T))
    d2 = d1 - sig_g * np.sqrt(T)
    if kind == "call":
        return np.exp(-r * T) * (S * np.exp(mu_g * T) * norm.cdf(d1) - K * norm.cdf(d2))
    return np.exp(-r * T) * (K * norm.cdf(-d2) - S * np.exp(mu_g * T) * norm.cdf(-d1))
