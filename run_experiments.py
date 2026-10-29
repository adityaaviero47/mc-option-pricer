"""Reproduces every number in README.md. Run: python run_experiments.py"""
import json, time, subprocess
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pricer import bs_price, bs_greeks, geometric_asian_price, european_mc, asian_mc, barrier_mc, mc_greeks
from pricer.mc import up_and_out_call_exact

S, K, T, r, sig = 100.0, 100.0, 1.0, 0.05, 0.20
R = {}

# 1) European call: accuracy + variance reduction at equal path budget (40 independent seeds)
N, SEEDS = 1_000_000, range(1, 41)
bs = bs_price(S, K, T, r, sig)
R["european"] = dict(bs=bs, paths=N, seeds=len(SEEDS))
eu = {}
for m in ["plain", "antithetic", "control"]:
    runs = [european_mc(S, K, T, r, sig, N, method=m, seed=sd) for sd in SEEDS]
    eu[m] = dict(se=float(np.mean([x["se"] for x in runs])))
    R["european"][m] = dict(mean_price=float(np.mean([x["price"] for x in runs])), mean_se=eu[m]["se"],
                            mean_abs_err_bp=float(np.mean([abs(x["price"] - bs) for x in runs]) / bs * 1e4),
                            share_within_2se=float(np.mean([abs(x["price"] - bs) < 2 * x["se"] for x in runs])))
R["european"]["vr_factor_antithetic"] = (eu["plain"]["se"] / eu["antithetic"]["se"]) ** 2
R["european"]["vr_factor_control"] = (eu["plain"]["se"] / eu["control"]["se"]) ** 2

# 2) Pricing grid: 45 strikes x maturities x vols, max error vs BS
errs = []
for k in [80, 90, 100, 110, 120]:
    for t in [0.25, 1.0, 2.0]:
        for s in [0.1, 0.2, 0.4]:
            for kind in ["call", "put"]:
                m = european_mc(S, k, t, r, s, 400_000, kind=kind, method="control", seed=7)
                ex = bs_price(S, k, t, r, s, kind)
                errs.append(dict(K=k, T=t, sigma=s, kind=kind, mc=m["price"], bs=ex, abs_err=abs(m["price"] - ex), z=(abs(m["price"] - ex) / m["se"]) if m["se"] > 0 else 0.0))
R["grid"] = dict(n_contracts=len(errs), max_abs_err=max(e["abs_err"] for e in errs),
                 mean_abs_err=float(np.mean([e["abs_err"] for e in errs])),
                 share_within_2se=float(np.mean([e["z"] < 2 for e in errs])))

# 3) Greeks: pathwise / CRN finite difference vs analytic
g = mc_greeks(S, K, T, r, sig, 1_000_000, seed=3)
ga = bs_greeks(S, K, T, r, sig)
R["greeks"] = {k: dict(mc=g[k][0], se=g[k][1], analytic=ga[k], rel_err_pct=abs(g[k][0] - ga[k]) / ga[k] * 100) for k in g}

# 4) Arithmetic Asian with geometric control variate
n_steps = 252
plain = asian_mc(S, K, T, r, sig, 200_000, n_steps, control=False, seed=5)
cv = asian_mc(S, K, T, r, sig, 200_000, n_steps, control=True, seed=5)
R["asian"] = dict(geometric_closed_form=geometric_asian_price(S, K, T, r, sig, n_steps),
                  plain=plain, control=cv, vr_factor=(plain["se"] / cv["se"]) ** 2)

# 5) Up-and-out barrier: discrete monitoring bias and Brownian-bridge correction
B = 130.0
exact = up_and_out_call_exact(S, K, B, T, r, sig)
naive = barrier_mc(S, K, B, T, r, sig, 200_000, 52, bridge=False, seed=9)
bridge = barrier_mc(S, K, B, T, r, sig, 200_000, 52, bridge=True, seed=9)
R["barrier"] = dict(B=B, exact_continuous=exact, weekly_naive=naive, weekly_bridge=bridge,
                    naive_bias_pct=(naive["price"] - exact) / exact * 100,
                    bridge_bias_pct=(bridge["price"] - exact) / exact * 100)

# 6) Speed: C++ loop vs vectorised NumPy, 10M paths, antithetic
t0 = time.perf_counter(); european_mc(S, K, T, r, sig, 10_000_000, method="antithetic", seed=11); py_ms = (time.perf_counter() - t0) * 1e3
cpp = json.loads(subprocess.run(["./cpp/mc_european", "100", "100", "1", "0.05", "0.2", "10000000", "42"], capture_output=True, text=True).stdout)
R["speed"] = dict(numpy_ms=py_ms, cpp_ms=cpp["ms"], cpp_price=cpp["price"], cpp_abs_err=cpp["abs_err"])

# 7) Convergence plot: error ~ 1/sqrt(N)
Ns = np.logspace(3, 6.5, 12).astype(int)
fig, ax = plt.subplots(figsize=(6, 4))
for m in ["plain", "antithetic", "control"]:
    ax.loglog(Ns, [european_mc(S, K, T, r, sig, n, method=m, seed=2)["se"] for n in Ns], marker="o", label=m)
ax.loglog(Ns, eu["plain"]["se"] * np.sqrt(N / Ns), "k--", lw=1, label=r"$1/\sqrt{N}$")
ax.set_xlabel("paths"); ax.set_ylabel("standard error"); ax.set_title("European call: MC standard error"); ax.legend()
fig.tight_layout(); fig.savefig("results/convergence.png", dpi=150)

json.dump(R, open("results/results.json", "w"), indent=2, default=float)
print(json.dumps(R, indent=2, default=float))
