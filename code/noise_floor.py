"""
Reproduce the perfect-sampler marginal-JS noise floor (Figure 2 / Section 5.3).

The floor is the marginal JS distance between a finite draw of size N taken
directly from a target and a 20,000-sample reference draw from the same target,
using the identical 100-bin / percentile-edge JS estimator used everywhere in
the paper (see compute_marginal_js_distance in run_controls.py). It is averaged
over the correlated_gaussian, bimodal_gaussian and funnel targets (20D).

Output: noise_floor_curve.csv  (n_samples, perfect_sampler_marginal_JS)
Verified anchors: ~0.085 at N=2,000 and ~0.043 at N=10,000 (factor ~1.97).
"""
import numpy as np
import pandas as pd
from scipy.special import kl_div

rng = np.random.default_rng(7)

def gen(target, n, dim=20):
    """Direct draws from the target generatives (match run_controls.py defs)."""
    if target == "correlated_gaussian":          # rho=0.95 on first 2 dims
        L = np.linalg.cholesky([[1, 0.95], [0.95, 1]])
        xy = rng.standard_normal((n, 2)) @ L.T
        return np.column_stack([xy, rng.standard_normal((n, dim - 2))])
    if target == "bimodal_gaussian":             # mode_distance=6 along dim 0
        off = 3.0
        mask = rng.random(n) < 0.5
        s = rng.standard_normal((n, dim))
        s[:, 0] += np.where(mask, -off, off)
        return s
    if target == "funnel":                       # Neal funnel, sigma_v=0.3
        v = rng.standard_normal(n) * 0.3
        s = np.zeros((n, dim)); s[:, 0] = v
        s[:, 1:] = rng.standard_normal((n, dim - 1)) * np.exp(v[:, None] / 2)
        return s
    raise ValueError(target)

def js_dist(s, r, nbins=100, pad=0.1):
    """Average marginal JS distance (identical to the paper's estimator)."""
    s = np.asarray(s); r = np.asarray(r); out = []
    for i in range(s.shape[1]):
        lo = min(np.percentile(s[:, i], 0.5),  np.percentile(r[:, i], 0.5))
        hi = max(np.percentile(s[:, i], 99.5), np.percentile(r[:, i], 99.5))
        rg = hi - lo; lo -= pad * rg; hi += pad * rg
        bins = np.linspace(lo, hi, nbins + 1)
        h1, _ = np.histogram(s[:, i], bins=bins, density=True)
        h2, _ = np.histogram(r[:, i], bins=bins, density=True)
        eps = 1e-12
        h1 = (h1 + eps) / (h1.sum() + eps * len(h1))
        h2 = (h2 + eps) / (h2.sum() + eps * len(h2))
        m = 0.5 * (h1 + h2)
        out.append(np.sqrt(0.5 * np.sum(kl_div(h1, m)) + 0.5 * np.sum(kl_div(h2, m))))
    return float(np.mean(out))

if __name__ == "__main__":
    Ns = [500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000]
    rows = []
    for N in Ns:
        vals = []
        for t in ["correlated_gaussian", "bimodal_gaussian", "funnel"]:
            for _ in range(6):
                vals.append(js_dist(gen(t, N), gen(t, 20000)))
        rows.append((N, round(float(np.mean(vals)), 4)))
        print(f"N={N:>5}: {rows[-1][1]:.4f}")
    pd.DataFrame(rows, columns=["n_samples", "perfect_sampler_marginal_JS"]).to_csv(
        "noise_floor_curve.csv", index=False)
