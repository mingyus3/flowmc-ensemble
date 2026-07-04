#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary controlled experiments for
"A Single Chain Suffices: Re-examining Diversity-Enhanced flowMC Ensembles
 under Equal Sampling Budgets".

All experiments here operate on EXACT draws from the target generative laws
(the same generators used to produce the paper's reference samples, see
generate_reference_samples in code/enhanced_flowmc_ensemble.py of [13]).
They therefore characterize the metric and the aggregation operators
themselves, independently of flowMC. Every reported number is produced by
actually running this script; per-run CSVs are written to ../results.

Conventions copied verbatim from the released repository [13]:
  * marginal JS distance: 100 bins, percentile(0.5/99.5) range over the union
    of the two samples, 10% padding, eps=1e-12 smoothing, JS divergence in
    nats via scipy.special.kl_div, sqrt, averaged over dimensions.
  * coverage-aware reweighting: _cluster_weights (KMeans, K=8, n_init=5,
    random_state=0, inverse-frequency weights, rare_boost=0.35).
  * quality-based member weighting: score = ESS^beta * exp(-gamma*JS)
    * exp(-lam*sigma_logp) with beta=1, gamma=2, lam=0.05; z-standardised;
    softmax with alpha_temp=0.35.
  * adaptive tempering schedule: T <- min(2, 1.1 T) if ESS ratio < 0.5 else
    T <- max(0.1, 0.95 T), starting from T=1 (healthy-member branch gives
    T_m = 1, 0.95, 0.9025, 0.857375, 0.81450625 for members 1..5).
  * resampling with replacement to N_target = n_per * n_members.
    (The released pipeline uses a fixed resampling seed 2025; here the
    resampling rng varies with the repetition so that error bars reflect
    resampling noise as well.)

Phases:
  floor    : E1 perfect-sampler JS floor vs N, analytic fit, bins dependence,
             floors for all 11 configurations at N=2,000 and N=10,000
  decomp   : E2 sampler-free decomposition of the aggregation bias
  members  : E3 ensemble-size scaling (M = 2, 5, 10, 20)
  energy   : E4 joint-metric (energy distance, unbiased U-statistic) check
  temper   : E5 exact effect of the adaptive-tempering schedule
  audit    : E6 reference-generator audit (Rosenbrock / ring), for review notes

Usage: python3 supp_experiments.py <phase> [...phases | all]
"""

import sys
import time
import os
import zlib
import numpy as np
import pandas as pd
from scipy.special import kl_div
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# ----------------------------------------------------------------------------
# Target generative laws (identical to generate_reference_samples in [13])
# ----------------------------------------------------------------------------

CONFIGS = [
    ("banana",              20, dict(a=1.0, b=100.0)),
    ("banana",              50, dict(a=1.0, b=100.0)),
    ("bimodal_gaussian",    20, dict(mode_distance=6.0)),
    ("bimodal_gaussian",    50, dict(mode_distance=6.0)),
    ("correlated_gaussian", 20, dict(rho=0.95)),
    ("correlated_gaussian", 50, dict(rho=0.95)),
    ("funnel",              20, dict(sigma_v=0.3)),
    ("funnel",              50, dict(sigma_v=0.3)),
    ("gaussian_ring",       20, dict(radius=5.0, K=8, std=0.7)),
    ("rosenbrock",          20, dict(a=1.0, b=100.0)),
    ("rosenbrock",          50, dict(a=1.0, b=100.0)),
]

def sample_target(name, dim, n, rng, p):
    """Exact draw from the generative law used for the paper's references."""
    if name == "banana":
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n) + p["a"] * x1**2 - p["b"]
        return np.column_stack([x1, x2, rng.standard_normal((n, dim - 2))])
    if name == "bimodal_gaussian":
        off = p["mode_distance"] / 2.0
        s = rng.standard_normal((n, dim))
        s[:, 0] += np.where(rng.random(n) < 0.5, -off, off)
        return s
    if name == "funnel":
        v = rng.standard_normal(n) * p["sigma_v"]
        s = np.empty((n, dim)); s[:, 0] = v
        s[:, 1:] = rng.standard_normal((n, dim - 1)) * np.exp(v[:, None] / 2.0)
        return s
    if name == "correlated_gaussian":
        rho = p["rho"]
        L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
        xy = rng.standard_normal((n, 2)) @ L.T
        return np.column_stack([xy, rng.standard_normal((n, dim - 2))])
    if name == "rosenbrock":
        # NOTE: this is the paper's *reference* generative law (x0 ~ N(a, 2^2),
        # y|x0 ~ N(x0^2, 0.5^2)); see the audit phase for the comparison with
        # the law implied by the Rosenbrock potential.
        x0 = rng.standard_normal(n) * 2.0 + p["a"]
        y = x0**2 + rng.standard_normal(n) * 0.5
        return np.column_stack([x0, y, rng.standard_normal((n, dim - 2))])
    if name == "gaussian_ring":
        # Paper's reference law: uniform angle, radius ~ N(R, std^2).
        th = rng.random(n) * 2.0 * np.pi
        r = p["radius"] + rng.standard_normal(n) * p["std"]
        return np.column_stack([r * np.cos(th), r * np.sin(th),
                                rng.standard_normal((n, dim - 2))])
    raise ValueError(name)

def sample_rosenbrock_potential_law(dim, n, rng, a=1.0, b=100.0):
    """Exact draw from the density defined by the Rosenbrock *potential*
    0.5[(a-x0)^2 + b(y-x0^2)^2 + sum rest^2]:
        x0 ~ N(a, 1),  y|x0 ~ N(x0^2, 1/b),  rest ~ N(0, I)."""
    x0 = rng.standard_normal(n) + a
    y = x0**2 + rng.standard_normal(n) / np.sqrt(b)
    return np.column_stack([x0, y, rng.standard_normal((n, dim - 2))])

def sample_ring_potential_law(dim, n, rng, radius=5.0, K=8, std=0.7):
    """Exact draw from the density defined by the ring *potential*:
    equal mixture of K isotropic Gaussians (std) on a circle of given radius."""
    k = rng.integers(0, K, size=n)
    th = 2.0 * np.pi * k / K
    xy = np.column_stack([radius * np.cos(th), radius * np.sin(th)])
    xy += rng.standard_normal((n, 2)) * std
    return np.column_stack([xy, rng.standard_normal((n, dim - 2))])

def sample_correlated_tempered(dim, n, rng, T, rho=0.95):
    """Exact draw from pi^{1/T} for the correlated Gaussian: covariance * T."""
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    xy = (rng.standard_normal((n, 2)) @ L.T) * np.sqrt(T)
    return np.column_stack([xy, rng.standard_normal((n, dim - 2)) * np.sqrt(T)])

def sample_funnel_tempered(dim, n, rng, T, sigma_v=0.3):
    """Exact draw from pi^{1/T} for Neal's funnel:
        v ~ N(sigma_v^2 (d-1)(T-1)/2, T sigma_v^2),  x_i|v ~ N(0, T e^v)."""
    mu_v = sigma_v**2 * (dim - 1) * (T - 1.0) / 2.0
    v = mu_v + rng.standard_normal(n) * sigma_v * np.sqrt(T)
    s = np.empty((n, dim)); s[:, 0] = v
    s[:, 1:] = rng.standard_normal((n, dim - 1)) * np.sqrt(T) * np.exp(v[:, None] / 2.0)
    return s

# ----------------------------------------------------------------------------
# Log-densities (numpy ports of the potentials in [13]; used for sigma_logp)
# ----------------------------------------------------------------------------

def log_post(name, X, p):
    X = np.asarray(X)
    rest2 = np.sum(X[:, 2:]**2, axis=1) if X.shape[1] > 2 else 0.0
    if name == "banana":
        t = 0.5 * X[:, 0]**2 + 0.5 * (X[:, 1] - p["a"] * X[:, 0]**2 + p["b"])**2
        return -(t + 0.5 * rest2)
    if name == "bimodal_gaussian":
        off = p["mode_distance"] / 2.0
        d1 = np.sum(X**2, axis=1) + 2 * off * X[:, 0] + off**2
        d2 = np.sum(X**2, axis=1) - 2 * off * X[:, 0] + off**2
        m = np.maximum(-0.5 * d1, -0.5 * d2)
        return m + np.log(np.exp(-0.5 * d1 - m) + np.exp(-0.5 * d2 - m))
    if name == "funnel":
        v = X[:, 0]
        r2 = np.sum(X[:, 1:]**2, axis=1)
        return -(0.5 * (v / p["sigma_v"])**2 + 0.5 * r2 / np.exp(v)
                 + 0.5 * (X.shape[1] - 1) * v)
    if name == "correlated_gaussian":
        rho = p["rho"]
        Z = (X[:, 0]**2 - 2 * rho * X[:, 0] * X[:, 1] + X[:, 1]**2) / (1 - rho**2)
        return -(0.5 * (Z + rest2) + 0.5 * np.log(1 - rho**2))
    if name == "rosenbrock":
        u = (p["a"] - X[:, 0])**2 + p["b"] * (X[:, 1] - X[:, 0]**2)**2
        return -0.5 * (u + rest2)
    if name == "gaussian_ring":
        th = np.linspace(0, 2 * np.pi, p["K"], endpoint=False)
        C = np.stack([p["radius"] * np.cos(th), p["radius"] * np.sin(th)], axis=1)
        d = np.sum((X[:, None, :2] - C[None]) ** 2, axis=-1)
        a = -0.5 * d / p["std"]**2
        mx = a.max(axis=1)
        ll = mx + np.log(np.exp(a - mx[:, None]).sum(axis=1)) - np.log(p["K"])
        return ll - 0.5 * rest2
    raise ValueError(name)

# ----------------------------------------------------------------------------
# Metrics (identical estimators to [13]; ESS uses an FFT autocorrelation that
# is numerically equivalent to the released np.correlate implementation)
# ----------------------------------------------------------------------------

def marginal_js(s, r, nbins=100, pad=0.1):
    s = np.asarray(s); r = np.asarray(r); out = []
    for i in range(s.shape[1]):
        lo = min(np.percentile(s[:, i], 0.5), np.percentile(r[:, i], 0.5))
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

def marginal_js_per_dim(s, r, nbins=100, pad=0.1):
    """Per-dimension JS values (same estimator), for the audit breakdown."""
    s = np.asarray(s); r = np.asarray(r); out = []
    for i in range(s.shape[1]):
        out.append(marginal_js(s[:, i:i+1], r[:, i:i+1], nbins=nbins, pad=pad))
    return np.array(out)

def _acf_fft(x):
    n = len(x); xc = x - x.mean()
    f = np.fft.rfft(xc, 2 * n)
    return np.fft.irfft(f * np.conj(f))[:n]

def ess_1d(x):
    n = len(x)
    ac = _acf_fft(np.asarray(x, float))
    v0 = float(ac[0])
    if not np.isfinite(v0) or abs(v0) < 1e-12:
        return float(n)
    ac = ac / max(v0, 1e-12)
    idx = np.where(ac < 0)[0]
    tau = 0.5 + np.sum(ac[1:(int(idx[0]) if idx.size else None)])
    return float(n / (2 * max(tau, 1e-9)))

def ess_ddim(S):
    S = np.asarray(S)
    return float(np.mean([ess_1d(S[:, i]) for i in range(S.shape[1])]))

def energy_distance_ustat(X, Y, rng, m=2000):
    """Unbiased (U-statistic) energy distance estimate on m-subsamples.
    E[.] = 0 when X and Y follow the same law."""
    xi = X[rng.choice(len(X), size=min(m, len(X)), replace=False)]
    yi = Y[rng.choice(len(Y), size=min(m, len(Y)), replace=False)]
    dxy = cdist(xi, yi).mean()
    dxx = cdist(xi, xi); dyy = cdist(yi, yi)
    nx, ny = len(xi), len(yi)
    mxx = dxx.sum() / (nx * (nx - 1))
    myy = dyy.sum() / (ny * (ny - 1))
    return float(2 * dxy - mxx - myy)

# ----------------------------------------------------------------------------
# Aggregation operators (verbatim ports from [13])
# ----------------------------------------------------------------------------

def cluster_weights(X, K=8, rare_boost=0.35, seed=0):
    K = max(2, min(K, max(2, X.shape[0] // 50)))
    km = KMeans(n_clusters=K, n_init=5, random_state=seed).fit(X)
    cid = km.labels_
    cnt = np.bincount(cid)
    inv = 1.0 / np.maximum(cnt, 1)
    w = inv[cid]
    w = w * (1.0 + rare_boost * (w / np.max(w) - 1.0))
    w = np.maximum(w, 0.0)
    return w / np.sum(w)

def member_weights(ess_list, js_list, slogp_list,
                   beta=1.0, gamma=2.0, lam=0.05, alpha_temp=0.35):
    ess = np.maximum(np.asarray(ess_list, float), 1e-6)
    js = np.maximum(np.asarray(js_list, float), 1e-6)
    slp = np.maximum(np.asarray(slogp_list, float), 1e-6)
    score = (ess**beta) * np.exp(-gamma * js) * np.exp(-lam * slp)
    z = (score - score.mean()) / (score.std() + 1e-8)
    w = np.exp(alpha_temp * z)
    return w / w.sum()

TEMPER_SCHEDULE = [1.0, 0.95, 0.9025, 0.857375, 0.81450625]  # healthy branch

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def rng_for(*ids):
    return np.random.default_rng(np.random.SeedSequence([20260704] + list(ids)))

def ref_for(ci, name, dim, p, n=20000):
    return sample_target(name, dim, n, rng_for(ci, 0, 0), p)

# ----------------------------------------------------------------------------
# E1: perfect-sampler floor, analytic fit, bins dependence
# ----------------------------------------------------------------------------

def phase_floor():
    t0 = time.time()
    rows = []
    # (a) floor curve vs N, averaged over correlated/bimodal/funnel 20D
    curve_targets = [("correlated_gaussian", 20, dict(rho=0.95)),
                     ("bimodal_gaussian", 20, dict(mode_distance=6.0)),
                     ("funnel", 20, dict(sigma_v=0.3))]
    Ns = [500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000]
    for N in Ns:
        for ti, (name, dim, p) in enumerate(curve_targets):
            ref = ref_for(100 + ti, name, dim, p)
            for rep in range(6):
                s = sample_target(name, dim, N, rng_for(100 + ti, N, rep + 1), p)
                rows.append(dict(kind="curve", target=name, dim=dim, N=N,
                                 nbins=100, rep=rep, JS=marginal_js(s, ref)))
        print(f"[floor curve] N={N} done  ({time.time()-t0:.0f}s)", flush=True)
    # (b) floors for all 11 configurations at N = 2,000 and 10,000
    for ci, (name, dim, p) in enumerate(CONFIGS):
        ref = ref_for(ci, name, dim, p)
        for N in (2000, 10000):
            for rep in range(5):
                s = sample_target(name, dim, N, rng_for(ci, N, rep + 1), p)
                rows.append(dict(kind="all11", target=name, dim=dim, N=N,
                                 nbins=100, rep=rep, JS=marginal_js(s, ref)))
    print(f"[floor all-11] done  ({time.time()-t0:.0f}s)", flush=True)
    # (c) bin-count dependence on the correlated Gaussian, 20D
    name, dim, p = "correlated_gaussian", 20, dict(rho=0.95)
    ref = ref_for(104, name, dim, p)
    for B in (25, 50, 100, 200):
        for N in (1000, 2000, 5000, 10000, 20000):
            for rep in range(5):
                s = sample_target(name, dim, N, rng_for(104, 7 * N + B, rep + 1), p)
                rows.append(dict(kind="bins", target=name, dim=dim, N=N,
                                 nbins=B, rep=rep, JS=marginal_js(s, ref, nbins=B)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/supp_floor.csv", index=False)
    # analytic fit c: JS^2 = c * (1/N + 1/M), M = 20000, through the origin
    M = 20000.0
    for kind, sub in df.groupby("kind"):
        if kind == "bins":
            for B, s2 in sub.groupby("nbins"):
                x = 1.0 / s2["N"].values + 1.0 / M
                c = float(np.sum(x * s2["JS"].values**2) / np.sum(x * x))
                print(f"  fit c (B={B}): {c:.3f}  -> B_eff ≈ {8*c+1:.1f}")
        else:
            x = 1.0 / sub["N"].values + 1.0 / M
            c = float(np.sum(x * sub["JS"].values**2) / np.sum(x * x))
            print(f"  fit c ({kind}, B=100): {c:.3f}  -> B_eff ≈ {8*c+1:.1f}")
    print(f"[floor] total {time.time()-t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------
# E2: sampler-free decomposition of the aggregation bias
# ----------------------------------------------------------------------------

def build_pool(name, dim, p, ci, rep, n_members=5, n_per=2000):
    members = [sample_target(name, dim, n_per, rng_for(ci, rep + 1, 10 + m), p)
               for m in range(n_members)]
    allX = np.vstack(members)
    memb_id = np.concatenate([[i] * n_per for i in range(n_members)])
    return members, allX, memb_id

def member_stats(members, ref, name, p):
    ess_l, js_l, slp_l = [], [], []
    for X in members:
        ess_l.append(ess_ddim(X))
        js_l.append(marginal_js(X, ref))
        slp_l.append(float(np.std(log_post(name, X, p))))
    return ess_l, js_l, slp_l

def phase_decomp():
    t0 = time.time()
    rows = []
    for ci, (name, dim, p) in enumerate(CONFIGS):
        ref = ref_for(ci, name, dim, p)
        for rep in range(5):
            rng = rng_for(ci, rep + 1, 999)
            members, allX, memb_id = build_pool(name, dim, p, ci, rep)
            N_target = len(allX)

            def add(variant, S, extra=None):
                rows.append(dict(target=name, dim=dim, rep=rep, variant=variant,
                                 JS=marginal_js(S, ref), **(extra or {})))

            # V0: one exact draw of 10,000 (single-chain analogue)
            single = sample_target(name, dim, N_target, rng_for(ci, rep + 1, 5), p)
            add("exact_single_10000", single)
            # V1: plain uniform pool (concatenation, no weights, no resampling)
            add("pool_uniform", allX, dict(pool_ESS=ess_ddim(allX)))
            # V2: uniform-weight resampling with replacement
            idx = rng.choice(N_target, size=N_target, replace=True)
            add("pool_resample_uniform", allX[idx])
            # member quality stats (used by V3/V5)
            ess_l, js_l, slp_l = member_stats(members, ref, name, p)
            w_alpha = member_weights(ess_l, js_l, slp_l)
            w_m = w_alpha[memb_id]; w_m = w_m / w_m.sum()
            # V3: member weighting + resampling
            idx = rng.choice(N_target, size=N_target, replace=True, p=w_m)
            add("pool_memberweight_resample", allX[idx],
                dict(w_min=float(w_alpha.min()), w_max=float(w_alpha.max())))
            # V4: coverage reweighting + resampling
            cov_w = cluster_weights(allX[:, :2])
            wc = cov_w / cov_w.sum()
            idx = rng.choice(N_target, size=N_target, replace=True, p=wc)
            add("pool_coverage_resample", allX[idx])
            # V5: full aggregation stack (member x coverage weights, resample)
            w_full = w_m * cov_w; w_full = w_full / w_full.sum()
            idx = rng.choice(N_target, size=N_target, replace=True, p=w_full)
            S5 = allX[idx]
            add("pool_full_aggregation", S5, dict(pool_ESS=ess_ddim(S5)))
        print(f"[decomp] {name} d={dim} done  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(f"{RESULTS}/supp_decomposition.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby("variant")["JS"].mean().round(4))
    print(f"[decomp] total {time.time()-t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------
# E3: ensemble-size scaling
# ----------------------------------------------------------------------------

def phase_members():
    t0 = time.time()
    targets = [("bimodal_gaussian", 20, dict(mode_distance=6.0)),
               ("correlated_gaussian", 20, dict(rho=0.95)),
               ("funnel", 20, dict(sigma_v=0.3)),
               ("gaussian_ring", 20, dict(radius=5.0, K=8, std=0.7))]
    rows = []
    for ti, (name, dim, p) in enumerate(targets):
        ref = ref_for(200 + ti, name, dim, p)
        for M in (2, 5, 10, 20):
            for rep in range(5):
                rng = rng_for(200 + ti, M, rep + 1)
                members, allX, memb_id = build_pool(name, dim, p, 200 + ti,
                                                    100 * M + rep, n_members=M)
                N_target = len(allX)
                # floor at the pooled size: one exact draw of N_target
                single = sample_target(name, dim, N_target,
                                       rng_for(200 + ti, M, 50 + rep), p)
                rows.append(dict(target=name, M=M, rep=rep,
                                 variant="exact_single", JS=marginal_js(single, ref)))
                rows.append(dict(target=name, M=M, rep=rep,
                                 variant="pool_uniform", JS=marginal_js(allX, ref)))
                ess_l, js_l, slp_l = member_stats(members, ref, name, p)
                w_alpha = member_weights(ess_l, js_l, slp_l)
                w_m = w_alpha[memb_id]; w_m = w_m / w_m.sum()
                cov_w = cluster_weights(allX[:, :2])
                w_full = w_m * cov_w; w_full = w_full / w_full.sum()
                idx = rng.choice(N_target, size=N_target, replace=True, p=w_full)
                rows.append(dict(target=name, M=M, rep=rep,
                                 variant="pool_full_aggregation",
                                 JS=marginal_js(allX[idx], ref)))
            print(f"[members] {name} M={M} done  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(f"{RESULTS}/supp_members.csv", index=False)
    print(f"[members] total {time.time()-t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------
# E4: joint-metric (energy distance) check, 20D configurations
# ----------------------------------------------------------------------------

def phase_energy():
    t0 = time.time()
    rows = []
    targets20 = [(n, d, p) for (n, d, p) in CONFIGS if d == 20]
    for ti, (name, dim, p) in enumerate(targets20):
        ref = ref_for(300 + ti, name, dim, p)
        for rep in range(5):
            rng = rng_for(300 + ti, rep + 1, 42)
            members, allX, memb_id = build_pool(name, dim, p, 300 + ti, rep)
            N_target = len(allX)
            single = sample_target(name, dim, N_target,
                                   rng_for(300 + ti, rep + 1, 5), p)
            ess_l, js_l, slp_l = member_stats(members, ref, name, p)
            w_alpha = member_weights(ess_l, js_l, slp_l)
            w_m = w_alpha[memb_id]; w_m = w_m / w_m.sum()
            cov_w = cluster_weights(allX[:, :2])
            w_full = w_m * cov_w; w_full = w_full / w_full.sum()
            variants = {
                "exact_single_10000": single,
                "pool_uniform": allX,
                "pool_resample_uniform": allX[rng.choice(N_target, N_target, True)],
                "pool_full_aggregation": allX[rng.choice(N_target, N_target, True,
                                                         p=w_full)],
            }
            for vname, S in variants.items():
                ed = energy_distance_ustat(S, ref, rng_for(300 + ti, rep + 1,
                                                           zlib.crc32(vname.encode()) % 1000))
                rows.append(dict(target=name, dim=dim, rep=rep, variant=vname,
                                 energy_distance=ed))
        print(f"[energy] {name} done  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(f"{RESULTS}/supp_energy.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby("variant")["energy_distance"].mean().round(4))
    print(f"[energy] total {time.time()-t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------
# E5: exact effect of the adaptive-tempering schedule (correlated + funnel)
# ----------------------------------------------------------------------------

def phase_temper():
    t0 = time.time()
    rows = []
    targets = [("correlated_gaussian", 20, dict(rho=0.95)),
               ("correlated_gaussian", 50, dict(rho=0.95)),
               ("funnel", 20, dict(sigma_v=0.3)),
               ("funnel", 50, dict(sigma_v=0.3))]
    for ti, (name, dim, p) in enumerate(targets):
        ref = ref_for(400 + ti, name, dim, p)
        for rep in range(5):
            rng = rng_for(400 + ti, rep + 1, 7)
            # exact members (T = 1 for all)
            members_e, allX_e, memb_id = build_pool(name, dim, p, 400 + ti, rep)
            # tempered members: member m drawn exactly from pi^{1/T_m}
            members_t = []
            for m, T in enumerate(TEMPER_SCHEDULE):
                r = rng_for(400 + ti, rep + 1, 60 + m)
                if name == "correlated_gaussian":
                    members_t.append(sample_correlated_tempered(dim, 2000, r, T,
                                                                rho=p["rho"]))
                else:
                    members_t.append(sample_funnel_tempered(dim, 2000, r, T,
                                                            sigma_v=p["sigma_v"]))
            allX_t = np.vstack(members_t)
            rows.append(dict(target=name, dim=dim, rep=rep,
                             variant="pool_uniform_exact",
                             JS=marginal_js(allX_e, ref)))
            rows.append(dict(target=name, dim=dim, rep=rep,
                             variant="pool_uniform_tempered",
                             JS=marginal_js(allX_t, ref)))
            # tempered + full aggregation stack
            ess_l, js_l, slp_l = member_stats(members_t, ref, name, p)
            w_alpha = member_weights(ess_l, js_l, slp_l)
            w_m = w_alpha[memb_id]; w_m = w_m / w_m.sum()
            cov_w = cluster_weights(allX_t[:, :2])
            w_full = w_m * cov_w; w_full = w_full / w_full.sum()
            idx = rng.choice(len(allX_t), size=len(allX_t), replace=True, p=w_full)
            rows.append(dict(target=name, dim=dim, rep=rep,
                             variant="tempered_full_aggregation",
                             JS=marginal_js(allX_t[idx], ref)))
        print(f"[temper] {name} d={dim} done  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(f"{RESULTS}/supp_tempering.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["target", "dim", "variant"])["JS"].mean().round(4))
    print(f"[temper] total {time.time()-t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------
# E6: reference-generator audit (for the review notes, not the paper)
# ----------------------------------------------------------------------------

def phase_audit():
    t0 = time.time()
    rows = []
    # Rosenbrock: potential law vs paper reference law
    for dim in (20, 50):
        ref = ref_for(500 + dim, "rosenbrock", dim, dict(a=1.0, b=100.0))
        for rep in range(5):
            s_pot = sample_rosenbrock_potential_law(dim, 10000,
                                                    rng_for(500 + dim, rep + 1, 1))
            s_ref = sample_target("rosenbrock", dim, 10000,
                                  rng_for(500 + dim, rep + 1, 2),
                                  dict(a=1.0, b=100.0))
            per_dim = marginal_js_per_dim(s_pot, ref)
            rows.append(dict(target="rosenbrock", dim=dim, rep=rep,
                             draw="potential_law_10000",
                             JS=float(per_dim.mean()),
                             JS_dim0=float(per_dim[0]), JS_dim1=float(per_dim[1]),
                             JS_rest=float(per_dim[2:].mean()),
                             ED=energy_distance_ustat(s_pot, ref,
                                                      rng_for(500 + dim, rep, 3))))
            rows.append(dict(target="rosenbrock", dim=dim, rep=rep,
                             draw="reference_law_10000",
                             JS=marginal_js(s_ref, ref),
                             JS_dim0=np.nan, JS_dim1=np.nan, JS_rest=np.nan,
                             ED=energy_distance_ustat(s_ref, ref,
                                                      rng_for(500 + dim, rep, 4))))
    # Ring: 8-mode mixture (potential) law vs continuous annulus reference law
    ref = ref_for(560, "gaussian_ring", 20, dict(radius=5.0, K=8, std=0.7))
    for rep in range(5):
        s_pot = sample_ring_potential_law(20, 10000, rng_for(560, rep + 1, 1))
        s_ref = sample_target("gaussian_ring", 20, 10000, rng_for(560, rep + 1, 2),
                              dict(radius=5.0, K=8, std=0.7))
        per_dim = marginal_js_per_dim(s_pot, ref)
        rows.append(dict(target="gaussian_ring", dim=20, rep=rep,
                         draw="potential_law_10000", JS=float(per_dim.mean()),
                         JS_dim0=float(per_dim[0]), JS_dim1=float(per_dim[1]),
                         JS_rest=float(per_dim[2:].mean()),
                         ED=energy_distance_ustat(s_pot, ref, rng_for(560, rep, 3))))
        rows.append(dict(target="gaussian_ring", dim=20, rep=rep,
                         draw="reference_law_10000", JS=marginal_js(s_ref, ref),
                         JS_dim0=np.nan, JS_dim1=np.nan, JS_rest=np.nan,
                         ED=energy_distance_ustat(s_ref, ref, rng_for(560, rep, 4))))
    pd.DataFrame(rows).to_csv(f"{RESULTS}/supp_reference_audit.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["target", "dim", "draw"])[["JS", "ED"]].mean().round(4))
    print(f"[audit] total {time.time()-t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------

PHASES = dict(floor=phase_floor, decomp=phase_decomp, members=phase_members,
              energy=phase_energy, temper=phase_temper, audit=phase_audit)

if __name__ == "__main__":
    # quick equivalence check: FFT autocorr ESS == released np.correlate ESS
    x = np.random.default_rng(0).standard_normal(500)
    ac_np = np.correlate(x - x.mean(), x - x.mean(), mode="full")[len(x) - 1:]
    assert np.allclose(ac_np, _acf_fft(x), atol=1e-8)
    args = sys.argv[1:] or ["all"]
    todo = list(PHASES) if args == ["all"] else args
    for ph in todo:
        print(f"===== phase: {ph} =====", flush=True)
        PHASES[ph]()
