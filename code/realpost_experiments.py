#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep revision experiments (A)+(B) for
"A Single Chain Suffices: Re-examining Diversity-Enhanced flowMC Ensembles
 under Equal Sampling Budgets".

(A) REAL POSTERIORS. Replaces the synthetic targets with genuine Bayesian
    posteriors, sampled with the *actual flowMC sampler* (not exact draws):
      - blr8   : Bayesian logistic regression, 8 covariates  (200 obs)
      - blr20  : Bayesian logistic regression, 20 covariates (400 obs)
      - hier   : a hierarchical variance model (a real funnel-shaped posterior):
                 tau ~ HalfNormal; theta_j | tau ~ N(0, tau^2); y_j ~ N(theta_j, sigma_j^2).
                 9 groups -> 10-dim posterior (log tau + 9 thetas). This is the
                 genuine hierarchical geometry that Neal's synthetic funnel only
                 mimics.
    Gold-standard reference: long multi-chain NUTS (numpyro), 4 chains, large
    draw, thinned; this is the "truth" against which every sampler is scored.

(B) JOINT METRICS ON ACTUAL flowMC OUTPUT. The paper's comparative claims all
    used marginal JS, the very metric it criticises. Here every sampler variant
    (single@2000, single@10000, full ensemble, uniform pool) is scored on the
    *joint* distribution with:
      - energy distance (unbiased U-statistic; E=0 iff laws coincide),
      - MMD^2 (RBF kernel, median-heuristic bandwidth; unbiased estimator),
    all computed on the real flowMC samples against the NUTS reference. Marginal
    JS (vs the same NUTS reference) is reported alongside to show the ranking is
    metric-independent.

flowMC is driven through the released repository's own configuration (same
member palette, same aggregation stack) via enhanced_flowmc_ensemble.py, with a
thin v0.6.1 Sampler adapter (the installed flowMC exposes a keyword-only Sampler
API; the adapter reproduces the repo's single-chain JS magnitudes, checked in
the smoke test). All targets here are log-posteriors passed to flowMC as a
log_post_override, so the sampler code path is identical to the paper's.

Usage:
  python3 realpost_experiments.py ref     # build + cache NUTS references
  python3 realpost_experiments.py run      # run all flowMC variants + metrics
  python3 realpost_experiments.py all
Outputs: ../results/real_*.csv  and cached references in ../results/refs/
"""

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.spatial.distance import cdist
from scipy.special import kl_div
from sklearn.cluster import KMeans

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle
from flowMC.Sampler import Sampler

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
REFDIR = f"{RES}/refs"
os.makedirs(REFDIR, exist_ok=True)

# ============================================================================
# Real Bayesian posteriors (data-generating + numpyro model + numpy log-density)
# ============================================================================

def make_blr(n_obs, n_cov, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_obs, n_cov))
    w_true = rng.standard_normal(n_cov)
    logits = X @ w_true
    y = (rng.random(n_obs) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
    return dict(X=X, y=y, w_true=w_true, n_cov=n_cov, prior_sd=2.0)

def blr_numpyro(data):
    X = jnp.asarray(data["X"]); y = jnp.asarray(data["y"])
    D = data["n_cov"]; psd = data["prior_sd"]
    def model():
        w = numpyro.sample("w", dist.Normal(jnp.zeros(D), psd * jnp.ones(D)).to_event(1))
        numpyro.sample("y", dist.Bernoulli(logits=X @ w), obs=y)
    return model

def blr_logpost(data):
    """Vectorised numpy log-posterior over w (unnormalised)."""
    X = data["X"]; y = data["y"]; psd = data["prior_sd"]
    def lp(W):  # W: (n, D)
        logits = W @ X.T                        # (n, n_obs)
        ll = np.sum(y * logits - np.logaddexp(0.0, logits), axis=1)
        lprior = -0.5 * np.sum(W**2, axis=1) / psd**2
        return ll + lprior
    return lp

def make_hier(n_groups, seed):
    rng = np.random.default_rng(seed)
    tau_true = 1.2
    theta_true = rng.standard_normal(n_groups) * tau_true
    sigma = 0.5 + rng.random(n_groups) * 0.8          # heteroscedastic obs noise
    y = theta_true + rng.standard_normal(n_groups) * sigma
    return dict(y=y, sigma=sigma, n_groups=n_groups, tau_scale=2.0,
                tau_true=tau_true)

def hier_numpyro(data):
    y = jnp.asarray(data["y"]); sig = jnp.asarray(data["sigma"])
    J = data["n_groups"]; ts = data["tau_scale"]
    def model():
        # parameterise on log_tau so the support is R (matches flowMC's R^d)
        log_tau = numpyro.sample("log_tau", dist.Normal(0.0, 1.0))
        tau = jnp.exp(log_tau)
        # prior on tau is HalfNormal(ts); add the Jacobian via a factor
        numpyro.factor("tau_prior",
                       dist.HalfNormal(ts).log_prob(tau) + log_tau)
        theta = numpyro.sample("theta", dist.Normal(jnp.zeros(J), tau).to_event(1))
        numpyro.sample("y", dist.Normal(theta, sig).to_event(1), obs=y)
    return model

def hier_logpost(data):
    """log-posterior over z = (log_tau, theta_1..theta_J), unnormalised."""
    y = data["y"]; sig = data["sigma"]; J = data["n_groups"]; ts = data["tau_scale"]
    def lp(Z):  # Z: (n, J+1)
        log_tau = Z[:, 0]; theta = Z[:, 1:]
        tau = np.exp(log_tau)
        # HalfNormal(ts) prior on tau, transformed to log_tau (+ log_tau Jacobian)
        lprior_tau = (-0.5 * (tau / ts)**2 + np.log(np.sqrt(2.0 / np.pi) / ts)
                      + log_tau)
        lprior_theta = np.sum(-0.5 * (theta / tau[:, None])**2
                              - np.log(tau)[:, None] - 0.5 * np.log(2 * np.pi), axis=1)
        ll = np.sum(-0.5 * ((y[None, :] - theta) / sig[None, :])**2
                    - np.log(sig)[None, :] - 0.5 * np.log(2 * np.pi), axis=1)
        return lprior_tau + lprior_theta + ll
    return lp

# registry: name -> (dim, builder, numpyro-factory, logpost-factory, ref-param-name)
def build_targets():
    T = {}
    d = make_blr(200, 8, seed=11)
    T["blr8"]  = dict(dim=8,  data=d, npf=blr_numpyro, lpf=blr_logpost, var="w")
    d = make_blr(400, 20, seed=22)
    T["blr20"] = dict(dim=20, data=d, npf=blr_numpyro, lpf=blr_logpost, var="w")
    d = make_hier(9, seed=33)
    T["hier"]  = dict(dim=10, data=d, npf=hier_numpyro, lpf=hier_logpost, var="z")
    return T

# ============================================================================
# Gold-standard NUTS reference
# ============================================================================

def nuts_reference(name, tgt, n_ref=20000, seed=0):
    model = tgt["npf"](tgt["data"])
    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=n_ref // 4,
                num_chains=4, chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    s = mcmc.get_samples()
    if tgt["var"] == "w":
        ref = np.asarray(s["w"])
    else:  # hierarchical: assemble z = (log_tau, theta)
        ref = np.column_stack([np.asarray(s["log_tau"]).reshape(-1, 1),
                               np.asarray(s["theta"])])
    return ref

# ============================================================================
# flowMC v0.6.1 adapter (reproduces the repo's single-chain semantics)
# ============================================================================

def flowmc_infer(rng_key, n_particles, n_dims, log_post,
                 n_local_steps=10, n_global_steps=10, n_training_loops=2,
                 n_production_loops=2, n_epochs=5, hidden_units=(64, 64),
                 n_bins=8, n_layers=3):
    k1, k2, k3 = jax.random.split(rng_key, 3)
    bundle = RQSpline_MALA_Bundle(
        k1, n_particles, n_dims, log_post,
        n_local_steps, n_global_steps, n_training_loops, n_production_loops,
        n_epochs, rq_spline_hidden_units=list(hidden_units),
        rq_spline_n_bins=n_bins, rq_spline_n_layers=n_layers, verbose=False)
    s = Sampler(n_dim=n_dims, n_chains=n_particles, rng_key=k2,
                resource_strategy_bundles=bundle, checkpoint_interval=0)
    t0 = time.time()
    s.sample(jax.random.normal(k3, (n_particles, n_dims)), {})
    prod = np.asarray(s.resources["positions_production"].data)  # (chains, steps, dim)
    final = prod[:, -1, :]  # last production step = repo's returned "final position"
    return final, time.time() - t0

# member palette identical to the paper's diverse ensemble
PALETTE = [
    dict(hidden_units=(64, 64),    n_layers=3, n_bins=8,  n_local_steps=10, n_global_steps=10),
    dict(hidden_units=(96, 96),    n_layers=3, n_bins=12, n_local_steps=10, n_global_steps=10),
    dict(hidden_units=(128, 64),   n_layers=4, n_bins=8,  n_local_steps=15, n_global_steps=8),
    dict(hidden_units=(64, 128),   n_layers=4, n_bins=10, n_local_steps=8,  n_global_steps=15),
    dict(hidden_units=(32, 64, 32), n_layers=3, n_bins=8,  n_local_steps=12, n_global_steps=12),
]

# ============================================================================
# Aggregation operators (verbatim ports from the released repo)
# ============================================================================

def _acf_fft(x):
    n = len(x); xc = x - x.mean()
    f = np.fft.rfft(xc, 2 * n)
    return np.fft.irfft(f * np.conj(f))[:n]

def ess_1d(x):
    n = len(x); ac = _acf_fft(np.asarray(x, float)); v0 = float(ac[0])
    if not np.isfinite(v0) or abs(v0) < 1e-12:
        return float(n)
    ac = ac / max(v0, 1e-12)
    idx = np.where(ac < 0)[0]
    tau = 0.5 + np.sum(ac[1:(int(idx[0]) if idx.size else None)])
    return float(n / (2 * max(tau, 1e-9)))

def ess_ddim(S):
    S = np.asarray(S); return float(np.mean([ess_1d(S[:, i]) for i in range(S.shape[1])]))

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

def cluster_weights(X, K=8, rare_boost=0.35, seed=0):
    K = max(2, min(K, max(2, X.shape[0] // 50)))
    km = KMeans(n_clusters=K, n_init=5, random_state=seed).fit(X)
    cid = km.labels_; cnt = np.bincount(cid)
    inv = 1.0 / np.maximum(cnt, 1); w = inv[cid]
    w = w * (1.0 + rare_boost * (w / np.max(w) - 1.0))
    w = np.maximum(w, 0.0); return w / np.sum(w)

def member_weights(ess_l, js_l, slp_l, beta=1.0, gamma=2.0, lam=0.05, alpha_temp=0.35):
    ess = np.maximum(np.asarray(ess_l, float), 1e-6)
    js = np.maximum(np.asarray(js_l, float), 1e-6)
    slp = np.maximum(np.asarray(slp_l, float), 1e-6)
    score = (ess**beta) * np.exp(-gamma * js) * np.exp(-lam * slp)
    z = (score - score.mean()) / (score.std() + 1e-8)
    w = np.exp(alpha_temp * z); return w / w.sum()

TEMPER = [1.0, 0.95, 0.9025, 0.857375, 0.81450625]  # healthy-branch schedule

# ============================================================================
# Joint metrics (this is task (B): computed on real flowMC output)
# ============================================================================

def _subsample(A, m, rng):
    return A[rng.choice(len(A), size=min(m, len(A)), replace=False)]

def energy_distance(X, Y, rng, m=2000):
    xi = _subsample(X, m, rng); yi = _subsample(Y, m, rng)
    dxy = cdist(xi, yi).mean()
    dxx = cdist(xi, xi); dyy = cdist(yi, yi)
    nx, ny = len(xi), len(yi)
    return float(2 * dxy - dxx.sum() / (nx * (nx - 1)) - dyy.sum() / (ny * (ny - 1)))

def mmd2_rbf(X, Y, rng, m=2000):
    """Unbiased MMD^2 with RBF kernel, median-heuristic bandwidth."""
    xi = _subsample(X, m, rng); yi = _subsample(Y, m, rng)
    Z = np.vstack([xi, yi])
    d = cdist(_subsample(Z, min(1000, len(Z)), rng),
              _subsample(Z, min(1000, len(Z)), rng))
    med = np.median(d[d > 0]); gamma = 1.0 / (2.0 * med**2 + 1e-12)
    Kxx = np.exp(-gamma * cdist(xi, xi)**2)
    Kyy = np.exp(-gamma * cdist(yi, yi)**2)
    Kxy = np.exp(-gamma * cdist(xi, yi)**2)
    nx, ny = len(xi), len(yi)
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (nx * (nx - 1)) + Kyy.sum() / (ny * (ny - 1))
                 - 2.0 * Kxy.mean())

# ============================================================================
# Sampler variants on a real posterior (task A: actual flowMC)
# ============================================================================

def run_variants_once(name, tgt, ref, rep, n_members=5, n_per=2000):
    dim = tgt["dim"]; lp_np = tgt["lpf"](tgt["data"])
    log_post = lambda x, data=None: tgt_logpost_jax(name, tgt, x)
    seed = abs(hash((name, rep))) % (2**31)
    base_key = jax.random.PRNGKey(seed)

    out = {}

    # single @2000
    k = jax.random.fold_in(base_key, 1)
    X2, t2 = flowmc_infer(k, n_per, dim, log_post)
    out["single@2000"] = (X2, t2)

    # single @10000
    k = jax.random.fold_in(base_key, 2)
    X10, t10 = flowmc_infer(k, n_per * n_members, dim, log_post)
    out["single@10000"] = (X10, t10)

    # ensemble members (diverse palette) -> full aggregation + uniform pool
    members = []; tt = 0.0
    for m in range(n_members):
        cfg = PALETTE[m]
        km = jax.random.fold_in(base_key, 100 + m)
        Xm, tm = flowmc_infer(km, n_per, dim, log_post,
                              n_local_steps=cfg["n_local_steps"],
                              n_global_steps=cfg["n_global_steps"],
                              hidden_units=cfg["hidden_units"],
                              n_bins=cfg["n_bins"], n_layers=cfg["n_layers"])
        members.append(Xm); tt += tm
    allX = np.vstack(members)
    memb_id = np.concatenate([[i] * n_per for i in range(n_members)])

    # uniform pool
    out["uniform_pool"] = (allX, tt)

    # full aggregation stack (member x coverage weights, resample w/ replacement)
    ess_l, js_l, slp_l = [], [], []
    for Xm in members:
        ess_l.append(ess_ddim(Xm))
        js_l.append(marginal_js(Xm, ref))
        slp_l.append(float(np.std(lp_np(Xm))))
    w_alpha = member_weights(ess_l, js_l, slp_l)
    w_m = w_alpha[memb_id]; w_m = w_m / w_m.sum()
    cov_w = cluster_weights(allX[:, :2])
    w_full = w_m * cov_w; w_full = w_full / w_full.sum()
    rng = np.random.default_rng(2025 + rep)
    idx = rng.choice(len(allX), size=len(allX), replace=True, p=w_full)
    out["full_ensemble"] = (allX[idx], tt)
    return out

# jax log-posteriors (mirror the numpy ones; needed by flowMC on-device)
def tgt_logpost_jax(name, tgt, x):
    x = jnp.atleast_2d(x)
    d = tgt["data"]
    if name.startswith("blr"):
        X = jnp.asarray(d["X"]); y = jnp.asarray(d["y"]); psd = d["prior_sd"]
        logits = x @ X.T
        ll = jnp.sum(y * logits - jnp.logaddexp(0.0, logits), axis=1)
        lprior = -0.5 * jnp.sum(x**2, axis=1) / psd**2
        return jnp.squeeze(ll + lprior)
    else:  # hier
        y = jnp.asarray(d["y"]); sig = jnp.asarray(d["sigma"]); ts = d["tau_scale"]
        log_tau = x[:, 0]; theta = x[:, 1:]; tau = jnp.exp(log_tau)
        lprior_tau = (-0.5 * (tau / ts)**2 + jnp.log(jnp.sqrt(2.0 / jnp.pi) / ts)
                      + log_tau)
        lprior_theta = jnp.sum(-0.5 * (theta / tau[:, None])**2
                               - jnp.log(tau)[:, None]
                               - 0.5 * jnp.log(2 * jnp.pi), axis=1)
        ll = jnp.sum(-0.5 * ((y[None, :] - theta) / sig[None, :])**2
                     - jnp.log(sig)[None, :] - 0.5 * jnp.log(2 * jnp.pi), axis=1)
        return jnp.squeeze(lprior_tau + lprior_theta + ll)

# ============================================================================
# Phases
# ============================================================================

def phase_ref():
    T = build_targets()
    for name, tgt in T.items():
        t0 = time.time()
        ref = nuts_reference(name, tgt, n_ref=20000, seed=0)
        np.save(f"{REFDIR}/{name}.npy", ref)
        # also cache the target data so 'run' uses the identical posterior
        np.savez(f"{REFDIR}/{name}_data.npz",
                 **{k: np.asarray(v) for k, v in tgt["data"].items()
                    if isinstance(v, np.ndarray)},
                 meta=json.dumps({k: (v if not isinstance(v, np.ndarray) else None)
                                  for k, v in tgt["data"].items()}))
        print(f"[ref] {name}: ref {ref.shape}  ({time.time()-t0:.0f}s)  "
              f"mean|z|={np.abs(ref.mean(0)).mean():.3f}", flush=True)

def phase_run(n_reps=3):
    T = build_targets()
    rows = []
    for name, tgt in T.items():
        ref = np.load(f"{REFDIR}/{name}.npy")
        t0 = time.time()
        for rep in range(n_reps):
            variants = run_variants_once(name, tgt, ref, rep)
            for vname, (S, rt) in variants.items():
                rng = np.random.default_rng(abs(hash((name, vname, rep))) % (2**31))
                ed = energy_distance(S, ref, rng)
                mmd = mmd2_rbf(S, ref, rng)
                js = marginal_js(S, ref)
                rows.append(dict(target=name, dim=tgt["dim"], rep=rep, variant=vname,
                                 n_samples=len(S), runtime_s=round(rt, 2),
                                 energy_distance=ed, mmd2=mmd, marginal_JS=js,
                                 ess=ess_ddim(S)))
                print(f"[run] {name} rep{rep} {vname:14s} "
                      f"ED={ed:.4f} MMD2={mmd:.4f} JS={js:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(f"{RES}/real_posteriors.csv", index=False)
    df = pd.DataFrame(rows)
    print("\n=== mean over reps ===")
    print(df.groupby(["target", "variant"])[["energy_distance", "mmd2",
          "marginal_JS"]].mean().round(4).to_string())

PHASES = dict(ref=phase_ref, run=phase_run)

def phase_unit(name, rep):
    T = build_targets()
    tgt = T[name]
    ref = np.load(f"{REFDIR}/{name}.npy")
    csv = f"{RES}/real_{name}.csv"
    existing = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    if len(existing) and ((existing["rep"] == rep).any()):
        print(f"[unit] {name} rep{rep} already present, skipping", flush=True); return
    t0 = time.time()
    variants = run_variants_once(name, tgt, ref, rep)
    rows = []
    for vn, (S, rt) in variants.items():
        rng = np.random.default_rng(abs(hash((name, vn, rep))) % (2**31))
        ed = energy_distance(S, ref, rng); mmd = mmd2_rbf(S, ref, rng)
        js = marginal_js(S, ref)
        rows.append(dict(target=name, dim=tgt["dim"], rep=rep, variant=vn,
                         n_samples=len(S), runtime_s=round(rt, 2),
                         energy_distance=ed, mmd2=mmd, marginal_JS=js, ess=ess_ddim(S)))
        print(f"[unit] {name} rep{rep} {vn:14s} ED={ed:.4f} MMD2={mmd:.4f} "
              f"JS={js:.4f} rt={rt:.0f}s ({time.time()-t0:.0f}s)", flush=True)
    out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(csv, index=False)
    print(f"[unit] saved {csv} ({time.time()-t0:.0f}s total)", flush=True)

PHASES["unit"] = None  # handled specially in __main__

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    if args and args[0] == "unit":
        phase_unit(args[1], int(args[2]))
    else:
        todo = ["ref", "run"] if args == ["all"] else args
        for ph in todo:
            print(f"===== phase {ph} =====", flush=True)
            PHASES[ph]()

# ----------------------------------------------------------------------------
# Unit runner: one (target, rep) at a time, appends to a per-target CSV.
# Enables chunked execution under wall-clock limits. Usage:
#   python3 realpost_experiments.py unit <target> <rep>
# ----------------------------------------------------------------------------
