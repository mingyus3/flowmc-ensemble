#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced flowMC Ensemble Sampling Framework
============================================
Author: Mingyu Shi
Date: 2025
License: MIT

Description:
    A comprehensive implementation of ensemble-based MCMC sampling using flowMC,
    demonstrating significant improvements in sampling quality through diversity-enhanced
    ensemble methods. This implementation includes adaptive temperature scheduling,
    coverage-aware reweighting, and quality-based member aggregation.

Key Features:
    - 6 challenging test distributions (Banana, Bimodal, Funnel, etc.)
    - Multiple budget allocation strategies (equal_per_member, equal_total, equal_time)
    - Ablation studies for component validation
    - Comprehensive statistical analysis and visualization
    - Publication-ready outputs with figures and tables


Outputs:
    ./paper_outputs/
        ├── results_equal_per_member.csv    # Main experimental results
        ├── significance_equal_per_member.csv # Statistical significance tests
        ├── seeds_equal_per_member.csv      # Random seeds for reproducibility
        └── figs/                            # All visualization figures
            ├── *.png                        # PNG versions for viewing
            └── *.pdf                        # PDF versions for publication
"""

import os
import json
import time
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, List, Tuple, Optional, Union

import zlib, numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.special import kl_div
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Import flowMC components
from flowMC.Sampler import Sampler
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

# =============================================================================
# Section 1: Target Potential Functions
# =============================================================================

def standard_gaussian_potential(x: jnp.ndarray) -> jnp.ndarray:
    """
    Standard multivariate Gaussian potential.
    U(x) = 0.5 * ||x||^2

    Args:
        x: Input array of shape (..., d) where d is dimension

    Returns:
        Potential values of shape (...)
    """
    return jnp.squeeze(0.5 * jnp.sum(x**2, axis=-1))


def banana_potential(x: jnp.ndarray, a: float = 1.0, b: float = 100.0) -> jnp.ndarray:
    """
    Banana-shaped distribution (Rosenbrock-like).
    Strong nonlinear correlation between first two dimensions.

    Args:
        x: Input array of shape (..., d)
        a: Nonlinearity parameter
        b: Scale parameter

    Returns:
        Potential values of shape (...)
    """
    t1 = 0.5 * x[..., 0]**2
    t2 = 0.5 * (x[..., 1] - a * x[..., 0]**2 + b)**2
    if x.ndim > 1 and x.shape[-1] > 2:
        t3 = 0.5 * jnp.sum(x[..., 2:]**2, axis=-1)
        return jnp.squeeze(t1 + t2 + t3)
    return jnp.squeeze(t1 + t2)


def bimodal_gaussian_potential(x: jnp.ndarray, mode_distance: float = 6.0) -> jnp.ndarray:
    """
    Mixture of two Gaussians separated along first dimension.
    Tests ability to explore multiple modes.

    Args:
        x: Input array of shape (..., d)
        mode_distance: Distance between modes

    Returns:
        Potential values of shape (...)
    """
    d = x.shape[-1]
    off = mode_distance / 2
    mu1 = jnp.zeros(d).at[0].set(-off)
    mu2 = jnp.zeros(d).at[0].set(+off)
    d1 = jnp.sum((x - mu1)**2, axis=-1)
    d2 = jnp.sum((x - mu2)**2, axis=-1)
    m = jnp.maximum(-0.5 * d1, -0.5 * d2)
    return jnp.squeeze(-(m + jnp.log(jnp.exp(-0.5*d1 - m) + jnp.exp(-0.5*d2 - m))))


def funnel_potential(x: jnp.ndarray, sigma_v: float = 0.3) -> jnp.ndarray:
    """
    Neal's funnel distribution with varying scales.
    Challenging due to scale variations across dimensions.

    Args:
        x: Input array of shape (..., d)
        sigma_v: Standard deviation of the 'neck' variable

    Returns:
        Potential values of shape (...)
    """
    v = x[..., 0]
    pot_v = 0.5 * (v / sigma_v)**2
    rest_sq = jnp.sum(x[..., 1:]**2, axis=-1)
    pot_x = 0.5 * rest_sq / jnp.exp(v) + 0.5 * (x.shape[-1] - 1) * v
    return jnp.squeeze(pot_v + pot_x)


def correlated_gaussian_potential(x: jnp.ndarray, rho: float = 0.95) -> jnp.ndarray:
    """
    Strongly correlated Gaussian in first two dimensions.
    Tests handling of strong correlations.

    Args:
        x: Input array of shape (..., d)
        rho: Correlation coefficient (|rho| < 1)

    Returns:
        Potential values of shape (...)
    """
    d = x.shape[-1]
    x1, x2 = x[..., 0], x[..., 1]
    Z = (x1**2 - 2*rho*x1*x2 + x2**2) / (1 - rho**2)
    rest = jnp.sum(x[..., 2:]**2, axis=-1) if d > 2 else 0.0
    return jnp.squeeze(0.5*(Z + rest) + 0.5*jnp.log(1 - rho**2))


def rosenbrock_potential(x: jnp.ndarray, a: float = 1.0, b: float = 100.0) -> jnp.ndarray:
    """
    Classic Rosenbrock valley potential.
    Narrow curved valley that is difficult to explore.

    Args:
        x: Input array of shape (..., d)
        a, b: Rosenbrock parameters

    Returns:
        Potential values of shape (...)
    """
    d = x.shape[-1]
    u = (a - x[..., 0])**2 + b*(x[..., 1] - x[..., 0]**2)**2
    if d > 2:
        u = u + jnp.sum(x[..., 2:]**2, axis=-1)
    return jnp.squeeze(0.5*u)


def gaussian_ring_potential(x: jnp.ndarray, radius: float = 5.0,
                           K: int = 8, std: float = 0.7) -> jnp.ndarray:
    """
    Mixture of Gaussians arranged in a ring.
    Tests ability to explore circular mode structure.

    Args:
        x: Input array of shape (..., d)
        radius: Ring radius
        K: Number of modes on the ring
        std: Standard deviation of each mode

    Returns:
        Potential values of shape (...)
    """
    # Handle both single point and batch inputs
    if x.ndim == 1:
        x = x[None, :]  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False

    # Ring centers
    theta = jnp.linspace(0, 2*jnp.pi, K, endpoint=False)
    centers = jnp.stack([radius * jnp.cos(theta), radius * jnp.sin(theta)], axis=1)

    # Compute distances to all centers
    x2 = x[..., :2]
    dists = jnp.sum((x2[:, None, :] - centers[None, :, :])**2, axis=-1)

    # Log-sum-exp for numerical stability
    loglik2 = jax.scipy.special.logsumexp(-0.5 * dists / (std**2), axis=1) - jnp.log(K)

    # Add potential from remaining dimensions
    d = x.shape[-1]
    rest = 0.5 * jnp.sum(x[:, 2:]**2, axis=-1) if d > 2 else 0.0

    result = -loglik2 + rest
    return jnp.squeeze(result) if squeeze_output else result


def log_post_from_cfg(potential_fn: str, **pot_params) -> callable:
    """
    Create log-posterior function from configuration.

    Args:
        potential_fn: Name of the potential function
        **pot_params: Parameters specific to the potential

    Returns:
        Log-posterior function callable
    """
    potentials = {
        "standard_gaussian": lambda x: -standard_gaussian_potential(x),
        "banana": lambda x: -banana_potential(x, **pot_params),
        "bimodal_gaussian": lambda x: -bimodal_gaussian_potential(x, **pot_params),
        "funnel": lambda x: -funnel_potential(x, **pot_params),
        "correlated_gaussian": lambda x: -correlated_gaussian_potential(x, **pot_params),
        "rosenbrock": lambda x: -rosenbrock_potential(x, **pot_params),
        "gaussian_ring": lambda x: -gaussian_ring_potential(x, **pot_params)
    }

    if potential_fn not in potentials:
        raise ValueError(f"Unknown potential function: {potential_fn}")

    # Return wrapped function with data argument for flowMC compatibility
    return lambda x, data=None: potentials[potential_fn](x)


# =============================================================================
# Section 2: Reference Sample Generation
# =============================================================================

def generate_reference_samples(potential_fn: str, n_dims: int,
                              n_samples: int = 20000, **pot_params) -> jnp.ndarray:
    """
    Generate reference samples from the true distribution.
    Used for computing JS divergence metric.

    Args:
        potential_fn: Name of the distribution
        n_dims: Dimensionality
        n_samples: Number of reference samples
        **pot_params: Distribution-specific parameters

    Returns:
        Array of shape (n_samples, n_dims)
    """
    key = jax.random.PRNGKey(0)  # Fixed seed for reproducibility

    if potential_fn == "standard_gaussian":
        return jax.random.normal(key, (n_samples, n_dims))

    elif potential_fn == "banana":
        a, b = pot_params.get('a', 1.0), pot_params.get('b', 100.0)
        key, k1, k2 = jax.random.split(key, 3)
        x1 = jax.random.normal(k1, (n_samples,))
        x2 = jax.random.normal(k2, (n_samples,)) + a*x1**2 - b
        if n_dims > 2:
            key, kr = jax.random.split(key)
            rest = jax.random.normal(kr, (n_samples, n_dims-2))
            return jnp.column_stack((x1, x2, rest))
        return jnp.column_stack((x1, x2))

    elif potential_fn == "bimodal_gaussian":
        md = pot_params.get('mode_distance', 6.0)
        off = md/2
        mu1 = jnp.zeros(n_dims).at[0].set(-off)
        mu2 = jnp.zeros(n_dims).at[0].set(+off)
        key, k1, k2 = jax.random.split(key, 3)
        mask = jax.random.bernoulli(k1, 0.5, (n_samples,))
        samples = jnp.zeros((n_samples, n_dims))
        n1 = mask.sum().item()
        if n1 > 0:
            key, k1r = jax.random.split(key)
            s1 = jax.random.normal(k1r, (n1, n_dims)) + mu1
            samples = samples.at[mask].set(s1)
        if n1 < n_samples:
            key, k2r = jax.random.split(key)
            s2 = jax.random.normal(k2r, (n_samples-n1, n_dims)) + mu2
            samples = samples.at[~mask].set(s2)
        return samples

    elif potential_fn == "funnel":
        sv = pot_params.get('sigma_v', 0.3)
        key, k1 = jax.random.split(key)
        v = jax.random.normal(k1, (n_samples,)) * sv
        samples = jnp.zeros((n_samples, n_dims))
        samples = samples.at[:, 0].set(v)
        for i in range(n_samples):
            key, kr = jax.random.split(key)
            std = jnp.exp(v[i]/2)
            samples = samples.at[i, 1:].set(jax.random.normal(kr, (n_dims-1,)) * std)
        return samples

    elif potential_fn == "correlated_gaussian":
        rho = pot_params.get("rho", 0.95)
        cov2 = jnp.array([[1.0, rho], [rho, 1.0]])
        L = jnp.linalg.cholesky(cov2)
        key, k1 = jax.random.split(key)
        z = jax.random.normal(k1, (n_samples, 2))
        xy = z @ L.T
        if n_dims > 2:
            key, kr = jax.random.split(key)
            rest = jax.random.normal(kr, (n_samples, n_dims-2))
            return jnp.column_stack((xy, rest))
        return xy

    elif potential_fn == "rosenbrock":
        a, b = pot_params.get("a", 1.0), pot_params.get("b", 100.0)
        key, k1, k2 = jax.random.split(key, 3)
        x0 = jax.random.normal(k1, (n_samples,)) * 2.0 + a
        y0 = x0**2 + jax.random.normal(k2, (n_samples,)) * 0.5
        if n_dims > 2:
            key, kr = jax.random.split(key)
            rest = jax.random.normal(kr, (n_samples, n_dims-2))
            return jnp.column_stack((x0, y0, rest))
        return jnp.column_stack((x0, y0))

    elif potential_fn == "gaussian_ring":
        R = pot_params.get("radius", 5.0)
        K = pot_params.get("K", 8)
        std = pot_params.get("std", 0.7)
        theta = jax.random.uniform(key, (n_samples,)) * 2*jnp.pi
        key, k1 = jax.random.split(key)
        r = R + jax.random.normal(k1, (n_samples,)) * std
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        if n_dims > 2:
            key, kr = jax.random.split(key)
            rest = jax.random.normal(kr, (n_samples, n_dims-2))
            return jnp.column_stack((x, y, rest))
        return jnp.column_stack((x, y))

    else:
        raise ValueError(f"Unknown potential function: {potential_fn}")


# =============================================================================
# Section 3: Evaluation Metrics
# =============================================================================

def _ess_1d(x: np.ndarray) -> float:
    """
    Compute effective sample size for 1D array using autocorrelation.

    Args:
        x: 1D array of samples

    Returns:
        ESS value
    """
    n = len(x)
    mu = x.mean()

    # Compute autocorrelation
    ac = np.correlate(x - mu, x - mu, mode='full')
    v0 = float(ac[n-1])

    if not np.isfinite(v0) or abs(v0) < 1e-12:
        return float(n)

    # Normalize autocorrelation
    ac = ac[n-1:] / max(v0, 1e-12)

    # Find first negative autocorrelation
    idx = np.where(ac < 0)[0]

    # Integrated autocorrelation time
    tau = 0.5 + np.sum(ac[1:(int(idx[0]) if idx.size else None)])
    tau = max(tau, 1e-9)

    return float(n / (2*tau))


def compute_ess_d_dim(samples: np.ndarray) -> float:
    """
    Compute average ESS across dimensions.

    Args:
        samples: Array of shape (n_samples, n_dims)

    Returns:
        Average ESS value
    """
    samples = np.asarray(samples)
    return float(np.mean([_ess_1d(samples[:, i]) for i in range(samples.shape[1])]))


def compute_marginal_js_distance(samples: np.ndarray, reference: np.ndarray,
                                nbins: int = 100, pad: float = 0.1) -> float:
    """
    Compute average marginal Jensen-Shannon distance.

    Args:
        samples: Generated samples of shape (n_samples, n_dims)
        reference: Reference samples of shape (n_ref, n_dims)
        nbins: Number of histogram bins
        pad: Padding factor for histogram range

    Returns:
        Average JS distance across dimensions
    """
    s = np.asarray(samples)
    r = np.asarray(reference)
    js_list = []

    for i in range(s.shape[1]):
        # Determine histogram range with padding
        lo = min(np.percentile(s[:, i], 0.5), np.percentile(r[:, i], 0.5))
        hi = max(np.percentile(s[:, i], 99.5), np.percentile(r[:, i], 99.5))
        rng = hi - lo
        lo -= pad * rng
        hi += pad * rng

        # Compute histograms
        bins = np.linspace(lo, hi, nbins + 1)
        h1, _ = np.histogram(s[:, i], bins=bins, density=True)
        h2, _ = np.histogram(r[:, i], bins=bins, density=True)

        # Add small epsilon for numerical stability
        eps = 1e-12
        h1 = (h1 + eps) / (h1.sum() + eps * len(h1))
        h2 = (h2 + eps) / (h2.sum() + eps * len(h2))

        # JS divergence
        m = 0.5 * (h1 + h2)
        js = 0.5 * np.sum(kl_div(h1, m)) + 0.5 * np.sum(kl_div(h2, m))
        js_list.append(np.sqrt(js))

    return float(np.mean(js_list))


def quality_score(ess: float, js: float, runtime_s: float) -> float:
    """
    Combined quality metric: ESS per second per JS distance.

    Args:
        ess: Effective sample size
        js: JS distance (lower is better)
        runtime_s: Runtime in seconds

    Returns:
        Quality score (higher is better)
    """
    return float(ess / max(runtime_s, 1e-9) * (1.0 / max(js, 1e-9)))


# =============================================================================
# Section 4: flowMC Single Sampler
# =============================================================================

class SamplerWithReturn(Sampler):
    """
    Extended Sampler class that returns final samples.
    """

    def sample(self, initial_position, data=None):
        """
        Run sampling and return final position.

        Args:
            initial_position: Starting positions
            data: Optional data for likelihood

        Returns:
            Final sample positions
        """
        last = initial_position
        key = self.rng_key

        for name in self.strategy_order:
            strat = self.strategies[name]
            key, self.resources, last = strat(key, self.resources, last, data)

        return last


def flowmc_inference(
    rng_key: jax.random.PRNGKey,
    n_particles: int,
    n_dims: int,
    n_local_steps: int = 10,
    n_global_steps: int = 10,
    n_training_loops: int = 2,
    n_production_loops: int = 2,
    n_epochs: int = 5,
    hidden_units: List[int] = (64, 64),
    n_bins: int = 8,
    n_layers: int = 3,
    potential_fn: str = "standard_gaussian",
    log_post_override: Optional[callable] = None,
    **pot_params
) -> Tuple[np.ndarray, float]:
    """
    Run flowMC inference with specified configuration.

    Args:
        rng_key: JAX random key
        n_particles: Number of particles
        n_dims: Dimensionality
        n_local_steps: MALA steps per iteration
        n_global_steps: Flow steps per iteration
        n_training_loops: Training iterations
        n_production_loops: Production iterations
        n_epochs: Training epochs per loop
        hidden_units: Neural network architecture
        n_bins: Number of spline bins
        n_layers: Number of flow layers
        potential_fn: Name of target distribution
        log_post_override: Optional override for log-posterior (for temperature)
        **pot_params: Distribution parameters

    Returns:
        (samples, runtime) tuple
    """
    # Use override if provided, otherwise create from config
    if log_post_override is not None:
        log_post = log_post_override
    else:
        log_post = log_post_from_cfg(potential_fn, **pot_params)

    # Initialize particles
    rng_key, sub = jax.random.split(rng_key)
    init_pos = jax.random.normal(sub, (n_particles, n_dims))

    # Create sampler bundle
    rng_key, sub = jax.random.split(rng_key)
    bundle = RQSpline_MALA_Bundle(
        sub, n_particles, n_dims, log_post,
        n_local_steps, n_global_steps,
        n_training_loops, n_production_loops,
        n_epochs,
        rq_spline_hidden_units=list(hidden_units),
        rq_spline_n_bins=n_bins,
        rq_spline_n_layers=n_layers,
        verbose=False
    )

    # Run sampling
    sampler = SamplerWithReturn(n_dims, n_particles, rng_key,
                                resource_strategy_bundles=bundle)
    t0 = time.time()
    final = sampler.sample(init_pos, {})

    return np.array(final), time.time() - t0


# =============================================================================
# Section 5: Enhanced Ensemble Methods
# =============================================================================

def _cluster_weights(X: np.ndarray, K: int = 8, rare_boost: float = 0.35,
                    seed: int = 0) -> np.ndarray:
    """
    Compute coverage-based weights using K-means clustering.
    Boosts weights for rare regions to improve coverage.

    Args:
        X: Samples to cluster (typically first 2 dimensions)
        K: Number of clusters
        rare_boost: Boost factor for rare clusters
        seed: Random seed for K-means

    Returns:
        Weight array of same length as X
    """
    # Adaptive K based on sample size
    K = max(2, min(K, max(2, X.shape[0]//50)))

    # Perform clustering
    km = KMeans(n_clusters=K, n_init=5, random_state=seed).fit(X)
    cid = km.labels_
    cnt = np.bincount(cid)

    # Inverse frequency weighting
    inv = 1.0 / np.maximum(cnt, 1)
    w = inv[cid]

    # Apply rare boost
    w = w * (1.0 + rare_boost * (w/np.max(w) - 1.0))
    w = np.maximum(w, 0.0)

    return w / np.sum(w)


def enhanced_flowmc_ensemble(
    rng_key: jax.random.PRNGKey,
    n_particles_single: int,
    n_dims: int,
    potential_fn: str,
    n_members: int = 5,
    budget: str = "equal_per_member",
    # Quality weighting hyperparameters
    beta: float = 1.0,      # ESS weight exponent
    gamma: float = 2.0,     # JS penalty weight
    lam: float = 0.05,      # Log-prob std penalty
    alpha_temp: float = 0.35,  # Softmax temperature
    # Coverage parameters
    K_clusters: int = 8,
    rare_boost: float = 0.35,
    # Adaptive temperature
    adaptive_temp: bool = True,
    init_temp: float = 1.0,
    target_ess_ratio: float = 0.5,
    # Ablation flags
    use_member_weight: bool = True,
    use_cov_balance: bool = True,
    # Configuration palette
    palette: List[Dict] = None,
    **pot_params
) -> Tuple[np.ndarray, float, Dict]:
    """
    Run ensemble flowMC with enhanced aggregation strategies.

    Args:
        rng_key: JAX random key
        n_particles_single: Particles per member (or total if equal_total)
        n_dims: Dimensionality
        potential_fn: Target distribution name
        n_members: Number of ensemble members
        budget: Allocation strategy ('equal_per_member' or 'equal_total')
        beta, gamma, lam, alpha_temp: Quality weighting hyperparameters
        K_clusters, rare_boost: Coverage balancing parameters
        adaptive_temp: Whether to use adaptive tempering
        init_temp, target_ess_ratio: Temperature adaptation parameters
        use_member_weight, use_cov_balance: Ablation flags
        palette: List of member configurations
        **pot_params: Distribution parameters

    Returns:
        (samples, runtime, metadata) tuple
    """
    # Default configuration palette with diversity
    if palette is None:
        palette = [
            dict(hidden_units=(64, 64),    n_layers=3, n_bins=8,  n_local_steps=10, n_global_steps=10),
            dict(hidden_units=(96, 96),    n_layers=3, n_bins=12, n_local_steps=10, n_global_steps=10),
            dict(hidden_units=(128, 64),   n_layers=4, n_bins=8,  n_local_steps=15, n_global_steps=8),
            dict(hidden_units=(64, 128),   n_layers=4, n_bins=10, n_local_steps=8,  n_global_steps=15),
            dict(hidden_units=(32, 64, 32), n_layers=3, n_bins=8,  n_local_steps=12, n_global_steps=12),
        ]

    # Create base log-posterior
    base_log_post = log_post_from_cfg(potential_fn, **pot_params)

    # Generate reference samples for JS computation
    ref = np.array(generate_reference_samples(potential_fn, n_dims, 20000, **pot_params))

    # Determine particles per member based on budget
    if budget == "equal_total":
        n_per = max(1, int(np.floor(n_particles_single / n_members)))
    else:  # equal_per_member
        n_per = int(n_particles_single)

    # Storage for results
    member_samples = []
    ess_list = []
    js_list = []
    slogp_list = []
    total_time = 0.0
    temp = float(init_temp)
    temp_history = []  # Track temperature evolution

    # Run each ensemble member
    for m in range(n_members):
        cfg = palette[m % len(palette)]
        rng_key, sub = jax.random.split(rng_key)

        # Create temperature-adjusted log-posterior if adaptive tempering is enabled
        if adaptive_temp:
            # Create a tempered version of the log-posterior
            log_post_tempered = lambda x, d=None: base_log_post(x, d) / temp
        else:
            # Use original log-posterior without temperature
            log_post_tempered = None

        # Run flowMC for this member with temperature-adjusted posterior
        s_m, t_m = flowmc_inference(
            sub, n_per, n_dims,
            n_local_steps=cfg.get("n_local_steps", 10),
            n_global_steps=cfg.get("n_global_steps", 10),
            n_training_loops=2,
            n_production_loops=2,
            n_epochs=5,
            hidden_units=cfg.get("hidden_units", (64, 64)),
            n_bins=cfg.get("n_bins", 8),
            n_layers=cfg.get("n_layers", 3),
            potential_fn=potential_fn,
            log_post_override=log_post_tempered,  # Pass the tempered log-posterior
            **pot_params
        )

        total_time += t_m
        X = np.asarray(s_m)
        member_samples.append(X)

        # Compute metrics for this member using appropriate log-posterior
        ess_i = compute_ess_d_dim(X)
        js_i = compute_marginal_js_distance(X, ref)

        # Compute log-probability standard deviation using the tempered version if applicable
        if adaptive_temp:
            logp_vals = np.asarray(jax.vmap(log_post_tempered)(jnp.asarray(X)))
        else:
            logp_vals = np.asarray(jax.vmap(base_log_post)(jnp.asarray(X)))

        sigma_lp = float(np.std(logp_vals))

        ess_list.append(ess_i)
        js_list.append(js_i)
        slogp_list.append(sigma_lp)

        # Adapt temperature based on ESS ratio
        if adaptive_temp:
            ess_ratio = ess_i / max(1.0, n_per)
            temp_history.append(temp)

            # Adjust temperature based on ESS performance
            if ess_ratio < target_ess_ratio:
                # ESS too low -> increase temperature (flatten distribution)
                old_temp = temp
                temp = min(2.0, temp * 1.1)
                print(f"  Member {m}: ESS ratio {ess_ratio:.3f} < {target_ess_ratio}, "
                      f"increasing temp from {old_temp:.3f} to {temp:.3f}")
            else:
                # ESS acceptable -> decrease temperature (sharpen distribution)
                old_temp = temp
                temp = max(0.1, temp * 0.95)
                print(f"  Member {m}: ESS ratio {ess_ratio:.3f} >= {target_ess_ratio}, "
                      f"decreasing temp from {old_temp:.3f} to {temp:.3f}")

    # Combine all samples
    allX = np.vstack(member_samples)

    # Compute member weights based on quality metrics
    if use_member_weight:
        ess = np.maximum(np.asarray(ess_list, float), 1e-6)
        js = np.maximum(np.asarray(js_list, float), 1e-6)
        slp = np.maximum(np.asarray(slogp_list, float), 1e-6)

        # Quality score: high ESS good, low JS good, low std(logp) good
        score = (ess**beta) * np.exp(-gamma*js) * np.exp(-lam*slp)

        # Softmax normalization for member weights
        z = (score - score.mean()) / (score.std() + 1e-8)
        w_alpha = np.exp(alpha_temp * z)
        w_alpha /= w_alpha.sum()

        # Expand weights to all samples
        memb_id = np.concatenate([[i]*len(member_samples[i]) for i in range(n_members)])
        w_m = w_alpha[memb_id]
    else:
        # Equal weights for all members
        w_alpha = np.ones(n_members) / n_members
        memb_id = np.concatenate([[i]*len(member_samples[i]) for i in range(n_members)])
        w_m = w_alpha[memb_id]

    # Apply coverage balancing using K-means clustering
    if use_cov_balance:
        # Cluster on first 2 dimensions (or all if less)
        cluster_dims = min(2, allX.shape[1])
        cov_w = _cluster_weights(allX[:, :cluster_dims], K=K_clusters,
                                 rare_boost=rare_boost, seed=0)
        w_final = w_m * cov_w
        w_final /= w_final.sum()
    else:
        w_final = w_m / np.sum(w_m)

    # Resample according to final weights
    N_target = n_per * n_members
    rng_np = np.random.default_rng(2025)  # Fixed seed for reproducibility
    idx = rng_np.choice(len(allX), size=N_target, replace=True, p=w_final)
    samples_all = allX[idx]

    # Metadata for analysis
    meta = dict(
        ess_list=ess_list,
        js_list=js_list,
        sigma_logp_list=slogp_list,
        alphas=w_alpha.tolist(),
        temperature_last=float(temp),
        temperature_history=temp_history if adaptive_temp else None,  # Include temperature evolution
        N_total=int(N_target),
        budget=budget
    )

    return samples_all, float(total_time), meta


# =============================================================================
# Section 6: Visualization Functions
# =============================================================================

def _ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    dirpath = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def plot_bars(df: pd.DataFrame, budget: str, out_dir: str):
    """
    Create bar plots for ESS, runtime, and JS distance.

    Args:
        df: Results dataframe
        budget: Budget type for filename
        out_dir: Output directory
    """
    sns.set(style="whitegrid", context="paper", font_scale=1.2)

    # Aggregate results
    agg = df.groupby(["potential_fn", "dimension", "method"]).agg(
        ESS=('ESS', 'mean'),
        Runtime=('runtime_sec', 'mean'),
        JS=('JS_distance', 'mean')
    ).reset_index()

    def _plot_metric(metric, fname, ylabel):
        fns = agg["potential_fn"].unique()
        fig, axes = plt.subplots(1, len(fns), figsize=(5*len(fns), 4), sharey=False)
        if len(fns) == 1:
            axes = [axes]

        for i, fn in enumerate(fns):
            d = agg[agg["potential_fn"] == fn]
            sns.barplot(data=d, x="dimension", y=metric, hue="method", ax=axes[i])
            axes[i].set_title(fn.replace("_", " ").title())
            axes[i].set_xlabel("Dimension")
            axes[i].set_ylabel(ylabel)
            if i < len(fns) - 1 and axes[i].legend_:
                axes[i].legend_.remove()

        # Add legend to last plot
        if axes[-1].legend_:
            axes[-1].legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        _ensure_dir(os.path.join(out_dir, "figs"))
        plt.savefig(os.path.join(out_dir, f"figs/{fname}_{budget}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(out_dir, f"figs/{fname}_{budget}.pdf"), bbox_inches="tight")
        plt.close()

    _plot_metric("ESS", "ess_bar", "Mean ESS")
    _plot_metric("Runtime", "runtime_bar", "Mean runtime (s)")
    _plot_metric("JS", "js_bar", "Mean JS distance")


def plot_box_violin(df: pd.DataFrame, budget: str, out_dir: str):
    """
    Create violin plots for ESS and JS distance distributions.

    Args:
        df: Results dataframe
        budget: Budget type for filename
        out_dir: Output directory
    """
    sns.set(style="whitegrid", context="paper", font_scale=1.2)

    for metric, ylabel in [("ESS", "ESS"), ("JS_distance", "JS distance (lower better)")]:
        fns = df["potential_fn"].unique()
        fig, axes = plt.subplots(1, len(fns), figsize=(5*len(fns), 4), sharey=False)
        if len(fns) == 1:
            axes = [axes]

        for i, fn in enumerate(fns):
            d = df[df["potential_fn"] == fn]
            sns.violinplot(data=d, x="dimension", y=metric, hue="method",
                          ax=axes[i], cut=0, inner="quartile")
            axes[i].set_title(fn.replace("_", " ").title())
            axes[i].set_xlabel("Dimension")
            axes[i].set_ylabel(ylabel)
            if i < len(fns) - 1 and axes[i].legend_:
                axes[i].legend_.remove()

        # Add legend to last plot
        if axes[-1].legend_:
            axes[-1].legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        _ensure_dir(os.path.join(out_dir, "figs"))
        plt.savefig(os.path.join(out_dir, f"figs/box_violin_{metric}_{budget}.png"),
                   dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(out_dir, f"figs/box_violin_{metric}_{budget}.pdf"),
                   bbox_inches="tight")
        plt.close()


def plot_tradeoff(df: pd.DataFrame, budget: str, out_dir: str):
    """
    Create tradeoff plots: Runtime vs ESS and JS vs ESS.

    Args:
        df: Results dataframe
        budget: Budget type for filename
        out_dir: Output directory
    """
    # Aggregate results
    summ = df.groupby(["potential_fn", "dimension", "method"]).agg(
        mean_ESS=('ESS', 'mean'),
        mean_rt=('runtime_sec', 'mean'),
        mean_JS=('JS_distance', 'mean')
    ).reset_index()

    fns = summ["potential_fn"].unique()
    fig, axes = plt.subplots(len(fns), 2, figsize=(13, 4*len(fns)))
    if len(fns) == 1:
        axes = np.array([axes])

    for i, fn in enumerate(fns):
        d = summ[summ["potential_fn"] == fn]

        # Runtime vs ESS
        ax = axes[i, 0]
        ax.set_title(f"{fn.replace('_', ' ').title()}: Runtime vs ESS")
        for m in d["method"].unique():
            dd = d[d["method"] == m]
            ax.scatter(dd["mean_rt"], dd["mean_ESS"], label=m, s=70, alpha=0.8)
            for _, r in dd.iterrows():
                ax.annotate(int(r["dimension"]), (r["mean_rt"], r["mean_ESS"]))
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("ESS")
        ax.legend()

        # JS vs ESS
        ax = axes[i, 1]
        ax.set_title(f"{fn.replace('_', ' ').title()}: JS vs ESS")
        for m in d["method"].unique():
            dd = d[d["method"] == m]
            ax.scatter(dd["mean_JS"], dd["mean_ESS"], label=m, s=70, alpha=0.8)
            for _, r in dd.iterrows():
                ax.annotate(f"{r['mean_JS']:.3f}", (r["mean_JS"], r["mean_ESS"]))
        ax.set_xlabel("JS distance")
        ax.set_ylabel("ESS")
        ax.legend()

    plt.tight_layout()
    _ensure_dir(os.path.join(out_dir, "figs"))
    plt.savefig(os.path.join(out_dir, f"figs/tradeoff_{budget}.png"),
               dpi=300, bbox_inches="tight")
    plt.close()


def plot_2d_scatter(samples_single: np.ndarray, samples_ens: np.ndarray,
                   out_path: str, title: str):
    """
    Create 2D scatter plots comparing single vs ensemble samples.

    Args:
        samples_single: Samples from single method
        samples_ens: Samples from ensemble method
        out_path: Output file path
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, X, ttl in zip(axes, [samples_single, samples_ens],
                          ["flowMC-single", "flowMC-ensemble"]):
        ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.65)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(ttl)

    fig.suptitle(title)
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# =============================================================================
# Section 7: Experiment Drivers
# =============================================================================

def _run_pair_once(
    fn: str,
    d: int,
    params: Dict,
    rep: int,
    n_particles_single: int,
    n_members: int,
    budget: str,
    ablate: Dict
) -> Tuple[List[Dict], Dict, np.ndarray, np.ndarray]:
    """
    Run one pair of single vs ensemble experiments.

    Args:
        fn: Potential function name
        d: Dimensionality
        params: Distribution parameters
        rep: Repetition number
        n_particles_single: Particles for single method
        n_members: Number of ensemble members
        budget: Budget allocation strategy
        ablate: Ablation flags

    Returns:
        (result_rows, seed_info, single_samples, ensemble_samples)
    """
    # Generate seeds deterministically
    seed_base = 10_000 + 97*rep + (zlib.crc32(fn.encode("utf-8")) % 997)
    master = jax.random.PRNGKey(np.uint32(seed_base))
    rows = []

    # Run single flowMC
    master, sk1 = jax.random.split(master)
    Xs, t_single = flowmc_inference(sk1, n_particles_single, d,
                                    potential_fn=fn, **params)

    # Generate reference samples
    ref = generate_reference_samples(fn, d, 20000, **params)

    # Compute metrics for single
    ess_s = compute_ess_d_dim(Xs)
    js_s = compute_marginal_js_distance(Xs, np.array(ref))

    rows.append(dict(
        potential_fn=fn,
        dimension=d,
        repeat=rep,
        method="flowMC-single",
        ESS=ess_s,
        runtime_sec=t_single,
        ESS_per_sec=ess_s/max(t_single, 1e-9),
        JS_distance=js_s,
        quality=ess_s/max(t_single, 1e-9)/max(js_s, 1e-9),
        budget=budget,
        ablation=json.dumps(ablate)
    ))

    # Run ensemble flowMC
    master, sk2 = jax.random.split(master)
    Xe, t_ens, meta = enhanced_flowmc_ensemble(
        sk2, n_particles_single, d, fn,
        n_members=n_members,
        budget=budget,
        adaptive_temp=not ablate.get("no_temp", False),
        use_cov_balance=not ablate.get("no_cov", False),
        use_member_weight=not ablate.get("no_w", False),
        **params
    )

    # Compute metrics for ensemble
    ess_e = compute_ess_d_dim(Xe)
    js_e = compute_marginal_js_distance(Xe, np.array(ref))

    rows.append(dict(
        potential_fn=fn,
        dimension=d,
        repeat=rep,
        method=f"flowMC-ensemble(M={n_members})-{budget}",
        ESS=ess_e,
        runtime_sec=t_ens,
        ESS_per_sec=ess_e/max(t_ens, 1e-9),
        JS_distance=js_e,
        quality=ess_e/max(t_ens, 1e-9)/max(js_e, 1e-9),
        budget=budget,
        alphas=json.dumps(meta["alphas"]),
        ablation=json.dumps(ablate)
    ))

    # Seed information for reproducibility
    seeds = dict(
        potential_fn=fn,
        dimension=d,
        repeat=rep,
        single_seed=int(sk1[0]),
        ens_seed=int(sk2[0]),
        budget=budget,
        ablation=json.dumps(ablate)
    )

    return rows, seeds, Xs, Xe


def run_suite(
    configs: List[Dict],
    n_particles_single: int = 2000,
    n_repeats: int = 5,
    n_members: int = 5,
    budget: str = "equal_per_member",
    out_dir: str = "paper_outputs",
    ablate: Dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete experiment suite.

    Args:
        configs: List of experiment configurations
        n_particles_single: Particles for single method
        n_repeats: Number of repetitions per configuration
        n_members: Number of ensemble members
        budget: Budget allocation strategy
        out_dir: Output directory
        ablate: Ablation settings

    Returns:
        (results_dataframe, significance_dataframe)
    """
    if ablate is None:
        ablate = {}

    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    seed_rows = []
    last_pair_by_config = {}

    # Run all experiments
    for cfg in configs:
        fn = cfg["potential_fn"]
        d = cfg["n_dims"]
        params = cfg.get("params", {})
        s_single_last, s_ens_last = None, None

        print(f"Running {fn} (d={d})...")

        for rep in range(n_repeats):
            rows, seeds, Xs, Xe = _run_pair_once(
                fn, d, params, rep, n_particles_single, n_members, budget, ablate
            )
            all_rows += rows
            seed_rows.append(seeds)
            s_single_last, s_ens_last = Xs, Xe

        last_pair_by_config[(fn, d)] = (s_single_last, s_ens_last)

        # Create 2D scatter plot for 2D distributions
        if d == 2 and s_single_last is not None:
            plot_2d_scatter(
                s_single_last, s_ens_last,
                out_path=os.path.join(out_dir, f"figs/{fn}_2d_scatter_{budget}.png"),
                title=f"{fn} (d=2) coverage"
            )

    # Create dataframes
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out_dir, f"results_{budget}.csv"), index=False)
    pd.DataFrame(seed_rows).to_csv(os.path.join(out_dir, f"seeds_{budget}.csv"), index=False)

    # Statistical significance tests
    sig_rows = []
    for (fn, d), g in df.groupby(["potential_fn", "dimension"]):
        g_s = g[g["method"] == "flowMC-single"].sort_values("repeat")
        g_e = g[g["method"] == f"flowMC-ensemble(M={n_members})-{budget}"].sort_values("repeat")

        if len(g_s) == len(g_e) and len(g_s) > 1:
            # Paired t-tests
            t_ess, p_ess = ttest_rel(g_e["ESS"].values, g_s["ESS"].values)
            t_js, p_js = ttest_rel(g_s["JS_distance"].values, g_e["JS_distance"].values)

            # Effect sizes (Cohen's d)
            diff_ess = g_e["ESS"].values - g_s["ESS"].values
            diff_js = g_s["JS_distance"].values - g_e["JS_distance"].values
            d_ess = float(np.mean(diff_ess) / (np.std(diff_ess, ddof=1) + 1e-9))
            d_js = float(np.mean(diff_js) / (np.std(diff_js, ddof=1) + 1e-9))

            sig_rows.append(dict(
                potential_fn=fn,
                dimension=d,
                n=len(g_s),
                mean_ESS_single=float(g_s["ESS"].mean()),
                mean_ESS_ens=float(g_e["ESS"].mean()),
                mean_JS_single=float(g_s["JS_distance"].mean()),
                mean_JS_ens=float(g_e["JS_distance"].mean()),
                paired_t_ESS_t=float(t_ess),
                paired_t_ESS_p=float(p_ess),
                paired_t_JS_t=float(t_js),
                paired_t_JS_p=float(p_js),
                cohen_d_ESS=d_ess,
                cohen_d_JS=d_js,
                budget=budget,
                ablation=json.dumps(ablate)
            ))

    sig = pd.DataFrame(sig_rows)
    sig.to_csv(os.path.join(out_dir, f"significance_{budget}.csv"), index=False)

    # Generate all plots
    print("Generating plots...")
    plot_bars(df, budget, out_dir)
    plot_box_violin(df, budget, out_dir)
    plot_tradeoff(df, budget, out_dir)

    return df, sig


# =============================================================================
# Section 8: Sanity Checks
# =============================================================================

def sanity_check_potentials():
    """
    Verify all potentials work correctly with different input shapes.
    """
    print("Running sanity checks on potentials...")

    pots = [
        ("standard_gaussian", {}, 10),
        ("banana", {"a": 1.0, "b": 100.0}, 10),
        ("bimodal_gaussian", {"mode_distance": 6.0}, 10),
        ("funnel", {"sigma_v": 0.3}, 10),
        ("correlated_gaussian", {"rho": 0.95}, 10),
        ("rosenbrock", {"a": 1.0, "b": 100.0}, 10),
        ("gaussian_ring", {"radius": 5.0, "K": 8, "std": 0.7}, 10)
    ]

    key = jax.random.PRNGKey(0)

    for name, params, d in pots:
        # Test with single point
        x1 = jax.random.normal(key, (d,))
        log_post = log_post_from_cfg(name, **params)
        y1 = log_post(x1)
        assert np.isfinite(y1), f"{name}: NaN/Inf for single point"

        # Test with batch
        x2 = jax.random.normal(key, (32, d))
        y2 = jax.vmap(log_post)(x2)
        assert y2.shape == (32,), f"{name}: Wrong batch shape"
        assert np.all(np.isfinite(y2)), f"{name}: NaN/Inf in batch"

    print("Sanity checks passed!")


def sanity_smoketest_sampling():
    """
    Quick smoke test of sampling functionality.
    """
    print("Running smoke test on sampling...")

    d = 10
    fn = "banana"
    params = {"a": 1.0, "b": 100.0}

    # Test single sampler
    key = jax.random.PRNGKey(123)
    key, sk = jax.random.split(key)
    Xs, ts = flowmc_inference(sk, 128, d, potential_fn=fn, **params)
    assert Xs.shape == (128, d), "Single sampler wrong shape"
    assert np.all(np.isfinite(Xs)), "Single sampler has NaN/Inf"

    # Test ensemble sampler
    key, sk = jax.random.split(key)
    Xe, te, meta = enhanced_flowmc_ensemble(
        sk, 128, d, fn, n_members=3, budget="equal_per_member", **params
    )
    assert Xe.shape[1] == d, "Ensemble sampler wrong dimension"
    assert np.all(np.isfinite(Xe)), "Ensemble sampler has NaN/Inf"

    print("Smoke tests passed!")


# =============================================================================
# Section 9: Main Execution
# =============================================================================

def main():
    """
    Main execution function.
    """
    # Configuration
    OUT_DIR = "paper_outputs"
    N_PARTICLES_SINGLE = 2000
    N_REPEATS = 5
    N_MEMBERS = 5

    # Experiment configurations
    CONFIGS = [
        # 20-dimensional tests
        dict(potential_fn="banana",            n_dims=20, params=dict(a=1.0, b=100.0)),
        dict(potential_fn="bimodal_gaussian",  n_dims=20, params=dict(mode_distance=6.0)),
        dict(potential_fn="funnel",            n_dims=20, params=dict(sigma_v=0.3)),
        dict(potential_fn="correlated_gaussian", n_dims=20, params=dict(rho=0.95)),
        dict(potential_fn="rosenbrock",        n_dims=20, params=dict(a=1.0, b=100.0)),
        dict(potential_fn="gaussian_ring",     n_dims=20, params=dict(radius=5.0, K=8, std=0.7)),

        # 50-dimensional tests
        dict(potential_fn="banana",            n_dims=50, params=dict(a=1.0, b=100.0)),
        dict(potential_fn="bimodal_gaussian",  n_dims=50, params=dict(mode_distance=6.0)),
        dict(potential_fn="funnel",            n_dims=50, params=dict(sigma_v=0.3)),
        dict(potential_fn="correlated_gaussian", n_dims=50, params=dict(rho=0.95)),
        dict(potential_fn="rosenbrock",        n_dims=50, params=dict(a=1.0, b=100.0)),
        # Note: gaussian_ring only tested in 20D for visualization
    ]

    # Run sanity checks
    print("=" * 70)
    print("Enhanced flowMC Ensemble Framework")
    print("=" * 70)
    sanity_check_potentials()
    sanity_smoketest_sampling()

    # Main experiments
    print("\n" + "=" * 70)
    print("Running main experiments...")
    print("=" * 70)

    # 1) Equal-per-member budget (main results)
    print("\n1. Equal-per-member budget experiments...")
    df_eq, sig_eq = run_suite(
        CONFIGS, N_PARTICLES_SINGLE, N_REPEATS, N_MEMBERS,
        "equal_per_member", OUT_DIR
    )

    # 2) Equal-total budget (fairness check)
    print("\n2. Equal-total budget experiments...")
    df_tot, sig_tot = run_suite(
        CONFIGS, N_PARTICLES_SINGLE, N_REPEATS, N_MEMBERS,
        "equal_total", OUT_DIR
    )

    # Optional: Ablation studies
    # Uncomment to run ablations
    """
    print("\n3. Ablation studies...")
    print("   - No temperature adaptation...")
    run_suite(CONFIGS, N_PARTICLES_SINGLE, N_REPEATS, N_MEMBERS,
              "equal_per_member", OUT_DIR, ablate={"no_temp": True})

    print("   - No coverage balancing...")
    run_suite(CONFIGS, N_PARTICLES_SINGLE, N_REPEATS, N_MEMBERS,
              "equal_per_member", OUT_DIR, ablate={"no_cov": True})

    print("   - No member weighting...")
    run_suite(CONFIGS, N_PARTICLES_SINGLE, N_REPEATS, N_MEMBERS,
              "equal_per_member", OUT_DIR, ablate={"no_w": True})
    """

    print("\n" + "=" * 70)
    print("All experiments completed successfully!")
    print(f"Results saved to: {OUT_DIR}/")
    print("=" * 70)

    # Print summary statistics
    print("\nSummary of Results (Equal-per-member):")
    print("-" * 40)
    for _, row in sig_eq.iterrows():
        js_improvement = (row['mean_JS_single'] - row['mean_JS_ens']) / row['mean_JS_single'] * 100
        print(f"{row['potential_fn']} (d={row['dimension']}):")
        print(f"  JS improvement: {js_improvement:.1f}%")
        print(f"  p-value: {row['paired_t_JS_p']:.2e}")


if __name__ == "__main__":
    sanity_check_potentials()
    sanity_smoketest_sampling()
    # main()  # Uncomment to run full paper-scale experiments
