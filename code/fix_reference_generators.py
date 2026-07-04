#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected reference generators for the two targets whose released reference
law does not match the law defined by the potential (see Section 5.7 of the
revised paper and REVIEW_NOTES).

  * rosenbrock  potential 0.5[(a-x0)^2 + b(x1-x0^2)^2 + sum rest^2]
        correct law:  x0 ~ N(a, 1),  x1|x0 ~ N(x0^2, 1/b),  rest ~ N(0, I)
        released law: x0 ~ N(a, 2^2), x1|x0 ~ N(x0^2, 0.5^2)   <-- too wide

  * gaussian_ring  potential = equal mixture of K Gaussians (std) on a circle
        correct law:  pick mode k ~ U{0..K-1}, xy ~ N(center_k, std^2 I)
        released law: uniform angle, radius ~ N(R, std^2)      <-- annulus

Drop-in replacements for the corresponding branches of
generate_reference_samples() in code/enhanced_flowmc_ensemble.py /
code/run_controls.py of the released repository. After swapping these in,
re-run the Rosenbrock (20D, 50D) and ring (20D) rows of Tables 1-3.
Only those rows change; all other targets' generators already match their
potentials exactly (verified in supp_experiments.py, phase 'audit').
"""

import numpy as np


def reference_rosenbrock(n_samples, n_dims, a=1.0, b=100.0, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n_samples) + a                 # N(a, 1)
    x1 = x0**2 + rng.standard_normal(n_samples) / np.sqrt(b)  # N(x0^2, 1/b)
    rest = rng.standard_normal((n_samples, n_dims - 2))
    return np.column_stack([x0, x1, rest])


def reference_gaussian_ring(n_samples, n_dims, radius=5.0, K=8, std=0.7,
                            seed=0):
    rng = np.random.default_rng(seed)
    k = rng.integers(0, K, size=n_samples)
    th = 2.0 * np.pi * k / K
    xy = np.column_stack([radius * np.cos(th), radius * np.sin(th)])
    xy += rng.standard_normal((n_samples, 2)) * std
    rest = rng.standard_normal((n_samples, n_dims - 2))
    return np.column_stack([xy, rest])


if __name__ == "__main__":
    # sanity: means/variances of the corrected laws
    r = reference_rosenbrock(200000, 20)
    print("rosenbrock  x0: mean %.3f (exp 1.0)  sd %.3f (exp 1.0)"
          % (r[:, 0].mean(), r[:, 0].std()))
    print("            x1|x0 residual sd %.3f (exp 0.1)"
          % (r[:, 1] - r[:, 0]**2).std())
    g = reference_gaussian_ring(200000, 20)
    rad = np.hypot(g[:, 0], g[:, 1])
    print("ring        mean radius %.3f (exp ~5.05 incl. curvature)"
          % rad.mean())
    ang = (np.degrees(np.arctan2(g[:, 1], g[:, 0])) + 360) % 45
    print("            angular concentration around modes: sd %.1f deg "
          "within a 45-deg sector (annulus would be ~13.0)" % ang.std())
