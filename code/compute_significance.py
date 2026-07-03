#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_significance.py — Reproduce significance_recomputed_on_paper_run.csv
from the main equal-per-member run (paired t-test and Cohen's d of the
ensemble-vs-single@2,000 JS comparison reported in Section 5.1).

Usage:
    python compute_significance.py --data_dir data --out significance_recomputed_on_paper_run.csv
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ORDER = [
    ("banana", 20), ("banana", 50),
    ("bimodal_gaussian", 20), ("bimodal_gaussian", 50),
    ("correlated_gaussian", 20), ("correlated_gaussian", 50),
    ("funnel", 20), ("funnel", 50),
    ("gaussian_ring", 20),
    ("rosenbrock", 20), ("rosenbrock", 50),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="significance_recomputed_on_paper_run.csv")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, "results_equal_per_member.csv"))
    rows = []
    for fn, d in ORDER:
        g = df[(df.potential_fn == fn) & (df.dimension == d)]
        s = g[g.method == "flowMC-single"].sort_values("repeat").JS_distance.values
        e = g[g.method.str.contains("ensemble", regex=False)].sort_values("repeat").JS_distance.values
        t, p = ttest_rel(s, e)                       # single minus ensemble
        diff = s - e
        cohen = float(np.mean(diff) / np.std(diff, ddof=1))
        rows.append(dict(potential_fn=fn, dimension=d,
                         mean_JS_single=float(s.mean()), mean_JS_ens=float(e.mean()),
                         paired_t_JS_t=float(t), paired_t_JS_p=float(p),
                         cohen_d_JS=cohen, n=len(s), source_run="equal_per_member__1_"))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out, f"({len(rows)} configs)")
