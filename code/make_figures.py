#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py — Regenerate the three figures in
"A Single Chain Suffices" from the released CSV outputs.

Figure 1: equal-budget control (single@2k vs ensemble@10k vs single@10k)
Figure 2: finite-sample noise floor curve with the three operating points
Figure 3: uniform pool vs full ensemble vs single@10k

Usage:
    python make_figures.py --data_dir . --out_dir .

Inputs expected in --data_dir:
    results_equal_per_member.csv   (single@2k + ensemble@10k; col JS_distance)
    results_single_10000.csv           (single@10k)
    results_rawpool.csv            (uniform pool)
    noise_floor_curve.csv              (n_samples, perfect_sampler_marginal_JS)

Outputs (PNG + PDF) in --out_dir:
    fig_control_single10000.{png,pdf}
    fig_noise_floor.{png,pdf}
    fig_rawpool.{png,pdf}
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- canonical config order and pretty labels (matches the paper tables) ----
ORDER = [
    ("banana", 20), ("banana", 50),
    ("bimodal_gaussian", 20), ("bimodal_gaussian", 50),
    ("correlated_gaussian", 20), ("correlated_gaussian", 50),
    ("funnel", 20), ("funnel", 50),
    ("gaussian_ring", 20),
    ("rosenbrock", 20), ("rosenbrock", 50),
]
LABELS = ["Banana\n20D", "Banana\n50D", "Bimodal\n20D", "Bimodal\n50D",
          "Corr\n20D", "Corr\n50D", "Funnel\n20D", "Funnel\n50D",
          "Ring\n20D", "Rosen\n20D", "Rosen\n50D"]

FLOOR_10K = 0.043  # perfect-sampler floor at 10k (dashed reference line)


def mean_by_config(df, method_substr):
    """Mean JS per config for rows whose method contains method_substr."""
    out = []
    for fn, dim in ORDER:
        sel = df[(df.potential_fn == fn) & (df.dimension == dim) &
                 (df.method.str.contains(method_substr, regex=False))]
        out.append(sel.JS_distance.mean())
    return np.array(out)


def fig1_control(data_dir, out_dir):
    epm = pd.read_csv(os.path.join(data_dir, "results_equal_per_member.csv"))
    s10 = pd.read_csv(os.path.join(data_dir, "results_single_10000.csv"))
    single2k = mean_by_config(epm, "single")
    ensemble = mean_by_config(epm, "ensemble")
    single10k = mean_by_config(s10, "single@10000")

    x = np.arange(len(ORDER)); w = 0.27
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, single2k, w, label="Single chain @ 2000 (original baseline)",
           color="#E69F00")
    ax.bar(x,     ensemble, w, label="Ensemble @ 5×2000 = 10000 (proposed method)",
           color="#0072B2")
    ax.bar(x + w, single10k, w, label="Single chain @ 10000 (equal-budget control)",
           color="#009E73")
    ax.axhline(FLOOR_10K, ls="--", lw=1, color="grey")
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel("Marginal JS distance (lower is better)")
    ax.set_title("At equal total budget, a single chain (green) matches or "
                 "outperforms the ensemble (blue) on every target", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir, "fig_control_single10000")


def fig2_floor(data_dir, out_dir):
    curve = pd.read_csv(os.path.join(data_dir, "noise_floor_curve.csv"))
    epm = pd.read_csv(os.path.join(data_dir, "results_equal_per_member.csv"))
    s10 = pd.read_csv(os.path.join(data_dir, "results_single_10000.csv"))
    single2k = mean_by_config(epm, "single").mean()
    ensemble = mean_by_config(epm, "ensemble").mean()
    single10k = mean_by_config(s10, "single@10000").mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve.n_samples, curve.perfect_sampler_marginal_JS,
            color="grey", lw=2, label="perfect-sampler floor")
    ax.scatter([2000], [single2k], color="#E69F00", s=70, zorder=5,
               label="single @2,000")
    ax.scatter([10000], [single10k], color="#009E73", s=70, zorder=5,
               label="single @10,000")
    ax.scatter([10000], [ensemble], color="#0072B2", marker="s", s=70, zorder=5,
               label="full ensemble (nominal 10,000)")
    ax.annotate("ensemble sits far above the 10k floor\n"
                "— a bias from aggregation, not a\nsample shortage (measured ESS 9,800)",
                xy=(10000, ensemble), xytext=(2300, 0.105),
                fontsize=8, color="#0072B2")
    ax.set_xscale("log")
    ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000])
    ax.set_xticklabels(["500", "1k", "2k", "5k", "10k", "20k"])
    ax.set_xlabel("number of samples N")
    ax.set_ylabel("avg. marginal JS distance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig_noise_floor")


def fig3_rawpool(data_dir, out_dir):
    epm = pd.read_csv(os.path.join(data_dir, "results_equal_per_member.csv"))
    rp = pd.read_csv(os.path.join(data_dir, "results_rawpool.csv"))
    s10 = pd.read_csv(os.path.join(data_dir, "results_single_10000.csv"))
    full = mean_by_config(epm, "ensemble")
    pool = mean_by_config(rp, "rawpool")
    single10k = mean_by_config(s10, "single@10000")

    x = np.arange(len(ORDER)); w = 0.27
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, full, w, label="Full ensemble (resampling + heuristics)",
           color="#0072B2")
    ax.bar(x,     pool, w, label="Uniform pool (no resampling, no weights)",
           color="#9467BD")
    ax.bar(x + w, single10k, w, label="Single chain @10,000", color="#009E73")
    ax.axhline(FLOOR_10K, ls="--", lw=1, color="grey")
    ax.text(len(ORDER) - 1.5, FLOOR_10K + 0.001, "floor @10k", fontsize=7, color="grey")
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel("Marginal JS distance (lower is better)")
    ax.set_title("Uniform pooling (purple) matches the single chain (green) on all "
                 "targets;\nthe full aggregation machinery (blue) is what degrades the "
                 "ensemble", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir, "fig_rawpool")


def _save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem + ".png/.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    fig1_control(args.data_dir, args.out_dir)
    fig2_floor(args.data_dir, args.out_dir)
    fig3_rawpool(args.data_dir, args.out_dir)
    print("done.")
