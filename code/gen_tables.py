#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate LaTeX table bodies for the revised paper from
(a) the released repository CSVs (means +- sd over the five repetitions) and
(b) the new supplementary-experiment CSVs. Output: ../paper/tables/*.tex"""

import os
import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "tables")

NAME = {"banana": "Banana", "bimodal_gaussian": "Bimodal",
        "correlated_gaussian": "Correlated", "funnel": "Funnel",
        "gaussian_ring": "Ring", "rosenbrock": "Rosenbrock"}
ORDER = [("banana", 20), ("banana", 50), ("bimodal_gaussian", 20),
         ("bimodal_gaussian", 50), ("correlated_gaussian", 20),
         ("correlated_gaussian", 50), ("funnel", 20), ("funnel", 50),
         ("gaussian_ring", 20), ("rosenbrock", 20), ("rosenbrock", 50)]

def ms(g, key, nd=3):
    m, s = g.loc[key, "mean"], g.loc[key, "std"]
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"

def stats(df, filt=None):
    d = df if filt is None else df[df.method.str.contains(filt)]
    return d.groupby(["potential_fn", "dimension"])["JS_distance"].agg(["mean", "std"])

# ---------------------------------------------------------------- Table 1
eq = pd.read_csv(f"{REPO}/results_equal_per_member.csv")
s10 = pd.read_csv(f"{REPO}/results_single_10000.csv")
g_s2 = stats(eq, "single"); g_en = stats(eq, "ensemble"); g_10 = stats(s10)
rows = []
for key in ORDER:
    m_en, m_10 = g_en.loc[key, "mean"], g_10.loc[key, "mean"]
    delta = 100 * (m_10 - m_en) / m_en
    dtxt = "tie" if key[0] == "banana" else f"$-{abs(delta):.0f}\\%$"
    rows.append(f"{NAME[key[0]]} & {key[1]} & {ms(g_s2, key)} & "
                f"{ms(g_en, key)} & {ms(g_10, key)} & {dtxt} \\\\")
per_rep = lambda df, f=None: (df if f is None else df[df.method.str.contains(f)]) \
    .groupby(["potential_fn", "dimension", "repeat"])["JS_distance"].mean() \
    .groupby("repeat").mean()
p2, pe, p10 = per_rep(eq, "single"), per_rep(eq, "ensemble"), per_rep(s10)
mean_delta = 100 * (p10.mean() - pe.mean()) / pe.mean()
rows.append("\\midrule")
rows.append(f"\\textbf{{Mean}} & & \\textbf{{{p2.mean():.3f}}} & "
            f"\\textbf{{{pe.mean():.3f}}} & \\textbf{{{p10.mean():.3f}}} & "
            f"\\textbf{{${mean_delta:.0f}\\%$}} \\\\")
open(f"{OUT}/tab_control_body.tex", "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Table 2
def rep_mean_sd(path, f=None):
    df = pd.read_csv(path)
    if f is not None:
        df = df[df.method.str.contains(f)]
    pr = df.groupby(["potential_fn", "dimension", "repeat"])["JS_distance"] \
           .mean().groupby("repeat").mean()
    return pr.mean(), pr.std()

m2, s2 = rep_mean_sd(f"{REPO}/results_equal_per_member.csv", "single")
mfull, sfull = rep_mean_sd(f"{REPO}/results_equal_per_member.csv", "ensemble")
mnc, snc = rep_mean_sd(f"{REPO}/results_equal_per_member_no_cov.csv", "ensemble")
mnt, snt = rep_mean_sd(f"{REPO}/results_equal_per_member_no_temp.csv", "ensemble")
mnw, snw = rep_mean_sd(f"{REPO}/results_equal_per_member_no_w.csv", "ensemble")
mho, sho = rep_mean_sd(f"{REPO}/results_homogeneous_ensemble.csv")
m10, s10v = rep_mean_sd(f"{REPO}/results_single_10000.csv")
def row2(label, m, s, bold=False):
    pv2 = 100 * (m - m2) / m2
    pvf = 100 * (m - mfull) / mfull
    c2 = "---" if abs(m - m2) < 1e-12 else f"${pv2:+.1f}\\%$"
    cf = "---" if abs(m - mfull) < 1e-12 else f"${pvf:+.1f}\\%$"
    txt = f"{label} & {m:.4f} $\\pm$ {s:.4f} & {c2} & {cf} \\\\"
    return f"\\textbf{{{label}}} & \\textbf{{{m:.4f} $\\pm$ {s:.4f}}} & " \
           f"\\textbf{{{c2}}} & \\textbf{{{cf}}} \\\\" if bold else txt
rows = [row2("Single chain @2{,}000", m2, s2),
        row2("Full ensemble", mfull, sfull),
        row2("No coverage reweighting", mnc, snc),
        row2("No adaptive tempering", mnt, snt),
        row2("No member weighting", mnw, snw),
        row2("Homogeneous ensemble ($5\\times$ identical)", mho, sho),
        row2("Single chain @10{,}000", m10, s10v, bold=True)]
open(f"{OUT}/tab_ablation_body.tex", "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Table 3
rp = pd.read_csv(f"{REPO}/results_rawpool.csv")
g_rp = stats(rp)
rows = []
for key in ORDER:
    rows.append(f"{NAME[key[0]]} & {key[1]} & {ms(g_en, key)} & "
                f"{ms(g_rp, key)} & {ms(g_10, key)} \\\\")
prp = per_rep(rp)
rows.append("\\midrule")
rows.append(f"\\textbf{{Mean}} & & \\textbf{{{pe.mean():.4f}}} & "
            f"\\textbf{{{prp.mean():.4f}}} & \\textbf{{{p10.mean():.4f}}} \\\\")
open(f"{OUT}/tab_rawpool_body.tex", "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Table 4
dec = pd.read_csv(f"{RES}/supp_decomposition.csv")
pr = dec.groupby(["variant", "rep"])["JS"].mean().groupby("variant").agg(["mean", "std"])
labels = [
    ("exact_single_10000", "Exact single draw of 10{,}000", "0.045"),
    ("pool_uniform", "Uniform pool of $5\\times2{,}000$ exact draws", "0.045"),
    ("pool_resample_uniform", "\\quad $+$ resampling with replacement (uniform weights)", "0.058"),
    ("pool_memberweight_resample", "\\quad $+$ member weighting $+$ resampling", "---"),
    ("pool_coverage_resample", "\\quad $+$ coverage reweighting $+$ resampling", "---"),
    ("pool_full_aggregation", "Full aggregation stack (member $\\times$ coverage, resampled)", "---"),
]
rows = []
for key, lab, pred in labels:
    m, s = pr.loc[key, "mean"], pr.loc[key, "std"]
    rows.append(f"{lab} & {m:.4f} $\\pm$ {s:.4f} & {pred} \\\\")
rows.append("\\midrule")
rows.append(f"flowMC ensemble, measured (Table~1) & {mfull:.4f} $\\pm$ {sfull:.4f} & --- \\\\")
open(f"{OUT}/tab_decomp_body.tex", "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Table 5
en = pd.read_csv(f"{RES}/supp_energy.csv")
piv = en.groupby(["variant", "target"])["energy_distance"].mean().unstack("target")
t20 = ["banana", "bimodal_gaussian", "correlated_gaussian", "funnel",
       "gaussian_ring", "rosenbrock"]
vord = [("exact_single_10000", "Exact single draw of 10{,}000"),
        ("pool_uniform", "Uniform pool"),
        ("pool_resample_uniform", "Pool $+$ uniform resampling"),
        ("pool_full_aggregation", "Full aggregation stack")]
rows = []
for key, lab in vord:
    vals = " & ".join(f"{piv.loc[key, t]:.3f}" for t in t20)
    rows.append(f"{lab} & {vals} \\\\")
open(f"{OUT}/tab_energy_body.tex", "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Table 6
au = pd.read_csv(f"{RES}/supp_reference_audit.csv")
ap = au[au.draw == "potential_law_10000"].groupby(["target", "dim"]) \
    [["JS", "JS_dim0", "JS_dim1", "JS_rest"]].mean()
fl = pd.read_csv(f"{RES}/supp_floor.csv")
fl11 = fl[(fl.kind == "all11") & (fl.N == 10000)].groupby(["target", "dim"])["JS"].mean()
rows = []
for key in [("rosenbrock", 20), ("rosenbrock", 50), ("gaussian_ring", 20)]:
    r = ap.loc[key]
    d12 = 0.5 * (r["JS_dim0"] + r["JS_dim1"])
    rows.append(f"{NAME[key[0]]} & {key[1]} & {r['JS']:.4f} & {d12:.3f} & "
                f"{r['JS_rest']:.4f} & {fl11.loc[key]:.4f} & "
                f"{g_10.loc[key,'mean']:.4f} $\\pm$ {g_10.loc[key,'std']:.4f} \\\\")
open(f"{OUT}/tab_audit_body.tex", "w").write("\n".join(rows) + "\n")

print("tables written to", OUT)
# quick echo of headline numbers used in the running text
print(f"deficit reproduced: {(pr.loc['pool_full_aggregation','mean']-0.0435)/(mfull-0.0435)*100:.0f}%")

# ---- wrap bodies into complete tabular environments used by main.tex ----
SPECS = {
 "tab_control":  ("llcccc", "Target & Dim & Single @2{,}000 & Ensemble @10{,}000 & Single @10{,}000 & Single@10k vs ens \\\\"),
 "tab_ablation": ("lccc",   "Variant & Mean JS & vs.\\ single @2{,}000 & vs.\\ full ensemble \\\\"),
 "tab_rawpool":  ("llccc",  "Target & Dim & Full ensemble & Uniform pool & Single @10{,}000 \\\\"),
 "tab_decomp":   ("lcc",    "Variant (exact members, $5\\times2{,}000$) & Mean JS & $\\chi^2$ prediction \\\\"),
 "tab_energy":   ("lcccccc","Variant & Banana & Bimodal & Correlated & Funnel & Ring & Rosenbrock \\\\"),
 "tab_audit":    ("llccccc","Target & Dim & Bound & JS dims 1--2 & JS rest & Floor @10k & Single @10{,}000 \\\\"),
}
for _name, (_cols, _head) in SPECS.items():
    _body = open(f"{OUT}/{_name}_body.tex").read().rstrip()
    open(f"{OUT}/{_name}.tex", "w").write(
        "\\begin{tabular}{%s}\n\\toprule\n%s\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n"
        % (_cols, _head, _body))
print("full tabulars written")
