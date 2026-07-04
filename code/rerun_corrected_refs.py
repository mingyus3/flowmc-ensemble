#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun_corrected_refs.py — Colab driver: rerun the Rosenbrock (20/50D) and
ring (20D) rows of the paper with CORRECTED reference generators (Section 5.7
/ fix_reference_generators.py), producing CSVs directly comparable to the
released ones (same seed scheme: seeds depend only on target name + repeat,
not on which configs are present).

USAGE (Colab, GPU runtime recommended):
    !git clone https://github.com/mingyus3/flowmc-ensemble.git
    %cd flowmc-ensemble
    !pip install flowMC jax pandas scikit-learn scipy matplotlib seaborn
    !python code/rerun_corrected_refs.py --smoke   # ~few min sanity pass first
    !python code/rerun_corrected_refs.py           # full run (~2-3 h on GPU)

Outputs to ./rerun_corrected/:
    results_equal_per_member.csv   (single@2k + full ensemble, 3 configs)
    results_equal_total.csv
    results_equal_per_member_no_cov/_no_temp/_no_w.csv
    plus whatever run_controls.py / run_rawpool.py write (single@10000,
    homogeneous, rawpool) — those two scripts are patched and run as-is.

NOTE: this driver has NOT been executed end-to-end (it needs flowMC+JAX);
run --smoke first and sanity-check shapes before the full run.
"""
import argparse, importlib.util, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "rerun_corrected")

KEEP = ("rosenbrock", "gaussian_ring")
CONFIGS3 = [
    dict(potential_fn="rosenbrock",    n_dims=20, params=dict(a=1.0, b=100.0)),
    dict(potential_fn="rosenbrock",    n_dims=50, params=dict(a=1.0, b=100.0)),
    dict(potential_fn="gaussian_ring", n_dims=20,
         params=dict(radius=5.0, K=8, std=0.7)),
]

# ---- the two corrected reference branches (see fix_reference_generators.py) --
ROSEN_OLD = """        x0 = jax.random.normal(k1, (n_samples,)) * 2.0 + a
        y0 = x0**2 + jax.random.normal(k2, (n_samples,)) * 0.5"""
ROSEN_NEW = """        x0 = jax.random.normal(k1, (n_samples,)) + a                      # N(a,1)  [ref fix]
        y0 = x0**2 + jax.random.normal(k2, (n_samples,)) / jnp.sqrt(b)    # N(x0^2,1/b)  [ref fix]"""

RING_OLD = """        theta = jax.random.uniform(key, (n_samples,)) * 2*jnp.pi
        key, k1 = jax.random.split(key)
        r = R + jax.random.normal(k1, (n_samples,)) * std
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)"""
RING_NEW = """        key, kc, k1, k2 = jax.random.split(key, 4)                        # [ref fix]
        comp = jax.random.randint(kc, (n_samples,), 0, K)                 # pick 1 of K modes
        ang = 2 * jnp.pi * comp / K
        x = R * jnp.cos(ang) + jax.random.normal(k1, (n_samples,)) * std
        y = R * jnp.sin(ang) + jax.random.normal(k2, (n_samples,)) * std"""



# ---- condensed-style anchors (run_controls.py / run_rawpool.py) --------------
ROSEN_OLD2 = "        x0=jax.random.normal(k1,(n_samples,))*2.0+a; y0=x0**2+jax.random.normal(k2,(n_samples,))*0.5"
ROSEN_NEW2 = "        x0=jax.random.normal(k1,(n_samples,))+a; y0=x0**2+jax.random.normal(k2,(n_samples,))/jnp.sqrt(b)  # [ref fix]"
RING_OLD2 = """        theta=jax.random.uniform(key,(n_samples,))*2*jnp.pi; key,k1=jax.random.split(key)
        r=R+jax.random.normal(k1,(n_samples,))*std; x=r*jnp.cos(theta); y=r*jnp.sin(theta)"""
RING_NEW2 = """        K=p.get("K",8); key,kc,k1,k2=jax.random.split(key,4)  # [ref fix]
        comp=jax.random.randint(kc,(n_samples,),0,K); ang=2*jnp.pi*comp/K
        x=R*jnp.cos(ang)+jax.random.normal(k1,(n_samples,))*std; y=R*jnp.sin(ang)+jax.random.normal(k2,(n_samples,))*std"""

def patch_references(path):
    """Apply the two corrected reference branches to a runner script copy."""
    t = open(path).read()
    n = 0
    if ROSEN_OLD in t:
        t = t.replace(ROSEN_OLD, ROSEN_NEW, 1); n += 1
    if RING_OLD in t:
        t = t.replace(RING_OLD, RING_NEW, 1); n += 1
    if ROSEN_OLD2 in t:
        t = t.replace(ROSEN_OLD2, ROSEN_NEW2, 1); n += 1
    if RING_OLD2 in t:
        t = t.replace(RING_OLD2, RING_NEW2, 1); n += 1
    open(path, "w").write(t)
    return n


def restrict_configs(path, keep=KEEP):
    """Comment out config-dict lines for targets we are not rerunning."""
    out, n = [], 0
    for line in open(path).read().splitlines(keepends=True):
        if re.search(r'dict\(potential_fn="([a-z_]+)"', line):
            name = re.search(r'potential_fn="([a-z_]+)"', line).group(1)
            if name not in keep and not line.lstrip().startswith("#"):
                line = re.sub(r"^(\s*)", r"\1# [rerun skip] ", line, count=1)
                n += 1
        out.append(line)
    open(path, "w").write("".join(out))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 repetition, 256 particles: fast sanity pass")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    work = os.path.join(OUTDIR, "_patched"); os.makedirs(work, exist_ok=True)

    # 1) make patched copies of the three runner scripts
    for f in ["enhanced_flowmc_ensemble.py", "run_controls.py", "run_rawpool.py"]:
        src, dst = os.path.join(HERE, f), os.path.join(work, f)
        shutil.copy(src, dst)
        print(f"{f}: {patch_references(dst)}/2 reference branches fixed, "
              f"{restrict_configs(dst)} config lines skipped")

    # 2) import the patched framework and run the suites we drive directly
    spec = importlib.util.spec_from_file_location(
        "efe", os.path.join(work, "enhanced_flowmc_ensemble.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    n_rep = 1 if args.smoke else 5
    n_part = 256 if args.smoke else 2000
    for budget in ["equal_per_member", "equal_total"]:
        print(f"== run_suite {budget} ==", flush=True)
        m.run_suite(CONFIGS3, n_part, n_rep, 5, budget, OUTDIR)
    for flag in ["no_cov", "no_temp", "no_w"]:
        print(f"== ablation {flag} ==", flush=True)
        m.run_suite(CONFIGS3, n_part, n_rep, 5, "equal_per_member", OUTDIR,
                    ablate={flag: True})
        shutil.move(os.path.join(OUTDIR, "results_equal_per_member.csv"),
                    os.path.join(OUTDIR, f"results_equal_per_member_{flag}.csv"))

    # rename the last plain equal_per_member back (ablations overwrote name)
    # -> rerun the plain suite last so the canonical file is the unablated one
    print("== run_suite equal_per_member (final, unablated) ==", flush=True)
    m.run_suite(CONFIGS3, n_part, n_rep, 5, "equal_per_member", OUTDIR)

    # 3) single@10000 / homogeneous / rawpool via the patched control scripts
    for f in ["run_controls.py", "run_rawpool.py"]:
        print(f"== {f} (patched, 3 configs) ==", flush=True)
        subprocess.run([sys.executable, os.path.join(work, f)],
                       cwd=OUTDIR, check=True)

    print("done. CSVs in", OUTDIR)
    print("Next: recompute the 3 rows of Tables 1-3, update Section 5.7 "
          "wording from 'we keep the original references' to 'corrected and "
          "rerun', and regenerate tables via code/gen_tables.py.")


if __name__ == "__main__":
    main()
