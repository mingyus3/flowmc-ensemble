# Enhanced flowMC Ensemble Sampling Framework

An ensemble-based sampling framework built on top of flowMC and JAX.

This repository provides:

- Multiple challenging target distributions (banana, bimodal, funnel, correlated Gaussian, Rosenbrock, gaussian ring)
- Single flowMC baseline sampler
- Enhanced ensemble sampling with:
  - adaptive temperature scheduling
  - coverage-aware reweighting (KMeans)
  - quality-based member aggregation (ESS / JS / logp-std)
- Statistical evaluation and figure generation

------------------------------------------------------------

Repository structure

scripts/enhanced_flowmc_ensemble.py
    Main script. Contains sanity checks, smoke test, and full experiments.

environment.yml
    Reproducible conda environment.

requirements.txt
    Pip fallback installation.

paper_outputs/
    Generated results (ignored by git).

------------------------------------------------------------

Environment setup (recommended)

Create environment:

conda env create -f environment.yml
conda activate flowmc_ens

------------------------------------------------------------

Quickstart (sanity + smoke test)

Run:

python scripts/enhanced_flowmc_ensemble.py

Expected output:

Sanity checks passed!
Smoke tests passed!

Note:
Install package name is "flowMC"
Import name must be "flowMC" (case-sensitive).

------------------------------------------------------------

Full experiments (optional)

The script also contains full paper-scale experiments which may take a long time.

To enable:
Edit scripts/enhanced_flowmc_ensemble.py and replace the __main__ block to call main().

Outputs will be written to:

paper_outputs/results_*.csv
paper_outputs/significance_*.csv
paper_outputs/seeds_*.csv
paper_outputs/figs/*.png
paper_outputs/figs/*.pdf

------------------------------------------------------------

License

MIT License. See LICENSE.

