# flowmc-ensemble

Code, data, and reproducibility artifacts for the paper:

> **A Single Chain Suffices: Re-examining Diversity-Enhanced flowMC Ensembles
> under Equal Sampling Budgets.** Mingyu Shi, 2026.

This is a negative-results paper. A five-member diverse flowMC ensemble with
three aggregation mechanisms (adaptive tempering, coverage-aware reweighting,
quality-based member weighting) appeared to reduce marginal JS distance by ~18%
over a single flowMC chain — until an equal-budget control showed the gain was
a sample-count artifact. At a matched 10,000-sample budget, a single chain
matches or beats the ensemble on all eleven configurations, runs about six
times faster, and plain uniform pooling recovers single-chain accuracy,
localizing the deficit to the aggregation machinery itself.

## Repository layout

```
code/
  enhanced_flowmc_ensemble.py   original experiment framework
                                (main equal-per-member / equal-total runs and ablations)
  run_controls.py               controls: single chain @10,000 and homogeneous ensemble
  run_rawpool.py                control: uniform pooling of raw member samples
  noise_floor.py                perfect-sampler marginal-JS noise floor curve
  compute_significance.py       paired t-tests / Cohen's d for the Section 5.1 comparison
  make_figures.py               regenerates Figures 1-3 from the CSVs in data/
  make_supp_figures.py          regenerates Figures 4-6 from the CSVs in data/
data/                           all CSV outputs behind Tables 1-3 and Figures 1-6
figures/                        the six paper figures (PDF and PNG)
paper/                          LaTeX source and compiled PDF
DATA_MANIFEST.md                per-file description of the data
```

## Reproducing the figures

All six paper figures are deterministic functions of the released CSVs:

```bash
pip install numpy pandas scipy matplotlib
python code/make_figures.py      --data_dir data --out_dir figures
python code/make_supp_figures.py --data_dir data --out_dir figures
python code/compute_significance.py --data_dir data
```

## Reproducing the experiments

The sampling runs require JAX and flowMC (see `requirements.txt` /
`environment.yml`). `code/enhanced_flowmc_ensemble.py` contains the target
definitions, member palette, deterministic seeding, and the main
equal-per-member / equal-total suites (ablations via the `ablate` flags).
The control scripts (`run_controls.py`, `run_rawpool.py`) rerun the
matched-budget single chain, the homogeneous ensemble, and the uniform-pooling
control with the same seeds. Per-repetition seeds are recorded in
`data/seeds_equal_per_member.csv`.

## License

MIT (see `LICENSE`).
