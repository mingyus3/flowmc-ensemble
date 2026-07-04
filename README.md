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

## Revision (July 2026): mechanism-level supplementary experiments

The paper was substantially revised with six sampler-free supplementary
experiments (E1-E6) that explain the ensemble's deficit mechanistically:
a closed-form finite-sample law for the histogram-JS floor, a sampler-free
decomposition of the aggregation bias, an exact isolation of the tempering
schedule, an unbiased energy-distance check, ensemble-size scaling up to
M=20, and an audit of the reference generators (Section 5.7).

Additional layout:

```
code/
  supp_experiments.py         all supplementary experiments (E1-E6); pure
                              NumPy/scikit-learn, CPU-only, ~2-3 minutes;
                              writes results/*.csv (fully deterministic)
  gen_tables.py               regenerates every LaTeX table body in
                              paper/tables/ from data/ and results/
  make_revision_figures.py    regenerates the four new figures
  fix_reference_generators.py corrected Rosenbrock / ring reference
                              generators (Section 5.7)
results/                      CSV outputs of the supplementary experiments
paper/                        revised paper: main.tex + tables/ + figures/
                              (upload the whole paper/ folder to Overleaf,
                              compile with pdfLaTeX)
```

To reproduce the supplementary experiments and rebuild tables and figures:

```bash
pip install numpy scipy scikit-learn pandas matplotlib
python code/supp_experiments.py all
python code/gen_tables.py
python code/make_revision_figures.py
```
