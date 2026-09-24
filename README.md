# Mixed-Precision Hamiltonian Dynamics

This repository contains the experiment code, measured data, figures, and paper
for **A Timestep Crossover Law for Stochastically Rounded Velocity Verlet on a
Harmonic Oscillator**.

The study compares ordinary FP32 rounding with stochastic rounding for the
velocity-Verlet method applied to a unit harmonic oscillator. It measures final
state error across timestep sizes and significand precisions, then evaluates
how the stochastic-rounding optimum scales with precision and simulation
duration.

## Reproduce the experiment

Requirements: Python 3.13 and XeLaTeX to build the paper. Python dependencies
are pinned in `pyproject.toml`.

```powershell
python -m pip install .
python experiment.py
xelatex -interaction=nonstopmode -halt-on-error paper.tex
xelatex -interaction=nonstopmode -halt-on-error paper.tex
```

The experiment writes tables, fit summaries, and figures to `results/`. The
paper uses the generated figures in that directory. Its compiled PDF is
`paper.pdf`.

## Repository contents

- `experiment.py`: deterministic and stochastic rounding simulations,
  analysis, bootstrap uncertainty estimates, and figure generation.
- `paper.tex` and `paper.pdf`: manuscript source and compiled paper.
- The manuscript is laid out for A4 printing with 25 mm margins.
- `results/`: measured CSV data, fit summaries, and figures used in the paper.
- `pyproject.toml`: project metadata, pinned NumPy and Matplotlib versions, and
  Ruff lint configuration.

The checked-in results are the outputs used for the manuscript. Rerunning the
experiment regenerates them.

## License

The code and accompanying materials are released under the MIT License. See
[`LICENSE`](LICENSE).
