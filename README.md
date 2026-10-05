# DFM-PSA case study: grey-box optimisation of a CO₂-to-methane flowsheet

This repository reproduces **Section 8.4** of the thesis: the optimisation of a
power-to-methane flowsheet with the A-TRFu solver. Its parts are a solid oxide
electrolyser, a dual-function-material (DFM) reactor that captures CO₂ and
methanates it, compression, gas cooling, a pressure-swing adsorption (PSA) unit,
a hydrogen recycle with a furnace, and external hydrogen supply. The DFM reactor
cycle and the PSA unit are expensive truth models; the rest is an
equation-oriented cost and balance model. The objective is the levelised cost of
methane (LCOCH₄, USD/kg). The model is stated in Appendix B of the thesis.

## Layout

```
flowsheet.py         the Pyomo model, the black-box wrappers and their cache,
                     the economics recomputed at a point (report_economics)
run.py               the 13 runs of Section 8.4 (python run.py --list)
atrfu/               the A-TRFu solver (as in the A-TRFu repository, plus gjh.exe)
reactor/             DFM reactor cycle model (cited work, see below)
psa/                 PSA model and its isotherm data (cited work, uses pyapep)
data/                initialisation workbook (initial values of the model)
start_points/        x0_units_fixed.json  the original start
                     x0_best_hvR1.json    the second start (last iterate of Run 1)
                     reseed_x0.py         rebuilds the original start
cache/               black-box evaluation cache (see "The cache")
results/
  make_table87.py    Table 8.7
  make_fig810.py     Figure 8.10
  make_cost_figure.py  Figure 8.11
  tr_activity_check.py  trust-region activity at the reported designs
  reference/         the reported runs and outputs
environment.yml
```

## Requirements

Python 3.12 with Pyomo 6.10 and pyapep 0.1.8, and the IPOPT executable (the
reported runs used IPOPT 3.14.13 with MUMPS, built with MinGW-w64 under MSYS2,
and Pyomo 6.10.1.dev0). On Linux and macOS `environment.yml` installs IPOPT; on
Windows the conda-forge package provides only the library, so put an
`ipopt.exe` on the `PATH`, for example from the COIN-OR Ipopt releases on
GitHub:

```
conda env create -f environment.yml
conda activate dfm-psa
```

The solver obtains derivatives from AMPL's `gjh`. `atrfu/gjh.exe` is the
Windows build used for the reported runs (driver 20160215, ASL 20151209), and
the solver uses it in preference to any other gjh. On other platforms, or if
it is removed, the solver uses a gjh on the `PATH` or downloads the current
build on first use (`ampl-module-gjh`) into `atrfu/`. The current build prints
derivatives that can differ from the old one in the last digit. Replaying
`hv_on_r1_reported` with it gave identical iterates, objectives and
infeasibilities, with the criticality measure differing in its last digit;
longer runs can drift.

## The runs

All runs use A-TRFu with the funnel, the reduced trust region, variable
scaling, Δ_min = 5 × 10⁻⁴ and the restoration proximity weight w = 10⁻⁶ (the
settings stated in Section 8.4.1). They differ in the hypervolume mechanism (HV),
the restoration strategy, the start, the feasibility tolerance ε and the
iteration budget:

| Run | HV | Restoration | Start | ε | Budget | In Section 8.4 |
|---|---|---|---|---|---|---|
| `hv_off_r1` | off | R1 | original | 10⁻³ | 1000 | reported design without HV (feasible exit, 3.02 USD/kg) |
| `hv_off_r1_duals` | off | R1 | original | 10⁻³ | 87 | multipliers at that design |
| `hv_off_r1_second_start` | off | R1 | second | 10⁻³ | 1000 | alternative, 4.33 USD/kg |
| `hv_off_r1_eps1e-5` | off | R1 | original | 10⁻⁵ | 1000 | the tolerance 10⁻⁵ is not met |
| `hv_on_r1_run1` | on | R1 | original | 6 × 10⁻³ | 63 | Run 1; its last iterate is the second start |
| `hv_on_r1_second_start` | on | R1 | second | 10⁻³ | 1000 | the restart, to its budget |
| `hv_on_r1_reported` | on | R1 | second | 10⁻³ | 54 | stops at the reported design (iteration 55, 2.82 USD/kg) |
| `hv_on_r1_duals` | on | R1 | second | 10⁻³ | 55 | multipliers at that design |
| `hv_on_r1_eps1e-5` | on | R1 | original | 10⁻⁵ | 1000 | the tolerance 10⁻⁵ is not met; Run 1's path to iteration 64 |
| `r2_hv_off_original`, `r2_hv_on_original` | off / on | R2 | original | 10⁻⁵ | 1000 | Strategy R2 |
| `r2_hv_off_second`, `r2_hv_on_second` | off / on | R2 | second | 10⁻³ | 1000 | Strategy R2 |

The hypervolume-on design comes from two consecutive runs, as Section 8.4.2
explains. Run 1 reached its lowest infeasibility at iteration 64 and then
repeated the same restoration step, so it was stopped there (`hv_on_r1_run1`)
and restarted from that point. In the restart (`hv_on_r1_second_start`), the
lowest infeasibility came at iteration 55; `hv_on_r1_reported` stops at that
iterate. Its evaluation count in Table 8.7 is the sum over Run 1 and the
restart up to that iterate.

A run whose name ends in `_duals` stops at the reported design, so its last
subproblem is the trust-region subproblem centred there. Its constraint
multipliers are saved, together with that subproblem's solution, which
`tr_activity_check.py` uses.

## Reproducing Section 8.4

```
python run.py hv_off_r1
python run.py hv_off_r1_duals
python run.py hv_on_r1_run1
python run.py hv_on_r1_reported
python run.py hv_on_r1_duals
python run.py hv_on_r1_eps1e-5
python results/make_table87.py
python results/make_fig810.py
python results/make_cost_figure.py
python results/tr_activity_check.py
```

and, for the remaining statements of Section 8.4.2:

```
python run.py hv_off_r1_second_start
python run.py hv_on_r1_second_start
python run.py hv_off_r1_eps1e-5
python run.py r2_hv_off_original
python run.py r2_hv_on_original
python run.py r2_hv_off_second
python run.py r2_hv_on_second
```

Each run writes `results/runs/<run>.log` (the solver output) and
`results/runs/<run>.json`. The JSON holds the result, with the iteration
trajectory, the multipliers for a duals run, and the economics recomputed at the
final point. Runs whose point is needed later also write `<run>.point.json`. The
result scripts read `results/runs/` and write their tables and figures there;
with `--runs results/reference` they work on the reported runs instead.

With the shipped cache every black-box evaluation of these runs is a cache hit,
and the time goes into the subproblem solves (IPOPT and gjh). The runs of 34 to
104 iterations then take under two minutes each, and a 1000-iteration run
about 15 minutes on an otherwise idle machine. The original runs, which
simulated each new point, took up to 18 hours each.

`start_points/reseed_x0.py` rebuilds the original start from the workbook. It
writes `results/runs/x0_units_fixed.json`, to compare with the shipped one:
33 of the 44 values agree exactly, seven more to round-off, and the remaining
four (hydrogen price and credit, external hydrogen and its cost) to a relative
5 × 10⁻⁶. Those are values the feasibility problem leaves free within its
tolerance. The runs use the shipped file.
`hv_on_r1_run1` writes the second start as `results/runs/hv_on_r1_run1.point.json`.

## The cache

Each black-box call is cached under its inputs rounded to 2 decimals (reactor:
hydrogenation time, feed per channel, channel diameter) or 6 decimals (PSA:
hydrogen fraction of the feed). The simulation runs at the exact inputs of the
first point that falls in a cell, and that value is returned for every later
point in the same cell. The cached values are therefore part of the case
definition. With the shipped `cache/` the runs above replay the reported paths
exactly. With an empty cache, the first points of each cell differ, so a run can
follow a different path to a different end point.

The cache holds every evaluation made by the reported runs (1,695 reactor and
1,479 PSA points). New points are added to it as they are simulated.

## Reference results

`results/reference/` holds, for each run in the table, its `.log`, `.json` and
(where written) `.point.json`. It also holds `table_8_7.csv`,
`table_8_7_raw.json`, `figure_8_10`, `figure_8_11` (PNG and SVG) and the two
`tr_activity_*.txt` files, all produced from these runs by the scripts above.

## Citation

Release 1.0.0 of this repository accompanies G. Hameed, *Rigorous Trust-Region Algorithms for Surrogate-based Grey-box Optimisation*, PhD thesis, University of Surrey, 2026; publications that use later releases are listed here as they appear. Please cite the release you used with the metadata in `CITATION.cff`; each release has its own Zenodo DOI, shown on the repository page.

## Cited models

The code in this repository is released under the MIT licence (`LICENSE`), except the cited DFM reactor model (`reactor/`), the PSA model (`psa/`) and the files derived from Pyomo, which remain the work of their authors and keep their own terms (see "Cited models").

The DFM reactor model (`reactor/`) and the PSA model (`psa/`, built on pyapep)
are the work of others, cited in the thesis. They are included unchanged from
the versions used for the thesis, so that the flowsheet runs end to end; the
reactor model's entry point takes a cycle cap (`max_cycles`, 60 here) and an
Excel-export switch (off).

`atrfu/gjh.exe` is AMPL's gjh, built on the AMPL Solver Library, as AMPL
distributes it. `atrfu/PyomoInterface.py`, `Logger.py` and `readgjh.py` derive
from the trust-region code distributed with Pyomo (3-clause BSD licence,
© 2017 National Technology and Engineering Solutions of Sandia, LLC); their
headers retain the notice.
