# -*- coding: utf-8 -*-
"""
Build the FEASIBLE initial point of the DFM-PSA flowsheet (the "original start").

The workbook initial values (DFM-PSA PFS Initialisation.xlsx, To_Use3) were
computed on an earlier unit basis of the flowsheet (per-cycle reactor amounts used as mol/s,
mmol/s H2 feed used as mol/s), so they violate the corrected balances. This
script:

  1. keeps the workbook's reactor decision point (Time_hyd, F_gas, Di) fixed;
  2. evaluates the two truth models there (disk-cache hits with the shipped
     cache);
  3. replaces every black-box-linked constraint by the same expression with the
     true black-box values substituted, fixes x_H2 at the value the reactor
     outlet implies, and solves the remaining glass-box system with IPOPT;
  4. checks every original constraint residual (black boxes at their true
     values) and every bound, and writes the point (by default to
     results/runs/x0_units_fixed.json, to compare with the shipped
     start_points/x0_units_fixed.json that run.py uses).

At the written point the glass-box model is consistent with y = t(w0). The solver
still seeds its output holders with its standard y0 = 1 (as for every other result in
the thesis): seeding y0 = t(w0) was tried and abandoned, because theta_0 = 0 sets the
funnel width to phi_min = 1e-8 (rule 5.2), which rejects every trial step.

Usage (from the repository root):  python start_points/reseed_x0.py [--out PATH]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import flowsheet as M                       # noqa: E402  (builds M.m at the workbook values)
from pyomo.environ import (Constraint, Objective, SolverFactory, Var,  # noqa: E402
                           minimize, value)

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'runs', 'x0_units_fixed.json'))
args = ap.parse_args()

m = M.m
th, fg, di = value(m.Time_hyd), value(m.F_gas), value(m.Di)
co2_in, ch4, h2, co2_out = M.TRF_blackbox_DFM(th, fg, di)[:4]
xh2 = h2 / (ch4 + h2)
psa = M.TRF_blackbox_PSA(xh2)
print(f"x0 decisions: Time_hyd={th}, F_gas={fg}, Di={di}; reactor CH4={ch4}, H2={h2} "
      f"mmol/cycle/channel; x_H2={xh2:.6f}; PSA={psa}")

for v in (m.Time_hyd, m.F_gas, m.Di):
    v.fix()
m.x_H2.fix(xh2)

bb_linked = ['DFM_c1', 'DFM_c3', 'Compressor_PSA_c1', 'Recycle_c1', 'CF_c2', 'CF_c3']
for name in bb_linked:
    getattr(m, name).deactivate()
hrs, n = M.operating_hours, M.plant_life
SC = M.SC
m.R_c1 = Constraint(expr=SC['DFM_n_total_outlet'] * m.DFM_n_total_outlet * m.total_cycle_time == (ch4 + h2) * 1e-3)
m.R_c2 = Constraint(expr=psa[1] * m.n_PSA_units == m.n_Compressor_out)
m.R_c3 = Constraint(expr=m.H2_recycled <= psa[2] * m.n_PSA_units)
m.R_c4 = Constraint(expr=SC['Profit_CH4'] * m.Profit_CH4 == psa[3] * 3.6 * m.n_PSA_units * m.CH4_price * hrs)
m.R_c5 = Constraint(expr=SC['Profit_H2'] * m.Profit_H2 == (psa[2] * m.n_PSA_units - m.H2_recycled) * 3.6 * m.H2_price * hrs)

m.LC.deactivate()
m.R_obj = Objective(expr=m.LCOCH4, sense=minimize)   # glass-box part only; decisions fixed

res = SolverFactory('ipopt').solve(m, tee=False, options={'max_iter': 5000, 'tol': 1e-10, 'bound_relax_factor': 0.0, 'honor_original_bounds': 'yes'})
print("IPOPT:", res.solver.status, res.solver.termination_condition)

# ---- verify: every ORIGINAL constraint with true black-box values, and all bounds
worst, worst_name = 0.0, None
for c in m.component_data_objects(Constraint, active=True):
    body = value(c.body)
    lo, up = c.lower, c.upper
    r = 0.0
    if lo is not None:
        r = max(r, value(lo) - body)
    if up is not None:
        r = max(r, body - value(up))
    scale = max(1.0, abs(body))
    if r / scale > worst:
        worst, worst_name = r / scale, c.name
bviol = []
for v in m.component_data_objects(Var):
    val = v.value
    if val is None:
        bviol.append((v.name, None)); continue
    if (v.lb is not None and val < v.lb - 1e-9) or (v.ub is not None and val > v.ub + 1e-9):
        bviol.append((v.name, val))
print(f"max scaled residual (active constraints) = {worst:.3e} at {worst_name}; bound violations = {bviol}")
if worst > 1e-6 or bviol or str(res.solver.termination_condition) != 'optimal':
    sys.exit("RESEED FAILED -- not writing the start point")

x0 = {v.name: float(v.value) for v in m.component_objects(Var) if v.is_indexed() is False}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(x0, f, indent=1)
print(f"wrote {args.out} ({len(x0)} variables)")

# restore the original constraints for the economics preview (true BB values)
for name in bb_linked:
    getattr(m, name).activate()
M.report_economics(m, header="ECONOMICS AT THE FEASIBLE START (corrected units)")
