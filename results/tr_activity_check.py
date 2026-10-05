# -*- coding: utf-8 -*-
"""
tr_activity_check.py -- is the trust-region bound active in the last
subproblem at each reported DFM-PSA design (thesis Section 8.4.2, dual
analysis)?

In the reduced trust region every x and z variable is boxed by
|v - v_k| <= Delta * (ub - lb). For each design this compares the reported
iterate (the centre) with the solution of the trust-region subproblem solved
there (saved by the 'duals' run) and lists each variable's move as a fraction
of its half-width; a fraction of about 1 means the bound is active.

    design            centre                         subproblem solution
    hypervolume off   hv_off_r1.point.json           hv_off_r1_duals.point.json
    hypervolume on    hv_on_r1_reported.point.json   hv_on_r1_duals.point.json

Delta is the radius of that last subproblem, read from the duals run. Writes
<runs>/tr_activity_hv_off.txt and tr_activity_hv_on.txt.

    python results/tr_activity_check.py [--runs results/reference]
"""
import argparse
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import flowsheet as F                        # noqa: E402  (variable bounds)

DESIGNS = {'hv_off': ('hv_off_r1', 'hv_off_r1_duals'),
           'hv_on': ('hv_on_r1_reported', 'hv_on_r1_duals')}


def check(runs, centre_run, duals_run):
    load = lambda n: json.load(open(os.path.join(runs, n), encoding='utf-8'))
    centre = load(centre_run + '.point.json')
    trial = load(duals_run + '.point.json')
    delta = load(duals_run + '.json')['trajectory']['trust_radius'][-1]
    out = io.StringIO()
    rows = []
    for name, c in centre.items():
        v = getattr(F.m, name)
        if v.lb is None or v.ub is None or v.ub <= v.lb:
            continue
        span = v.ub - v.lb
        at = ('lower' if abs(c - v.lb) <= 1e-6 * span else
              'upper' if abs(c - v.ub) <= 1e-6 * span else '')
        rows.append((abs(trial[name] - c) / (delta * span), name, c, trial[name], at))
    rows.sort(reverse=True)
    print(f'Delta = {delta:.6g}\n', file=out)
    print(f"{'variable':<22}{'move/TR':>9}{'centre':>14}{'trial':>14}  bound at centre",
          file=out)
    for frac, name, c, t, at in rows:
        print(f'{name:<22}{frac:9.3f}{c:14.6g}{t:14.6g}  {at}', file=out)
    active = [r[1] for r in rows if r[0] > 0.99]
    print(f'\nTR bound active (move >= 0.99 of half-width) for {len(active)} of '
          f'{len(rows)} variables: {active}', file=out)
    print('At a variable bound at the reported point:', [r[1] for r in rows if r[4]],
          file=out)
    return out.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=os.path.join(HERE, 'runs'))
    args = ap.parse_args()
    for design, (centre_run, duals_run) in DESIGNS.items():
        text = check(args.runs, centre_run, duals_run)
        path = os.path.join(args.runs, f'tr_activity_{design}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'== {design}\n{text}')


if __name__ == '__main__':
    main()
