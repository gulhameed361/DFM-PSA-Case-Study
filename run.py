# -*- coding: utf-8 -*-
"""
run.py -- the A-TRFu runs behind thesis Section 8.4 (DFM-PSA flowsheet).

Every run uses the funnel strategy with the reduced trust region, variable
scaling, Delta_min = 5e-4 and the restoration proximity weight 1e-6 (both
stated in Section 8.4.1); they differ in the hypervolume mechanism, the
restoration strategy, the start point, the feasibility tolerance ep_i and the
iteration budget:

  name                    HV  R   start          ep_i   max_it  role in Section 8.4
  hv_off_r1               off R1  original       1e-3   1000    reported design, HV off
  hv_off_r1_duals         off R1  original       1e-3     87    duals at that design
  hv_off_r1_second_start  off R1  second         1e-3   1000    alternative, 4.33 USD/kg
  hv_off_r1_eps1e-5       off R1  original       1e-5   1000    tolerance 1e-5 not met
  hv_on_r1_run1           on  R1  original       6e-3     63    Run 1; its last iterate is the second start
  hv_on_r1_second_start   on  R1  second         1e-3   1000    the restart, run to its budget
  hv_on_r1_reported       on  R1  second         1e-3     54    stops at the reported iterate (iteration 55)
  hv_on_r1_duals          on  R1  second         1e-3     55    duals at that design
  hv_on_r1_eps1e-5        on  R1  original       1e-5   1000    tolerance 1e-5 not met
  r2_hv_off_original      off R2  original       1e-5   1000    Strategy R2
  r2_hv_on_original       on  R2  original       1e-5   1000    Strategy R2
  r2_hv_off_second        off R2  second         1e-3   1000    Strategy R2
  r2_hv_on_second         on  R2  second         1e-3   1000    Strategy R2

original = start_points/x0_units_fixed.json (the workbook design, reseeded by
reseed_x0.py); second = start_points/x0_best_hvR1.json (the last iterate of
hv_on_r1_run1). A 'duals' run stops at the reported iterate, so its last
subproblem is the trust-region subproblem centred there; its constraint
multipliers are saved, and so is that subproblem's solution (for
results/tr_activity_check.py).

Each run writes results/runs/<name>.log (full solver output) and <name>.json
(solver result with the iteration trajectory, the multipliers when requested,
and the economics recomputed at the final point); runs marked below also write
<name>.point.json (the final point, in the start-point format).

    python run.py <name>          # one run
    python run.py --list
"""
import argparse
import contextlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
START = os.path.join(HERE, 'start_points')
OUT = os.path.join(HERE, 'results', 'runs')

ORIGINAL = os.path.join(START, 'x0_units_fixed.json')
SECOND = os.path.join(START, 'x0_best_hvR1.json')


def _run(hv, r, start, ep_i, max_it, duals=False, point=False):
    return dict(hv=hv, r=r, start=start, ep_i=ep_i, max_it=max_it,
                duals=duals, point=point)


RUNS = {
    'hv_off_r1':              _run(0, 1, ORIGINAL, 1e-3, 1000, point=True),
    'hv_off_r1_duals':        _run(0, 1, ORIGINAL, 1e-3, 87, duals=True, point=True),
    'hv_off_r1_second_start': _run(0, 1, SECOND, 1e-3, 1000),
    'hv_off_r1_eps1e-5':      _run(0, 1, ORIGINAL, 1e-5, 1000),
    'hv_on_r1_run1':          _run(1, 1, ORIGINAL, 6e-3, 63, point=True),
    'hv_on_r1_second_start':  _run(1, 1, SECOND, 1e-3, 1000),
    'hv_on_r1_reported':      _run(1, 1, SECOND, 1e-3, 54, point=True),
    'hv_on_r1_duals':         _run(1, 1, SECOND, 1e-3, 55, duals=True, point=True),
    'hv_on_r1_eps1e-5':       _run(1, 1, ORIGINAL, 1e-5, 1000),
    'r2_hv_off_original':     _run(0, 2, ORIGINAL, 1e-5, 1000),
    'r2_hv_on_original':      _run(1, 2, ORIGINAL, 1e-5, 1000),
    'r2_hv_off_second':       _run(0, 2, SECOND, 1e-3, 1000),
    'r2_hv_on_second':        _run(1, 2, SECOND, 1e-3, 1000),
}


def solve(name):
    cfg = RUNS[name]
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, name)
    with open(stem + '.log', 'w', encoding='utf-8') as log, \
            contextlib.redirect_stdout(log):
        import flowsheet as F                    # builds the model
        from TRF import TrustRegionSolver
        F.load_point(F.m, cfg['start'])
        solver = TrustRegionSolver(
            solver='ipopt',
            globalization_strategy=1,
            multiple_black_boxes=cfg['hv'],
            restoration_strategy=cfg['r'],
            scaling=0,
            trust_radius=1.0,
            sample_radius=0.1,
            ep_i=cfg['ep_i'],
            ep_s=1e-4,
            delta_min=5e-4,
            frp_proximity=1e-6,
            max_it=cfg['max_it'],
            report_duals=int(cfg['duals']),
        )
        t0 = time.time()
        result = solver.solve(F.m, F.efmap)
        result['wall_s'] = time.time() - t0
        result['economics'] = F.report_economics(F.m)
        result['settings'] = dict(cfg, start=os.path.basename(cfg['start']))
        if not cfg['duals']:
            result.pop('duals', None)
        if cfg['point']:
            F.dump_point(F.m, stem + '.point.json')
    with open(stem + '.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', nargs='?', choices=list(RUNS))
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    if args.list or not args.name:
        for n, c in RUNS.items():
            print(f"{n:24s} HV={'on ' if c['hv'] else 'off'} R{c['r']} "
                  f"start={os.path.basename(c['start']):22s} ep_i={c['ep_i']:.0e} "
                  f"max_it={c['max_it']}")
        return
    r = solve(args.name)
    print(f"{args.name}: {r['status']} | iterations {r['iterations']} | "
          f"theta {r['final_theta']:.3e} | LCOCH4 {r['final_obj']:.4f} USD/kg | "
          f"evaluations {r['bb_evals']} {r['bb_evals_detail']} | {r['wall_s']:.0f} s")


if __name__ == '__main__':
    main()
