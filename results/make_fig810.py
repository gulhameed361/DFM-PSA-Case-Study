# -*- coding: utf-8 -*-
"""
make_fig810.py -- thesis Figure 8.10: convergence of A-TRFu (Strategy R1) on the
DFM-PSA flowsheet, (a) without and (b) with the hypervolume mechanism.
Infeasibility, sampling radius and criticality on a logarithmic axis, regret
|f_k - f*| on a linear axis, with one reference f* for both panels: the lower
of the two reported costs (the hypervolume-on design, iteration 55).

Panel (b) joins two runs: Run 1 up to iteration 64, and the restart from its
last iterate up to iteration 55. A run's trajectory ends at the last iteration
it began, so each segment is read from a run that passes through its end point
on the identical path: Run 1 from hv_on_r1_eps1e-5 (same start and path; the
script checks that it agrees with hv_on_r1_run1), the restart from
hv_on_r1_duals.

Reads results/runs/ (or --runs) and writes <runs>/figure_8_10.png and .svg.

    python results/make_fig810.py [--runs results/reference]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def trajectory(runs, name, last_it=None):
    t = json.load(open(os.path.join(runs, name + '.json'), encoding='utf-8'))['trajectory']
    keep = [i for i, k in enumerate(t['iter']) if last_it is None or k <= last_it]
    # A run that stops at a termination test records that iteration before its
    # sampling radius is set (NaN); it repeats the previous iterate, so drop it.
    while keep and t['sample_radius'][keep[-1]] != t['sample_radius'][keep[-1]]:
        keep.pop()
    return {key: [t[key][i] for i in keep] for key in ('iter', 'theta', 'sample_radius',
                                                        'chi', 'obj')}


def panel(ax1, t, fstar, reg_max, restart=None, title=''):
    its = t['iter']
    chi = [c if c < 1e7 else float('nan') for c in t['chi']]   # iteration 0: placeholder
    reg = [abs(f - fstar) for f in t['obj']]
    ax2 = ax1.twinx()
    p1, = ax1.plot(its, t['theta'], color='#1f77b4', linestyle='-', marker='o',
                   markersize=4, markevery=5, label='Infeasibility')
    p2, = ax1.plot(its, t['sample_radius'], color='#2ca02c', linestyle='--', marker='s',
                   markersize=4, markevery=5, label='Sampling Region')
    p3, = ax1.plot(its, chi, color='#d62728', linestyle='-.', marker='^', markersize=4,
                   markevery=5, label='Criticality')
    p4, = ax2.plot(its, reg, color='#111111', linestyle=':', linewidth=2, label='Regret')
    ax2.set_ylim(-0.03 * reg_max, reg_max)
    ax2.annotate(f'final regret {reg[-1]:.3f}', xy=(its[-1], reg[-1]),
                 xytext=(-60, 60), textcoords='offset points', ha='right', fontsize=11,
                 arrowprops=dict(arrowstyle='->', color='#111111', lw=1))
    if restart is not None:
        ax1.axvline(restart, color='#7f7f7f', linestyle='-', linewidth=1)
        ax1.text(restart, 1.5e2, ' restart', color='#7f7f7f', fontsize=11, va='top')
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Infeasibility / Sampling Region / Criticality (log scale)', fontsize=12)
    ax2.set_ylabel('Regret (linear scale)', fontsize=12, color='#111111')
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)
    ax1.set_xlim(0, its[-1])
    ax1.set_title(title, fontsize=12, loc='left')
    lines = [p1, p2, p3, p4]
    ax1.legend(lines, [p.get_label() for p in lines], loc='upper right', framealpha=0.9,
               edgecolor='none')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=os.path.join(HERE, 'runs'))
    args = ap.parse_args()

    a = trajectory(args.runs, 'hv_off_r1')
    run1 = trajectory(args.runs, 'hv_on_r1_eps1e-5', last_it=64)
    check = trajectory(args.runs, 'hv_on_r1_run1')
    assert run1['theta'][:len(check['theta'])] == check['theta'], \
        'hv_on_r1_eps1e-5 and hv_on_r1_run1 do not share a path'
    restart = trajectory(args.runs, 'hv_on_r1_duals', last_it=55)
    off = run1['iter'][-1]
    b = {k: run1[k] + restart[k][1:] for k in run1}
    b['iter'] = run1['iter'] + [off + k for k in restart['iter'][1:]]

    fstar = restart['obj'][-1]
    reg_max = 1.05 * max(abs(f - fstar) for f in a['obj'] + b['obj'])
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10, 11))
    panel(ax_a, a, fstar, reg_max, title='(a) Without the hypervolume mechanism (R1)')
    panel(ax_b, b, fstar, reg_max, restart=off,
          title='(b) With the hypervolume mechanism (R1), restarted from the best point '
                'of the first run')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        path = os.path.join(args.runs, f'figure_8_10.{ext}')
        fig.savefig(path, dpi=300)
        print('wrote', path)
    print(f'f* = {fstar:.4f} USD/kg; panel (b): {len(b["iter"])} points, restart at {off}')


if __name__ == '__main__':
    main()
