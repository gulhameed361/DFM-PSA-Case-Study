# -*- coding: utf-8 -*-
"""
make_cost_figure.py -- thesis Figure 8.11: (a) capital and (b) annual operating
cost by unit of the two reported DFM-PSA designs (logarithmic axes, since the
electrolyser carries about 90% of the capital), and (c) the total annualised
cost with the levelised cost of methane.

External hydrogen is bought from year 2 (the year-1 purchase is financed with
the capital, following the cost model), so it has an operating bar in (b) and
no capital bar in (a).

Reads <runs>/table_8_7_raw.json (written by make_table87.py) and writes
<runs>/figure_8_11.png and .svg.

    python results/make_cost_figure.py [--runs results/reference]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ['Hypervolume off (R1)', 'Hypervolume on (R1)']
UNITS = [('SOEL', 'SOEL'), ('H2_EXT', 'External H$_2$ purchase'), ('DFM', 'DFM reactor'),
         ('COMP', 'Compressor'), ('PSA', 'PSA'), ('GC', 'Gas cooler'), ('FURNACE', 'Furnace')]
COLOURS = {'Hypervolume off (R1)': '#1f77b4', 'Hypervolume on (R1)': '#d62728'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=os.path.join(HERE, 'runs'))
    args = ap.parse_args()
    data = json.load(open(os.path.join(args.runs, 'table_8_7_raw.json'), encoding='utf-8'))
    for c in COLS:
        data[c]['OPEX_H2_EXT'] = data[c]['CAPEX_OPEX_H2_EXT']
        data[c]['CAPEX_H2_EXT'] = 0.0

    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    fig = plt.figure(figsize=(17, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05])
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    axc = fig.add_subplot(gs[0, 2])
    y = np.arange(len(UNITS))
    h = 0.38
    for ax, kind, xlabel in ((axes[0], 'CAPEX', 'Total capital cost (million USD, log scale)'),
                             (axes[1], 'OPEX', 'Operating cost (million USD yr$^{-1}$, log scale)')):
        for k, c in enumerate(COLS):
            raw = [data[c][f'{kind}_{u}'] / 1e6 for u, _ in UNITS]
            vals = [v if v >= 1e-3 else float('nan') for v in raw]          # zero duty: no bar
            bars = ax.barh(y + (k - 0.5) * h, vals, height=h, color=COLOURS[c], label=c)
            for bar, v in zip(bars, raw):
                label = f'{v:.3g}' if v >= 1e-3 else ('–' if kind == 'CAPEX' else '0 (furnace off)')
                xpos = v * 1.15 if v >= 1e-3 else 1.2e-2
                ax.text(xpos, bar.get_y() + bar.get_height() / 2, label, va='center', fontsize=10)
        ax.set_xscale('log')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.grid(True, axis='x', which='major', linestyle='--', alpha=0.5)
        ax.set_xlim(1e-2, 4e2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([n for _, n in UNITS])
    axes[0].invert_yaxis()
    axes[0].set_title('(a) Capital cost', fontsize=12, loc='left')
    axes[1].set_title('(b) Annual operating cost', fontsize=12, loc='left')
    plt.setp(axes[1].get_yticklabels(), visible=False)

    # (c) total annualised cost = annualised capital + average annual operating cost
    for k, c in enumerate(COLS):
        cap, op = data[c]['CAPEX_annualised'] / 1e6, data[c]['OPEX_avg'] / 1e6
        axc.barh(k, cap, height=0.55, color='#7f7f7f', label='Annualised capital' if k == 0 else None)
        axc.barh(k, op, left=cap, height=0.55, color='#bcbd22', label='Operating (average)' if k == 0 else None)
        axc.text(cap / 2, k, f'{cap:.2f}', ha='center', va='center', fontsize=10, color='white')
        axc.text(cap + op / 2, k, f'{op:.2f}', ha='center', va='center', fontsize=10)
        axc.text(cap + op + 0.3, k, f'TAC {cap + op:.2f} M USD yr$^{{-1}}$\n'
                 f'LCOCH$_4$ {data[c]["LCOCH4"]:.2f} USD kg$^{{-1}}$', va='center', fontsize=10)
    axc.set_yticks(np.arange(len(COLS)))
    axc.set_yticklabels(['Hypervolume off\n(R1)', 'Hypervolume on\n(R1)'])
    axc.invert_yaxis()
    axc.set_xlim(0, 32)
    axc.set_xlabel('Total annualised cost (million USD yr$^{-1}$)', fontsize=12)
    axc.grid(True, axis='x', which='major', linestyle='--', alpha=0.5)
    axc.set_title('(c) Total annualised cost', fontsize=12, loc='left')
    axc.legend(loc='lower right', framealpha=0.9, edgecolor='none', fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               framealpha=0.9, edgecolor='none', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ('png', 'svg'):
        path = os.path.join(args.runs, f'figure_8_11.{ext}')
        fig.savefig(path, dpi=300)
        print('wrote', path)


if __name__ == '__main__':
    main()
