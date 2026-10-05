# -*- coding: utf-8 -*-
"""
make_table87.py -- thesis Table 8.7: the two reported DFM-PSA designs (Strategy
R1, hypervolume off and on), in physical units.

Reads, from results/runs/ (or the directory given with --runs):
  hv_off_r1.json / .point.json           design without the hypervolume mechanism
  hv_on_r1_reported.json / .point.json   design with it (iterate of iteration 55)
  hv_on_r1_run1.json                     Run 1, whose last iterate started the
                                         run above (its counts are added)
Writes <runs>/table_8_7.csv and <runs>/table_8_7_raw.json (all values, used by
make_cost_figure.py).

Simulations are true black-box evaluations divided by the outputs each unit
returns per simulation (reactor 2, PSA 3).

    python results/make_table87.py [--runs results/reference]
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import flowsheet as F                        # noqa: E402  (model scaling SC, efmap)

COLS = {'Hypervolume off (R1)': ['hv_off_r1'],
        'Hypervolume on (R1)': ['hv_on_r1_run1', 'hv_on_r1_reported']}
VARS = ('CAPEX_SOEL', 'OPEX_SOEL', 'CAPEX_DFM', 'OPEX_DFM', 'CAPEX_COMP', 'OPEX_COMP',
        'CAPEX_GC', 'OPEX_GC', 'CAPEX_PSA', 'OPEX_PSA', 'CAPEX_FURNACE', 'OPEX_FURNACE',
        'CAPEX_OPEX_H2_EXT', 'Time_hyd', 'F_gas', 'Di', 'total_channels', 'N_blocks_num',
        'V_react', 'x_H2', 'n_PSA_units', 'W_comp', 'Q_cooler', 'Q_recovered', 'Q_Furnace',
        'H2_electrolyser', 'H2_external', 'H2_recycled', 'n_Compressor_out', 'A',
        'total_cycle_time')


def usd(v):
    return f"{round(v, -3):,.0f}"


def sci(v):
    m, e = f'{v:.1e}'.split('e')
    return f'{m} × 10^{int(e)}'


def column(runs, names):
    res = [json.load(open(os.path.join(runs, n + '.json'), encoding='utf-8'))
           for n in names]
    point = json.load(open(os.path.join(runs, names[-1] + '.point.json'), encoding='utf-8'))
    final, e = res[-1], res[-1]['economics']
    g = {k: point[k] * F.SC.get(k, 1.0) for k in VARS}
    g.update(SOEL_kW=e['SOEL_kW'], CH4_kt=e['CH4_produced_kg_yr'] / 1e6,
             TAC=e['TAC_USD_yr'], CAPEX_tot=e['CAPEX_total_USD'], LCOCH4=e['LCOCH4_USD_kg'],
             LCOCH4_cr=e['LCOCH4_with_H2_credit_USD_kg'],
             CAPEX_annualised=e['CAPEX_annualised_USD_yr'], OPEX_avg=e['OPEX_avg_USD_yr'],
             theta=final['final_theta'], iterations=[r['iterations'] for r in res],
             final_status=final['status'])
    for blk, unit in (('block1', 'reactor'), ('block2', 'psa')):
        g[f'{unit}_simulations'] = sum(r['bb_evals_detail'][blk] for r in res) \
            // len(F.efmap[blk])
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=os.path.join(HERE, 'runs'))
    args = ap.parse_args()

    data = {c: column(args.runs, n) for c, n in COLS.items()}
    with open(os.path.join(args.runs, 'table_8_7_raw.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)

    OFF, ON = data['Hypervolume off (R1)'], data['Hypervolume on (R1)']
    two = lambda fmt, k: [fmt(OFF[k]), fmt(ON[k])]
    f0, f1, f2, f3 = (lambda v: f'{v:.0f}'), (lambda v: f'{v:.1f}'), \
        (lambda v: f'{v:.2f}'), (lambda v: f'{v:.3f}')
    rows = [
        ['Hydrogen production: SOEL', '', '', ''],
        ['Total capital cost', 'USD', *two(usd, 'CAPEX_SOEL')],
        ['Operating cost', 'USD/yr', *two(usd, 'OPEX_SOEL')],
        ['Electrical load', 'MW', *[f'{v / 1e3:.1f}' for v in (OFF['SOEL_kW'], ON['SOEL_kW'])]],
        ['Hydrogen produced', 'mol/s', *two(f1, 'H2_electrolyser')],
        ['Adsorption-reaction unit: DFM', '', '', ''],
        ['Total capital cost', 'USD', *two(usd, 'CAPEX_DFM')],
        ['Operating cost', 'USD/yr', *two(usd, 'OPEX_DFM')],
        ['Hydrogenation time', 's', *two(f0, 'Time_hyd')],
        ['Cycle time', 's', *two(f0, 'total_cycle_time')],
        ['Feed per channel', 'cm3/s', *two(f1, 'F_gas')],
        ['Channel diameter', 'cm', *two(f2, 'Di')],
        ['Volume per channel', 'cm3', *two(f2, 'V_react')],
        ['Number of channels', '-', *two(usd, 'total_channels')],
        ['Number of blocks', '-', *two(f1, 'N_blocks_num')],
        ['Compression and gas cooling', '', '', ''],
        ['Compressor capital cost', 'USD', *two(usd, 'CAPEX_COMP')],
        ['Compressor operating cost', 'USD/yr', *two(usd, 'OPEX_COMP')],
        ['Compressor power', 'MW', *two(f2, 'W_comp')],
        ['Gas-cooler capital cost', 'USD', *two(usd, 'CAPEX_GC')],
        ['Gas-cooler operating cost', 'USD/yr', *two(usd, 'OPEX_GC')],
        ['Heat recovered', 'MW', *two(f2, 'Q_recovered')],
        ['Separation: PSA', '', '', ''],
        ['Total capital cost', 'USD', *two(usd, 'CAPEX_PSA')],
        ['Operating cost', 'USD/yr', *two(usd, 'OPEX_PSA')],
        ['Hydrogen fraction of the feed', '-', *two(f3, 'x_H2')],
        ['Recycle heating: furnace', '', '', ''],
        ['Total capital cost', 'USD', *two(usd, 'CAPEX_FURNACE')],
        ['Furnace duty', 'MW', *two(lambda v: '0' if v < 5e-3 else f'{v:.2f}', 'Q_Furnace')],
        ['Hydrogen supply', '', '', ''],
        ['External hydrogen', 'mol/s', *two(f1, 'H2_external')],
        ['External hydrogen cost (from year 2)', 'USD/yr', *two(usd, 'CAPEX_OPEX_H2_EXT')],
        ['Recycled hydrogen', 'mol/s', *two(f1, 'H2_recycled')],
        ['Plant performance', '', '', ''],
        ['Methane produced', 'kt/yr', *two(f2, 'CH4_kt')],
        ['Total capital cost', 'USD', *two(usd, 'CAPEX_tot')],
        ['Total annualised cost', 'USD/yr', *two(usd, 'TAC')],
        ['Levelised cost of methane', 'USD/kg', *two(f2, 'LCOCH4')],
        ['Levelised cost with hydrogen credit', 'USD/kg', *two(f2, 'LCOCH4_cr')],
        ['Algorithm', '', '', ''],
        ['Final infeasibility', '-', *two(sci, 'theta')],
        ['Iterations', '-', ' + '.join(map(str, OFF['iterations'])),
         ' + '.join(map(str, ON['iterations'])) + ' (two runs)'],
        ['Reactor / PSA simulations', '-',
         *[f"{g['reactor_simulations']} / {g['psa_simulations']}" for g in (OFF, ON)]],
    ]
    header = ['Variable', 'Unit', 'Hypervolume off', 'Hypervolume on']
    path = os.path.join(args.runs, 'table_8_7.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for line in [header] + rows:
        print('  '.join(str(x).ljust(w) for x, w in zip(line, widths)))
    print('\nwrote', path)


if __name__ == '__main__':
    main()
