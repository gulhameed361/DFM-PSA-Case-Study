# -*- coding: utf-8 -*-
"""
DFM-PSA flowsheet of thesis Section 8.4 (model in Appendix B).

Couples two expensive truth models through a glass-box flowsheet and cost
model, and minimises the levelised cost of methane (LCOCH4, USD/kg):
  block1 = DFM  dual-function-material reactor cycle (CO2 capture and
                methanation; reactor/, cited model), 2 outputs used
  block2 = PSA  pressure-swing adsorption H2/CH4 separator (psa/, cited
                model, needs pyapep), 3 outputs used

Importing this module builds the Pyomo model `m` at the workbook's initial
values (data/DFM-PSA PFS Initialisation.xlsx), `efmap`, and the black-box
wrappers. Start points are applied with load_point(); run.py solves.

Black-box evaluations are cached on disk (cache/*.json) under keys rounded to
2 dp (reactor inputs) and 6 dp (PSA input). A simulation is run at the exact
inputs of the first point that falls in a key's cell, and that value is then
returned for every later point in the same cell. The cached values are
therefore part of the case definition: with the shipped cache the reported
runs replay exactly; with an empty cache a run can take a different path.
"""
import json
import math
import os
import sys
from math import pi, log

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ('atrfu', 'reactor', 'psa'):
    sys.path.insert(0, os.path.join(_HERE, _sub))
_CACHEDIR = os.path.join(_HERE, 'cache')

from pyomo.environ import (                                  # noqa: E402
    ConcreteModel, Param, Var, Constraint, Objective, ExternalFunction,
    NonNegativeReals, minimize, value)
from openpyxl import load_workbook                           # noqa: E402

from cycle_model_CSS_X import DFM_Model                      # noqa: E402
from PSA_real_simulation_EM import PSA_simulation            # noqa: E402

DFM_CYCLES = 60      # cyclic-steady-state cycle cap of the reactor model

# ============================================================================
# Plant / costing parameters
# ============================================================================
operating_hours = 330 * 24
interest_rate = 0.000000000001
plant_life = 20

kWh_cost = 0.05
Water_cost = 1.1 * 1.72e-3 * 18.02e-3 * 1000
W_cost = kWh_cost / 3600
H2_ext_price = 20.016

SOEL_unit_cost = 269900
SOEL_CAPEX = 4200

DFM_unit_cost = 156.2
DFM_SPWeight = 85.43

Comp_cost_ref = 100000
W_comp_ref = 0.1
n_sf_comp = 0.7
F_m_comp = 2.5

T_Cooler_in = 600
T_Cooler_out = 323
C_p_gc = 29.1
U = 200
T_cold_in = 298.15
T_cold_out = 448.15
delta_T_lm = (T_Cooler_in - T_cold_out - (T_Cooler_out - T_cold_in)) / log(
    (T_Cooler_in - T_cold_out) / (T_Cooler_out - T_cold_in))
P_heat = 0.015
eta_recovery = 0.85
C_cost_ref = 50000
A_C_ref = 50
n_sf_C = 0.6
F_m_C = 2

PSA_vessel_Di = 0.3
PSA_vessel_P = 9
PSA_vessel_L = 1.35
PSA_vessel_V = (pi * (PSA_vessel_Di / 2) ** 2) * PSA_vessel_L
rho = 1324
B1 = 2.25
B2 = 1.82
F_m = 3.1
F_p = (((PSA_vessel_P * PSA_vessel_Di) / (2 * (850 - 0.6 * PSA_vessel_P))) + 0.00315) / 0.0063
K1 = 3.4974
K2 = 0.4485
K3 = 0.1074
_LC_precost = log(PSA_vessel_V)
C_p = 10 ** (K1 + K2 * _LC_precost + K3 * (_LC_precost ** 2))
C_bm_PSA = C_p * (B1 + B2 * F_m * F_p)

kg_price_adsorbent = 2.5
Annualised_Adsorbents_Cost_per_unit = ((PSA_vessel_V * rho) / 2) * kg_price_adsorbent

C_p_H2 = 29.1

CEPCI_2025 = 950
CEPCI_2013 = 567
CEPCI_2018 = 603.1
CEPCI_2007 = 509.7
CEPCI_2001 = 394.3


def annualized_capital(i, n):
    return (i * (1 + i) ** n) / ((1 + i) ** n - 1)


ACCR = annualized_capital(interest_rate, plant_life)

# ============================================================================
# Initial values (from the initialisation workbook)
# ============================================================================
FILE = os.path.join(_HERE, 'data', 'DFM-PSA PFS Initialisation.xlsx')
if not os.path.exists(FILE):
    raise FileNotFoundError(FILE)

wb = load_workbook(FILE, data_only=True)
init_to_use = wb['To_Use3']

OPEX_SOEL_INIT = init_to_use['M31'].value
CAPEX_SOEL_INIT = init_to_use['M32'].value

Time_hyd = init_to_use['M7'].value
F_gas = init_to_use['M8'].value
Di = init_to_use['M9'].value
DFM_n_total_outlet = init_to_use['M10'].value
V_react = init_to_use['M11'].value

H2O_op = init_to_use['M14'].value
total_cycle_time = init_to_use['M15'].value
total_channels = init_to_use['M16'].value
total_frontal_area = init_to_use['M17'].value
N_blocks_num = init_to_use['M18'].value
CAPEX_DFM = init_to_use['M34'].value
OPEX_DFM = init_to_use['M35'].value

W_comp = init_to_use['M20'].value
n_Compressor_out = init_to_use['M22'].value
CAPEX_COMP = init_to_use['M38'].value
OPEX_COMP = init_to_use['M39'].value

Q_cooler = init_to_use['M21'].value
A_init = init_to_use['M42'].value
Q_recovered = init_to_use['M43'].value
CAPEX_GC = init_to_use['M46'].value
OPEX_GC = init_to_use['M47'].value

x_CH4 = init_to_use['M25'].value
x_H2 = init_to_use['M24'].value
n_PSA_units = init_to_use['M26'].value
CAPEX_PSA = init_to_use['M48'].value
OPEX_PSA = init_to_use['M49'].value

H2_price = init_to_use['D30'].value
CH4_price = init_to_use['D31'].value

Profit_CH4 = init_to_use['M53'].value
Profit_H2 = init_to_use['M54'].value

CAPEX = init_to_use['M57'].value
OPEX_Y1 = init_to_use['M58'].value
OPEX_Y2 = init_to_use['M59'].value

LCOCH4 = init_to_use['M67'].value

H2_electrolyser = init_to_use['M73'].value
H2_external = init_to_use['M74'].value
H2_recycled = init_to_use['M75'].value
CAPEX_OPEX_H2_EXT = init_to_use['M76'].value

H2_ext_price = init_to_use['D35'].value

Q_H2_required = init_to_use['M79'].value

Q_Furnace = init_to_use['M81'].value
CAPEX_FURNACE = init_to_use['M83'].value
OPEX_FURNACE = init_to_use['M84'].value

# --- Nudge initial values that landed exactly on a variable bound (a
# zero-gradient stall risk) 1% inward. Q_Furnace was AT its lower bound
# (1e-5, 10000); H2_price (6, 15) and CH4_price (5, 10) were AT their upper
# bounds. ---
Q_Furnace = max(Q_Furnace, 1e-5 + 0.01 * (10000 - 1e-5))
H2_price = min(H2_price, 15 - 0.01 * (15 - 6))
CH4_price = min(CH4_price, 10 - 0.01 * (10 - 5))

# ============================================================================
# Black-box wrappers: in-memory rounded-key cache, persisted to disk
# ============================================================================


def _disk_cache_path(name):
    return os.path.join(_CACHEDIR, f'{name}.json')


def _load_disk_cache(name):
    path = _disk_cache_path(name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    return {tuple(json.loads(k)): v for k, v in raw.items()}


def _save_disk_cache(name, cache):
    """Write the cache atomically (temporary file, then os.replace). On Windows
    the replace can fail briefly while another program (e.g. a sync client)
    holds the file; retry, and if it still fails keep the entries in memory
    for the next save -- a cache write must never end a run."""
    import time
    os.makedirs(_CACHEDIR, exist_ok=True)
    path = _disk_cache_path(name)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({json.dumps(list(k)): v for k, v in cache.items()}, f)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    os.remove(tmp)
    print(f"[flowsheet] WARNING cache save skipped for {name} (file locked)")


TRF_CACHE_DFM = _load_disk_cache('dfm_cache')
TRF_CACHE_PSA = _load_disk_cache('psa_cache')
print(f"[flowsheet] cache: {len(TRF_CACHE_DFM)} reactor and "
      f"{len(TRF_CACHE_PSA)} PSA evaluation(s) loaded")


def _validate_bb(vals, names, where):
    """Return None if every output is physically admissible, else a reason.

    A non-finite or negative flow fed into the finite-difference ROM would
    give meaningless gradients, so such a point is never cached or used as is.
    """
    for n, v in zip(names, vals):
        if not math.isfinite(v):
            return f"{where}: {n} = {v} (not finite)"
        if v < 0.0:
            return f"{where}: {n} = {v:.6g} (negative, non-physical)"
    return None


_DFM_OUT_NAMES = ('CO2_in', 'CH4_out', 'H2_out', 'CO2_out')
_PSA_OUT_NAMES = ('n_H2_feed', 'n_total_gas_inlet_ads', 'n_H2_total', 'n_CH4_total')


def TRF_blackbox_DFM(*args):
    args = [x for x in args if x is not None]
    args = args[0] if isinstance(args[0], list) else args
    inputs = [value(x) if hasattr(x, 'is_expression_type') else float(x) for x in args]
    # Key: the inputs rounded to 2 dp plus the CSSC cycle cap (60).
    key = tuple(round(float(x), 2) for x in inputs) + (DFM_CYCLES,)

    if key in TRF_CACHE_DFM:
        return TRF_CACHE_DFM[key]

    Time_hyd_, F_gas_, Di_ = inputs
    result = [float(v) for v in DFM_Model(
        Time_hyd_, F_gas_, Di_,
        max_cycles=DFM_CYCLES, export_excel=False,
    )]

    bad = _validate_bb(result, _DFM_OUT_NAMES, f"DFM{tuple(round(v, 3) for v in inputs)}")
    if bad:
        # Retry once at double resolution before giving up on the point.
        print(f"[flowsheet] WARNING invalid black-box output -- {bad}; "
              f"retrying at 2x cycle budget")
        result = [float(v) for v in DFM_Model(
            Time_hyd_, F_gas_, Di_,
            max_cycles=DFM_CYCLES * 2, export_excel=False,
        )]
        bad = _validate_bb(result, _DFM_OUT_NAMES,
                           f"DFM{tuple(round(v, 3) for v in inputs)} (retry)")

    if bad:
        # Never raise -- an exception here aborts the whole solve. Refuse to
        # cache or return the bad point; clamp to a small positive floor so the
        # surrogate sees a finite, physically-admissible value and the funnel
        # can reject the step on its own terms.
        print(f"[flowsheet] ERROR invalid after retry -- {bad}; "
              f"clamping to floor and NOT caching")
        result = [max(v, 1e-12) if math.isfinite(v) else 1e-12 for v in result]
        return result

    TRF_CACHE_DFM[key] = result
    _save_disk_cache('dfm_cache', TRF_CACHE_DFM)
    print(f"[flowsheet] DFM cache miss -> {len(TRF_CACHE_DFM)} cached evaluation(s)")
    return result


def TRF_blackbox_PSA(*args):
    args = [x for x in args if x is not None]
    args = args[0] if isinstance(args[0], list) else args
    inputs = [value(x) if hasattr(x, 'is_expression_type') else float(x) for x in args]
    # 6 dp (not the 2 dp of the reactor's wide-range inputs): CF_c2/c3/c9
    # multiply the PSA outputs by large economic factors, so a coarser key would
    # reuse a stale result for a meaningfully different x_H2.
    key = tuple(round(float(x), 6) for x in inputs)

    if key in TRF_CACHE_PSA:
        return TRF_CACHE_PSA[key]

    mH2 = inputs
    result = [float(v) for v in PSA_simulation(mH2)]

    bad = _validate_bb(result, _PSA_OUT_NAMES, f"PSA(x_H2={inputs[0]:.6g})")
    if bad:
        # PSA has no time-stepping to refine, so there is no retry to make --
        # refuse to cache, clamp to a finite positive floor, and let the funnel
        # judge the resulting step.
        print(f"[flowsheet] ERROR invalid black-box output -- {bad}; "
              f"clamping to floor and NOT caching")
        return [max(v, 1e-12) if math.isfinite(v) else 1e-12 for v in result]

    TRF_CACHE_PSA[key] = result
    _save_disk_cache('psa_cache', TRF_CACHE_PSA)
    print(f"[flowsheet] PSA cache miss -> {len(TRF_CACHE_PSA)} cached evaluation(s)")
    return result


# ============================================================================
# Pyomo model
# ============================================================================
m = ConcreteModel()

# DFM parameters
m.L = Param(default=5)
# Hydrogenation-feed H2 concentration, from the reactor model's own inlet
# definition (hydrogenation_model_X.py: C0_H2 = (30/100) * P/(R*T), P = 3 atm,
# T = 600 K).
_P_HYD_ATM, _T_REACT_K, _H2_VOLPCT = 3.0, 600.0, 30.0
C0_H2 = (_H2_VOLPCT / 100.0) * (_P_HYD_ATM * 101325.0) / (8.314 * _T_REACT_K) * 1e-3   # mmol/cm^3
m.C_feed_hyd_H2 = Param(default=C0_H2)
m.W_DFM = Param(default=0.010)
# Plant scale: a 15 m^3/s feed. The pilot-unit specification (Q_cap = 15 L/s,
# 500-2500 channels) is scaled by PLANT_SCALE in the feed capacity and in the
# bounds on the channel count and frontal area; every other quantity follows
# from the balances.
PLANT_SCALE = 1000
m.Q_cap = Param(default=15000 * PLANT_SCALE)   # cm^3/s
# Scale-up by block size: the number of reactor blocks does not scale; each
# block holds PLANT_SCALE x more channels and is costed as one larger vessel
# (the 0.8 exponent of DFM_c10 then gives economies of scale).
m.N_block_size = Param(default=100 * PLANT_SCALE)
m.T_K = Param(default=600)
m.t_ads = Param(default=50)
m.t_prg1 = Param(default=50)
m.t_prg2 = Param(default=50)
m.H2_ext_price = Param(default=H2_ext_price)

m.P_Comp_in = Param(initialize=3)
m.P_Comp_out = Param(initialize=9)
m.R = Param(initialize=8.314)
m.kappa = Param(initialize=1.4)
m.eta_comp = Param(initialize=0.85)

# Variables (initial values from the workbook, with the three nudges above)
m.H2_electrolyser = Var(domain=NonNegativeReals, initialize=H2_electrolyser, bounds=(0, 1100))
m.H2O_op = Var(domain=NonNegativeReals, bounds=(0, 1100), initialize=H2O_op)
m.OPEX_SOEL = Var(domain=NonNegativeReals, initialize=OPEX_SOEL_INIT, bounds=(1e-1, 1e15))
m.CAPEX_SOEL = Var(domain=NonNegativeReals, initialize=CAPEX_SOEL_INIT, bounds=(1e-1, 1e15))

m.Time_hyd = Var(domain=NonNegativeReals, bounds=(100, 730), initialize=Time_hyd)
m.F_gas = Var(domain=NonNegativeReals, bounds=(6, 30), initialize=F_gas)
m.Di = Var(domain=NonNegativeReals, bounds=(0.8, 4), initialize=Di)

m.DFM_n_total_outlet = Var(domain=NonNegativeReals, bounds=(1e-9, 1000), initialize=DFM_n_total_outlet)  # mol/s per channel (cycle-averaged)

m.V_react = Var(domain=NonNegativeReals, bounds=(2.5, 65), initialize=V_react)

m.total_cycle_time = Var(domain=NonNegativeReals, bounds=(250, 880), initialize=total_cycle_time)
m.total_channels = Var(domain=NonNegativeReals, bounds=(500 * PLANT_SCALE, 2500 * PLANT_SCALE), initialize=total_channels * PLANT_SCALE)
m.total_frontal_area = Var(domain=NonNegativeReals, bounds=(200 * PLANT_SCALE, 35000 * PLANT_SCALE), initialize=total_frontal_area * PLANT_SCALE)
m.N_blocks_num = Var(domain=NonNegativeReals, bounds=(5, 25), initialize=N_blocks_num)
m.CAPEX_DFM = Var(initialize=CAPEX_DFM, bounds=(1e5, 1e13))
m.OPEX_DFM = Var(initialize=OPEX_DFM, bounds=(1e5, 1e13))

m.W_comp = Var(initialize=W_comp, bounds=(1e-5, 1000))
m.n_Compressor_out = Var(initialize=n_Compressor_out, bounds=(1e-5, 1e5))
m.CAPEX_COMP = Var(initialize=CAPEX_COMP, bounds=(1e-1, 1e15))
m.OPEX_COMP = Var(initialize=OPEX_COMP, bounds=(1e-1, 1e15))

m.Q_cooler = Var(initialize=Q_cooler, bounds=(1e-5, 1000))
m.A = Var(initialize=A_init, bounds=(1e-5, 1e15))
m.Q_recovered = Var(initialize=Q_recovered, bounds=(1e-5, 1000))
m.CAPEX_GC = Var(initialize=CAPEX_GC, bounds=(1e-1, 1e15))
m.OPEX_GC = Var(initialize=OPEX_GC, bounds=(1e-1, 1e15))

m.x_CH4 = Var(initialize=x_CH4, bounds=(1e-6, 1))
m.x_H2 = Var(initialize=x_H2, bounds=(0.4, 1))
m.n_PSA_units = Var(initialize=n_PSA_units, bounds=(1e-6, 1e5))
m.CAPEX_PSA = Var(initialize=CAPEX_PSA, bounds=(1e-1, 1e15))
m.OPEX_PSA = Var(initialize=OPEX_PSA, bounds=(1e-1, 1e15))

m.H2_external = Var(domain=NonNegativeReals, initialize=H2_external, bounds=(0, 1100))
m.H2_recycled = Var(domain=NonNegativeReals, initialize=H2_recycled, bounds=(0, 10000))
m.Q_H2_required = Var(bounds=(0, 1000), initialize=Q_H2_required)

m.CAPEX_OPEX_H2_EXT = Var(domain=NonNegativeReals, initialize=CAPEX_OPEX_H2_EXT, bounds=(1e-3, 1e15))

m.Q_Furnace = Var(bounds=(1e-5, 10000), initialize=Q_Furnace)
m.CAPEX_FURNACE = Var(bounds=(1e4, 1e9), initialize=CAPEX_FURNACE)
m.OPEX_FURNACE = Var(bounds=(1, 1e9), initialize=OPEX_FURNACE)

m.H2_price = Var(initialize=H2_price, bounds=(6, 15))
m.CH4_price = Var(initialize=CH4_price, bounds=(5, 10))

m.Profit_CH4 = Var(domain=NonNegativeReals, initialize=Profit_CH4, bounds=(1e-7, 1e15))
m.Profit_H2 = Var(domain=NonNegativeReals, initialize=Profit_H2, bounds=(1e-7, 1e15))

m.CAPEX = Var(domain=NonNegativeReals, initialize=CAPEX, bounds=(1e-1, 1e15))
m.OPEX_Y1 = Var(domain=NonNegativeReals, initialize=OPEX_Y1, bounds=(1e-1, 1e15))
m.OPEX_Y2 = Var(domain=NonNegativeReals, initialize=OPEX_Y2, bounds=(1e-1, 1e15))

m.LCOCH4 = Var(domain=NonNegativeReals, initialize=LCOCH4, bounds=(1e-6, 1e7))

# Black-box components: the five truth-model outputs used by the constraints.
def blackbox_DFM_CH4(a, b, c):
    return TRF_blackbox_DFM(a, b, c)[1]


DFM_n_total_CH4_out = ExternalFunction(blackbox_DFM_CH4)


def blackbox_DFM_H2(a, b, c):
    return TRF_blackbox_DFM(a, b, c)[2]


DFM_n_total_H2_out = ExternalFunction(blackbox_DFM_H2)


def blackbox_PSA_gas_inlet(a):
    return TRF_blackbox_PSA(a)[1]


PSA_n_total_gas_inlet_ads = ExternalFunction(blackbox_PSA_gas_inlet)


def blackbox_PSA_H2(a):
    return TRF_blackbox_PSA(a)[2]


PSA_n_H2_total = ExternalFunction(blackbox_PSA_H2)


def blackbox_PSA_CH4(a):
    return TRF_blackbox_PSA(a)[3]


PSA_n_CH4_total = ExternalFunction(blackbox_PSA_CH4)

# ============================================================================
# VARIABLE SCALING. The criticality measure chi is computed on a unit
# box in the model's own units (PyomoInterface.criticalityCheck). With costs in USD
# (1e5-1e8), 2.5e6 channels and a per-channel outlet of 1e-5 mol/s, that box let the
# cost and channel variables move by ~1 unit, which made chi ~1e-7 everywhere and
# produced spurious "optimal" exits. Every variable is therefore held in units that
# make it O(0.01-1000): costs in M USD (annual quantities in M USD/yr), channels and
# frontal area in millions, the per-channel outlet in mmol/s. Each constraint below
# multiplies a scaled variable back by its factor, so the physics is unchanged.
# ============================================================================
SC = {'OPEX_SOEL': 1e6, 'CAPEX_SOEL': 1e6, 'CAPEX_DFM': 1e6, 'OPEX_DFM': 1e6, 'CAPEX_COMP': 1e6, 'OPEX_COMP': 1e6, 'CAPEX_GC': 1e6, 'OPEX_GC': 1e6, 'CAPEX_PSA': 1e6, 'OPEX_PSA': 1e6, 'CAPEX_OPEX_H2_EXT': 1e6, 'CAPEX_FURNACE': 1e6, 'OPEX_FURNACE': 1e6, 'Profit_CH4': 1e6, 'Profit_H2': 1e6, 'CAPEX': 1e6, 'OPEX_Y1': 1e6, 'OPEX_Y2': 1e6, 'total_channels': 1e6, 'total_frontal_area': 1e6, 'DFM_n_total_outlet': 1e-3}
_COST_UB_MUSD = 1e5          # 100 billion USD: finite, non-binding bound for every cost variable
for _name, _f in SC.items():
    _v = getattr(m, _name)
    _lb, _ub = _v.lb, _v.ub
    _v.setlb(None if _lb is None else _lb / _f)
    _v.setub(None if _ub is None else _ub / _f)
    if _v.value is not None:
        _v.set_value(_v.value / _f)
    if _f == 1e6 and _name not in ('total_channels', 'total_frontal_area'):
        _v.setub(_COST_UB_MUSD)

# Constraints
m.SOEL_c1 = Constraint(expr=m.H2_electrolyser == m.H2O_op)
m.SOEL_c2 = Constraint(expr=(SC['OPEX_SOEL'] * m.OPEX_SOEL) == (m.H2_electrolyser * 3.6 * W_cost * SOEL_unit_cost + m.H2O_op * 3.6 * Water_cost) * operating_hours)
m.SOEL_c3 = Constraint(expr=(SC['CAPEX_SOEL'] * m.CAPEX_SOEL) == (m.H2_electrolyser * 3.6) * SOEL_unit_cost * SOEL_CAPEX / 3600)

# F_gas [cm^3/s] x C0_H2 [mmol/cm^3] is mmol/s per channel during hydrogenation;
# x1e-3 -> mol/s, x Time_hyd/total_cycle_time -> cycle-averaged demand (electrolyser runs continuously).
m.DFM_c0 = Constraint(expr=(m.H2_recycled + m.H2_electrolyser + m.H2_external) * m.total_cycle_time == m.F_gas * m.C_feed_hyd_H2 * 1e-3 * (SC['total_channels'] * m.total_channels) * m.Time_hyd)
# The reactor returns mmol PER CYCLE per channel; x1e-3 -> mol per cycle,
# / total_cycle_time -> cycle-averaged mol/s per channel.
m.DFM_c1 = Constraint(expr=(SC['DFM_n_total_outlet'] * m.DFM_n_total_outlet) * m.total_cycle_time == (DFM_n_total_CH4_out(m.Time_hyd, m.F_gas, m.Di) * (10 ** (-3))) + (DFM_n_total_H2_out(m.Time_hyd, m.F_gas, m.Di) * (10 ** (-3))))
m.DFM_c3 = Constraint(expr=m.x_H2 * (SC['DFM_n_total_outlet'] * m.DFM_n_total_outlet) * m.total_cycle_time == DFM_n_total_H2_out(m.Time_hyd, m.F_gas, m.Di) * (10 ** (-3)))  # same per-cycle basis as DFM_c1
m.DFM_c4 = Constraint(expr=m.V_react == (3.14 * (m.Di ** 2) / 4) * m.L)
m.DFM_c6 = Constraint(expr=m.total_cycle_time == m.t_ads + m.t_prg1 + m.Time_hyd + m.t_prg2)
m.DFM_c7 = Constraint(expr=(SC['total_channels'] * m.total_channels) == m.Q_cap / m.F_gas)
m.DFM_c8 = Constraint(expr=(SC['total_frontal_area'] * m.total_frontal_area) == (SC['total_channels'] * m.total_channels) * (22 / 7) * ((m.Di / 2) ** 2))
m.DFM_c9 = Constraint(expr=m.N_blocks_num == (SC['total_channels'] * m.total_channels) / m.N_block_size)
m.DFM_c10 = Constraint(expr=(SC['CAPEX_DFM'] * m.CAPEX_DFM) == 2 * ((53000 + (28000 * ((m.V_react * 1e-6 * m.N_block_size) ** 0.8))) * m.N_blocks_num * CEPCI_2025 / CEPCI_2007))
m.DFM_c11 = Constraint(expr=(SC['OPEX_DFM'] * m.OPEX_DFM) == 0.18 * (SC['CAPEX_DFM'] * m.CAPEX_DFM) + (DFM_unit_cost * m.W_DFM * (SC['total_channels'] * m.total_channels) / 10))

m.DFM_Compressor_c1 = Constraint(expr=(SC['DFM_n_total_outlet'] * m.DFM_n_total_outlet) * (SC['total_channels'] * m.total_channels) == m.n_Compressor_out)

m.GC_c1 = Constraint(expr=m.Q_cooler * 1e6 == m.n_Compressor_out * C_p_gc * (T_Cooler_in - T_Cooler_out))
m.GC_c2 = Constraint(expr=m.A == (m.Q_cooler * 1e6) / (U * delta_T_lm))
m.GC_c3 = Constraint(expr=m.Q_recovered == eta_recovery * m.Q_cooler)
m.GC_c6 = Constraint(expr=(SC['CAPEX_GC'] * m.CAPEX_GC) == 1.18 * (C_cost_ref * ((m.A / A_C_ref) ** n_sf_C) * F_m_C) * (CEPCI_2025 / CEPCI_2018))
m.GC_c7 = Constraint(expr=(SC['OPEX_GC'] * m.OPEX_GC) == 0.18 * (SC['CAPEX_GC'] * m.CAPEX_GC))

m.Compressor_c1 = Constraint(expr=m.W_comp * 1e6 == (m.n_Compressor_out * m.R * T_Cooler_out / m.eta_comp) * (m.kappa / (m.kappa - 1)) * ((m.P_Comp_out / m.P_Comp_in) ** ((m.kappa - 1) / m.kappa) - 1))
m.Compressor_c3 = Constraint(expr=(SC['CAPEX_COMP'] * m.CAPEX_COMP) == 1.18 * (Comp_cost_ref * ((m.W_comp / W_comp_ref) ** n_sf_comp) * F_m_comp) * (CEPCI_2025 / CEPCI_2001))
m.Compressor_c4 = Constraint(expr=(SC['OPEX_COMP'] * m.OPEX_COMP) == 0.18 * (SC['CAPEX_COMP'] * m.CAPEX_COMP))

m.Compressor_PSA_c1 = Constraint(expr=PSA_n_total_gas_inlet_ads(m.x_H2) * m.n_PSA_units == m.n_Compressor_out)

m.PSA_c1 = Constraint(expr=m.x_CH4 + m.x_H2 == 1)
m.PSA_c4 = Constraint(expr=(SC['CAPEX_PSA'] * m.CAPEX_PSA) == 1.18 * (m.n_PSA_units * C_bm_PSA) * (CEPCI_2025 / CEPCI_2001))
m.PSA_c5 = Constraint(expr=(SC['OPEX_PSA'] * m.OPEX_PSA) == (0.18 * (SC['CAPEX_PSA'] * m.CAPEX_PSA)) + (Annualised_Adsorbents_Cost_per_unit * m.n_PSA_units))

m.Recycle_c1 = Constraint(expr=m.H2_recycled <= PSA_n_H2_total(m.x_H2) * m.n_PSA_units)
m.Recycle_c2 = Constraint(expr=m.Q_H2_required * 1e6 == m.H2_recycled * C_p_H2 * (T_Cooler_in - T_Cooler_out))

m.Furnace_c1 = Constraint(expr=m.Q_Furnace + m.Q_recovered >= m.Q_H2_required)
# The furnace never supplies more than the whole H2 preheat duty (Appendix B).
# Without this the duty is bounded only by its 10,000 MW variable bound.
m.Furnace_c2 = Constraint(expr=m.Q_Furnace <= m.Q_H2_required)
# The correlation a + b*S^0.8 takes the duty S in MW (Q_Furnace is in MW).
m.Furnace_c3 = Constraint(expr=(SC['CAPEX_FURNACE'] * m.CAPEX_FURNACE) == (68500 + 93000 * (m.Q_Furnace ** 0.8)) * (CEPCI_2025 / CEPCI_2007))
m.Furnace_c4 = Constraint(expr=(SC['OPEX_FURNACE'] * m.OPEX_FURNACE) == (m.Q_Furnace * 1e3) * operating_hours * kWh_cost)

m.External_H2_c1 = Constraint(expr=(SC['CAPEX_OPEX_H2_EXT'] * m.CAPEX_OPEX_H2_EXT) == m.H2_external * 3.6 * m.H2_ext_price * operating_hours)

m.CF_c2 = Constraint(expr=(SC['Profit_CH4'] * m.Profit_CH4) == (PSA_n_CH4_total(m.x_H2) * 3.6 * m.n_PSA_units * m.CH4_price) * operating_hours)
m.CF_c3 = Constraint(expr=(SC['Profit_H2'] * m.Profit_H2) == (((PSA_n_H2_total(m.x_H2) * m.n_PSA_units) - m.H2_recycled) * 3.6 * m.H2_price) * operating_hours)

m.CF_c5 = Constraint(expr=(SC['CAPEX'] * m.CAPEX) == ((SC['CAPEX_OPEX_H2_EXT'] * m.CAPEX_OPEX_H2_EXT) + (SC['CAPEX_SOEL'] * m.CAPEX_SOEL) + (SC['CAPEX_DFM'] * m.CAPEX_DFM) + (SC['CAPEX_COMP'] * m.CAPEX_COMP) + (SC['CAPEX_GC'] * m.CAPEX_GC) + (SC['CAPEX_PSA'] * m.CAPEX_PSA) + (SC['CAPEX_FURNACE'] * m.CAPEX_FURNACE)) * ACCR)
m.CF_c6 = Constraint(expr=(SC['OPEX_Y1'] * m.OPEX_Y1) == (SC['OPEX_SOEL'] * m.OPEX_SOEL) + (SC['OPEX_DFM'] * m.OPEX_DFM) + (SC['OPEX_COMP'] * m.OPEX_COMP) + (SC['OPEX_GC'] * m.OPEX_GC) + (SC['OPEX_PSA'] * m.OPEX_PSA) + (SC['OPEX_FURNACE'] * m.OPEX_FURNACE))
m.CF_c6_0 = Constraint(expr=(SC['OPEX_Y2'] * m.OPEX_Y2) == (SC['CAPEX_OPEX_H2_EXT'] * m.CAPEX_OPEX_H2_EXT) + (SC['OPEX_SOEL'] * m.OPEX_SOEL) + (SC['OPEX_DFM'] * m.OPEX_DFM) + (SC['OPEX_COMP'] * m.OPEX_COMP) + (SC['OPEX_GC'] * m.OPEX_GC) + (SC['OPEX_PSA'] * m.OPEX_PSA) + (SC['OPEX_FURNACE'] * m.OPEX_FURNACE))

# Objective: levelised cost per kg of methane PRODUCED by the reactor, i.e. the
# CH4 entering the PSA (x_CH4 * n_Compressor_out). The PSA model's CH4 output is
# not used as the basis: it does not close the PSA's own CH4 balance.
m.CF_c9 = Constraint(expr=m.LCOCH4 * (m.x_CH4 * m.n_Compressor_out * 3.6 * operating_hours * 16.043) / 1e6 == (m.CAPEX + ((m.OPEX_Y1 + ((plant_life - 1) * m.OPEX_Y2)) / plant_life)))  # M USD/yr on both sides

m.LC = Objective(expr=m.LCOCH4, sense=minimize)


def report_economics(model, header="POST-SOLVE ECONOMICS AND BALANCE CHECK"):
    """Recompute the reported economics from the converged flows, and check the
    reactor's H2 and carbon balances, on the cycle-averaged basis used above.
    Returns a dict (also printed)."""
    # physical value: scaled variables are multiplied back by their factor (SC)
    v = lambda c: float(value(c)) * SC.get(getattr(c, 'local_name', ''), 1.0)
    th, fg, di = v(model.Time_hyd), v(model.F_gas), v(model.Di)
    co2_in, ch4_out, h2_out, co2_out = TRF_blackbox_DFM(th, fg, di)[:4]   # mmol / cycle / channel
    h2_fed = fg * C0_H2 * th                                               # mmol / cycle / channel
    xh2 = v(model.x_H2)
    psa = TRF_blackbox_PSA(xh2)
    n_psa = v(model.n_PSA_units)
    ch4_prod = psa[3] * n_psa                                              # mol/s, PSA model's CH4 output (does not close)
    ch4_kg_yr = ch4_prod * 3.6 * operating_hours * 16.043
    ch4_psa_in = v(model.n_Compressor_out) * (1.0 - xh2)                   # mol/s entering the PSA
    capex_ann = v(model.CAPEX)
    opex = (v(model.OPEX_Y1) + (plant_life - 1) * v(model.OPEX_Y2)) / plant_life
    tac = capex_ann + opex
    soel_kw = v(model.H2_electrolyser) * 3.6 * SOEL_unit_cost / 3600.0
    capex_units = {k: v(getattr(model, 'CAPEX_' + k)) for k in ('SOEL', 'DFM', 'COMP', 'GC', 'PSA', 'FURNACE')}
    opex_units = {k: v(getattr(model, 'OPEX_' + k)) for k in ('SOEL', 'DFM', 'COMP', 'GC', 'PSA', 'FURNACE')}
    cap_tot, op_tot = sum(capex_units.values()), sum(opex_units.values())
    r = dict(
        CH4_produced_kg_yr=ch4_psa_in * 3.6 * operating_hours * 16.043, CH4_into_PSA_mol_s=ch4_psa_in,
        CH4_PSA_output_mol_s=ch4_prod,
        PSA_CH4_recovery=(ch4_prod / ch4_psa_in if ch4_psa_in > 0 else float('nan')),
        H2_electrolyser_mol_s=v(model.H2_electrolyser), SOEL_kW=soel_kw,
        H2_to_CH4_stoich_ratio=(v(model.H2_electrolyser) / (4 * ch4_psa_in) if ch4_psa_in > 0 else float('nan')),
        PSA_CH4_balance_gap=(1.0 - psa[3] / ((1.0 - xh2) * psa[1]) if psa[1] > 0 else float('nan')),
        CAPEX_total_USD=cap_tot, CAPEX_annualised_USD_yr=capex_ann, OPEX_avg_USD_yr=opex, TAC_USD_yr=tac,
        LCOCH4_PSA_output_basis_USD_kg=(tac / ch4_kg_yr if ch4_kg_yr > 0 else float('nan')),
        # TAC over the CH4 ENTERING the PSA (PSA gas in x n_PSA x x_CH4); the
        # variant with an H2 credit values the PSA H2 product at 15 USD/kmol.

        LCOCH4_USD_kg=(tac / (ch4_psa_in * 3.6 * operating_hours * 16.043) if ch4_psa_in > 0 else float('nan')),
        LCOCH4_with_H2_credit_USD_kg=((tac - (psa[2] * n_psa - v(model.H2_recycled)) * 3.6 * operating_hours * 15.0)
                                             / (ch4_psa_in * 3.6 * operating_hours * 16.043) if ch4_psa_in > 0 else float('nan')),
        CAPEX_share={k: c / cap_tot for k, c in capex_units.items()},
        OPEX_share={k: c / op_tot for k, c in opex_units.items()},
        H2_balance_rel_err=((h2_fed - h2_out - 4.0 * ch4_out) / h2_fed if h2_fed > 0 else float('nan')),
        # carbon captured but not in the hydrogenation outlet (leaves in the purge steps, which the
        # reactor model does not return) -- reported, not a closure test
        C_unaccounted_frac=((co2_in - ch4_out - co2_out) / co2_in if co2_in > 0 else float('nan')),
        reactor_per_cycle_mmol=dict(CO2_in=co2_in, CH4_out=ch4_out, H2_out=h2_out, CO2_out=co2_out, H2_fed=h2_fed),
    )
    print("=" * 70); print(header); print("=" * 70)
    for k, val in r.items():
        print(f"  {k}: {val}")
    print("ECONOMICS_DICT " + json.dumps(r))
    return r

efmap = {
    "block1": [DFM_n_total_CH4_out, DFM_n_total_H2_out],
    "block2": [PSA_n_total_gas_inlet_ads, PSA_n_H2_total, PSA_n_CH4_total],
}


def load_point(model, path):
    """Set the model variables named in a JSON point file (name -> value)."""
    with open(path, encoding='utf-8') as f:
        point = json.load(f)
    for name, val in point.items():
        getattr(model, name).set_value(val)
    print(f"[flowsheet] initialised {len(point)} variable(s) from "
          f"{os.path.basename(path)}")


def dump_point(model, path):
    """Write every scalar variable (name -> value), the format load_point reads."""
    point = {v.local_name: float(value(v))
             for v in model.component_objects(Var, descend_into=False)
             if not v.is_indexed() and v.value is not None}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(point, f, indent=1)
    print(f"[flowsheet] wrote {len(point)} variable(s) to {path}")
