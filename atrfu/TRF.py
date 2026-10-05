from math import pow

import time

from pyomo.common.dependencies import numpy as np

from pyomo.environ import Objective, Var, value
from pyomo.common.config import (
    ConfigBlock, ConfigValue, PositiveInt, PositiveFloat, NonNegativeFloat, In)

from filterMethod import (
    FilterElement, Filter)
from funnelMethod import Funnel
# Importing GJHPseudoSolver also registers the 'local.gjh' SolverFactory
# entry, so any entry script that reaches TRF() gets it automatically.
from GJHPseudoSolver import ensure_gjh
from helper import (cloneXYZ, packXYZ, system_report)
from Logger import Logger
from PyomoInterface import PyomoInterface
from HV_calculations import calculate_hypervolume, update_pareto_front


# =============================================================================
# Sampling-region update (Eq 14): hypervolume-driven 3-case rule
# =============================================================================

def update_sampling_region(config, reduction_ratio_HV, theta_values, HV, HVk,
                           k_population):
    """Update sigma_{k+1} from the HV reduction ratio (Eq 14).

    rho_HV > eta3   -> expand  (sigma *= sigma_inc, capped at the trust radius)
    0 < rho <= eta3 -> maintain
    rho_HV <= 0     -> shrink  (sigma *= sigma_dec, floored at delta_min)

    The Pareto front and HVk are updated only when rho_HV > 0 (Eq 14 text).
    """
    eta3      = config.sigma_eta3
    sigma_inc = config.sigma_inc
    sigma_dec = config.sigma_dec
    if reduction_ratio_HV > 0:
        if reduction_ratio_HV > eta3:
            config.sample_radius = min(sigma_inc * config.sample_radius,
                                       config.trust_radius)
        else:
            config.sample_radius = min(config.sample_radius,
                                       config.trust_radius)
        k_population = update_pareto_front(k_population, np.array(theta_values))
        HVk = HV
    else:
        config.sample_radius = min(
            max(sigma_dec * config.sample_radius, config.delta_min),
            config.trust_radius)
    return k_population, HVk


# =============================================================================
# Hypervolume reduction ratio (Eqs 11-13)
# =============================================================================

def hv_reduction_ratio(k_population, ref_point, error_vec_plus, HVk):
    """Return (rho_HV, candidate_front, HV_actual).

    actual    = HV(update(P_k, phi_{k+1})) - HV_k        (Eq 11)
    predicted = HV(update(P_k, 0)) - HV_k                (Eq 12)
    rho_HV    = actual / predicted, 0 if predicted <= 0  (Eq 13)
    """
    temp_act = update_pareto_front(k_population, np.array(error_vec_plus))
    HV_act = calculate_hypervolume(temp_act, ref_point)
    actual = HV_act - HVk

    temp_pred = update_pareto_front(
        k_population, np.zeros_like(np.asarray(error_vec_plus, dtype=float)))
    HV_pred = calculate_hypervolume(temp_pred, ref_point)
    predicted = HV_pred - HVk

    rho = 0.0 if predicted <= 0 else actual / predicted
    return rho, temp_act, HV_act


# =============================================================================
# Delta update on a REJECTED step (shared by both globalization strategies)
# =============================================================================

def radius_on_reject(config, restoration_mode, rhok):
    """Update config.trust_radius for a rejected trial step.

    R1 / DEFAULT (restoration strategy R1): every rejected step
    contracts UNCONDITIONALLY, restoration or not. That is the convergence anchor
    -- "infinitely many rejections => Delta -> 0", which with a kappa-fully-linear
    ROM forces the criticality measure to zero.

    R2 (restoration strategy R2): a RESTORATION iteration
    (>=2 consecutive rejections) uses the theta-reduction ratio rhok instead, so
    persistent rejections at a point that is feasible for the TRSP but not for the
    true problem cannot strangle Delta and trap the run at an unrecoverable
    infeasible point. Restoration sits outside the f-step / theta-step / rejection
    legs of the convergence argument, and n_restoration_max keeps the phase finite
    -- see the entry guard in TRF().
    """
    if restoration_mode and config.restoration_strategy == 2:
        if rhok < config.eta1:
            config.trust_radius = max(config.gamma_c * config.trust_radius,
                                      config.delta_min)
        elif rhok >= config.eta2:
            config.trust_radius = min(config.gamma_e * config.trust_radius,
                                      config.radius_max)
        # eta1 <= rhok < eta2 -> maintain
    else:
        config.trust_radius = max(config.gamma_c * config.trust_radius,
                                  config.delta_min)


# =============================================================================
# Main TRF function
# =============================================================================

def TRF(m, eflist, config):

    start_time = time.time()
    # Whole-run timers: wall clock (perf_counter) and process CPU time.
    _t_wall0 = time.perf_counter()
    _t_cpu0 = time.process_time()

    # Make sure the gjh executable is available before anything else runs:
    # existing PATH installation > bundled repo copy > Pyomo config dir >
    # automatic download (then prepended to PATH for this process).
    ensure_gjh(verbose=True)

    logger = Logger()
    filteR = Filter()
    problem = PyomoInterface(m, eflist, config)
    x, y, z = problem.getInitialValue()

    # ---- startup banner: configuration + problem size -------------------
    # x = black-box INPUT variables, y = black-box OUTPUT (surrogate / tear)
    # variables, z = remaining glass-box decision variables; total = x+y+z.
    _strat_name = {0: 'Filter', 1: 'Funnel'}.get(
        config.globalization_strategy, '?')
    _frp_name = {1: 'R1 classic (contract on reject)',
                 2: 'R2 relaxed (entry reset + rho radius)'}.get(
                     config.restoration_strategy, '?')
    _lx0, _ly0, _lz0 = problem.lx, problem.ly, problem.lz
    _neq0, _nineq0 = problem.count_constraints()
    print("========== A-TRFu SOLVER ==========")
    print(f"Globalization strategy: {config.globalization_strategy} "
          f"({_strat_name})")
    print(f"Hypervolume sampling  : multiple_black_boxes="
          f"{config.multiple_black_boxes}")
    print(f"Restoration strategy  : {config.restoration_strategy} "
          f"({_frp_name})")
    print(f"Problem size          : {_lx0 + _ly0 + _lz0} variables  "
          f"(x|y|z = {_lx0}|{_ly0}|{_lz0});  "
          f"{len(problem.TRF.y)} black-box block(s)")
    print("    x = black-box inputs | y = black-box outputs | z = remaining")
    print(f"Constraints           : {_neq0 + _nineq0}  "
          f"(equality {_neq0} | inequality {_nineq0})")
    print(f"Tolerances            : ep_i={config.ep_i:.1e}, "
          f"ep_chi={config.ep_chi:.1e}, ep_delta={config.ep_delta:.1e}, "
          f"max_it={config.max_it}")
    print("===================================\n")

    iteration = -1

    if config.scaling == 0:
        config.trust_radius = 1
        config.radius_max = 1

    romParam, yr = problem.buildROM(x, config.sample_radius, config.scaling)
    g, J, varlist, conlist = \
        problem.grad_hess_calc(x, y, z, romParam, config.scaling)
    rebuildROM = False

    xk, yk, zk = cloneXYZ(x, y, z)

    chik = 1e8

    thetak = np.linalg.norm(yr - yk, 1)
    IT = None
    objk = problem.evaluateObj(x, y, z)

    # NOTE: PyomoInterface normalizes maximize objectives at construction
    # (expr is negated, sense set to minimize), so evaluateObj() already
    # returns minimize-convention values and obj_sense is +1 by invariant.
    # It is threaded through the classification tests below purely as a
    # guard, in case that normalization is ever removed.
    obj_sense = int(problem.objective.sense)


    if config.multiple_black_boxes == 1:
        yr_k, info_k = problem.evaluateDx(xk, with_info=True)
        block_thetas_k = {}
        # sorted(): deterministic BB component ordering (set iteration order
        # of strings varies across runs with PYTHONHASHSEED)
        for block in sorted(set(b for b, _, _, _ in info_k)):
            indices = [i for i, (b, _, _, _) in enumerate(info_k) if b == block]
            block_thetas_k[block] = np.linalg.norm(
                yr_k[indices] - yk[indices], 1)

        thetak_values = np.array(list(block_thetas_k.values()), dtype=float)

        k_population  = [thetak_values]
        ref_point     = thetak_values * 2 + 10
        HVk           = calculate_hypervolume(k_population, ref_point)


    stepNorm = 1e10

    if config.globalization_strategy == 1:
        funnel = Funnel(
            phi_init             = thetak,
            f_best_init          = obj_sense * objk,
            phi_min              = config.phi_min,
            kappa_f              = config.kappa_f,
            beta                 = config.beta,
            switch_coeff         = config.mu_s,
            switch_exponent      = config.funnel_exponent,
            eta                  = config.eta,
            kappa_r              = config.kappa_r)

    consecutive_rejections = 0
    radius_resets = 0
    # Consecutive restoration (FRP) iterations within the current run, and the
    # theta the run started from. Capped by n_restoration_max, which TERMINATES
    # the solve -- that finiteness is what lets the restoration Delta rule
    # deviate from the unconditional contraction without losing convergence.
    restoration_iters = 0
    restoration_theta0 = None
    # TOTAL FRP solves in the run, and the number of distinct restoration PHASES
    # entered. Both are distinct from the 'restorations' metric, which counts
    # only restoration iterations that were REJECTED (the iterlog.restoration
    # flag is set solely in the reject branches) -- an FRP step that gets
    # accepted is tallied as an f-/theta-step, so 'restorations' systematically
    # undercounts how often restoration actually ran.
    frp_solves = 0
    restoration_phases = 0
    subopt_flag = False   # defined before the first termination test


    # Per-iteration wall time (diagnostic, logged via logger.newIter): measured
    # once at the top of each pass as the duration of the previous pass. This
    # replaces the end_time/IT bookkeeping formerly duplicated at every exit
    # path of the loop body. exit_msg records the termination headline for the
    # result object returned to the caller.
    exit_msg = "UNKNOWN"
    _prev_iter_start = start_time

    # Radius carried out of the most recent ACCEPTED step (assigned at every
    # acceptance site AFTER that site's Delta update, so it is the radius the
    # NEXT trust-region subproblem was solved at -- including the one that then
    # got rejected). Restoration resets Delta from this, rather than from the
    # twice-contracted value the rejections left behind. Also refreshed by the
    # wedge escape, so an escape is not silently undone by a later restoration.
    accepted_radius = config.trust_radius

    while True:
        if iteration >= 0:
            logger.printIteration(iteration)
        _now = time.time()
        IT = _now - _prev_iter_start
        _prev_iter_start = _now

        iteration = iteration + 1
        if iteration > config.max_it:
            exit_msg = "Maximum iterations"
            print("EXIT: Maximum iterations\n")
            break

        if iteration == 1:
            config.sample_region = False

        # Keep Sample Region within Trust Region
        if config.trust_radius < config.sample_radius:
            config.sample_radius = max(
                config.sample_radius_adjust * config.trust_radius,
                config.delta_min)
            rebuildROM = True

        # Generate a ROM r_k(x) that is kappa-fully linear on the sampling
        # region sigma_k
        if rebuildROM:
            romParam, yr = problem.buildROM(x, config.sample_radius,
                                            config.scaling)
            g, J, varlist, conlist = \
                problem.grad_hess_calc(x, y, z, romParam, config.scaling)

        # Criticality Check
        #
        # chi is always measured on a UNIT box (thesis (3.21), ||zeta|| <= 1),
        # for every strategy and scaling mode. A Delta-sized box makes chi
        # proportional to Delta, which is self-referential: the loop condition
        # Delta > xi*chi degenerates to a constant test and collapses Delta at
        # non-stationary points, and the chi termination test passes trivially
        # as Delta -> 0 (seen on strategy 1 with scaling=0: a spurious
        # "optimal" exit with chi == Delta == 3.9e-6 on the DFM-PSA case).
        criticality_radius = 1

        if iteration > 0:
            flag, chik = problem.criticalityCheck(
                x, y, z, romParam, g, J, varlist, conlist,
                criticality_radius=criticality_radius)
            if not flag:
                raise Exception("Criticality Check fails!\n")

        # Feasibility status: an iterate is FEASIBLE only when thetak < ep_i
        # (strict; never upgraded). Used by every termination test below.
        feas_ok = thetak < config.ep_i

        # Save the iteration information to the logger
        logger.newIter(iteration, xk, yk, zk, thetak, objk, chik, IT,
                       config.print_variables)

        # Check for Termination (accuracy radius: the sampling radius sigma).
        # Every exit reports thetak so feasibility claims are auditable.
        if (feas_ok and
                chik < config.ep_chi and
                config.sample_radius < config.ep_delta):
            exit_msg = "OPTIMAL SOLUTION FOUND"
            print(f"EXIT: OPTIMAL SOLUTION FOUND "
                  f"(thetak={thetak:.3e}, chik={chik:.3e})")
            break

        if (feas_ok and stepNorm < config.ep_s):
            exit_msg = "POSSIBLY AN OPTIMAL SOLUTION IS FOUND"
            print(f"EXIT: POSSIBLY AN OPTIMAL SOLUTION IS FOUND "
                  f"(thetak={thetak:.3e}, chik={chik:.3e})")
            break

        if config.trust_radius <= config.delta_min and feas_ok:
            if subopt_flag:
                exit_msg = "FEASIBLE SOLUTION FOUND"
                print(f"EXIT: FEASIBLE SOLUTION FOUND "
                      f"(thetak={thetak:.3e})")
                break
            else:
                subopt_flag = True
        else:
            subopt_flag = False

        # New criticality phase (sigma-based)
        if not config.sample_region:
            if config.sample_radius > chik * config.criticality_check:
                config.sample_radius = config.sample_radius / 2
        else:
            config.sample_radius = max(
                min(config.sample_radius, chik * config.criticality_check),
                config.delta_min)

        logger.setCurIter(trustRadius=config.trust_radius,
                          sampleRadius=config.sample_radius)

        # ----------------------------------------------------------
        # TRSP solve
        # ----------------------------------------------------------
        # Restoration (the FRP, (7.11)) is entered after two consecutive
        # rejections. 'restoration enabled' = 0 switches it off, so the
        # plain TRSP keeps being solved (Table 7.2, no-restoration column).
        restoration_mode = (consecutive_rejections >= 2
                            and int(config.restoration_enabled) == 1)

        # Restoration STATIONARITY guard -- and the restoration ENTRY.
        #
        # The cap is load-bearing for global convergence, not a convenience,
        # which is why it applies under R2 only rather than
        # being always-on: the convergence argument needs "infinitely many
        # rejections => Delta -> 0", which ONLY the relaxed restoration Delta
        # rule breaks. When that rule is active, hitting the cap TERMINATES
        # the run (the break), so an infinite tail of rejections with no
        # intervening acceptance is impossible: after the last acceptance
        # consecutive_rejections climbs to 2, restoration engages, and the cap
        # fires within n_restoration_max iterations. Hence infinitely many
        # rejections implies infinitely many acceptances, and the f-step /
        # theta-step legs of the proof carry the result. With the default
        # (unconditional contraction) the anchor holds on its own and no cap
        # is needed. Tolerances are NOT relaxed either way.
        if not restoration_mode:
            # Any accepted step (or a wedge reset) zeroes
            # consecutive_rejections, which drops us out of restoration_mode
            # here -- so this single reset covers every exit from a
            # restoration run, at every acceptance site.
            restoration_iters = 0
            restoration_theta0 = None
        else:
            if restoration_theta0 is None:
                restoration_theta0 = thetak
            if (config.restoration_strategy == 2
                    and restoration_iters >= config.n_restoration_max):
                problem.setVarValue(x=x, y=y, z=z)
                # The cap fires on a STALLED restoration, which is not the
                # same thing as an infeasible one: the iterate can already
                # satisfy thetak < ep_i and simply be unable to improve
                # further. Reporting that as "NOT FEASIBLE" mislabels a
                # feasible answer as a failure (measured: multimodal s1
                # exited NOT FEASIBLE at theta=7.97e-09 against ep_i=1e-5).
                # Check feas_ok first, exactly as the wedge-stall exit below
                # and the three termination tests above already do.
                if feas_ok:
                    exit_msg = "FEASIBLE SOLUTION FOUND (restoration stationary)"
                    print(f"EXIT: FEASIBLE SOLUTION FOUND "
                          f"(restoration stationary after "
                          f"{restoration_iters} consecutive restoration "
                          f"iterations, thetak={thetak:.3e} < "
                          f"ep_i={config.ep_i:.1e}; theta over the "
                          f"restoration run: {restoration_theta0:.3e} -> "
                          f"{thetak:.3e})")
                    break
                exit_msg = "RESTORATION STATIONARY -- NOT FEASIBLE"
                print(f"EXIT: RESTORATION STATIONARY -- NOT FEASIBLE "
                      f"({restoration_iters} consecutive restoration "
                      f"iterations, thetak={thetak:.3e} >= "
                      f"ep_i={config.ep_i:.1e}; theta over the restoration "
                      f"run: {restoration_theta0:.3e} -> {thetak:.3e})")
                break

            if restoration_iters == 0:
                # A new restoration PHASE begins. The counter is a pure
                # metric and is always maintained; the Delta reset below is
                # opt-in.
                restoration_phases += 1

                if config.restoration_strategy == 2:
                    # ENTERING restoration, opt-in path. This happens BEFORE
                    # the FRP solve below, or the first (and most expensive)
                    # FRP solve would run at the strangled radius and the
                    # intent would take effect one iteration late.
                    #
                    # Delta has been contracted by the two rejections that
                    # triggered restoration, so restore it to accepted_radius
                    # -- the radius carried out of the last accepted step --
                    # and elongate once. accepted_radius is the radius the
                    # subsequently-rejected TRSP was actually solved at, so
                    # re-using it unchanged would just re-fail; gamma_e gives
                    # the FRP room the plain TRSP did not have.
                    #
                    # CAVEAT: under scaling=0 (the default) radius_max == 1
                    # and accepted_radius returns to that ceiling after any
                    # f-step, so this clamps to the FULL variable-range box on
                    # essentially every entry -- the gamma_e factor is inert.
                    _delta_pre_restoration = config.trust_radius
                    config.trust_radius = min(
                        config.gamma_e * accepted_radius,
                        config.radius_max)

                    # Filter only: the current iterate is being abandoned, so
                    # forbid returning to its neighbourhood. The filter only
                    # ever grows, so this is monotone and costs nothing
                    # theoretically.
                    #
                    # DELIBERATELY R2-ONLY. It sits inside the entry-reset
                    # block because it is the counterpart of that reset: R2
                    # re-opens Delta, and the filter insertion is what stops
                    # the re-opened box from simply walking back to the point
                    # just abandoned. R1 leaves the filter untouched on
                    # restoration entry and must stay that way -- it is the
                    # reference policy the whole battery was solved with.
                    #
                    # Funnel (1): deliberately NOTHING.
                    # The funnel width phi is only updated on ACCEPTED steps,
                    # by a convex shrink, and the theta -> 0 proof rides on
                    # phi being non-increasing -- widening it would break it.
                    if config.globalization_strategy == 0:
                        fe = FilterElement(
                            obj_sense * objk - config.gamma_f * thetak,
                            (1 - config.gamma_theta) * thetak)
                        filteR.addToFilter(fe)

                    print(f"[FRP] entering restoration phase "
                          f"{restoration_phases}: Delta "
                          f"{_delta_pre_restoration:.3e} -> "
                          f"{config.trust_radius:.3e} "
                          f"(gamma_e * accepted_radius="
                          f"{accepted_radius:.3e}"
                          f"{', CLAMPED at radius_max' if config.trust_radius >= config.radius_max else ''}"
                          f"), thetak={thetak:.3e}")

        # Solve TRSP_k or restoration
        if not restoration_mode:
            flag, obj, infeas_measure = problem.TRSPk(
                x, y, z, xk, yk, zk,
                romParam, config.trust_radius, config.scaling)
        else:
            # Soft, problem-independent restoration: minimise the surrogate
            # mismatch ||r_k(w) - y||^2 with the glass-box h=0, g<=0 active
            # and y free of the TR box, within ||u-u_k|| <= Delta. y is
            # coupled to x through the glass-box linking constraints, so
            # this genuinely drives the iterate toward BB-consistency.
            flag, obj, infeas_measure = problem.compatibilityCheck(
                x, y, z, xk, yk, zk,
                romParam, config.trust_radius, config.scaling)
            obj = problem.evaluateObj(x, y, z)
            restoration_iters += 1
            frp_solves += 1
            print(f"[FRP] solve {restoration_iters} of phase "
                  f"{restoration_phases}, thetak={thetak:.3e}, "
                  f"Delta={config.trust_radius:.3e}"
                  f"{' (at radius_max)' if config.trust_radius >= config.radius_max else ''}")

        # ----------------------------------------------------------
        # Handle TRSP/FRP solver failure (IPOPT infeasible/diverged).
        # Both degrade gracefully to a rejected step: contract and
        # retry. At the radius floor, the wedge logic decides between
        # an honest exit and an ESCALATING radius reset.
        # ----------------------------------------------------------
        if not flag:
            which = "restoration" if restoration_mode else "TRSP"
            print(f"WARNING: {which} solve failed; "
                  f"treating as a rejected step")
            x, y, z = cloneXYZ(xk, yk, zk)
            if config.trust_radius <= config.delta_min:
                if feas_ok:
                    # Push the accepted iterate back into the Pyomo
                    # model: after a failed solve the model vars may
                    # hold IPOPT's failed intermediate point, and the
                    # final value copy-back reads the model, not xk.
                    problem.setVarValue(x=x, y=y, z=z)
                    exit_msg = (f"FEASIBLE SOLUTION FOUND "
                                f"({which} stalled at minimum radius)")
                    print(f"EXIT: FEASIBLE SOLUTION FOUND "
                          f"({which} stalled at minimum radius, "
                          f"thetak={thetak:.3e})")
                    break
                if radius_resets < config.n_radius_resets:
                    # Infeasible WEDGE: theta is large but Delta has
                    # collapsed, so neither the TRSP nor the FRP can
                    # move y back onto the truth within the box (the
                    # linking constraints need O(theta) moves in u).
                    # Escape with an ESCALATING reset (1e-2, 1e-1, 1.0,
                    # ... capped at radius_max): closing a large theta
                    # gap needs a radius commensurate with it.
                    # Tolerances are NOT relaxed.
                    radius_resets += 1
                    config.trust_radius = min(
                        config.radius_max,
                        max(1e-2, 100.0 * config.delta_min)
                        * (10.0 ** (radius_resets - 1)))
                    consecutive_rejections = 0
                    rebuildROM = True
                    # The escape radius becomes the new reference for a
                    # later restoration entry; leaving accepted_radius at
                    # its stale pre-wedge value would silently undo the
                    # escape the moment restoration fires again.
                    accepted_radius = config.trust_radius
                    print(f"WARNING: infeasible wedge at minimum radius "
                          f"(thetak={thetak:.3e}); radius reset "
                          f"{radius_resets}/{config.n_radius_resets} "
                          f"to {config.trust_radius:.1e}")
                    continue
                problem.setVarValue(x=x, y=y, z=z)
                exit_msg = f"{which.upper()} STALLED -- NOT FEASIBLE"
                print(f"EXIT: {which.upper()} STALLED -- NOT FEASIBLE "
                      f"(thetak={thetak:.3e} >= ep_i={config.ep_i:.1e}, "
                      f"radius resets exhausted)")
                break
            config.trust_radius = max(
                config.gamma_c * config.trust_radius,
                config.delta_min)
            rebuildROM = True
            logger.iterlog.rejected = True
            consecutive_rejections += 1
            continue

        # ----------------------------------------------------------
        # Evaluate true BB at trial point
        # ----------------------------------------------------------
        yr, info = problem.evaluateDx(x, with_info=True)

        stepNorm = np.linalg.norm(packXYZ(x - xk, y - yk, z - zk),
                                  np.inf)
        logger.setCurIter(stepNorm=stepNorm)

        theta = np.linalg.norm(yr - y, 1)

        # Per-BB infeasibility vector (multiple black-boxes); sorted()
        # keeps component ordering deterministic and consistent with init
        if config.multiple_black_boxes == 1:
            block_thetas = {}
            for block in sorted(set(b for b, _, _, _ in info)):
                indices = [i for i, (b, _, _, _) in enumerate(info)
                           if b == block]
                block_thetas[block] = np.linalg.norm(
                    yr[indices] - y[indices], 1)

            theta_values = np.array(list(block_thetas.values()),
                                    dtype=float)

        # Hypervolume-driven sampling-region (sigma) management
        if config.multiple_black_boxes == 1:

            reduction_ratio_HV, temp_pop_act, HV = hv_reduction_ratio(
                k_population, ref_point, theta_values, HVk)

            # Eq 14: hypervolume-driven sampling-region update (module-level)
            k_population, HVk = update_sampling_region(
                config, reduction_ratio_HV, theta_values, HV, HVk,
                k_population)

        # Calculate rho for theta-step trust-region update
        rhok = 1 - ((theta - config.ep_i) /
                    max(thetak, config.ep_i))

        if restoration_mode and config.restoration_strategy == 2:
            # rhok is measured against the CENTRE's thetak, and a rejected
            # step does not move the centre -- so within one restoration run
            # thetak is constant and rhok compares each FRP trial against the
            # same reference. Logged so the study can see whether theta+ is
            # actually walking down over the run or just re-scoring.
            print(f"[FRP] solve {restoration_iters} outcome: "
                  f"theta+={theta:.3e} (thetak={thetak:.3e}), "
                  f"rhok={rhok:.3e} -> "
                  f"{'contract' if rhok < config.eta1 else ('expand' if rhok >= config.eta2 else 'maintain')}")

        # ----------------------------------------------------------
        # Filter method (globalization_strategy == 0)
        # ----------------------------------------------------------
        if config.globalization_strategy == 0:

            fe = FilterElement(obj_sense * obj, theta)

            if (not filteR.checkAcceptable(fe, config.theta_max)
                    and iteration > 0):
                radius_on_reject(config, restoration_mode, rhok)
                rebuildROM = True
                x, y, z = cloneXYZ(xk, yk, zk)

                if restoration_mode:
                    logger.iterlog.restoration = True
                else:
                    logger.iterlog.rejected = True
                consecutive_rejections += 1

                continue

            # Switching Condition and Trust Region update
            if ((obj_sense * (objk - obj) >= config.kappa_theta *
                 pow(thetak, config.gamma_s))
                    and (thetak < config.theta_min)):
                logger.iterlog.fStep = True
                consecutive_rejections = 0
                config.trust_radius = min(
                    config.gamma_e * config.trust_radius,
                    config.radius_max)
                accepted_radius = config.trust_radius

            else:
                logger.iterlog.thetaStep = True
                consecutive_rejections = 0
                fe = FilterElement(
                    obj_sense * obj - config.gamma_f * theta,
                    (1 - config.gamma_theta) * theta)
                filteR.addToFilter(fe)

                if rhok < config.eta1:
                    config.trust_radius = max(
                        config.gamma_c * config.trust_radius,
                        config.delta_min)
                elif rhok >= config.eta2:
                    config.trust_radius = min(
                        config.gamma_e * config.trust_radius,
                        config.radius_max)
                accepted_radius = config.trust_radius

        # ----------------------------------------------------------
        # Funnel method (globalization_strategy == 1)
        # ----------------------------------------------------------
        elif config.globalization_strategy == 1:
            status = funnel.classify_step(
                thetak, theta, obj_sense * objk, obj_sense * obj,
                config.trust_radius)

            if status == 'f':
                funnel.accept_f(theta, obj_sense * obj)
                logger.iterlog.fStep = True
                consecutive_rejections = 0
                config.trust_radius = min(
                    config.gamma_e * config.trust_radius,
                    config.radius_max)
                accepted_radius = config.trust_radius

            elif status in ('theta', 'theta-relax'):
                if status == 'theta':
                    funnel.accept_theta(theta)
                    logger.iterlog.thetaStep = True
                else:   # legacy outside-funnel relaxed theta step
                    funnel.relax_theta(theta)
                    logger.iterlog.relaxthetaStep = True
                consecutive_rejections = 0
                if rhok < config.eta1:
                    config.trust_radius = max(
                        config.gamma_c * config.trust_radius,
                        config.delta_min)
                elif rhok >= config.eta2:
                    config.trust_radius = min(
                        config.gamma_e * config.trust_radius,
                        config.radius_max)
                accepted_radius = config.trust_radius

            else:   # 'reject'
                radius_on_reject(config, restoration_mode, rhok)
                rebuildROM = True
                x, y, z = cloneXYZ(xk, yk, zk)

                if restoration_mode:
                    logger.iterlog.restoration = True
                else:
                    logger.iterlog.rejected = True
                consecutive_rejections += 1

                continue


        rebuildROM = True
        xk, yk, zk = cloneXYZ(x, y, z)
        # Reuse the true BB outputs already evaluated for the trial point at
        # x (== xk after the clone above); avoids one full BB sweep per step.
        yk = yr.copy()
        if config.multiple_black_boxes == 1:
            thetak_values = theta_values.copy()
        thetak = theta
        objk = obj

    logger.printVectors()
#    problem.reverseTransform()

    # ---------------------------------------------------------------------
    # Run summary: wall/CPU time, exact external (black-box) evaluations
    # per block + total, and the machine the run executed on. Printed once
    # at the end.
    # ---------------------------------------------------------------------
    wall_s = time.perf_counter() - _t_wall0
    cpu_s = time.process_time() - _t_cpu0
    print("\n========== RUN SUMMARY ==========")
    print(f"Iterations            : {iteration}")
    # Problem size under trust-region management. x = black-box INPUT
    # variables, y = black-box OUTPUT (surrogate / tear) variables, z =
    # remaining glass-box decision variables; total = x + y + z.
    _lx, _ly, _lz = problem.lx, problem.ly, problem.lz
    _neq, _nineq = problem.count_constraints()
    print(f"Problem size          : {_lx + _ly + _lz} variables  "
          f"(x|y|z = {_lx}|{_ly}|{_lz})")
    print("    x = black-box inputs | y = black-box outputs | z = remaining")
    _y_per_block = ", ".join(f"{blk}={len(problem.TRF.y[blk])}"
                             for blk in sorted(problem.TRF.y))
    print(f"Black-box blocks      : {len(problem.TRF.y)}  "
          f"(outputs per block: {_y_per_block})")
    print(f"Constraints           : {_neq + _nineq}  "
          f"(equality {_neq} | inequality {_nineq})")
    print(f"Wall time (s)         : {wall_s:.3f}")
    print(f"CPU time (s)          : {cpu_s:.3f}   (process_time)")
    print("External BB evaluations (true black-box callback calls):")
    for blk in sorted(problem.bb_eval_counts):
        print(f"    {blk:<12}: {problem.bb_eval_counts[blk]}")
    print(f"    {'TOTAL':<12}: {problem.bb_eval_total}")
    print(f"    (evaluateDx sweeps: {problem.countDx})")
    # Cache statistics (solver-level memoization of BB calls)
    if getattr(problem, 'bb_cache_enabled', False):
        hits = problem.bb_cache_hits
        misses = problem.bb_eval_total
        requests = hits + misses
        rate = (100.0 * hits / requests) if requests else 0.0
        print(f"BB cache              : {hits} hits / {requests} requests "
              f"({rate:.1f}% hit rate); {misses} true evaluations")
    else:
        print("BB cache              : disabled")

    # Optional: constraint dual values (multipliers) from the final successful
    # subproblem solve. Reports the count plus the largest-magnitude duals.
    # Captured only when config 'report duals' = 1; guarded so a reporting
    # hiccup can never abort an otherwise-successful solve.
    duals_dict = dict(getattr(problem, 'last_duals', {}) or {})
    if getattr(problem, 'report_duals', False):
        print(f"Dual values           : {len(duals_dict)} captured "
              f"(last subproblem solve); largest |dual|:")
        for name, val in sorted(duals_dict.items(), key=lambda t: -abs(t[1]))[:15]:
            print(f"    {name:<32}: {val:.6g}")

    print("System:")
    print(system_report())
    print("=================================\n")

    # Structured result for the caller (solve() augments it with the
    # user-sense objective read off the original model). feasible is the
    # STRICT test thetak < ep_i -- never silently upgraded from a noise-level
    # exit, consistent with the strict feasibility reporting policy.
    return {
        'status':          exit_msg,
        'iterations':      iteration,
        'feasible':        bool(thetak < config.ep_i),
        'final_theta':     float(thetak),
        'final_chi':       float(chik),
        'bb_evals':        int(problem.bb_eval_total),
        'bb_evals_detail': {blk: int(problem.bb_eval_counts[blk])
                            for blk in sorted(problem.bb_eval_counts)},
        'bb_cache_hits':   int(getattr(problem, 'bb_cache_hits', 0)),
        'bb_cache_enabled': bool(getattr(problem, 'bb_cache_enabled', False)),
        # Step-type tallies, summed from the per-iteration logger flags (the
        # same booleans that drive the "f-type step"/"step rejected"/... prints)
        # so callers get them without re-parsing stdout. theta_steps counts the
        # plain theta-type acceptances (relax-theta steps are a strategy-1 legacy
        # sub-case and were never tallied separately).
        'f_steps':         sum(1 for it in logger.iters if it.fStep),
        'theta_steps':     sum(1 for it in logger.iters if it.thetaStep),
        'rejected':        sum(1 for it in logger.iters if it.rejected),
        'restorations':    sum(1 for it in logger.iters if it.restoration),
        # Actual FRP solves (see frp_solves above); 'restorations' counts only
        # the REJECTED ones, so the two differ whenever an FRP step is accepted.
        'frp_solves':      int(frp_solves),
        # Distinct restoration PHASES entered (each is 1..n_restoration_max
        # solves), i.e. how many times the run had to escape a rejection stall.
        'restoration_phases': int(restoration_phases),
        # Which named restoration policy produced the above (1 = R1, 2 = R2),
        # so a results row is self-describing without the caller tracking it.
        'restoration_strategy': int(config.restoration_strategy),
        'wall_s':          float(wall_s),
        'cpu_s':           float(cpu_s),
        'n_vars':          int(_lx + _ly + _lz),
        'n_vars_xyz':      (int(_lx), int(_ly), int(_lz)),
        'n_eq_constraints':   int(_neq),
        'n_ineq_constraints': int(_nineq),
        'n_constraints':      int(_neq + _nineq),
        'duals':              duals_dict,
        # Per-iteration trajectory from the logger (for convergence plots /
        # paper figures). obj is reported in the user's sense (obj_sense undoes
        # the internal minimize negation), consistent with final_obj.
        'trajectory': {
            'iter':  [int(it.iteration) for it in logger.iters],
            'theta': [float(it.thetak) for it in logger.iters],
            'obj':   [float(obj_sense * it.objk) for it in logger.iters],
            'chi':   [float(it.chik) for it in logger.iters],
            'trust_radius': [float('nan') if it.trustRadius is None
                             else float(it.trustRadius) for it in logger.iters],

            'sample_radius': [float('nan') if it.sampleRadius is None
                              else float(it.sampleRadius)
                              for it in logger.iters],
            'step_norm': [float('nan') if it.stepNorm is None
                          else float(it.stepNorm) for it in logger.iters],
        },
    }

# =============================================================================
# TrustRegionSolver -- the user-facing solver wrapper + full configuration.
# Consolidated here (was duplicated across RunFile.py and the HDA_main_*.py
# case studies, which had drifted out of sync). Case-study / run scripts now
# do `from TRF import TrustRegionSolver` and contain only the model + run.
# =============================================================================

class TrustRegionSolver:
    CONFIG = ConfigBlock('Trust Region')

    CONFIG.declare('solver', ConfigValue(
        default='ipopt',
        description='subproblem solver (e.g. ipopt); user solver_options are merged on top of the curated IPOPT settings in PyomoInterface.solveModel',
    ))

    CONFIG.declare('solver_options', ConfigBlock(
        implicit=True,
        description='Options to pass to the subproblem solver',
    ))

    CONFIG.declare('max it', ConfigValue(
        default=100,
        domain=PositiveInt,
        description='Maximum number of trust-region iterations.',
    ))

    # Initialize trust radius
    CONFIG.declare('trust radius', ConfigValue(
        default=1.0,
        domain=PositiveFloat,
        description='Initial trust-region radius Delta_0 (on the scaled '
                    'unit box when scaling=0).',
    ))

    # Initialize sample region and radius
    CONFIG.declare('sample region', ConfigValue(
        default=True,
        domain=bool,
        description='Tie the ROM sampling region to the criticality measure '
                    'chi during the criticality phase.',
    ))

    CONFIG.declare('sample radius', ConfigValue(
        default=0.1,
        domain=PositiveFloat,
        description='Initial ROM sampling radius sigma_0.',
    ))

    # Placeholder for 'radius max', value to be set in __init__
    CONFIG.declare('radius max', ConfigValue(
        default=None,
        domain=PositiveFloat,
        description='Upper bound on the trust-region radius; '
                    'defaults to 1000 * trust radius.',
    ))

    # Termination tolerances
    CONFIG.declare('ep i', ConfigValue(
        default=1e-5,
        domain=PositiveFloat,
        description='Feasibility tolerance: an iterate is reported FEASIBLE '
                    'only when thetak < ep i (strict; never upgraded).',
    ))

    CONFIG.declare('ep s', ConfigValue(
        default=1e-4,
        domain=PositiveFloat,
        description='Step-norm tolerance for the "possibly optimal" '
                    'termination test (||s_k|| < ep s).',
    ))

    CONFIG.declare('ep delta', ConfigValue(
        default=1e-3,
        domain=PositiveFloat,
        description='Trust-region radius tolerance for the optimal-'
                    'termination test.',
    ))

    CONFIG.declare('ep chi', ConfigValue(
        default=1e-3,
        domain=PositiveFloat,
        description='Criticality (stationarity) tolerance for the optimal-'
                    'termination test (chik < ep chi).',
    ))

    CONFIG.declare('delta min', ConfigValue(
        default=1e-6,
        domain=PositiveFloat,
        description='delta min <= ep delta',
    ))

    # Criticality Check Parameters
    CONFIG.declare('criticality check', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
    ))

    # Trust region update parameters
    CONFIG.declare('gamma c', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
        description='Trust-region CONTRACTION factor (<1), applied on '
                    'rejected or poor (rho<eta1) steps.',
    ))

    CONFIG.declare('gamma e', ConfigValue(
        default=2.5,
        domain=PositiveFloat,
        description='Trust-region EXPANSION factor (>1), applied on '
                    'successful f-type (or good rho>=eta2) steps.',
    ))

    # Switching Condition
    CONFIG.declare('gamma s', ConfigValue(
        default=2.0,
        domain=PositiveFloat,
        description='Filter switching exponent (strategy 0): an f-type step '
                    'needs f_k - f+ >= kappa_theta * theta_k^gamma_s.',
    ))

    CONFIG.declare('kappa theta', ConfigValue(
        default=0.1,
        domain=PositiveFloat,
        description='Filter switching coefficient (strategy 0).',
    ))

    CONFIG.declare('kappa f', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
        description='funnel‑shrink factor after f‑type',
    ))

    CONFIG.declare('kappa r', ConfigValue(
        default=1.1,
        domain=PositiveFloat,
        description='funnel width factor of a theta-type step taken outside '
                    'the funnel (theta+ <= kappa_r * phi)',
    ))

    CONFIG.declare('theta min', ConfigValue(
        default=1e-4,
        domain=PositiveFloat,
    ))

    CONFIG.declare('phi min', ConfigValue(
        default=1e-8,
        domain=PositiveFloat,
        description='hard floor on funnel width',
    ))

    # Filter
    CONFIG.declare('gamma f', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='gamma_f and gamma_theta in (0,1) are fixed parameters',
    ))

    CONFIG.declare('gamma theta', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='gamma_f and gamma_theta in (0,1) are fixed parameters',
    ))

    CONFIG.declare('theta max', ConfigValue(
        default=50,
        domain=PositiveInt,
    ))


    CONFIG.declare('beta', ConfigValue(
        default=0.8,
        domain=PositiveFloat,
        description='extra shrink required for theta‑type',
    ))

    CONFIG.declare('mu s', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='funnel switching coefficient: an f-type step needs '
                    'f_k - f+ >= mu_s * theta_k^(funnel exponent)',
    ))

    CONFIG.declare('eta', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='funnel Armijo coefficient: an f-type step needs '
                    'f_k - f+ >= eta * Delta_k',
    ))

    CONFIG.declare('funnel exponent', ConfigValue(
        default=2.0,
        domain=PositiveFloat,
        description='funnel switching exponent',
    ))

    # Ratio test parameters (for theta steps)
    CONFIG.declare('eta1', ConfigValue(
        default=0.05,
        domain=PositiveFloat,
    ))

    CONFIG.declare('eta2', ConfigValue(
        default=0.2,
        domain=PositiveFloat,
    ))

    # Hypervolume-based sampling-region update (Eq 14)
    CONFIG.declare('sigma eta3', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
        description='HV reduction-ratio expand threshold eta3 (Eq 14); in (0,1)',
    ))

    CONFIG.declare('sigma inc', ConfigValue(
        default=1.5,
        domain=PositiveFloat,
        description='sampling-region expansion factor gamma_sigma_inc (>1)',
    ))

    CONFIG.declare('sigma dec', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
        description='sampling-region contraction factor gamma_sigma_dec (in (0,1))',
    ))

    CONFIG.declare('n radius resets', ConfigValue(
        default=3,
        domain=NonNegativeFloat,
        description='max radius resets allowed when restoration wedges at the '
                    'minimum radius while infeasible (escape hatch; tolerances '
                    'are never relaxed)',
    ))

    # --- Restoration phase and its Delta policy ---
    CONFIG.declare('restoration enabled', ConfigValue(
        default=1,
        domain=In([0, 1]),
        description='1 = enter the feasibility restoration phase (7.11) after '
                    '>=2 consecutive step rejections (the default). 0 = never '
                    'enter restoration and keep solving the plain TRSP '
                    '(Table 7.2, no-restoration column).',
    ))

    #
    # Two strategies, selected by 'restoration strategy':
    #
    #   R1 (default) -- classic. The FRP (compatibilityCheck) is solved at the
    #       current Delta, and a rejected restoration iteration contracts Delta
    #       unconditionally by gamma_c like any other rejection.
    #   R2 -- relaxed. On ENTERING a phase Delta is reset to
    #       min(gamma_e * accepted_radius, radius_max) before the first FRP
    #       solve (and, under the filter, the abandoned iterate is pushed into
    #       the filter); a rejected restoration iteration then updates Delta by
    #       the theta-reduction ratio rhok, with the n_restoration_max cap
    #       active to keep the phase finite.
    #
    # Both solve the SAME subproblem -- compatibilityCheck, the TR-boxed FRP.
    # They differ only in the Delta policy. R1 is the default (Table 7.3).
    CONFIG.declare('restoration strategy', ConfigValue(
        default=1,
        domain=In([1, 2]),
        description='Feasibility-restoration policy. '
                    '1 = R1 classic (FRP at the current Delta; a rejected '
                    'restoration iteration contracts unconditionally), '
                    '2 = R2 relaxed (entry reset to gamma_e*accepted_radius '
                    'plus filter insertion; a rejected restoration iteration '
                    'updates Delta by rhok; n_restoration_max cap active).',
    ))

    # Opt-in regularisation of the feasibility restoration problem (7.11).
    # (7.11) minimises only the surrogate residual, so any variable that does
    # not affect it (e.g. an over-bounded cost-side duty) is left free and the
    # NLP solver may place it anywhere within the trust region. A positive
    # weight adds  w * sum(((v - v_k)/(ub - lb))^2)  over the x and z variables
    # with finite bounds, which pins such variables at the centre and barely
    # touches the residual-reducing ones. Default 0 = the thesis's (7.11)
    # exactly (Chapter 7 results). Used only for the DFM-PSA case study.
    CONFIG.declare('frp proximity', ConfigValue(
        default=0.0,
        domain=float,
        description='Weight of the range-normalised proximity term added to '
                    'the restoration objective (0 = off, the default)',
    ))

    CONFIG.declare('n restoration max', ConfigValue(
        default=10,
        domain=NonNegativeFloat,
        description='max CONSECUTIVE feasibility-restoration (FRP) iterations '
                    'before the run exits as RESTORATION STATIONARY. Only takes '
                    'effect under R2 (restoration strategy 2), because it '
                    'is LOAD-BEARING for that mode rather than a diagnostic cap: '
                    'the relaxed rule breaks the "infinitely many rejections => '
                    'Delta -> 0" leg of the proof, which survives only because '
                    'hitting this cap TERMINATES the run, making an infinite tail '
                    'of rejections without an intervening acceptance impossible. '
                    'Raise it if theta is still improving when the cap is hit',
    ))

    # Output level (replace with real print levels)
    CONFIG.declare('print variables', ConfigValue(
        default=False,
        domain=bool,
    ))

    # Built-in black-box evaluation cache (memoization of true BB callback
    # calls, exact-key -> transparent to results; reports hits/misses).
    CONFIG.declare('bb cache', ConfigValue(
        default=1,
        domain=In([0, 1]),
        description='0 = off, 1 = memoise true black-box evaluations in the '
                    'solver (exact input key; never changes results, only the '
                    'evaluation count). Reported as cache hits/misses in the '
                    'run summary. Problems with their own block-level cache '
                    '(e.g. HDA) keep it -- the two layers are complementary.',
    ))

    # Report constraint dual values (multipliers) from the final successful
    # subproblem solve in the run summary. Off by default (attaches an IMPORT
    # dual Suffix). The (equality | inequality) constraint COUNTS are always
    # reported in the banner / run summary regardless of this flag.
    CONFIG.declare('report duals', ConfigValue(
        default=0,
        domain=In([0, 1]),
        description='0 = off, 1 = capture and report constraint dual values '
                    '(Lagrange multipliers) from the final subproblem solve in '
                    'the run summary (and the solve() result dict under "duals").',
    ))


    # Sample Radius reset parameter
    CONFIG.declare('sample radius adjust', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
    ))

    # Default globalization strategy
    CONFIG.declare('globalization strategy', ConfigValue(
        default=0,
        domain=In([0, 1]),
        description='0 = Filter, 1 = Funnel',
    ))

    # Default problem scaling
    CONFIG.declare('scaling', ConfigValue(
        default=0,
        domain=In([0, 1]),
        description='0 = Yes, 1 = No',
    ))

    # Default number of black-boxes setting
    CONFIG.declare('multiple black boxes', ConfigValue(
        default=0,
        domain=In([0, 1]),
        description='0 = one aggregate infeasibility, 1 = per-black-box '
                    'infeasibility vector with hypervolume sampling-region '
                    'management (Eq 14)',
    ))

    def __init__(self, **kwargs):
        self.config = self.CONFIG(kwargs, preserve_implicit=True)

        # Set 'radius max' if not already provided
        if self.config['radius max'] is None:
            self.config['radius max'] = 1000.0 * self.config['trust radius']

    def solve(self, model, eflist, **kwds):
        # Store all data needed to change in the original model
        model._tmp_trf_data = (list(model.component_data_objects(Var)), eflist, self.config)

        # Clone the model to work on it. NOTE: clone() deep-copies the attached
        # _tmp_trf_data tuple too, so inst gets its OWN copy of self.config. The
        # main loop mutates config (trust_radius, sample_radius, ...) as run
        # state, but only on that copy -- self.config is left untouched, so
        # calling solve() again on this solver starts fresh (verified). Do not
        # "optimise" this by passing self.config directly without a copy.
        inst = model.clone()

        # Call the TRF function on the cloned model
        result = TRF(inst, inst._tmp_trf_data[1], inst._tmp_trf_data[2])

        # Copy potentially changed variable values back to the original model
        for inst_var, orig_var in zip(inst._tmp_trf_data[0], model._tmp_trf_data[0]):
            orig_var.set_value(value(inst_var))

        # Add the objective in the user's own sense, read off the original
        # model's active objective after copy-back.
        objective = next(model.component_data_objects(Objective, active=True))
        result['final_obj'] = value(objective)
        return result
