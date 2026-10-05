#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import numpy as np

from pyomo.common.collections import ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
    Block, Var, Param, VarList, ConstraintList, Constraint, Objective,
    RangeSet, value, ConcreteModel, Reals, sqrt, minimize, maximize, inequality,
    Suffix
)
# The expression API moved from pyomo.core.expr.current up into
# pyomo.core.expr (the .current submodule was deprecated in Pyomo 6.6.2).
# Prefer the new location when the symbols are present (>= ~6.7, incl. 6.10,
# and avoids the deprecation warning); fall back to .current on the 6.2/6.6
# baseline where they are not yet exposed there.
import pyomo.core.expr as EXPR
if not hasattr(EXPR, 'ExternalFunctionExpression'):
    from pyomo.core.expr import current as EXPR
from pyomo.core.base.external import PythonCallbackFunction
from pyomo.core.base.numvalue import nonpyomo_leaf_types
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from helper import maxIgnoreNone, minIgnoreNone

from pyomo.core.expr.numvalue import native_types

# Sentinel distinguishing "absent" from a cached value of None.
_CACHE_MISS = object()

class Scaling:
    Yes = 0
    No = 1

class ReplaceEFVisitor(EXPR.ExpressionReplacementVisitor):
    def __init__(self, trf_block, efDict):
        super(ReplaceEFVisitor, self).__init__(
            descend_into_named_expressions=True,
            remove_named_expressions=False)
        self.trf = trf_block
        self.efDict = efDict

    def beforeChild(self, node, child, child_idx):
        """
            Called during expression tree traversal.
            If node is an ExternalFunctionExpression listed in efDict,
            replace it with an auxiliary variable y specific to its block.
        """
        descend, result = super().beforeChild(node, child, child_idx)
        if (
            not descend
            and result.__class__ not in native_types
            and result.is_variable_type()
        ):
            self.trf.all_variables.add(result)
        return descend, result

    def exitNode(self, node, values):
        new_node = super().exitNode(node, values)
        if new_node.__class__ is not EXPR.ExternalFunctionExpression:
            return new_node
        if id(new_node._fcn) not in self.efDict:
            return new_node
        # At this point we know this is an ExternalFunctionExpression
        # node that we want to replace with an auliliary variable (y)

        # Determine which black-box this EF belongs to
        block_name = self.efDict[id(new_node._fcn)]

        # Ensure the TRF block has containers for this block
        if block_name not in self.trf.y:
            # Lazily create VarList and ConstraintList
            self.trf.y[block_name] = VarList()
            self.trf.x[block_name] = VarList()
            self.trf.conset[block_name] = ConstraintList()
            self.trf.external_fcns[block_name] = []
            self.trf.exfn_xvars[block_name] = []

        new_args = []
        seen = ComponentSet()
        # TODO: support more than PythonCallbackFunctions
        assert isinstance(new_node._fcn, PythonCallbackFunction)
        #
        # Pyomo encodes the PythonCallbackFunction's integer function-ID as an
        # extra constant leaf in the EF argument list.  Its POSITION changed
        # across Pyomo versions: it was prepended (args = [fid, x, y, ...]) up
        # to ~6.2, and is appended (args = [x, y, ..., fid], wrapped as a
        # _PythonCallbackFunctionID) from ~6.4 on.  We therefore strip it by
        # identity rather than by a fixed index, so this works on 6.2-6.10.
        for arg in self._ef_input_args(new_node):
            if type(arg) in nonpyomo_leaf_types or arg.is_fixed():
                # We currently do not allow constants or parameters for
                # the external functions.
                raise RuntimeError(
                    "TrustRegion does not support black boxes with "
                    "constant or parameter inputs\n\tExpression: %s"
                    % (new_node,) )
            if arg.is_expression_type():
                # All expressions (including simple linear expressions)
                # are replaced with a single auxiliary variable (and
                # eventually an additional constraint equating the
                # auxiliary variable to the original expression)
                _x = self.trf.x[block_name].add()
                _x.set_value(value(arg))
                self.trf.conset[block_name].add(_x == arg)
                new_args.append(_x)
            else:
                # The only thing left is bare variables: check for duplicates.
                if arg in seen:
                    raise RuntimeError(
                        "TrustRegion does not support black boxes with "
                        "duplicate input arguments\n\tExpression: %s"
                        % (new_node,) )
                seen.add(arg)
                new_args.append(arg)

        # Create tear variable for this EF (per block)
        _y = self.trf.y[block_name].add()
        # Record bookkeeping for this block
        self.trf.external_fcns[block_name].append(new_node)
        self.trf.exfn_xvars[block_name].append(new_args)
        return _y

    @staticmethod
    def _ef_input_args(ef_node):
        """Return the user input arguments of a PythonCallbackFunction
        ExternalFunctionExpression, with the encoded function-ID leaf removed.

        The fid is the single constant leaf (neither a variable nor an
        expression) whose value equals ``_fcn._fcn_id``; it was prepended in
        Pyomo <= 6.2 and appended (as ``_PythonCallbackFunctionID``) in
        Pyomo >= 6.4, so we drop it by identity rather than by index.
        """
        fid = ef_node._fcn._fcn_id
        out = []
        stripped = False
        for a in ef_node.args:
            if not stripped:
                is_var = getattr(a, 'is_variable_type', lambda: False)()
                is_expr = getattr(a, 'is_expression_type', lambda: False)()
                if not is_var and not is_expr:
                    try:
                        if int(value(a)) == fid:
                            stripped = True
                            continue
                    except (TypeError, ValueError):
                        pass
            out.append(a)
        return out

class PyomoInterface(object):
    '''
    Initialize with a pyomo model m.
    This is used in TRF.py, same requirements for m apply

    m is reformulated into form for use in TRF algorithm

    Specified ExternalFunction() objects are replaced with new variables
    All new attributes (including these variables) are stored on block
    "tR"
    '''

    stream_solver = False # True prints solver output to screen
    keepfiles = False  # True prints intermediate file names (.nl,.sol,...)
    countDx = -1

    def __init__(self, m, efmap, config):

        self.config = config
        self.model = m;
        self.TRF = self.transformForTrustRegion(self.model,efmap)

        # Build global index → (block, local_index) mapping for y variables
        self.index_map_global_to_block = {}
        count = 0
        for block, ylist in self.TRF.y.items():
            for local_idx in ylist.keys():
                self.index_map_global_to_block[count] = (block, local_idx)
                count += 1


        self.lx = len(self.TRF.xvars)
        self.lz = len(self.TRF.zvars)
        self.ly = sum(len(vlist) for vlist in self.TRF.y.values())

        # Accurate external (black-box) evaluation counters. Every true BB
        # callback invocation -- in evaluateDx AND in the finite-difference ROM
        # build -- goes through _eval_bb, so these are exact (the legacy
        # `countDx` only counted evaluateDx *sweeps*, missing the ROM-build
        # perturbation calls).
        self.bb_eval_counts = {block: 0 for block in self.TRF.y}
        self.bb_eval_total = 0

        # Built-in black-box evaluation cache (memoization). Keyed EXACTLY on
        # (block, output-function id, input tuple) -- exact keys can never
        # return a value for a different point, so caching is transparent: it
        # changes evaluation COUNT, never results. Hits are counted but do not
        # increment bb_eval_total. Enabled by config 'bb cache' (default on).
        # NOTE: this memoises per OUTPUT function; problems whose multiple
        # outputs share one expensive simulation (e.g. HDA's reactor ODE) keep
        # their own block-level cache too -- the two are complementary.
        self.bb_cache = {}
        self.bb_cache_hits = 0
        self.bb_cache_enabled = (int(getattr(self.config, 'bb_cache', 1)) == 1)

        # Optional constraint dual values (multipliers). When requested (config
        # 'report duals' = 1) attach an IMPORT dual Suffix so each subproblem
        # solve populates the constraint multipliers; the last successful
        # subproblem's duals are snapshotted in solveModel and reported in the
        # run summary. Off by default (the Suffix has a small overhead and the
        # later gjh evaluation clears model.dual anyway).
        self.report_duals = (int(getattr(self.config, 'report_duals', 0)) == 1)
        self.last_duals = {}
        if self.report_duals and not hasattr(self.model, 'dual'):
            self.model.dual = Suffix(direction=Suffix.IMPORT)

        # Subproblem solvers, built once and reused across every solve in this
        # run (see _build_subproblem_solver): the main TRSP solver, the lighter
        # restoration solver, and the gjh pseudo-solver used for gradients.
        self.opt = self._build_subproblem_solver()
        self.opt_restoration = self._build_subproblem_solver(restoration=True)
        self.optGJH = SolverFactory('local.gjh')

        self.createParam()
        self.createRomConstraint()
        self.createCompCheckObjective()
        self.cacheBound()

    def count_constraints(self):
        """(n_equality, n_inequality) over the model's own active constraints,
        EXCLUDING the internal trust-region block (ROM / comp-check / tear /
        trust-region-box constraints that the solver adds). Counts the user's
        glass-box constraints plus the black-box link constraints -- i.e. the
        problem as the user posed it."""
        neq = nineq = 0
        for con in self.model.component_data_objects(Constraint, active=True):
            if con.parent_block() is self.TRF:
                continue
            if con.equality:
                neq += 1
            else:
                nineq += 1
        return neq, nineq

    def _eval_bb(self, callable_fcn, values, block):
        """Invoke one true black-box callback, with caching + counting.

        On a cache hit the stored value is returned (counted as a hit, not a
        true evaluation). On a miss the callback is invoked, counted (per block
        + total), and stored. Exact-key cache => results are identical to
        running with the cache off.
        """
        if self.bb_cache_enabled:
            key = (block, id(callable_fcn), tuple(values))
            cached = self.bb_cache.get(key, _CACHE_MISS)
            if cached is not _CACHE_MISS:
                self.bb_cache_hits += 1
                return cached
        out = callable_fcn(*values)
        self.bb_eval_counts[block] = self.bb_eval_counts.get(block, 0) + 1
        self.bb_eval_total += 1
        if self.bb_cache_enabled:
            self.bb_cache[key] = out
        return out


    def transformForTrustRegion(self, model, efmap):
        """
        Transform the model into TRF form for multiple black-boxes.

        Args:
           model : Pyomo model containing ExternalFunctions
           efmap : dict { block_name: [ExternalFunction, ...] }
        """

        # Flatten mapping: id(ef) -> block_name
        efDict = {id(ef): block for block, funclist in efmap.items() for ef in funclist}

        TRF = Block()
        TRF.all_variables = ComponentSet()

        # Attach TRF block to model
        model.add_component(unique_component_name(model, 'tR'), TRF)

        # Create per-block containers and register them as TRF components so .add() works
        TRF.y = {}
        TRF.x = {}
        TRF.conset = {}
        TRF.external_fcns = {}
        TRF.exfn_xvars = {}

        for block in efmap:
            # unique names attached to the TRF block (prevents name collisions)
            name_y = unique_component_name(TRF, f"y_{block}")
            name_x = unique_component_name(TRF, f"x_{block}")
            name_con = unique_component_name(TRF, f"con_{block}")

            # create VarList / ConstraintList and attach to TRF so they are constructed
            vlist_y = VarList()
            vlist_x = VarList()
            clist = ConstraintList()

            TRF.add_component(name_y, vlist_y)
            TRF.add_component(name_x, vlist_x)
            TRF.add_component(name_con, clist)

            # keep dictionary references for the rest of the code
            TRF.y[block] = vlist_y
            TRF.x[block] = vlist_x
            TRF.conset[block] = clist
            TRF.external_fcns[block] = []
            TRF.exfn_xvars[block] = []


        # Also keep global lists for compatibility with solver
        TRF.xvars = []
        TRF.yvars = []
        TRF.zvars = []

        # Substitute EF expressions in constraints & objective
        for con in model.component_data_objects(Constraint, active=True):
            con.set_value((con.lower,
                       ReplaceEFVisitor(TRF, efDict).walk_expression(con.body),
                       con.upper))

        for obj in model.component_data_objects(Objective, active=True):
            obj.set_value(ReplaceEFVisitor(TRF, efDict).walk_expression(obj.expr))
            self.objective = obj
            if self.objective.sense == maximize:
                self.objective.expr = -1 * self.objective.expr
                self.objective.sense = minimize

        # Collect all original model variables
        allVariables = list(model.component_data_objects(Var))

        # Build global xvars list (inputs to EFs)
        seenVar = set()
        for block, varss in TRF.exfn_xvars.items():
            for varlist in varss:
                for var in varlist:
                    if id(var) not in seenVar:
                        seenVar.add(id(var))
                        TRF.xvars.append(var)

        # Build global yvars list (outputs of EFs)
        for block, ylist in TRF.y.items():
            for yvar in ylist.values():
                TRF.yvars.append(yvar)
                seenVar.add(id(yvar))

        # Build zvars as remaining variables (not EF inputs, not EF outputs)
        for var in allVariables:
            if id(var) not in seenVar:
                TRF.zvars.append(var)
                seenVar.add(id(var))

        # Build index map (per EF input → index in xvars)
        self.exfn_xvars_ind = []
        for block, varss in TRF.exfn_xvars.items():
            for varlist in varss:
                listtmp = []
                for var in varlist:
                    for i in range(len(TRF.xvars)):
                        if id(var) == id(TRF.xvars[i]):
                            listtmp.append(i)
                            break
                self.exfn_xvars_ind.append(listtmp)

        return TRF


    def getInitialValue(self):
        x = np.zeros(self.lx, dtype=float)
        y = np.zeros(self.ly, dtype=float)
        z = np.zeros(self.lz, dtype=float)
        for i in range(0, self.lx):
            x[i] = value(self.TRF.xvars[i])
        for i in range(0, self.ly):
            y[i] = 1
        for i in range(0, self.lz):
            self.TRF.zvars[i].value = value(self.TRF.zvars[i]) if self.TRF.zvars[i].value is not None else 0.0
            z[i] = value(self.TRF.zvars[i])
        return x, y, z

    def createParam(self):
        self.TRF.ind_lx=RangeSet(0,self.lx-1)
        self.TRF.ind_ly=RangeSet(0,self.ly-1)
        self.TRF.ind_lz=RangeSet(0,self.lz-1)
        self.TRF.px0 = Param(self.TRF.ind_lx,mutable=True,default=0)
        self.TRF.pz0 = Param(self.TRF.ind_lz,mutable=True,default=0)
        self.TRF.plrom = Param(self.TRF.ind_ly,range(self.lx+1),mutable=True,default=0)

    def ROMlinear(self,model,i):
        ind = self.exfn_xvars_ind[i]
        y1 = (model.plrom[i,0] + sum(model.plrom[i,j+1] * (model.xvars[ind[j]] - model.px0[ind[j]]) for j in range(0, len(ind))))
        return y1

    def createRomConstraint(self):
        def consROMl(model, i):
            block, local_idx = self.index_map_global_to_block[i]
            return model.y[block][local_idx] == self.ROMlinear(model, i)
        self.TRF.romL = Constraint(self.TRF.ind_ly, rule=consROMl)

    def _frp_proximity_term(self):
        """Opt-in (config 'frp proximity' > 0): w * sum(((v - v_k)/range)^2)
        over the x and z variables with finite bounds. Returns 0 when off, so
        the restoration objective is then exactly the residual of (7.11)."""
        w = float(getattr(self.config, 'frp_proximity', 0.0) or 0.0)
        if w <= 0.0:
            return 0
        model = self.TRF
        term = 0
        for vars_, centre in ((model.xvars, model.px0), (model.zvars, model.pz0)):
            for i, v in enumerate(vars_):
                if v.lb is not None and v.ub is not None and v.ub > v.lb:
                    term += ((v - centre[i]) / (v.ub - v.lb)) ** 2
        return w * term

    def createCompCheckObjective(self):
        prox = self._frp_proximity_term()
        obj = prox
        model = self.TRF
        for i in range(0, self.ly):
            block, local_idx = self.index_map_global_to_block[i]
            obj += (self.ROMlinear(model, i) - model.y[block][local_idx]) ** 2
        model.objCompCheckL = Objective(expr=obj)

    def cacheBound(self):
        self.TRF.xvarlo = []
        self.TRF.xvarup = []
        self.TRF.zvarlo = []
        self.TRF.zvarup = []
        for x in self.TRF.xvars:
            self.TRF.xvarlo.append(x.lb)
            self.TRF.xvarup.append(x.ub)
        for z in self.TRF.zvars:
            self.TRF.zvarlo.append(z.lb)
            self.TRF.zvarup.append(z.ub)


    def setParam(self, x0=None, z0=None, rom_params=None):
        if x0 is not None:
            for i in range(self.lx):
                self.TRF.px0[i] = x0[i]

        # pz0 holds the centre z_k; written when a solve passes z0, harmless
        # otherwise.
        if z0 is not None:
            for i in range(self.lz):
                self.TRF.pz0[i] = z0[i]

        if rom_params is not None:
            for i in range(self.ly):
                for j in range(len(rom_params[i])):
                    self.TRF.plrom[i,j] = rom_params[i][j]

    def setVarValue(self, x=None, y=None, z=None):
        if x is not None:
            if(len(x) != self.lx):
                raise Exception(
                    "setValue: The dimension of x is not consistant!\n")
            for i in range(0, self.lx):
                self.TRF.xvars[i].set_value(x[i])
        if y is not None:
            if(len(y) != self.ly):
                raise Exception(
                    "setValue: The dimension of y is not consistant!\n")
            for i in range(0, self.ly):
                block, local_idx = self.index_map_global_to_block[i]
                self.TRF.y[block][local_idx].set_value(y[i])

        if z is not None:
            if(len(z) != self.lz):
                raise Exception(
                    "setValue: The dimension of z is not consistant!\n")
            for i in range(0, self.lz):
                self.TRF.zvars[i].set_value(z[i])

    def setBound_reduced_DOF(self, x0, y0, z0, radius, scaling):
        if scaling in [Scaling.No]:
            for i in range(0,self.lx):
                lb = maxIgnoreNone(x0[i] - radius, self.TRF.xvarlo[i])
                ub = minIgnoreNone(x0[i] + radius, self.TRF.xvarup[i])
                # Guard: current value must be strictly inside the box
                if lb is not None and x0[i] < lb:
                    lb = x0[i]
                if ub is not None and x0[i] > ub:
                    ub = x0[i]
                self.TRF.xvars[i].setlb(lb)
                self.TRF.xvars[i].setub(ub)

            for i in range(0,self.ly):
                block, local_idx = self.index_map_global_to_block[i]
                self.TRF.y[block][local_idx].setlb(None)
                self.TRF.y[block][local_idx].setub(None)
            for i in range(0,self.lz):
                lb = maxIgnoreNone(z0[i] - radius, self.TRF.zvarlo[i])
                ub = minIgnoreNone(z0[i] + radius, self.TRF.zvarup[i])
                # Guard: current value must be strictly inside the box
                if lb is not None and z0[i] < lb:
                    lb = z0[i]
                if ub is not None and z0[i] > ub:
                    ub = z0[i]
                self.TRF.zvars[i].setlb(lb)
                self.TRF.zvars[i].setub(ub)

        elif scaling in [Scaling.Yes]:
            for i in range(0,self.lx):
                lb = maxIgnoreNone(x0[i] - radius*(self.TRF.xvarup[i]-self.TRF.xvarlo[i]),self.TRF.xvarlo[i])
                ub = minIgnoreNone(x0[i] + radius*(self.TRF.xvarup[i]-self.TRF.xvarlo[i]),self.TRF.xvarup[i])
                # Guard: current value must be strictly inside the box
                if lb is not None and x0[i] < lb:
                    lb = x0[i]
                if ub is not None and x0[i] > ub:
                    ub = x0[i]
                self.TRF.xvars[i].setlb(lb)
                self.TRF.xvars[i].setub(ub)

            for i in range(0,self.ly):
                block, local_idx = self.index_map_global_to_block[i]
                self.TRF.y[block][local_idx].setlb(None)
                self.TRF.y[block][local_idx].setub(None)
            for i in range(0,self.lz):
                lb = maxIgnoreNone(z0[i] - radius*(self.TRF.zvarup[i]-self.TRF.zvarlo[i]),self.TRF.zvarlo[i])
                ub = minIgnoreNone(z0[i] + radius*(self.TRF.zvarup[i]-self.TRF.zvarlo[i]),self.TRF.zvarup[i])
                # Guard: current value must be strictly inside the box
                if lb is not None and z0[i] < lb:
                    lb = z0[i]
                if ub is not None and z0[i] > ub:
                    ub = z0[i]
                self.TRF.zvars[i].setlb(lb)
                self.TRF.zvars[i].setub(ub)


    def evaluateDx(self,x,with_info=False):
        # This is messy, currently redundant with
        # some lines in buildROM()
        self.countDx += 1
        ans = []
        info = []
        for i in range(0,self.ly):
            block, local_idx = self.index_map_global_to_block[i]
            fcn = self.TRF.external_fcns[block][local_idx-1]._fcn
            values = []
            for j in self.exfn_xvars_ind[i]:
                values.append(x[j])

            ans.append(self._eval_bb(fcn._fcn, values, block))
            info.append((block, local_idx, values,ans[i]))
        if with_info:
            return np.array(ans), info
        else:
            return np.array(ans)

    def evaluateObj(self, x, y, z):
        if(len(x) != self.lx or len(y) != self.ly or len(z) != self.lz):
            raise Exception("evaluateObj: The dimension is not consistent with the initialization \n")
        self.setVarValue(x=x,y=y,z=z)
        return self.objective()

    def deactiveExtraConObj(self):
        self.TRF.objCompCheckL.deactivate()
        self.TRF.romL.deactivate()
        self.objective.activate()

    def activateRomCons(self,x0, rom_params):
        self.setParam(x0=x0,rom_params=rom_params)
        self.TRF.romL.activate()

    def activateCompCheckObjective(self, x0, z0, rom_params):
        self.setParam(x0=x0,z0=z0,rom_params=rom_params)
        self.TRF.objCompCheckL.activate()
        self.objective.deactivate()


    def _build_subproblem_solver(self, restoration=False):
        """Create and configure a subproblem solver ONCE per run.

        Previously the SolverFactory object + its option dict were re-created on
        every solve call (each trust-region subproblem AND each restoration
        solve); a run does hundreds of these, so that was pure overhead. The
        objects are now built once in __init__ and reused.

        Two variants, kept BEHAVIOUR-IDENTICAL to the previous inline code:
          * main (restoration=False): the full curated IPOPT option set used by
            solveModel (TRSP solves);
          * restoration=True: the lighter option set the compatibilityCheck /
            restoration solve used (only halt_on_ampl_error + max_iter on top of
            the user options).

        Solver-specific options are gated on the solver name, so swapping in a
        different subproblem solver later (e.g. conopt, uno) is just adding
        another block here; user-supplied `solver_options` apply to ANY solver.
        """
        opt = SolverFactory(self.config.solver)
        if restoration:
            if self.config.solver == 'ipopt':
                # order preserved from the original restoration solve: user
                # options first, then halt_on_ampl_error + max_iter on top.
                opt.options.update(self.config.solver_options)
                opt.options['halt_on_ampl_error'] = 'yes'
                opt.options['max_iter'] = 20000
            return opt

        if self.config.solver == 'ipopt':
            # UPDATE (not overwrite) so the curated IPOPT settings,
            # halt_on_ampl_error, max_iter AND the user-supplied
            # config.solver_options all survive (a wholesale `opt.options = {}`
            # would silently discard the latter).
            opt.options.update({
                'halt_on_ampl_error': 'yes',
                'max_iter': 20000,
                'constr_viol_tol': 1e-6,  # Allow small constraint violations
                'bound_relax_factor': 1e-8,  # Relax variable bounds slightly
                'tol': 1e-6,  # Overall solver tolerance
                'dual_inf_tol': 1e-4,  # Dual infeasibility tolerance
                'compl_inf_tol': 1e-6,
                'acceptable_tol': 1e-4,  # Acceptable tolerance for convergence
                'acceptable_constr_viol_tol': 1e-4,
                'acceptable_iter': 8,     # Exit early on acceptable convergence
                # 'nlp_scaling_method': 'gradient-based',  # Enable scaling
                # 'linear_solver': 'mumps',  # 'ma27','ma57','pardiso'
                # 'mu_strategy' : 'adaptive',
            })
        # else: a future conopt/uno block goes here.

        # User-supplied options take precedence (and apply to any solver).
        opt.options.update(self.config.solver_options)
        return opt

    def solveModel(self, x, y, z):
        model = self.model
        opt = self.opt    # built once in __init__, reused across all solves

        EPSILON_FEAS = 1e-6

        # load_solutions=False is REQUIRED, not a preference. With the default
        # (True), Pyomo loads the solution inside solve() and raises
        # "Cannot load a SolverResults object with bad status: error" on a hard
        # solver failure -- escaping this function entirely and killing the run,
        # even though the caller already handles a False flag gracefully
        # (contract and retry, then the wedge escape). Deferring the load to the
        # guarded block below turns a hard IPOPT error into a rejected step.
        try:
            results = opt.solve(model, keepfiles=self.keepfiles,
                                tee=self.stream_solver, load_solutions=False)
        except Exception as e:
            print(f"\n--- Solver raised {type(e).__name__}: {e} ---")
            print("--- treating as a failed subproblem solve ---")
            return False, 0, float('inf')

        try:
            model.solutions.load_from(results)
            solution_loaded = True
        except Exception:
            solution_loaded = False

        # Snapshot constraint duals from THIS subproblem solve (only the user's
        # own constraints, not the internal TRF block). Kept on the interface
        # because the later gjh evaluation in grad_hess_calc clears model.dual,
        # so we hold onto the last successful subproblem's multipliers for the
        # run summary. Guarded so a reporting hiccup never aborts the solve.
        if self.report_duals and solution_loaded and hasattr(model, 'dual'):
            snap = {}
            try:
                for con, val in model.dual.items():
                    if con.parent_block() is self.TRF:
                        continue
                    if val is not None:
                        snap[con.name] = float(val)
            except Exception:
                snap = {}
            if snap:
                self.last_duals = snap

        def _max_violation(model):
            max_viol = 0.0
            for con in model.component_data_objects(Constraint, active=True):
                try:
                    body_val = value(con.body)
                    if con.lower is not None:
                        max_viol = max(max_viol, max(0.0, value(con.lower) - body_val))
                    if con.upper is not None:
                        max_viol = max(max_viol, max(0.0, body_val - value(con.upper)))
                except Exception:
                    pass
            return max_viol

        infeas_measure = _max_violation(model) if solution_loaded else float('inf')

        solver_ok = (
            results.solver.status == SolverStatus.ok
            and results.solver.termination_condition in [
                TerminationCondition.optimal,
                TerminationCondition.locallyOptimal,
                TerminationCondition.feasible,
            ]
        )
        numerically_feasible = solution_loaded and (infeas_measure <= EPSILON_FEAS)

        if solver_ok or numerically_feasible:
            for i in range(self.lx):
                x[i] = value(self.TRF.xvars[i])
            for i in range(self.ly):
                block, local_idx = self.index_map_global_to_block[i]
                y[i] = value(self.TRF.y[block][local_idx])
            for i in range(self.lz):
                z[i] = value(self.TRF.zvars[i])

            for obj in model.component_data_objects(Objective, active=True):
                return True, obj(), infeas_measure
        else:
            print(f"\n--- Constraint violation: max_violation={infeas_measure:.3e} ---")
            print(f"Solver: {results.solver.status}, {results.solver.termination_condition}")
            if solution_loaded:
                log_infeasible_constraints(model, log_expression=True, log_variables=True)
            return False, 0, infeas_measure


    def TRSPk(self, x, y, z, x0, y0, z0, rom_params, radius, scaling):
        if(len(x) != self.lx or len(y) != self.ly or len(z) != self.lz or
                len(x0) != self.lx or len(y0) != self.ly or len(z0) != self.lz):
            raise Exception(
                "TRSP_k: The dimension is not consistant with the initialization!\n")

        self.setBound_reduced_DOF(x0, y0, z0, radius, scaling)
        self.setVarValue(x, y, z)
        self.deactiveExtraConObj()
        self.activateRomCons(x0, rom_params)
        return self.solveModel(x, y, z)

    def compatibilityCheck(self, x, y, z, x0, y0, z0, rom_params, radius, scaling):
        """Feasibility restoration problem (7.11): minimise the surrogate
        mismatch ||r_k(w) - y||^2 within the trust region, glass-box active."""
        if(len(x) != self.lx or len(y) != self.ly or len(z) != self.lz or
                len(x0) != self.lx or len(y0) != self.ly or len(z0) != self.lz):
            raise Exception(
                "Compatibility_Check: The dimension is not consistant with the initialization!\n")

        self.setBound_reduced_DOF(x0, y0, z0, radius, scaling)
        self.setVarValue(x, y, z)
        self.deactiveExtraConObj()
        self.activateCompCheckObjective(x0, z0, rom_params)
        return self.solveModel(x, y, z)

    def grad_hess_calc(self, x, y, z, rom_params, scaling):
        model = self.model

        self.setBound_reduced_DOF(x, y, z, 1e10, scaling)
        self.setVarValue(x=x,y=y,z=z)
        self.deactiveExtraConObj()
        self.activateRomCons(x, rom_params)

        self.optGJH.solve(model, tee=False, symbolic_solver_labels=True)

        # Only the objective gradient g and constraint Jacobian J (plus the
        # var/constraint name lists) are consumed downstream -- by the
        # criticality LP. The full Hessian and its eigen-decomposition that
        # this routine used to compute were never used by the TRSP (which is
        # driven by rom_params), so they are not computed.
        g, J, H, varlist, conlist = model._gjh_info
        return g, J, varlist, conlist

    def criticalityCheck(self, x, y, z, rom_params, g, J, varlist, conlist, criticality_radius):
        """
        Perform a criticality check by solving a linearized trust-region subproblem (TRSP).
        The constraints are normalized by scaling with the largest coefficient in each constraint.
        """

        model = self.model
        l = ConcreteModel()

        # Define variables
        l.v = Var(varlist, domain=Reals)
        for i in varlist:
            l.v[i] = 0.0
            l.v[i].setlb(-criticality_radius)
            l.v[i].setub(criticality_radius)

        # Define constraints
        def linConMaker(l, i):
            """
            Construct normalized constraints for the criticality check.
            """
            con_i = model.find_component(conlist[i])
            if con_i is None:
                return Constraint.Skip

            isEquality = con_i.equality
            lo = con_i.lower
            up = con_i.upper

            # --- Skip inactive inequalities ---
            if not isEquality:
                lslack = con_i.lslack() if lo is not None else float('inf')
                uslack = con_i.uslack() if up is not None else float('inf')
                # Only include if nearly active (slack < threshold)
                if min(abs(lslack), abs(uslack)) > 1e-4:
                    return Constraint.Skip
            # --- ---

            try:
                # Extract coefficients for the constraint
                coefficients = [x[2] for x in J if x[0] == i]
                scale_factor = max(abs(c) for c in coefficients)  # Largest coefficient

                if scale_factor == 0:
                    return Constraint.Skip

                if scale_factor < 1e-8:
                    return Constraint.Skip

                # Normalize the coefficients
                Jv = sum((x[2] / scale_factor) * l.v[varlist[x[1]]] for x in J if x[0] == i)

                # Normalize the bounds
                if lo is not None:
                    lo_scaled = lo / scale_factor
                else:
                    lo_scaled = None

                if up is not None:
                    up_scaled = up / scale_factor
                else:
                    up_scaled = None

                # Construct normalized constraints
                if isEquality:
                    return Jv == 0  # Equality constraint
                elif lo_scaled is not None and up_scaled is not None:
                    return inequality(lo_scaled, Jv, up_scaled)  # Inequality with both bounds
                elif lo_scaled is not None:
                    return Jv >= lo_scaled  # Lower bound only
                elif up_scaled is not None:
                    return Jv <= up_scaled  # Upper bound only
                else:
                    raise ValueError(f"Constraint {conlist[i]} has no bounds.")

            except Exception as e:
                print(f"❌ Error computing Jv for constraint {i}: {e}")
                return Constraint.Skip

        try:
            l.lincons = Constraint(range(len(conlist) - 1), rule=linConMaker)
        except Exception as e:
            print(f"❌ Error defining constraints: {e}")

        # Define objective
        try:
            # Normalize gradient; guard against a (near-)zero reduced gradient
            gfnorm = max(sqrt(sum(x[1]**2 for x in g)), 1e-12)
            l.obj = Objective(expr=sum(x[1] / gfnorm * l.v[varlist[x[0]]] for x in g if varlist[x[0]] in l.v))
        except Exception as e:
            print(f"❌ Error defining objective function: {e}")

        # Solve the model (restoration solver: built once in __init__, lighter
        # option set than the TRSP solver -- preserved for behaviour parity).
        opt = self.opt_restoration
        results = opt.solve(l)

        if results.solver.status == SolverStatus.ok and results.solver.termination_condition in [
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        ]:
            l.solutions.load_from(results)
            return True, abs(l.obj()) / gfnorm if gfnorm > 1 else abs(l.obj())
        else:
            print("⚠️ Criticality check failed!")
            return False, float("inf")

    def buildROM(self, x, radius_base, scaling):
        """
        Builds a linear ROM near x by one-sided finite differences.
        Perturbation step is scaled by bound width when scaling=Yes,
        or used as raw physical step when scaling=No.
        """

        y1 = self.evaluateDx(x)
        rom_params = []

        lo = np.array(self.TRF.xvarlo)
        up = np.array(self.TRF.xvarup)

        dist_to_upper = up - x
        dist_to_lower = x - lo

        for i in range(0, self.ly):
            rom_params.append([])
            rom_params[i].append(y1[i])

            block, local_idx = self.index_map_global_to_block[i]
            fcn = self.TRF.external_fcns[block][local_idx-1]._fcn
            values = []
            for j in self.exfn_xvars_ind[i]:
                values.append(x[j])

            for j_local, j_global in enumerate(self.exfn_xvars_ind[i]):
                # Compute step size consistent with trust-region box scaling
                if scaling in [Scaling.Yes]:
                    bound_width = self.TRF.xvarup[j_global] - self.TRF.xvarlo[j_global]
                    step_size = radius_base * bound_width
                else:  # Scaling.No
                    step_size = radius_base

                # Directional check: move forward if room, backward otherwise
                if dist_to_upper[j_global] >= step_size:
                    actual_step = step_size
                else:
                    actual_step = -min(step_size, dist_to_lower[j_global])

                # Guard against zero step
                if abs(actual_step) < 1e-8:
                    actual_step = 1e-8 if dist_to_upper[j_global] > 0 else -1e-8

                values[j_local] = values[j_local] + actual_step
                y2 = self._eval_bb(fcn._fcn, values, block)
                rom_params[i].append((y2 - y1[i]) / actual_step)
                values[j_local] = values[j_local] - actual_step

        return rom_params, y1
