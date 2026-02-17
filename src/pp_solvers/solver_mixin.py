"""This module contains the `IterativeSolverMixin` class, which provides the capabilitiy
of using iterative linear solvers to a PorePy model.

"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from time import time
from typing import Callable
from warnings import warn

import numpy as np
import porepy as pp
import scipy.sparse as sps
from porepy.viz.solver_statistics import SolverStatistics

from pp_solvers.dof_manager import DofManager
from pp_solvers.equation_variable_groups import (
    ContactMechanicsGroup,
    EnergyBalanceTemperatureGroup,
    InterfaceForceBalanceGroup,
)
from pp_solvers.mat_utils import csr_ones, inv_block_diag
from pp_solvers.options_parsers import LinearTransformedScheme, PetscKSPScheme
from pp_solvers.preconditioners import (
    PetscKspPcConfiguration,
    hm_factory,
    mass_balance_factory,
    momentum_balance_factory,
    th_factory,
    thm_factory,
)
from pp_solvers.solver_selection.selector import SolverSelector

from .block_linear_system import BlockLinearSystem, LinearSystemIndexer

from .block_linear_system import BlockLinearSystem, LinearSystemIndexer

__all__ = [
    "IterativeSolverMixin",
]

"""Below are methods that are used to create specific schemes for different equations.
Note that these consider PETSc configurations, and have no responsibility for
taking care of equations etc. (CURRENT IMPLEMENTATION IS NOT RIGHT). This means they are
essentially bearers of options for the solver.
"""

FAILURE_COUNTER = count(0)


def save_with_pickle(object, name: str):
    import pickle

    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(object, f)


def transform_contact_block(
    J: BlockLinearSystem, row_group: int, col_group: int, nd: int
):
    """Assemble the right linear transformation."""
    # Sorted according to groups. If not done, the matrix can be in porepy order,
    # which does not guarantee that diagonal groups are truly on diagonals.
    Qright = J.empty_container()[:]

    if len(J.indexer.dofs_row[row_group]) == 0:
        # If the relevant row group is empty (case without fractures), the
        # transformation is the identity matrix, nothing should be done.
        Qright.mat = csr_ones(Qright.shape[0])
        return Qright

    # Pick out the block matrix corresponding to the interface force balance equation
    # (the row index) and the interface displacement variable (the column index). There
    # is an underlying assumption that the groups in the preconditioner ordering are so
    # that this equtaion-variable pair is on the diagonal of the matrix.
    J55 = J[col_group, col_group].mat

    # The contribution from the interface displacement variable to the force balance
    # should be diagonally dominant, reflecting that the interface displacement has the
    # strongest influence on the force on its own cell (and less so on the neighboring
    # cell, though, with the MPSA stencil, the latter will not be zero). Note that there
    # is no connection between the two sides of a fracture; this is represented in a
    # different block of the full matrix. Approximate the stencil by a block diagonal,
    # and calculate the inverse cheaply.
    J55_inv = inv_block_diag(J55, nd=nd)

    Qright.mat = csr_ones(Qright.shape[0])
    # Extract the block matrix corresponding to the impact of the contact forces on the
    # force balance equation.
    J54 = J[col_group, row_group].mat

    # The transformation is given like this, see papers by Zabegaev for the details.
    tmp = -J55_inv @ J54
    Qright[col_group, row_group] = tmp
    return Qright


def scale_energy_transform(J, row_groups: list[int], model: pp.PorePyModel):
    """Assemble the right linear transformation for scaling energy fluxes."""
    # Sorted according to groups. If not done, the matrix can be in porepy order,
    # which does not guarantee that diagonal groups are truly on diagonals.
    Q = J.empty_container()[:]

    subdomains = model.mdg.subdomains()
    vols = 1.0 / model.equation_system.evaluate(model.specific_volume(subdomains))

    Q.mat = sps.eye(Q.shape[0], format="csr")
    if len(subdomains) == 0:
        # No subdomains, hence no scaling.
        return Q
    Q.set_diagonal(groups=row_groups, values=vols, additive=False)

    return Q


@dataclass
class LinearSolverStatistics(SolverStatistics):
    """A dataclass to store statistics about the linear solver.

    Currently, PorePy only has stastics for the nonlinear solver, so we create an
    extension to store the linear solver statistics.
    """

    linsolve_construction_time: list[float] = None
    linsolve_solve_time: list[float] = None
    petsc_converged_reason: list[int] = None
    num_krylov_iters: list[int] = None

    def __post_init__(self):
        self.linsolve_construction_time = []
        self.linsolve_solve_time = []
        self.petsc_converged_reason = []
        self.num_krylov_iters = []


class IterativeSolverMixin(pp.PorePyModel):
    """Intended usage:

    (i) Plug in the `IterativeSolverMixin` to the PorePy model inheritance chain below
    your methods that override `solve_linear_system` and `assemble_linear_system`, e.g.
    for logging purposes.

    (ii) Insert an additional option to the model options dictionary:
    ```
    model_options["linear_solver"] = {
        "options": {
            {
                # Uncomment below to enable convergence logging:
                # "gmres: {
                #     "ksp_monitor": None,
                # }
            }
        }
    }
    ```
    The linear solver can be customized via the `"options"` sub-dictionary, see
    the examples folder for details.

    """

    def _determine_solver_options(self) -> dict:
        # The options are either provided by a used manually, or determined by
        # machine learning in the solver selector.

        solver_selector: SolverSelector | None = self.params["linear_solver"].get(
            "solver_selector", None
        )
        solver_options: dict = self.params["linear_solver"].get("options", {})
        if solver_selector is None:
            # No solver selection. Taking manually provided options.
            return solver_options

        # Warn if both the solver selector and the options are present.
        if len(solver_options) > 0:
            warn(
                'Parameters "options" and "solver_selector" are mutually exclusive. '
                "Solver selection may override manually provided parameters."
            )

        characteristics = np.array([self.ad_time_step.value(self.equation_system)])

        solver_selection_opts, solver_id = solver_selector.select_linear_solver_scheme(
            characteristics=characteristics, active_solver_idx=-1
        )

        return solver_options | solver_selection_opts

    def solve_linear_system(self) -> None:
        solver_selector: SolverSelector | None = self.params["linear_solver"].get(
            "solver_selector", None
        )
        solver_options = self._determine_solver_options()

        try:
            solution = self._solve_linear_system(solver_options=solver_options)
        except RuntimeError as e:
            success = False
            raise e
        else:
            success = True
        finally:
            if solver_selector is not None:
                # The way of accessing these values should be changed when they find a
                # better accommodation.
                solve_time = self.nonlinear_solver_statistics.linsolve_solve_time[-1]
                construct_time = (
                    self.nonlinear_solver_statistics.linsolve_construction_time[-1]
                )
                solver_selector.provide_performance_feedback(
                    solve_time=solve_time,
                    construct_time=construct_time,
                    success=success,
                )
        return solution

    def _solve_linear_system(self, solver_options: dict) -> None:
        # Check for NaN or Inf in the RHS.
        # The rhs inside the linear system object is rearranged to match the matrix.
        rhs = self.bmat.rhs
        if np.any(np.isnan(rhs) | np.isinf(rhs)):
            raise ValueError("RHS contains NaN or Inf values")

        solver_options = self.params["linear_solver"].get("options", {})
        ksp_factory = self._solver_factory

        t0 = time()
        try:
            solver = ksp_factory.make_solver(self.bmat, solver_options)
        except Exception as e:
            raise RuntimeError(
                "Failed to create solver with the provided preconditioner",
                solver_options,
            ) from e
        self.nonlinear_solver_statistics.linsolve_construction_time.append(time() - t0)

        # Project the right hand side to the local block matrix ordering, as was done
        # for the block matrix during assembly. We need to do this on the reordered rhs
        # vector (with contact eqs reordered).
        t0 = time()
        x_loc = solver.solve(rhs)
        self.nonlinear_solver_statistics.linsolve_solve_time.append(time() - t0)

        info = solver.ksp.getConvergedReason()

        # Project the solution back to the PorePy ordering. For clarity, no contact
        # reordering here, since only the equations (rows) and not the variables
        # (columns) were reordered.
        x = self.bmat.permute_right_vector_to_original(x_loc)
        self.nonlinear_solver_statistics.petsc_converged_reason.append(info)
        self.nonlinear_solver_statistics.num_krylov_iters.append(
            len(solver.get_residuals())
        )

        if info <= 0:
            save_with_pickle(self.bmat, f"failed_matrix_{next(FAILURE_COUNTER)}")
            print("Saving failed matrix")
            raise RuntimeError(
                f"Solver did not converge. Reason: {info}. "
                "Check the solver options and the problem setup."
            )

        x = np.atleast_1d(x)
        # this is not a responsibility of the iterative linear solver!
        if self._apply_schur_complement_reduction():
            x = self.equation_system.expand_schur_complement_solution(x)
        return x

    def assemble_linear_system(self):
        super().assemble_linear_system()  # type: ignore[misc]

        dof_manager = self._solver_factory.dof_manager
        # Get the linear system from the equation system.

        # TODO: Replace this with a different type of plugin
        mat, rhs = self.linear_system

        # Creating the indices of DoFs for the BlockLinearSystem class.
        bmat = BlockLinearSystem(
            mat=mat,
            rhs=rhs,
            indexer=LinearSystemIndexer(
                dofs_row=dof_manager.eq_dofs(),
                dofs_col=dof_manager.var_dofs(),
                group_names_row=dof_manager.equation_names(),
                group_names_col=dof_manager.variable_names(),
            ),
        )

        # Store the linear system in the solver mixin *and*, by calling [:], rearrange
        # the blocks (and thereby the underlying matrix) to match the ordering defined
        # by the # `dof_manager`.
        self.bmat = bmat[:]

    def _initialize_linear_solver(self):
        # Set up preconditioner.

        precond_factory: Callable[[], PetscKspPcConfiguration]
        linear_solver_params = self.params.get("linear_solver", {})
        precond_factory = linear_solver_params.get("preconditioner_factory", None)
        if precond_factory is None:
            precond_factory = default_preconditioner_factory(self)

        petsc_ksp_pc_configuration: PetscKspPcConfiguration = precond_factory()

        dof_manager = DofManager(model=self, groups=petsc_ksp_pc_configuration.groups)

        try:
            contact_ind, u_intf_ind = dof_manager.indices_of_groups(
                [ContactMechanicsGroup(), InterfaceForceBalanceGroup()]
            )
        except ValueError:
            contact_ind, u_intf_ind = None, None

        # TODO: Logic below should not be hard-coded in SolverMixin. precond_factory()
        # should return some object that determines pre-processing before solving.
        ksp_factory = PetscKSPScheme(
            petsc_ksp_pc_configuration=petsc_ksp_pc_configuration,
            dof_manager=dof_manager,
        )
        contact_transform, thermal_transform = None, None
        if contact_ind is not None and u_intf_ind is not None:
            # If there is a contact group, we need to use a linear solver that takes
            # care of potential singularities in the contact block.
            contact_transform = [
                lambda bmat: transform_contact_block(
                    bmat, contact_ind, u_intf_ind, self.nd
                )
            ]
        try:
            energy_balance_groups = dof_manager.indices_of_groups(
                [EnergyBalanceTemperatureGroup()]
            )
        except ValueError:
            energy_balance_groups = []
        if len(energy_balance_groups) > 0:
            thermal_transform = [
                lambda bmat: scale_energy_transform(
                    bmat, row_groups=energy_balance_groups, model=self
                )
            ]

        if contact_transform is not None or thermal_transform is not None:
            solver_factory = LinearTransformedScheme(
                inner=ksp_factory,
                right_transformations=contact_transform,
                left_transformations=thermal_transform,
            )

        else:
            # A standard KSP solver will do.
            solver_factory = ksp_factory

        self._solver_factory = solver_factory

    def set_solver_statistics(self) -> None:
        """Override the method to set the solver statistics, so that we also get fields
        for the linear solver.

        This is certainly not the intended way of doing this, and it hacky, but the
        current PorePy implementation only caters to statistics objects being sent
        as part of the parameter class, which would require modification of all
        runscripts. Instead, we do it dirty for now.

        """
        # The name of the attribute is really not meaningful..
        self.nonlinear_solver_statistics = LinearSolverStatistics()


def default_preconditioner_factory(
    model: pp.PorePyModel,
) -> Callable[[], PetscKspPcConfiguration]:
    if isinstance(model, pp.SinglePhaseFlow):
        return mass_balance_factory
    if isinstance(model, pp.MomentumBalance):
        return momentum_balance_factory
    if isinstance(model, pp.MassAndEnergyBalance):
        return th_factory
    if isinstance(model, pp.Poromechanics):
        return hm_factory
    if isinstance(model, pp.Thermoporomechanics):
        return thm_factory
    raise ValueError()
