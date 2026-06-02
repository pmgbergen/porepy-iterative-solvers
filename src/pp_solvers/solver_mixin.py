"""This module contains the `IterativeSolverMixin` class, which provides the capabilitiy
of using iterative linear solvers to a PorePy model.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import time
from typing import Callable, TypedDict

import numpy as np
import porepy as pp
import scipy.sparse as sps
from porepy.viz.solver_statistics import SolverStatistics

from pp_solvers.block_linear_system import (
    BlockLinearSystem,
    LinearSystemIndexer,
)
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

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


__all__ = [
    "IterativeSolverMixin",
    "LinearSolverParams",
]

"""Below are methods that are used to create specific schemes for different equations.
Note that these consider PETSc configurations, and have no responsibility for
taking care of equations etc. (CURRENT IMPLEMENTATION IS NOT RIGHT). This means they are
essentially bearers of options for the solver.
"""

type PETScKspConvergedReason = int
"""A type alias for PETSc return codes. See
https://petsc.org/release/manualpages/KSP/KSPConvergedReason/"""


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

    linsolve_construction_time: list[float] = field(default_factory=list)
    linsolve_solve_time: list[float] = field(default_factory=list)
    petsc_converged_reason: list[int] = field(default_factory=list)
    num_krylov_iters: list[int] = field(default_factory=list)


class LinearSolverParams(TypedDict, total=False):
    """A dictionary of linear solver parameters, stored in
    `model.params['linear_solver']`.

    The argument `total=False` states that all entries are optional.

    """

    options: dict
    """A dict of parameters to tune the solver configuration. See examples for the
    structure."""
    solver_selector: SolverSelector
    """A solver selector object providing multiple linear solver configurations."""
    delete_matrices: bool
    """Delete the linear solver matrix when it is not needed to free the memory as early
    as possible. Defaults to True.

    """
    preconditioner_factory: PetscKspPcConfiguration
    """A factory to build a PETSc preconditioned linear solver from. Using the default
    factory if not passed.

    """


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

    def linear_solver_params(self) -> LinearSolverParams:
        """Access linear solver parameters dictionary."""
        try:
            linear_solver_params = self.params.get("linear_solver", dict())
        except KeyError as e:
            logger.exception("You must specify `linear_solver` in the model params.")
            raise e
        if not isinstance(linear_solver_params, dict):
            raise ValueError(
                "model_params['linear_solver'] must be a dictionary when used together "
                "with the IterativeSolverMixin."
            )
        return linear_solver_params

    def solve_linear_system(self) -> np.ndarray:
        """Solve the linear system.

        This function returns a solution array even if the underlying linear solver did
        not converge. A warning will be logged in this case. It may also return nans if
        things go particularly bad. It is the caller's responsibility to validate the
        returned values, same as for the direct linear solver counterpart.

        Dispatches to one of two paths:

        - No ML selection: calls `_solve_linear_system` directly.
        - With ML selection: delegates to `_solve_linear_system_with_solver_selection`.

        Returns:
            Solution array of the linear system.
        """
        linear_solver_params = self.linear_solver_params()

        solver_selector = linear_solver_params.get("solver_selector", None)
        solver_options = linear_solver_params.get("options", {})
        if solver_selector is None:
            solution, petsc_converged_reason = self._solve_linear_system(
                solver_options=solver_options
            )
        else:
            solution, petsc_converged_reason = (
                self._solve_linear_system_with_solver_selection(
                    solver_selector=solver_selector, solver_options=solver_options
                )
            )

        if petsc_converged_reason <= 0:
            logger.warning(
                f"Linear solver did not converge. Reason: %d. "
                "Check the solver options and the problem setup. "
                "See detailed description of PETSc error codes: "
                "https://petsc.org/release/manualpages/KSP/KSPConvergedReason/",
                petsc_converged_reason,
            )

        return solution

    def _solve_linear_system_with_solver_selection(
        self, solver_selector: SolverSelector, solver_options: dict
    ) -> tuple[np.ndarray, PETScKspConvergedReason]:
        """Use ML-based solver selection to solve the linear system and update the
        model.

        Selects solver options via `solver_selector`, merging them with any manually
        provided `solver_options` (manual options may be overridden). After solving,
        feeds back performance metrics and updates the ML model with them.

        Parameters:
            solver_selector: Selects the solver scheme based on system characteristics.
            solver_options: Manually provided solver options; may be overridden by the
                selected scheme.

        Returns:
            A tuple of two elements:
                - Solution array of the linear system.
                - PETSc KSP converged reason
        """
        if hasattr(self.model, "solver_selection_characteristics"):
            characteristics: np.ndarray = self.model.solver_selection_characteristics()
        else:
            characteristics = np.array([])
        # Perform the ML selection.
        solver_selection_opts, solver_id = solver_selector.select_linear_solver_scheme(
            characteristics=characteristics, active_solver_idx=-1
        )

        # Check that the ML model does not override the manually provided options. Warn
        # if so and merge the options into a single dict.
        intersecting_keys = set(solver_options).intersection(solver_selection_opts)
        if len(intersecting_keys) > 0:
            logger.warning(
                "Solver selection override manually provided solver options:",
                intersecting_keys,
            )
        solver_selection_opts = solver_options | solver_selection_opts

        # Solve the linear system.
        solution, petscConvergedReason = self._solve_linear_system(
            solver_options=solver_selection_opts
        )

        # The way of accessing these values should be changed when they find a
        # better accommodation.
        solve_time = self.nonlinear_solver_statistics.linsolve_solve_time
        construct_time = self.nonlinear_solver_statistics.linsolve_construction_time
        # Providing feedback to the ML model.
        solver_selector.provide_performance_feedback(
            solve_time=solve_time,
            construct_time=construct_time,
            success=petscConvergedReason > 0,
        )
        return solution, petscConvergedReason

    def _solve_linear_system(
        self, solver_options: dict
    ) -> tuple[np.ndarray, PETScKspConvergedReason]:
        """Assembles the PETSc linear solver and solves the linear system.

        Parameters:
            solver_options: Manually provided solver options.

        Returns:
            A tuple of two elements:
                - Solution array of the linear system.
                - PETSc KSP converged reason
        """
        # Check for NaN or Inf in the RHS.
        # The rhs inside the linear system object is rearranged to match the matrix.
        rhs = self.bmat.rhs
        if np.any(np.isnan(rhs) | np.isinf(rhs)):
            # This should never be the case, as this situation should cut off by the
            # nonlinear convergence criterion from the earliear nonlinear iteration. We
            # keep this safeguard until the iterative solver is in a more mature state.
            raise ValueError("RHS contains NaN or Inf values")

        t0 = time()
        try:
            solver = self._solver_factory.make_solver(self.bmat, solver_options)
        except Exception as e:
            logger.warning("Failed to create solver with the provided preconditioner.")
            return np.full(rhs.shape, np.nan), -9999
        elapsed = time() - t0
        self.nonlinear_solver_statistics.linsolve_construction_time.append(elapsed)
        logger.info("Linear solver constructed in %.2f seconds.", elapsed)

        # Project the right hand side to the local block matrix ordering, as was done
        # for the block matrix during assembly. We need to do this on the reordered rhs
        # vector (with contact eqs reordered).
        t0 = time()
        x_loc = solver.solve(rhs)
        elapsed = time() - t0
        self.linear_solver_statistics.linsolve_solve_time.append(elapsed)
        num_it = len(solver.get_residuals())
        logger.info(
            "Linear system solved in %.2f seconds with %d iterations.",
            elapsed,
            num_it,
        )

        info: PETScKspConvergedReason = solver.ksp.getConvergedReason()
        # Project the solution back to the global (PorePy) ordering. For clarity, no
        # contact reordering here, since only the equations (rows) and not the variables
        # (columns) were reordered.
        x = self.bmat.permute_right_vector_to_original(x_loc)
        self.linear_solver_statistics.petsc_converged_reason.append(info)
        self.linear_solver_statistics.num_krylov_iters.append(num_it)
        if self.linear_solver_params().get("delete_matrices", True):
            del self.bmat
        if info < 0:
            x = np.full_like(x, np.nan)
        return np.atleast_1d(x), info

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
        # Delete the original linear system to save memory unless instructed not to.
        if self.linear_solver_params().get("delete_matrices", True):
            del self.linear_system

    def _initialize_linear_solver(self):
        # Set up preconditioner.

        # Add fields for the linear solver statistics to the nonlinear solver statistics
        # object.
        self.nonlinear_solver_statistics.linsolve_construction_time = []
        self.nonlinear_solver_statistics.linsolve_solve_time = []
        self.nonlinear_solver_statistics.petsc_converged_reason = []
        self.nonlinear_solver_statistics.num_krylov_iters = []

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

    def set_nonlinear_solver_statistics(self) -> None:
        """Override the method to set the solver statistics, so that we also get fields
        for the linear solver.

        This is certainly not the intended way of doing this, and it hacky, but the
        current PorePy implementation only caters to statistics objects being sent
        as part of the parameter class, which would require modification of all
        runscripts. Instead, we do it dirty for now.

        """
        super().set_nonlinear_solver_statistics()  # type: ignore[misc]
        # The name of the attribute is really not meaningful..
        self.linear_solver_statistics = LinearSolverStatistics()


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
    raise ValueError(f"Unknown model:", type(model))
