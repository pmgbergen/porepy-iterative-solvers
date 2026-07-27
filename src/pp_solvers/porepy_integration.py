"""PETSc-based iterative linear solver for PorePy models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import time
from typing import Callable, Optional

import numpy as np
import porepy as pp
from porepy.numerics.solvers import (
    LinearSolverStatus,
    LinearSolverStatusFailure,
    LinearSolverStatusSuccess,
)

from pp_solvers.block_linear_system import BlockLinearSystem, LinearSystemIndexer
from pp_solvers.dof_manager import DofManager
from pp_solvers.equation_variable_groups import EquationVariableGroup
from pp_solvers.options_parsers import initialize_petsc_ksp
from pp_solvers.preconditioners import (
    LinearSolverConfiguration,
    PetscKspPcConfiguration,
    hm_factory,
    mass_balance_factory,
    momentum_balance_factory,
    th_factory,
    thm_factory,
    thm_tpsa_factory,
    validate_all_keys_are_unique,
)
from pp_solvers.solver_selection.selector import SolverSelector
from pp_solvers.transformations import (
    LinearSystemTransformation,
    PorePyArrangementTransformation,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


__all__ = [
    "IterativeLinearSolver",
    "PETScKspConvergedReason",
    "IterativeLinearSolverSuccess",
    "IterativeLinearSolverFailure",
]

type PETScKspConvergedReason = int
"""A type alias for PETSc return codes. See
https://petsc.org/release/manualpages/KSP/KSPConvergedReason/

"""


@dataclass
class IterativeLinearSolverSuccess(LinearSolverStatusSuccess):
    """Status of a successful PETSc iterative solve."""

    construct_time: float
    """Time it took to construct a linear solver."""
    petsc_converged_reason: PETScKspConvergedReason
    """A status returned by PETSc KSP. See
    https://petsc.org/release/manualpages/KSP/KSPConvergedReason/

    """
    num_krylov_iters: int
    """Number of Krylov subspace-based iterative method iterations."""


@dataclass
class IterativeLinearSolverFailure(LinearSolverStatusFailure):
    """Status of a failed PETSc iterative solve.

    PETSc details are optional because a failure can occur before KSP construction.

    """

    reason: str
    """Human readible failure description."""
    construct_time: float
    """Wall-clock time spent constructing the linear solver, in seconds."""
    solve_time: float
    """Wall-clock time spent solving the linear system, in seconds."""
    petsc_converged_reason: Optional[PETScKspConvergedReason] = None
    """A status returned by PETSc KSP. See
    https://petsc.org/release/manualpages/KSP/KSPConvergedReason/

    """
    num_krylov_iters: int = 0
    """Number of Krylov subspace-based iterative method iterations."""


class IterativeLinearSolver(pp.solvers.LinearSolverBase):
    """Solve PorePy linear systems with configurable PETSc preconditioners.

    Parameters:
        solver_options: Parameters used to tune the solver configuration. See examples
            for the expected structure.
        solver_selector: A solver selector object providing multiple linear solver
            configurations. If not passed (default), ML solver selection is disabled.
        delete_matrices: Delete the linear solver matrix when it is not needed to free
            the memory as early as possible. Defaults to True.
        configuration_factory: A factory that defines the PETSc solver, variable groups,
            and linear-system transformations. If None (default), using a default
            factory for a given model.

    """

    def __init__(
        self,
        solver_options: Optional[dict] = None,
        solver_selector: Optional[SolverSelector] = None,
        delete_matrices: bool = True,
        configuration_factory: Optional[Callable[[], LinearSolverConfiguration]] = None,
    ):
        if solver_options is None:
            solver_options = {}
        self.solver_options: dict = solver_options
        """A dict of parameters to tune the solver configuration. See examples for the
        structure.

        """
        self.solver_selector: Optional[SolverSelector] = solver_selector
        """A solver selector object providing multiple linear solver configurations. If
        None, ML solver selection is disabled.
    
        """
        self.delete_matrices: bool = delete_matrices
        """Delete the linear solver matrix when it is not needed to free the memory as
        early as possible.
    
        """
        self.configuration_factory: Optional[
            Callable[[], LinearSolverConfiguration]
        ] = configuration_factory
        """A factory to build a PETSc preconditioned linear solver. If None, a default
        factory will be set for a given model in :meth:`initialize_linear_solver`.

        """
        self.petsc_ksp_pc_configuration: Optional[PetscKspPcConfiguration] = None
        """PETSc solver and preconditioner configuration, set in
        :meth:`initialize_with_model`.

        """
        self.transformations: list[LinearSystemTransformation] = []
        """Transformations applied before solving and reversed on the solution, set in
        :meth:`initialize_with_model`.

        """
        self.dof_manager: Optional[DofManager] = None
        """Mapping between PorePy degrees of freedom and configured solver groups, set
        from the first assembled linear system.

        """
        self._groups: list[EquationVariableGroup] | None = None
        """Groups of equations and variables. Set in :meth:`initialize_with_model."""
        self._model: pp.PorePyModel | None = None
        """PorePy model. Set in :meth:`initialize_with_model."""

    def initialize_with_model(self, model: pp.PorePyModel) -> None:
        """Initialize configuration, transformations, and DoF mappings for ``model``."""

        if self.configuration_factory is None:
            self.configuration_factory = default_preconditioner_factory(model)

        configuration = self.configuration_factory()
        validate_all_keys_are_unique(configuration.solver)
        self.petsc_ksp_pc_configuration = configuration.solver
        # The PorePyArrangementTransformation permutes the linear system from the PorePy
        # ordering to the ordering declared by the DofManager. Then it transforms the
        # solution back to the PorePy ordering. It is included by default for all the
        # problems.
        self.transformations = [
            PorePyArrangementTransformation()
        ] + configuration.transformations

        self._model = model
        self._groups = configuration.groups
        self.dof_manager = None

    def construct_dof_manager(
        self,
        equation_indexer: pp.ad.EquationIndexer,
        variable_indexer: pp.ad.VariableIndexer,
    ) -> DofManager:
        """Initialize or validate DoF mappings from an assembled linear system."""
        if self._groups is None or self._model is None:
            raise ValueError(
                "The linear solver must be initialized with a model first."
            )

        return DofManager(
            model=self._model,
            equation_indexer=equation_indexer,
            variable_indexer=variable_indexer,
            groups=self._groups,
        )

    def solve_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> tuple[np.ndarray, LinearSolverStatus]:
        """Solve a linear system and return its solution and solver status.

        This function returns a solution array even if the underlying linear solver did
        not converge, and it might contain nans. A warning will be logged in this case,
        and the returned status is set to "failure". It is the caller's responsibility
        to check the returned status.

        Dispatches to one of two paths:

        - No ML selection: calls `_solve_linear_system` directly.
        - With ML selection: delegates to `_solve_linear_system_with_solver_selection`.

        Parameters:
            linear_system: PorePy's container for a linear system. If the attribute
                :attr:`delete_matrices` is `True`, it modifies the linear system by
                dropping the matrix reference.

        Returns:
            Solution array of the linear system and solver status.

        """
        assert self.petsc_ksp_pc_configuration is not None, (
            "The linear solver must be initialized with a model before solving."
        )

        # If the rhs contains nans or infs, exiting early.
        if np.any(np.isnan(linear_system.rhs) | np.isinf(linear_system.rhs)):
            error_msg = "RHS contains NaN or Inf values"
            logger.warning(error_msg)
            status = IterativeLinearSolverFailure(
                reason=error_msg, solve_time=0.0, construct_time=0.0
            )
            dtype = linear_system.rhs.dtype
            return np.full(linear_system.rhs.size, np.nan, dtype=dtype), status

        block_linear_system = self.construct_block_linear_system(linear_system)

        if self.solver_selector is None:
            return self._solve_linear_system(
                block_linear_system, solver_options=self.solver_options
            )
        else:
            return self._solve_linear_system_with_solver_selection(block_linear_system)

    def construct_block_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> BlockLinearSystem:
        """Construct and transform a block representation of a linear system.

        Parameters:
            linear_system: PorePy's container for a linear system. If the attribute
                :attr:`delete_matrices` is `True`, it modifies the linear system by
                dropping the matrix reference.

        Returns:
            The linear system transformed to the ordering and scaling expected by the
            configured iterative solver.

        """
        assert linear_system.matrix is not None, (
            "The linear system must contain an assembled matrix."
        )
        if self.dof_manager is None:
            self.dof_manager = self.construct_dof_manager(
                equation_indexer=linear_system.equation_indexer,
                variable_indexer=linear_system.variable_indexer,
            )

        # Creating the indices of DoFs for the BlockLinearSystem class.
        block_linear_system = BlockLinearSystem(
            mat=linear_system.matrix,
            rhs=linear_system.rhs,
            indexer=LinearSystemIndexer(
                dofs_row=self.dof_manager.eq_dofs(),
                dofs_col=self.dof_manager.var_dofs(),
                group_names_row=self.dof_manager.equation_names(),
                group_names_col=self.dof_manager.variable_names(),
            ),
        )

        # Delete the original linear system to save memory unless instructed not to.
        if self.delete_matrices:
            linear_system.release_matrix_reference()

        # Apply transformations to the linear systems before passing it to the solver.
        for transformation in self.transformations:
            block_linear_system = transformation.transform_matrix_rhs(
                block_linear_system, dof_manager=self.dof_manager
            )

        return block_linear_system

    def _solve_linear_system_with_solver_selection(
        self, linear_system: BlockLinearSystem
    ) -> tuple[np.ndarray, LinearSolverStatus]:
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
        assert self.solver_selector is not None, (
            "Solver selection was requested without a configured solver selector."
        )
        characteristics = np.array([])  # Not implemented yet.

        # Perform the ML selection.
        solver_selection_opts, solver_id = (
            self.solver_selector.select_linear_solver_scheme(
                characteristics=characteristics, active_solver_idx=-1
            )
        )

        # Check that the ML model does not override the manually provided options. Warn
        # if so and merge the options into a single dict.
        intersecting_keys = set(self.solver_options).intersection(solver_selection_opts)
        if len(intersecting_keys) > 0:
            logger.warning(
                "Solver selection override manually provided solver options:",
                intersecting_keys,
            )
        solver_selection_opts = self.solver_options | solver_selection_opts

        # Solve the linear system.
        solution, status = self._solve_linear_system(
            linear_system=linear_system, solver_options=solver_selection_opts
        )

        # Providing feedback to the ML model.
        self.solver_selector.provide_performance_feedback(
            solve_time=status.solve_time,
            construct_time=status.construct_time,
            success=(
                status.petsc_converged_reason is not None
                and status.petsc_converged_reason > 0
            ),
        )
        return solution, status

    def _solve_linear_system(
        self, linear_system: BlockLinearSystem, solver_options: dict
    ) -> tuple[np.ndarray, IterativeLinearSolverSuccess | IterativeLinearSolverFailure]:
        """Assembles the PETSc linear solver and solves the linear system.

        Parameters:
            solver_options: Manually provided solver options.

        Returns:
            A tuple of two elements:
                - Solution array of the linear system.
                - Status containing timing, iteration, and PETSc convergence details.
        """
        assert (
            self.dof_manager is not None and self.petsc_ksp_pc_configuration is not None
        ), "The linear solver must be initialized with a model before solving."

        t0 = time()
        try:
            solver = initialize_petsc_ksp(
                block_linear_system=linear_system,
                dof_manager=self.dof_manager,
                petsc_ksp_pc_configuration=self.petsc_ksp_pc_configuration,
                user_options=solver_options,
                delete_matrices=self.delete_matrices,
            )
        except Exception:
            error_msg = (
                "Failed to build a PETSc linear solver based on the given linear system"
            )
            logger.exception(error_msg)
            nans = np.full(
                self.dof_manager.num_dofs, np.nan, dtype=linear_system.rhs.dtype
            )
            return nans, IterativeLinearSolverFailure(
                solve_time=0.0, construct_time=time() - t0, reason=error_msg
            )
        construct_time = time() - t0
        logger.info("Linear solver constructed in %.2f seconds.", construct_time)

        # Project the right hand side to the local block matrix ordering, as was done
        # for the block matrix during assembly. We need to do this on the reordered rhs
        # vector (with contact eqs reordered).
        t0 = time()
        x = solver.solve(linear_system.rhs)
        solve_time = time() - t0
        num_it = len(solver.get_residuals())
        info: PETScKspConvergedReason = solver.ksp.getConvergedReason()
        logger.info(
            "Linear system solved in %.2f seconds with %d iterations, "
            "converged reason: %d.",
            solve_time,
            num_it,
            info,
        )

        if info <= 0:
            logger.warning(
                "Linear solver did not converge. Reason: %d. "
                "Check the solver options and the problem setup. "
                "See detailed description of PETSc error codes: "
                "https://petsc.org/release/manualpages/KSP/KSPConvergedReason/",
                info,
            )
            status = IterativeLinearSolverFailure(
                reason="Linear solver did not converge.",
                solve_time=solve_time,
                construct_time=construct_time,
                petsc_converged_reason=info,
                num_krylov_iters=num_it,
            )
        else:
            status = IterativeLinearSolverSuccess(
                solve_time=solve_time,
                construct_time=construct_time,
                petsc_converged_reason=info,
                num_krylov_iters=num_it,
            )

        # Transform the solution back to PorePy ordering.
        for transformation in reversed(self.transformations):
            x = transformation.transform_solution(x)

        return np.atleast_1d(x), status


def default_preconditioner_factory(
    model: pp.PorePyModel,
) -> Callable[[], LinearSolverConfiguration]:
    is_tpsa = isinstance(model, pp.poromechanics.TpsaPoromechanicsMixin)
    if isinstance(model, pp.SinglePhaseFlow):
        return mass_balance_factory
    if isinstance(model, pp.MomentumBalance):
        return momentum_balance_factory
    if isinstance(model, pp.MassAndEnergyBalance):
        return th_factory
    if isinstance(model, pp.Poromechanics):
        return hm_factory
    if isinstance(model, pp.Thermoporomechanics) and not is_tpsa:
        return thm_factory
    if isinstance(model, pp.Thermoporomechanics) and is_tpsa:
        return thm_tpsa_factory
    raise ValueError(f"Unknown model:", type(model))
