import sys
import time
from functools import cached_property
from typing import Sequence

import numpy as np
import porepy as pp
import scipy.sparse as sps

from .block_matrix import BlockMatrixStorage, FieldSplitScheme, KSPScheme
from .stats import LinearSolveStats, StatisticsSavingMixin


__all__ = [
    "IterativeLinearSolver",
]


class IterativeLinearSolver(StatisticsSavingMixin, pp.PorePyModel):
    """Mixin for iterative linear solvers."""

    nd: int
    mdg: pp.MixedDimensionalGrid
    params: dict
    equation_system: pp.ad.EquationSystem
    linear_system: tuple[sps.spmatrix, np.ndarray]

    _linear_solve_stats = LinearSolveStats()
    """A placeholder to statistics. The solver mixin only writes in it, not reads."""

    bmat: BlockMatrixStorage
    """The current Jacobian."""

    @cached_property
    def variable_groups(self) -> list[list[int]]:
        raise NotImplementedError("This method should be implemented in the subclass.")

    @cached_property
    def equation_groups(self) -> list[list[int]]:
        """Define the groups of equation in the specific order, that we will use in
        the block Jacobian to access the submatrices.

        Returns:
            List of lists of integers. Each list contains the indices of the equations
                in the group.

        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def group_row_names(self) -> list[str] | None:
        """Return the names of the equation groups. Used for visualization purposes.
        See subclasses for examples.

        Returns:
            List of strings. Each string is the name of the group of equations.

        """
        return None

    def group_col_names(self) -> list[str] | None:
        """Return the names of the column groups. Used for visualization purposes. See
        subclasses for examples.

        Returns:
            List of strings. Each string is the name of a group of variables.

        """
        return None

    def make_solver_scheme(self) -> FieldSplitScheme | KSPScheme:
        # TOOD: Should this be an abstract method?
        raise NotImplementedError("This method should be implemented in the subclass.")

    def assemble_linear_system(self) -> None:
        """Assemble the linear system. Also build a block matrix representation of the
        matrix.
        """
        super().assemble_linear_system()  # type: ignore[misc]

        row_permutation = self._linear_solver_scheme_maker.row_indices()
        mat, rhs = self.linear_system

        # Apply the `contact_permutation`.
        mat = mat[row_permutation]
        rhs = rhs[row_permutation]

        scheme_maker = self._linear_solver_scheme_maker

        bmat = BlockMatrixStorage(
            mat=self.linear_system[0],
            global_dofs_row=scheme_maker.eq_dofs,
            global_dofs_col=scheme_maker.var_dofs,
            groups_to_blocks_row=scheme_maker.equation_groups,
            groups_to_blocks_col=scheme_maker.variable_groups,
            group_names_row=self.group_row_names(),  # TODO: Move to the scheme
            group_names_col=self.group_col_names(),
        )

        self.bmat = bmat

    def solve_linear_system(self) -> np.ndarray:
        """Solve the linear system using the defined iterative scheme.

        Raises:
            ValueError: If the solver construction or solve fails.

        """
        # Check that rhs is finite.
        mat, rhs = self.linear_system
        if not np.all(np.isfinite(rhs)):
            # TODO: We should rather raise an exception here and let the caller handle
            # it.
            self._linear_solve_stats.krylov_iters = 0
            result = np.zeros_like(rhs)
            result[:] = np.nan
            return result

        # Check if we reached steady state and no solve needed.
        # residual_norm = self.compute_residual_norm(rhs, None)
        # if residual_norm < self.params["nl_convergence_tol_res"]:
        #     result = np.zeros_like(rhs)
        #     return result

        config = self.params.get("linear_solver_config", {})

        if config.get("save_matrix", False):
            self.save_matrix_state()

        scheme = self._linear_solver_scheme_maker.make_solver_scheme()
        # Construct the solver.
        bmat = self.bmat[scheme.get_groups()]

        t0 = time.time()
        try:
            solver = scheme.make_solver(bmat)
        except Exception as e:
            self.save_matrix_state()
            raise ValueError("Solver construction failed") from e

        if config.get("logging", False):
            print("Construction took:", round(time.time() - t0, 2))

        # Permute the rhs groups to match mat_permuted.
        rhs_local = bmat.project_rhs_to_local(rhs)

        t0 = time.time()
        try:
            sol_local = solver.solve(rhs_local)
        except Exception as e:
            self.save_matrix_state()
            raise ValueError("Solver solve failed") from e

        if config.get("logging", False):
            print("Solve took:", round(time.time() - t0, 2))

        info = solver.ksp.getConvergedReason()

        # Permute the solution groups to match the original porepy arrangement.
        sol = bmat.project_solution_to_global(sol_local)

        # Verify that the original problem is solved and we did not do anything wrong.
        true_residual_nrm_drop = abs(mat @ sol - rhs).max() / abs(rhs).max()

        if info <= 0:
            # TODO: Raise an exception here and let the caller handle it.
            print(f"GMRES failed, {info=}", file=sys.stderr)
            if info == -9:
                sol[:] = np.nan
        else:
            if true_residual_nrm_drop >= 1:
                # TODO: This should be a warning.
                print("True residual did not decrease")

        # Write statistics
        self._linear_solve_stats.petsc_converged_reason = info
        self._linear_solve_stats.krylov_iters = len(solver.get_residuals())
        return np.atleast_1d(sol)

    def _initialize_linear_solver(self) -> None:
        """Initialize the linear solver.

        This method fetches the linear solver scheme class (essentially a factory class
        for a linear solver) from the parameters and creates an instance of it.
        """
        scheme_maker_cls = self.params.get("linear_solver_scheme")
        self._linear_solver_scheme_maker = scheme_maker_cls(self, self.params)
