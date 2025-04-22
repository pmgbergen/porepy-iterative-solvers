import sys
import time
from functools import cached_property
from typing import Sequence
import logging

import numpy as np
import porepy as pp
import scipy.sparse as sps

from .block_matrix import BlockMatrixStorage, FieldSplitScheme, KSPScheme
from .stats import LinearSolveStats, StatisticsSavingMixin


logger = logging.getLogger(__name__)

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
    def var_dofs(self) -> list[np.ndarray]:
        """Variable degrees of freedom (columns of the Jacobian) in the PorePy order
        (how they are arranged in the model).

        Returns:
            List of numpy arrays. Each array contains the global degrees of freedom for
                one variable on one grid and provides the fine-scale (actual column
                indices) of the variable.

        """
        var_dofs: list[np.ndarray] = []
        for var in self.equation_system.variables:
            var_dofs.append(self.equation_system.dofs_of([var]))
        return var_dofs

    @cached_property
    def eq_dofs(self) -> list[np.ndarray]:
        """Equation indices (rows of the Jacobian) in the order defined by the PorePy
        EquationSystem.

        Returns:
            List of numpy arrays. Each list entry correspond to one equation on one
                grid, and provides the fine-scale (actual row indices) of the equation.

        """
        eq_dofs: list[np.ndarray] = []
        offset = 0
        for data in self.equation_system._equation_image_space_composition.values():
            local_offset = 0
            for dofs in data.values():
                eq_dofs.append(dofs + offset)
                local_offset += len(dofs)
            offset += local_offset
        return eq_dofs

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

        bmat = BlockMatrixStorage(
            mat=self.linear_system[0],
            global_dofs_row=self.eq_dofs,
            global_dofs_col=self.var_dofs,
            groups_to_blocks_row=self.equation_groups,
            groups_to_blocks_col=self.variable_groups,
            group_names_row=self.group_row_names(),
            group_names_col=self.group_col_names(),
        )

        self.bmat = bmat

    def solve_linear_system(self) -> np.ndarray:
        """Solve the linear system using the defined iterative scheme.

        Raises:
            ValueError: If the solver construction or solve fails.

        """
        t_0 = time.time()
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

        scheme = self.make_solver_scheme()
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

        logger.info(f"Solved linear system in {time.time() - t_0:.2e} seconds.")

        # Write statistics
        self._linear_solve_stats.petsc_converged_reason = info
        self._linear_solve_stats.krylov_iters = len(solver.get_residuals())
        return np.atleast_1d(sol)


def get_variables_group_ids(
    model: pp.PorePyModel,
    md_variables_groups: Sequence[
        Sequence[pp.ad.MixedDimensionalVariable | pp.ad.Variable]
    ],
) -> list[list[int]]:
    """Used to assemble the index that will later help accessing the submatrix
    corresponding to a group of variables, which may include one or more variable.

    Example: Group 0 corresponds to the pressure on all the subdomains. It will contain
    indices [0, 1, 2] which point to the pressure variable dofs on sd1, sd2 and sd3,
    respectively. Combination of different variables in one group is also possible.

    Parameters:
        model: The PorePy model. The model should have the EquationSystem defined.
        md_variables_groups: The order of the groups of variables. Each group is a
            sequence of variables (either MixedDimensionalVariable or Variable).

    Returns:
        List of lists of integers. Each inner list contains the indices of the variables
            in defined in the respective item in md_variables_groups.

    """
    # Create a 0-based index for each variable.
    variable_to_idx = {var: i for i, var in enumerate(model.equation_system.variables)}
    indices = []
    for md_var_group in md_variables_groups:
        group_idx = []
        for md_var in md_var_group:
            # If we ever get a variable in here, we need to handle it directly, and not
            # call sub_vars.
            assert isinstance(md_var, pp.ad.MixedDimensionalVariable)
            group_idx.extend([variable_to_idx.pop(var) for var in md_var.sub_vars])
        indices.append(group_idx)
    assert len(variable_to_idx) == 0, "Some variables are not used."
    return indices


def get_equations_group_ids(
    model: pp.PorePyModel,
    equations_group_order: Sequence[Sequence[tuple[str, pp.GridLikeSequence]]],
) -> list[list[int]]:
    """Used to assemble the index that will later help accessing the submatrix
    corresponding to a group of equation, which may include one or more equation.

    Parameters:
        model: The PorePy model. The model should have the EquationSystem defined.
        equations_group_order: The order of the groups of equations. Each group is a
            sequence of tuples. Each tuple contains the name of the equation and the
            domain where it is applied.

    Returns:
        List of lists of integers. Each inner list contains the indices of the equations
            in defined in the respective item in equations_group_order. The indices
            refer to the block indices defined in
            model.equation_system._equation_image_space_composition.

    """
    # Assign a unique index to each equation-domain pair.
    equation_to_idx: dict[tuple[str, pp.GridLike], int] = {}
    idx: int = 0
    for (
        eq_name,
        domains,
    ) in model.equation_system._equation_image_space_composition.items():
        for domain in domains:
            equation_to_idx[(eq_name, domain)] = idx
            idx += 1

    indices: list[list[int]] = []
    # The outer loop define different groups of equations (to become blocks in the
    # block matrix).
    for group in equations_group_order:
        group_idx = []
        # Items in the group will contain a single equation defined on one or more
        # domains (subdomains or interfaces). Loop over equations an over all their
        # domains to add the indices to the group.
        for eq_name, domains_of_eq in group:
            for domain in domains_of_eq:
                if (eq_name, domain) in equation_to_idx:
                    group_idx.append(equation_to_idx.pop((eq_name, domain)))
        indices.append(group_idx)

    # TODO EK: Added this assert just to verify that my understanding of the function
    # is correct. Delete it later.
    assert len(indices) == len(equations_group_order)
    assert len(equation_to_idx) == 0, "Some equations are not used."

    return indices
