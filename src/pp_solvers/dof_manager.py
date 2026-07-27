"""This module defines the class DofManager - a layer of translation between a PorePy
model and the equation-variable groups defined in `equation_variable_groups.py`.
Given the `MassBalancePressureGroup()` as example, the DofManager can tell us:
- Is this group present in the problem?
- If yes, what PorePy DoFs correspond to this equation?

This is done by:
```
dof_manager = DofManager(...)
mass_balance_group = dof_manager.indices_of_groups([MassBalancePressureGroup()])[0]

dofs_mass_balance_eq = dof_manager.eq_dofs()[mass_balance_group]
dofs_pressure_var = dof_manager.var_dofs()[mass_balance_group]

# These dofs now can be used to slice the matrix, produced by the PorePy model:
mat, rhs = model.linear_system

submatrix_mass_balance_pressure = mat[dofs_mass_balance_eq, dofs_pressure_var]
rhs_mass_balance = rhs[dofs_mass_balance_eq]
```

"""

from __future__ import annotations

from collections import Counter

import numpy as np
import porepy as pp

from pp_solvers.block_linear_system import concatenate_dof_indices
from pp_solvers.equation_variable_groups import (
    CONTACT_MECHANICS_EQUATION_TAG,
    EquationVariableGroup,
)

__all__ = ["DofManager"]


class DofManager:
    """Takes care of translation of PorePy equations and variables (from EquationSystem
    format) to group indices, suited to construct a `BlockLinearSystem` to be solved
    with an iterative solver.

    One particular "exception" or "edge case" is the contact mechanics equation, which
    requires extra care, since PorePy treats tangential and normal equations as
    different entities, and it requires reordering before solving with an iterative
    solver. This reordering is done in this class.

    A general problem would outsource the contact reordering to a subclass, but right
    now we have no reason to do so, since it's the only known exception.

    """

    def __init__(
        self,
        model: pp.PorePyModel,
        equation_indexer: pp.ad.EquationIndexer,
        variable_indexer: pp.ad.VariableIndexer,
        groups: list[EquationVariableGroup],
    ):
        """Constructs the DoFs mapping for the passed groups of equations and
        variables.

        Raises:
            ValueError: If a group defines a variable or an equation on the same domain
            more than once.

        """
        self.model: pp.PorePyModel = model
        """The PorePy model of the given problem."""

        self._groups: list[EquationVariableGroup] = groups
        """Groups that define the DofManager."""

        # TODO YZ: ensure docstring up to date
        # Collecting and validation equation and variable groups. This ensures no
        # duplicates. More validation regarding meaningful dofs is made in
        # BlockLinearSystem constructor.
        # Assembling DoFs that correspond to each group:
        # 1. PorePy provides us with a list of arrays, each array corresponds to the
        #   DoFs of a single equation/variable on a single (not mixed-dimensional)
        #   grid.
        # 2. We construct a mapping from what PorePy provided to the groups, which
        #   equations/variables on which collection grids we will treat monolithically
        #   in this DofManager.
        # 3. We concatenate arrays of DoFs, so that now a single array correspond to a
        #   single group.

        self._eq_dofs, self._equations_per_group = _collect_group_dofs(
            indexer=equation_indexer,
            tags_by_group=[group.equation_tag for group in groups],
            model=model,
        )
        self._var_dofs, self._variables_per_group = _collect_group_dofs(
            indexer=variable_indexer,
            tags_by_group=[group.variable_tag for group in groups],
            model=model,
        )

    def groups(self) -> list[EquationVariableGroup]:
        """Groups of equations and variables that define the DofManager."""
        return self._groups

    def indices_of_groups(self, groups: list[EquationVariableGroup]) -> list[int]:
        """Return unique numerical identifiers of ``groups``.

        Raises:
            ValueError: If a group is absent or repeated.

        """
        indices = [self._groups.index(x) for x in groups]
        if len(indices) != len(set(indices)):
            # YZ does not see a situation when this can be desired behavior, but
            # clearly sees how it can lead to bugs later on. This can be caused by
            # comparison of custom EquationVariableGroups, when they are not equal but
            # treated as so.
            raise ValueError(f"Repeating group indices are produced by {self.groups}.")
        return indices

    def equation_names(self) -> list[str]:
        """Get the names of equations in the DofManager. These names are not generally
        equal to the PorePy model equation names, and are intended for debugging and
        matrix visualization.

        Returns:
            A list of strings containing the names of equations in the DofManager.

        """
        return [group.equation_tag.name for group in self._groups]

    def variable_names(self) -> list[str]:
        """Get the names of variables in the DofManager. These names are not generally
        equal to the PorePy model variable names, and are intended for debugging and
        matrix visualization.

        Returns:
            A list of strings containing the names of equations in the DofManager.

        """
        return [group.variable_tag.name for group in self._groups]

    def eq_dofs(self) -> list[np.ndarray]:
        """List of arrays, i-th array contains the DoFs of the i-th equation group."""
        return self._eq_dofs

    def var_dofs(self) -> list[np.ndarray]:
        """List of arrays, i-th array contains the DoFs of the i-th variable group."""
        return self._var_dofs


def _collect_group_dofs[T: (pp.ad.EquationOnDomain, pp.ad.Variable)](
    indexer: pp.ad.Indexer[T],
    tags_by_group: list[pp.solvers.OperatorTag[T]],
    model: pp.PorePyModel,
) -> tuple[list[np.ndarray], list[list[T]]]:
    """Collect indexer DoFs in group and tag order and validate a partition."""
    dofs_by_groups: list[np.ndarray] = []
    operators_by_groups: list[list[T]] = []

    for tag in tags_by_group:
        if not tag == CONTACT_MECHANICS_EQUATION_TAG:
            selected, _ = indexer.filter_by_tags(tags=[tag], model=model)
            dofs_selected = concatenate_dof_indices(
                [indexer.operators_to_dofs[operator] for operator in selected]
            )
        else:
            normal_operators, _ = indexer.filter_by_tags(
                tags=[pp.solvers.DefaultEquationTags.normal_fracture_deformation],
                model=model,
            )
            normal_dofs = concatenate_dof_indices(
                [indexer.operators_to_dofs[op] for op in normal_operators]
            )
            tangential_operators, _ = indexer.filter_by_tags(
                tags=[pp.solvers.DefaultEquationTags.tangential_fracture_deformation],
                model=model,
            )
            tangential_dofs = concatenate_dof_indices(
                [indexer.operators_to_dofs[op] for op in tangential_operators]
            )
            dofs_selected = _permute_contact_dofs(
                normal_dofs=normal_dofs,
                tangential_dofs=tangential_dofs,
            )
            selected = normal_operators + tangential_operators

        operators_by_groups.append(selected)
        dofs_by_groups.append(dofs_selected)

    # Each operator of the indexer must be encountered exactly once.
    encountered = Counter(op for group in operators_by_groups for op in group)
    for operator, count in encountered.items():
        if count < 1:
            raise ValueError(
                "Equation / Variable in the assembled linear system is not covered by "
                f"the requested solver configuration: {operator}."
            )
        if count > 1:
            raise ValueError(
                "Equation / Variable is duplicated in the requested solver "
                f"configuration: {operator}."
            )

    return dofs_by_groups, operators_by_groups


def _permute_contact_dofs(
    normal_dofs: np.ndarray, tangential_dofs: np.ndarray
) -> np.ndarray:
    """Get a permuted array of the DoFs in the contact group.

    This is used to reorder the equations so that the contact equations for single
    fracture cells form a diagonal block.

    The PorePy arrangement in 3D is:

        [C_n^0, C_n^1, ..., C_n^K, C_y^0, C_z^0, C_y^1, C_z^1, ..., C_z^K, C_z^k],

    where `C_n` is a normal component, `C_y` and `C_z` are two tangential
    components. The superscript corresponds to cell index. We permute it to

        `[C_n^0, C_y^0, C_z^0, ..., C_n^K, C_y^K, C_z^K]`.

    Parameters:
        contact_group: The group index of the contact mechanics equations.

    Raises:
        ValueError: If the model dimension is not 2 or 3.

    Returns:
        A numpy array with the permuted DoFs for the contact group.

    """
    if len(normal_dofs) == len(tangential_dofs):
        # 2D
        return np.vstack([normal_dofs, tangential_dofs]).ravel("F")
    elif len(normal_dofs) * 2 == len(tangential_dofs):
        # 3D
        dofs_y = tangential_dofs[::2]
        dofs_z = tangential_dofs[1::2]
        return np.vstack([normal_dofs, dofs_y, dofs_z]).ravel("F")
    else:
        raise ValueError("Unknown dimension, must be either 2D or 3D.")
