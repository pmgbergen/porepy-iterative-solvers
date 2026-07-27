"""This module defines the class DofManager - a layer of translation between a PorePy
model and the equation-variable groups defined in `equation_variable_groups.py`.

Consider the linear system split by two groups: `mass_balance_pressure_group` and
`mechanics_group`, which can be taken from `pp_solvers.DefaultEquationVariableGroups`:
```
linear_system = model.equation_system.assemble(...)

dof_manager = DofManager(
    model=model,
    equation_indexer=linear_system.equation_indexer,
    variable_indexer=linear_system.variable_indexer,
    groups=[mass_balance_pressure_group, mechanics_group],
)
```
The DofManager can tell us:
- Is this group present in the problem?
- If yes, what PorePy DoFs correspond to this equation?
```
mass_balance_group = dof_manager.indices_of_groups([mass_balance_pressure_group])[0]
dofs_mass_balance_eq = dof_manager.eq_dofs()[mass_balance_group]
dofs_pressure_var = dof_manager.var_dofs()[mass_balance_group]
```
These dofs now can be used to slice the matrix, produced by the PorePy model:
```
mat = linear_system.matrix
rhs = linear_system.rhs
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

        self.num_dofs: int = equation_indexer.num_dofs
        """Total number of DoFs in the linear system."""

        # Collecting and validation equation and variable DoFs. This ensures no
        # duplicates. More validation regarding meaningful dofs is made in
        # BlockLinearSystem constructor. The contact mechanics special case is treated
        # here.
        self._eq_dofs, self.equations_per_group = _collect_group_dofs(
            indexer=equation_indexer,
            tags_by_group=[group.equation_tag for group in groups],
            model=model,
        )
        self._var_dofs, self.variables_per_group = _collect_group_dofs(
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


def _collect_group_dofs[
    EquationOrVariableType: (pp.ad.EquationOnDomain, pp.ad.Variable)
](
    indexer: pp.ad.Indexer[EquationOrVariableType],
    tags_by_group: list[pp.solvers.OperatorTag[EquationOrVariableType]],
    model: pp.PorePyModel,
) -> tuple[list[np.ndarray], list[list[EquationOrVariableType]]]:
    """Collect DoFs in groups and validate a partition.

    Treatment of the contact mechanics special case is localized here, see the docstring
    of :class:`DofManager`.

    Parameters:
        indexer: Indexer of the corresponding linear systems.
        tags_by_groups: Equation or variable tags, one per group.
        model: PorePy model.

    Raises:
        ValueError: If an equation / variable in the indexer is not covered by any
            group.
        ValueError: If groups have overlapping equations / variables.

    Returns:
        A tuple of 2 elements:
        - A list of numpy arrays, each is the DoF indices of the corresponding group.
        - A list of lists of atomic equations / variable. Each inner lists corresponds
            to a group.

    """
    dofs_by_groups: list[np.ndarray] = []
    operators_by_groups: list[list[EquationOrVariableType]] = []

    for tag in tags_by_group:
        if tag != CONTACT_MECHANICS_EQUATION_TAG:
            # A general case. Get atomic equations / variables corresponding to the tag.
            selected, _ = indexer.filter_by_tags(tags=[tag], model=model)
            # Get and concatenate the dofs of these atomic equations / variables.
            dofs_selected = concatenate_dof_indices(
                [indexer.operators_to_dofs[operator] for operator in selected]
            )
        else:
            # Contact mechanics special case. Need to treat normal and tangential
            # separately. The same logic as the general case applied twice.

            # Get atomic equations for the normal contact mecahnics equation.
            normal_operators, _ = indexer.filter_by_tags(
                tags=[pp.solvers.DefaultEquationTags.normal_fracture_deformation],
                model=model,
            )
            # Get dofs for the normal equation.
            normal_dofs = concatenate_dof_indices(
                [indexer.operators_to_dofs[op] for op in normal_operators]
            )
            # Get atomic equations for the tangential equation.
            tangential_operators, _ = indexer.filter_by_tags(
                tags=[pp.solvers.DefaultEquationTags.tangential_fracture_deformation],
                model=model,
            )
            # Get dofs for the tangential equation.
            tangential_dofs = concatenate_dof_indices(
                [indexer.operators_to_dofs[op] for op in tangential_operators]
            )
            # Apply the permutation.
            dofs_selected = _permute_contact_dofs(
                normal_dofs=normal_dofs,
                tangential_dofs=tangential_dofs,
            )
            # Register normal and tangential contact equation as seen.
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
        normal_dofs: An flat array with the normal equation dofs C_n.
        tangentail_dofs: An flat array with the tangential equation dofs. In the 2D
            case, must be of the same shape as `normal_dofs`. In the 3D case, must be
            double the size of the `tangential_dofs`.

    Raises:
        ValueError: If the shape of `tangential_dofs` does not align with the shape of
            `normal_dofs`.

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
