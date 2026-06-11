"""Tests for the linear-system transformation classes in transformations.py.

Coverage:
  - PorePyArrangementTransformation: block reordering
  - SchurComplementReduction: reduced system correctness, solution reconstruction
  - ContactLinearTransformation: contact singularity, correctness
  - ScaleSpecificVolume: group scaling and isolation, solution correctness (THM only)
"""

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sp
from porepy.applications.test_utils.models import add_mixin
from scipy.sparse.linalg import spsolve, inv

from pp_solvers.block_linear_system import (
    BlockLinearSystem,
    LinearSystemIndexer,
)
from pp_solvers.dof_manager import DofManager
from pp_solvers.equation_variable_groups import (
    ContactMechanicsGroup,
    EnergyBalanceTemperatureGroup,
    EquationVariableGroup,
    MassBalancePressureFracturesGroup,
    MassBalancePressureIntersectionsGroup,
)
from pp_solvers.mat_utils import inv_block_diag
from pp_solvers.solver_mixin import IterativeSolverMixin, LinearSolverParams
from pp_solvers.transformations import (
    ContactLinearTransformation,
    LinearSystemTransformation,
    PorePyArrangementTransformation,
    ScaleSpecificVolume,
    SchurComplementReduction,
)
from testing_utils import (
    MockDofManager,
    generate_block_linear_system,
)


@pytest.fixture(scope="module", params=[False, True])
def with_fractures(request) -> bool:
    return request.param


@pytest.fixture(scope="module", params=["flow", "THM"])
def model_kind(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def model(model_kind: str, with_fractures: bool):
    """Instantiate a model for the test suites in this file."""
    match model_kind:
        case "flow":
            model_type = pp.SinglePhaseFlow
        case "THM":
            model_type = pp.Thermoporomechanics
        case default:
            raise ValueError(default)

    class TailoredClass(
        IterativeSolverMixin, pp.model_geometries.SquareDomainOrthogonalFractures
    ):
        """Common base class for all models in this test suite."""

        def meshing_arguments(self):
            return {"cell_size": self.params["cell_size"]}

    params = {
        "cell_size": 0.5,
        "cartesian": True,
        "fracture_indices": [0, 1] if with_fractures else [],
        "linear_solver": LinearSolverParams(
            delete_matrices=False,
        ),
    }
    model_class = add_mixin(TailoredClass, model_type)
    model = model_class(params=params)
    model.prepare_simulation()
    model.before_time_step()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    model.assemble_linear_system()

    mat, rhs = model.linear_system
    rhs[:] = np.arange(rhs.size) + 1

    return model


@pytest.fixture
def linear_system(model: IterativeSolverMixin):
    mat, rhs = model.linear_system
    dof_manager: DofManager = model._dof_manager
    return BlockLinearSystem(
        mat=mat.copy(),
        rhs=rhs.copy(),
        indexer=LinearSystemIndexer(
            dofs_row=dof_manager.eq_dofs(),
            dofs_col=dof_manager.var_dofs(),
            group_names_row=dof_manager.equation_names(),
            group_names_col=dof_manager.variable_names(),
        ),
    )


def test_porepy_arrangement_transformation(
    model: IterativeSolverMixin,
    linear_system: BlockLinearSystem,
    model_kind: str,
    with_fractures: bool,
):
    """Solve the unpermuted system, permute it, solve, permute back, and compare."""
    dof_manager: DofManager = model._dof_manager

    sol = spsolve(linear_system.mat.tocsc(), linear_system.rhs)

    transformation = PorePyArrangementTransformation()
    permuted_linear_system = transformation.transform_matrix_rhs(
        linear_system, dof_manager=dof_manager
    )
    # The flow model without fractures has a single group, so should be no difference.
    should_permute = model_kind != "flow" or with_fractures
    assert np.allclose(linear_system.rhs, permuted_linear_system.rhs) != should_permute

    sol_permuted = spsolve(
        permuted_linear_system.mat.tocsc(), permuted_linear_system.rhs
    )
    assert np.allclose(sol, sol_permuted) != should_permute

    actual_sol = transformation.transform_solution(sol_permuted)
    np.testing.assert_allclose(sol, actual_sol)


@pytest.mark.parametrize("block_size", [1, 2, 3])
@pytest.mark.parametrize(
    "primary_secondary_groups",
    [
        (["g1", "g2", "g3"], ["g4", "g5"]),
        (["g4", "g5"], ["g1", "g2", "g3"]),
    ],
)
def test_schur_complement_reduction(
    block_size: int, primary_secondary_groups: tuple[list, list]
):
    """Solve via Schur complement reduction and verify the residual is near zero."""
    primary_groups = primary_secondary_groups[0]
    secondary_groups = primary_secondary_groups[1]
    # Some arbitrary numbers of dofs.
    num_dofs_primary = [5 + i for i in range(len(primary_groups))]
    # Same for secondary, but multiplied by block_size for no division remainder.
    num_dofs_secondary = [(3 + i) * block_size for i in range(len(secondary_groups))]
    A = generate_block_linear_system(num_dofs_primary + num_dofs_secondary)
    dof_manager = MockDofManager(
        block_linear_system=A, groups=primary_groups + secondary_groups
    )

    # making secondary groups block-diagonal
    secondary_idx = dof_manager.indices_of_groups(secondary_groups)
    num_blocks = sum(num_dofs_secondary) // block_size
    mats = [sp.csr_matrix(np.ones((block_size, block_size))) for i in range(num_blocks)]
    mask = sp.block_diag(mats, format="csr")
    A[secondary_idx] = A[secondary_idx].mat.multiply(mask)

    reduction = SchurComplementReduction(
        primary_groups=primary_groups,
        secondary_groups=secondary_groups,
        invertor=lambda mat: inv_block_diag(mat, nd=block_size),
    )
    S = reduction.transform_matrix_rhs(A, dof_manager)

    # The reduced system covers only the primary groups.
    assert S.mat.shape == (sum(num_dofs_primary), sum(num_dofs_primary))
    x_reduced = spsolve(S.mat.tocsc(), S.rhs)
    x_full = reduction.transform_solution(x_reduced)

    np.testing.assert_allclose(A.mat @ x_full, A.rhs, atol=1e-16)


def test_contact_transformation(
    model: IterativeSolverMixin,
    linear_system: BlockLinearSystem,
    model_kind: str,
    with_fractures: bool,
):
    """Solve the unpermuted system, permute, solve, permute back, and compare.

    Also checks that the contact submatrix is singular before the transformation
    and non-singular after, and that the rest of the matrix is unchanged.
    """
    dof_manager: DofManager = model._dof_manager

    sol = spsolve(linear_system.mat.tocsc(), linear_system.rhs)

    # Contact transformation requires that groups are sorted, so we first appply the
    # PorePyArrangementTransformation, which does that.
    transformations = [PorePyArrangementTransformation(), ContactLinearTransformation()]
    permuted_linear_system = linear_system
    for transformation in transformations:
        permuted_linear_system = transformation.transform_matrix_rhs(
            permuted_linear_system, dof_manager=dof_manager
        )
    # The difference should be only for the THM model with fractures.
    should_do_something = model_kind != "flow" and with_fractures

    if should_do_something:
        contact_idx = dof_manager.indices_of_groups([ContactMechanicsGroup()])
        contact_submat = linear_system[contact_idx]
        permuted_contact_submat = permuted_linear_system[contact_idx]
        # Matrix should be singular.
        with pytest.raises(RuntimeError):
            _ = inv(contact_submat.mat.tocsc())
        # Transformed matrix should be non-singular.
        _ = inv(permuted_contact_submat.mat.tocsc())

    # Check that the rest of the matrix did not change.
    unchanged_groups = [g for g in dof_manager.groups() if g != ContactMechanicsGroup()]
    unchanged_groups_idx = dof_manager.indices_of_groups(unchanged_groups)
    original_submat = linear_system[unchanged_groups_idx].mat
    submat_after_transformation = permuted_linear_system[unchanged_groups_idx].mat
    assert (original_submat - submat_after_transformation).data.size == 0

    sol_permuted = spsolve(
        permuted_linear_system.mat.tocsc(), permuted_linear_system.rhs
    )
    actual_sol = sol_permuted
    for transformation in reversed(transformations):
        actual_sol = transformation.transform_solution(actual_sol)
    np.testing.assert_allclose(sol, actual_sol)


@pytest.mark.parametrize(
    "groups_to_scale",
    [
        [EnergyBalanceTemperatureGroup()],
        [MassBalancePressureIntersectionsGroup(), MassBalancePressureFracturesGroup()],
    ],
)
def test_scale_specific_volume(
    model: IterativeSolverMixin,
    linear_system: BlockLinearSystem,
    model_kind: str,
    with_fractures: bool,
    groups_to_scale: list[EquationVariableGroup],
):
    """Solve the unpermuted THM system, permute, solve, permute back, and compare.

    Also checks that the targeted groups are actually scaled and the rest of the
    matrix is unchanged. Skipped for the flow model (it does not have tested groups).
    """
    if model_kind == "flow":
        return
    dof_manager: DofManager = model._dof_manager

    sol = spsolve(linear_system.mat.tocsc(), linear_system.rhs)

    # ScaleSpecificVolume requires that groups are sorted, so we first appply the
    # PorePyArrangementTransformation, which does that.
    transformations = [
        PorePyArrangementTransformation(),
        ScaleSpecificVolume(groups=groups_to_scale),
    ]
    permuted_linear_system = linear_system
    for transformation in transformations:
        permuted_linear_system = transformation.transform_matrix_rhs(
            permuted_linear_system, dof_manager=dof_manager
        )

    # Should change only if fractures are present (specific volume != 1).
    should_change = with_fractures

    # Check that the groups we want to scale actually scaled.
    groups_changed_idx = dof_manager.indices_of_groups(groups_to_scale)
    changed_submat = permuted_linear_system[groups_changed_idx].mat
    original_submat = linear_system[groups_changed_idx].mat
    assert (
        np.allclose((changed_submat - original_submat).data, 0, atol=1e-16)
        != should_change
    )

    # Check that the groups we do not want to change remain the same.
    groups_not_changed = [g for g in dof_manager.groups() if g not in groups_to_scale]
    groups_not_changed_idx = dof_manager.indices_of_groups(groups_not_changed)
    unchanged_submat = permuted_linear_system[groups_not_changed_idx].mat
    original_submat = linear_system[groups_not_changed_idx].mat
    assert (unchanged_submat - original_submat).size == 0

    sol_permuted = spsolve(
        permuted_linear_system.mat.tocsc(), permuted_linear_system.rhs
    )
    actual_sol = sol_permuted
    for transformation in reversed(transformations):
        actual_sol = transformation.transform_solution(actual_sol)
    np.testing.assert_allclose(sol, actual_sol)


@pytest.mark.parametrize(
    "transformation",
    [
        ContactLinearTransformation(),
        ScaleSpecificVolume(groups=[EnergyBalanceTemperatureGroup()]),
    ],
)
def test_transformations_with_unsorted_dofs(
    transformation: LinearSystemTransformation,
    linear_system: BlockLinearSystem,
    model_kind: str,
    model: IterativeSolverMixin,
):
    """Transformations that require sorted DoFs raise ValueError when given an
    unsorted (raw PorePy) linear system."""
    if model_kind == "flow":
        return
    dof_manager: DofManager = model._dof_manager

    with pytest.raises(
        ValueError,
        match="Use .+ after PorePyArrangementTransformation.",
    ):
        _ = transformation.transform_matrix_rhs(linear_system, dof_manager)
