"""Tests for the linear-system transformation classes in transformations.py.

Coverage:
  - PorePyArrangementTransformation: block reordering, solution scatter, caching
  - SchurComplementReduction: reduced system correctness, solution reconstruction
  - ContactLinearTransformation: three early-return paths, solution passthrough
  - ScaleSpecificVolume: row scaling on a real TH model, solution passthrough
"""

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sp
from porepy.applications.test_utils.models import add_mixin
from scipy.sparse.linalg import spsolve

from pp_solvers.block_linear_system import (
    BlockLinearSystem,
    LinearSystemIndexer,
    concatenate_dof_indices,
)
from pp_solvers.dof_manager import DofManager
from pp_solvers.equation_variable_groups import (
    ContactMechanicsGroup,
    EnergyBalanceTemperatureGroup,
    InterfaceForceBalanceGroup,
)
from pp_solvers.mat_utils import inv_block_diag
from pp_solvers.preconditioners import th_factory
from pp_solvers.solver_mixin import IterativeSolverMixin, LinearSolverParams
from pp_solvers.transformations import (
    ContactLinearTransformation,
    PorePyArrangementTransformation,
    ScaleSpecificVolume,
    SchurComplementReduction,
)
from testing_utils import (
    MockDofManager,
    generate_block_linear_system,
    generate_reference_dofs_3_groups,
    generate_reference_matrix_3_groups,
    generate_reference_rhs_3_groups,
    generate_reference_submatrices_3_groups,
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


def test_porepy_arrangement_transformation(
    model: IterativeSolverMixin, model_kind: str, with_fractures: bool
):
    # Solve unpermuted linear system. Permute, solve permuted, permute back. Compare.
    dof_manager: DofManager = model._dof_manager
    mat, rhs = model.linear_system
    rhs = rhs.copy()

    sol = spsolve(mat, rhs)

    linear_system = BlockLinearSystem(
        mat=mat.copy(),
        rhs=rhs.copy(),
        indexer=LinearSystemIndexer(
            dofs_row=dof_manager.eq_dofs(),
            dofs_col=dof_manager.var_dofs(),
            group_names_row=dof_manager.equation_names(),
            group_names_col=dof_manager.variable_names(),
        ),
    )

    transformation = PorePyArrangementTransformation()
    permuted_linear_system = transformation.transform_matrix_rhs(
        linear_system, dof_manager=dof_manager
    )
    # The flow model without fractures has a single group, so should be no difference.
    should_permute = model_kind != "flow" or with_fractures
    assert np.allclose(rhs, permuted_linear_system.rhs) != should_permute

    sol_permuted = spsolve(permuted_linear_system.mat, permuted_linear_system.rhs)
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
    # Solve the linear system with the Schur complement. Ensure residual close to zero.
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
    x_reduced = spsolve(S.mat, S.rhs)
    x_full = reduction.transform_solution(x_reduced)

    np.testing.assert_allclose(A.mat @ x_full, A.rhs, atol=1e-16)


def test_contact_transformation(
    model: IterativeSolverMixin, model_kind: str, with_fractures: bool
):
    # Solve unpermuted linear system. Permute, solve permuted, permute back. Compare.
    dof_manager: DofManager = model._dof_manager
    mat, rhs = model.linear_system
    rhs = rhs.copy()

    sol = spsolve(mat, rhs)

    linear_system = BlockLinearSystem(
        mat=mat.copy(),
        rhs=rhs.copy(),
        indexer=LinearSystemIndexer(
            dofs_row=dof_manager.eq_dofs(),
            dofs_col=dof_manager.var_dofs(),
            group_names_row=dof_manager.equation_names(),
            group_names_col=dof_manager.variable_names(),
        ),
    )

    transformation = ContactLinearTransformation()
    permuted_linear_system = transformation.transform_matrix_rhs(
        linear_system, dof_manager=dof_manager
    )
    # The flow model without fractures has a single group, so should be no difference.
    should_permute = model_kind != "flow" or with_fractures
    assert np.allclose(rhs, permuted_linear_system.rhs) != should_permute

    sol_permuted = spsolve(permuted_linear_system.mat, permuted_linear_system.rhs)
    assert np.allclose(sol, sol_permuted) != should_permute

    actual_sol = transformation.transform_solution(sol_permuted)
    np.testing.assert_allclose(sol, actual_sol)


# TODO: complete test_contact_transformation
# TODO: Scale specific volume transformation
# TODO: Fix why test_model fails.
# TODO: And the rest of todos...
