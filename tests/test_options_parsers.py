"""Tests below are used to check that the PETSc preconditioners are correctly assembled.
For this, we compare the expected chunks of text with the result produced by PetscPCView
and PetscKSPView.

"""
# ruff: noqa: E501
# The line silences ruff about long lines in this file. We need long lines to describe
# expected PETSc output.

import re

import numpy as np
import porepy as pp
import pytest
from petsc4py import PETSc
from porepy.applications.test_utils.models import add_mixin

import pp_solvers
from pp_solvers.block_linear_system import BlockLinearSystem
from pp_solvers.dof_manager import DofManager
from pp_solvers.options_parsers import (
    LinearTransformedScheme,
    MultiPhysicsPreconditioner,
    PetscKSPScheme,
)
from pp_solvers.preconditioners import SinglePhysicsPreconditioner


@pytest.fixture(scope="module", params=[False, True])
def with_fractures(request) -> bool:
    return request.param


@pytest.fixture(scope="module", params=["flow", "mechanics", "TH", "HM", "THM"])
def model_kind(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def model(model_kind, with_fractures) -> pp.PorePyModel:
    """Instantiate a model for the test suites in this file."""
    match model_kind:
        case "flow":
            model_type = pp.SinglePhaseFlow
        case "mechanics":
            model_type = pp.MomentumBalance
        case "TH":
            model_type = pp.MassAndEnergyBalance
        case "HM":
            model_type = pp.Poromechanics
        case "THM":
            model_type = pp.Thermoporomechanics
        case default:
            raise ValueError(default)

    class TailoredClass(
        pp_solvers.IterativeSolverMixin,
        pp.model_geometries.SquareDomainOrthogonalFractures,
    ):
        """Common base class for all models in this test suite."""

        def meshing_arguments(self):
            return {"cell_size": self.params["cell_size"]}

    params = {
        "cell_size": 0.25,
        "cartesian": True,
        "fracture_indices": [0, 1] if with_fractures else [],
        "linear_solver": {},  # YZ: Default is "pypardiso". Do we want this collision?
    }
    model_class = add_mixin(TailoredClass, model_type)
    model = model_class(params=params)
    model.prepare_simulation()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    model.assemble_linear_system()
    return model


@pytest.fixture(scope="module")
def solvers(model_kind: str) -> list[SinglePhysicsPreconditioner]:
    match model_kind:
        case "flow":
            return pp_solvers.mass_balance_factory()
        case "mechanics":
            return pp_solvers.momentum_balance_factory()
        case "TH":
            return pp_solvers.th_factory()
        case "HM":
            return pp_solvers.hm_factory()
        case "THM":
            return pp_solvers.thm_factory()
        case default:
            raise ValueError(default)


@pytest.fixture(scope="module")
def dof_manager(
    model: pp.PorePyModel, solvers: list[SinglePhysicsPreconditioner]
) -> DofManager:
    return DofManager(model, solvers)


@pytest.fixture
def jacobian(model: pp.PorePyModel, dof_manager: DofManager) -> BlockLinearSystem:
    bmat = model.bmat
    contact = dof_manager.identify_contact_group()

    # Given the current discretization, the contact group is singular. We simply fill
    # the diagonal to avoid numerical issues. This does not represent realistic physics.
    if contact is not None:
        bmat.set_diagonal([contact], 1)
    return bmat


@pytest.fixture
def pc(jacobian: BlockLinearSystem) -> PETSc.PC:
    petsc_mat = pp_solvers.csr_to_petsc(jacobian.mat)
    pc = PETSc.PC().create()
    pc.setOperators(petsc_mat, petsc_mat)
    pp_solvers.petsc_utils.clear_petsc_options()

    yield pc

    # Teardown.
    pc.destroy()
    petsc_mat.destroy()


@pytest.fixture
def petsc_stdout(
    capfd,  # This is a pytest object to capture the os-level stdout, needed for PETSc.
    jacobian: BlockLinearSystem,
    pc: PETSc.PC,
    model: pp.PorePyModel,
    dof_manager: DofManager,
    solvers: list[SinglePhysicsPreconditioner],
) -> str:
    pp_solvers.petsc_utils.clear_petsc_options()
    preconditioner_scheme = MultiPhysicsPreconditioner(
        components=solvers, dof_manager=dof_manager, model=model
    )

    # user_options = {
    #     "fieldsplit_contact_cpl_fieldsplit_intf_mass_energy_flx_pc_type": "sor",
    # }

    preconditioner_scheme.configure(
        bmat=jacobian,
        pc=pc,
        user_options=None,
        precond_list=solvers,
    )

    pc.view()
    out, err = capfd.readouterr()
    return out


@pytest.fixture(scope="module")
def patterns_to_compare(model_kind: str) -> list[str]:
    r"""Regular expressions are used to allow for any group names, since the same
    sub-algorithm can have different petsc prefixes depending on the model we solver.
    E.g. the mechanics amg can be a schur complement to the contact mechanics, or to
    the interface flow, if the latter is present in the model. The following expressions
    are used:
    - ".*", where "." means "any symbol" and "*" means "zero or more times";

    I did not find a good way to distinguish between amg for mechanics and amg for
    the mass balance, as group names can vary and we cannot rely on them. Therefore,
    the tests check that the number of matches is >= 1, not exactly == 1.
    """
    contact = [
        r"""PC Object: 1 MPI process
type: fieldsplit
FieldSplit with Schur preconditioner, factorization UPPER
Preconditioner for the Schur complement formed from Sp, an assembled approximation to S, which uses A00's block diagonal's inverse
""",
        r"""PC Object: .*contact.* 1 MPI process
type: pbjacobi
""",
    ]
    interface_flow = [
        r"""PC Object:.*1 MPI process
type: fieldsplit
FieldSplit with Schur preconditioner, factorization UPPER
Preconditioner for the Schur complement formed from Sp, an assembled approximation to S, which uses A00's diagonal's inverse
""",
        r"""PC Object: .* 1 MPI process
type: ilu
PC has not been set up so information may be incomplete
out-of-place factorization
0 levels of fill""",
    ]
    fixed_stress = [
        r"""PC Object: .* 1 MPI process
type: fieldsplit
FieldSplit with Schur preconditioner, factorization UPPER
Preconditioner for the Schur complement formed from user provided matrix""",
    ]
    mechanics_amg = [
        r"""PC Object: .* 1 MPI process
type: hypre""",
    ]
    isothermal_flow = [
        r"""PC Object: .* 1 MPI process
type: hypre
"""
    ]
    thermal_flow = [
        r"""PC Object: .* 1 MPI process
type: composite
Composite PC type - MULTIPLICATIVE""",
        r"""PC Object: .*sub_0_.* 1 MPI process
type: fieldsplit
FieldSplit with ADDITIVE composition: total splits = 2""",
        r"""PC Object: .*mass_bal_.* 1 MPI process
type: hypre""",
        r"""PC Object: .*mass_bal_cpl_.* 1 MPI process
type: none""",
        r"""PC Object: .*sub_1_.* 1 MPI process
type: python
Python: pp_solvers.petsc_solvers.PcPythonPermutation""",
        r"""PC Object: .*python_.* 1 MPI process
type: ilu""",
    ]
    if model_kind == "flow":
        return interface_flow + isothermal_flow
    if model_kind == "mechanics":
        return contact + mechanics_amg
    if model_kind == "TH":
        return interface_flow + thermal_flow
    if model_kind == "HM":
        return contact + fixed_stress + isothermal_flow + mechanics_amg
    if model_kind == "THM":
        return contact + interface_flow + fixed_stress + thermal_flow + mechanics_amg

    raise NotImplementedError


def find_in_petsc_output(pattern: str, petsc_stdout: str):
    # \s* at the start of each new line means arbitrary number of whitespaces.
    pattern = "\n".join(r"\s*" + line for line in pattern.splitlines())
    return re.findall(pattern, petsc_stdout)


def test_ksp_none(petsc_stdout: str, model_kind: str):
    """Ensure that the default preconditioners do not involve inner Krylov solvers.

    In PETSc, this is done by replacing the Krylov solver with "preonly". We also count
    "preonly" to estimate the number of inner preconditioners.

    """
    matches = find_in_petsc_output(r"KSP Object:.*\n *type: (.+)", petsc_stdout)
    if model_kind in ["flow", "mechanics"]:
        assert len(matches) == 2
    elif model_kind == "HM":
        assert len(matches) == 6
    elif model_kind == "TH":
        assert len(matches) == 4
    elif model_kind == "THM":
        assert len(matches) == 8
    else:
        raise NotImplementedError(model_kind)

    assert all(x == "preonly" for x in matches)


def test_petsc_options(petsc_stdout: str, patterns_to_compare):
    """Compare the output of `pc.view()` - PETSc report of how the preconditioner is
    configured, to the expected values. We search for a few key lines in the report.

    """
    for pattern in patterns_to_compare:
        found = find_in_petsc_output(pattern, petsc_stdout)
        assert len(found) >= 1, f"Not found:\n{pattern}\n\n{petsc_stdout}\n\n"


def test_pass_user_options(
    capfd,  # This is a pytest object to capture the os-level stdout, needed for PETSc.
    jacobian: BlockLinearSystem,
    pc: PETSc.PC,
    model: pp.PorePyModel,
    dof_manager: DofManager,
    solvers: list[SinglePhysicsPreconditioner],
    model_kind: str,
) -> str:
    if model_kind == "mechanics":
        expected_group_name = "mechanics"
    else:
        expected_group_name = "mass_balance"

    user_options = {
        expected_group_name: {
            "pc_type": "jacobi",
        }
    }

    preconditioner_scheme = MultiPhysicsPreconditioner(
        components=solvers, dof_manager=dof_manager, model=model
    )
    preconditioner_scheme.configure(
        bmat=jacobian,
        pc=pc,
        user_options=user_options,
        precond_list=solvers,
    )

    pc.view()
    petsc_stdout, _ = capfd.readouterr()

    pattern = "PC Object: .* 1 MPI process\ntype: jacobi"

    matches = find_in_petsc_output(pattern, petsc_stdout)
    assert len(matches) == 1, f"Not found:\n{pattern}\n\n{petsc_stdout}\n\n"


def test_petsc_ksp_scheme(
    capfd,  # This is a pytest object to capture the os-level stdout, needed for PETSc.
    jacobian: BlockLinearSystem,
    model: pp.PorePyModel,
    dof_manager: DofManager,
    solvers: list[SinglePhysicsPreconditioner],
    patterns_to_compare: list[str],
):
    preconditioner_scheme = MultiPhysicsPreconditioner(
        components=solvers, dof_manager=dof_manager, model=model
    )

    user_options = {"ksp_type": "bcgs"}

    ksp_scheme = PetscKSPScheme(
        preconditioner=preconditioner_scheme, petsc_options=user_options
    )

    user_options_2 = {"ksp_rtol": 5e-5}
    solver = ksp_scheme.make_solver(mat_orig=jacobian, options=user_options_2)

    solver.ksp.view()
    petsc_stdout, _ = capfd.readouterr()

    # Checking that the preconditioner is correctly initialized.
    for pattern in patterns_to_compare:
        found = find_in_petsc_output(pattern, petsc_stdout)
        assert len(found) >= 1, f"Not found:\n{pattern}\n\n{petsc_stdout}\n\n"

    # Checking that the ksp is correctly initialized, including user options.
    pattern_ksp = r"""KSP Object: 1 MPI process
type: bcgs
maximum iterations=120, initial guess is zero
tolerances: relative=5e-05, absolute=1e-50, divergence=10000.
right preconditioning
using UNPRECONDITIONED norm type for convergence test
"""
    found = find_in_petsc_output(pattern_ksp, petsc_stdout)
    assert len(found) == 1, f"Not found:\n{pattern_ksp}\n\n{petsc_stdout}\n\n"


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("right", [True, False])
def test_linear_transformed_scheme(
    jacobian: BlockLinearSystem, left: bool, right: bool
):
    # Sorting the blocks in the matrix, same as it is done in the solver code.
    jacobian = jacobian[:]

    # Generating some transformation matrices.
    left_transformations = []
    right_transformations = []
    expected = jacobian.mat
    if left:
        Qleft = jacobian.copy()
        Qleft2 = jacobian.copy()
        Qleft2.mat *= 2
        left_transformations = [lambda _: Qleft, lambda _: Qleft2]
        expected = Qleft.mat @ Qleft2.mat @ expected
    if right:
        Qright = jacobian.copy()
        Qright2 = jacobian.copy()
        Qright2.mat *= 2
        right_transformations = [lambda _: Qright, lambda _: Qright2]
        expected = expected @ Qright.mat @ Qright2.mat

    # Initializing the KSP with transformations, without the preconditioner.
    solver_scheme = LinearTransformedScheme(
        inner=PetscKSPScheme(
            preconditioner=None,
            # Default preconditioner is ILU, it can complain for some matrices.
            petsc_options={"pc_type": "jacobi"},
        ),
        left_transformations=left_transformations,
        right_transformations=right_transformations,
    )
    solver = solver_scheme.make_solver(mat_orig=jacobian)

    result_mat = pp_solvers.petsc_to_csr(solver.ksp.getOperators()[0])

    # They should be exactly equal, numerical error may appear due to different order of
    # matrix multiplication.
    np.testing.assert_almost_equal(result_mat.toarray(), expected.toarray())
