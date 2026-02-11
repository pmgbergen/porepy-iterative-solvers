"""This module tests that the petsc options are generated correctly by various
combinations of configuration classes, defined in `preconditioners.py`.

"""

import numpy as np
import pytest
from petsc4py import PETSc
from scipy.sparse import csr_matrix
from testing_utils import (
    MockDofManager,
    generate_reference_dofs_3_groups,
    generate_reference_matrix_3_groups,
    generate_reference_rhs_3_groups,
)

from pp_solvers.block_linear_system import BlockLinearSystem, LinearSystemIndexer
from pp_solvers.options_parsers import PetscKSPScheme
from pp_solvers.petsc_utils import (
    clear_petsc_options,
    csr_to_petsc,
    insert_petsc_options,
)
from pp_solvers.preconditioners import (
    AMG,
    GMRES,
    ILU,
    BlockDiagonalInvertor,
    BlockDiagonalPreconditioner,
    CompositePreconditioner,
    DiagonalInvertor,
    FieldSplitAdditive,
    FieldSplitSchur,
    FixedStressInvertor,
    Identity,
    PetscInvertor,
    PetscKspPcConfiguration,
    PythonPermutationWrapper,
)


@pytest.fixture
def ksp() -> PETSc.KSP:
    ksp = PETSc.KSP().create()
    mat = [
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
    ]
    petsc_matrix = csr_to_petsc(csr_matrix(np.array(mat)))
    ksp.setOperators(petsc_matrix)
    yield ksp

    ksp.destroy()
    petsc_matrix.destroy()


CONFIGURATIONS_FOR_PETSC = [
    ILU(groups=["g1"], key="custom_key"),
    AMG(groups=["g1"], key="custom_key"),
    Identity(groups=["g1"], key="custom_key"),
    BlockDiagonalPreconditioner(groups=["g1"], key="custom_key"),
    GMRES(
        preconditioner=Identity(groups=["g1"]),
        key="custom_key",
    ),
    CompositePreconditioner(
        subsolvers=[
            Identity(groups=["g1", "g2"]),
            Identity(groups=["g1", "g2"]),
        ],
        key="custom_key",
    ),
]
"""List of shallow configurations used for tests, which invoke `ksp.setUp()`."""

CONFIGURATIONS_ALL = CONFIGURATIONS_FOR_PETSC + [
    FieldSplitSchur(
        subsolver=Identity(groups=["g1"]),
        complement=Identity(groups=["g2"]),
        approximate_invertor=DiagonalInvertor(),
        key="custom_key",
    ),
    FieldSplitAdditive(
        subsolvers=[
            Identity(groups=["g1"]),
            Identity(groups=["g2"]),
            Identity(groups=["g3"]),
        ],
        key="custom_key",
    ),
    PythonPermutationWrapper(
        key="custom_key",
        permutation_groups=[["g1", "g2"]],
        inner_subsolver=Identity(groups=["g1", "g2"]),
    ),
]
"""List of shallow configurations used for the rest of the tests."""


@pytest.mark.parametrize("configuration", CONFIGURATIONS_FOR_PETSC)
def test_default_petsc_options(configuration: PetscKspPcConfiguration, ksp: PETSc.KSP):
    """We test that the leaf PetscKspPcConfiguration produce correct default PETSc
    options."""
    clear_petsc_options()
    petsc_options = configuration.petsc_options(
        user_options={}, prefix="", dof_manager=MockDofManager()
    )
    assert isinstance(petsc_options, dict)
    insert_petsc_options(petsc_options)

    ksp.setFromOptions()
    # If the bad options were passed, it will raise here.
    ksp.setUp()

    # It should either set up a ksp or a pc.
    assert "ksp_type" in petsc_options or "pc_type" in petsc_options
    if "ksp_type" in petsc_options:
        assert ksp.type == petsc_options["ksp_type"]
    if "pc_type" in petsc_options:
        assert ksp.getPC().type == petsc_options["pc_type"]


@pytest.mark.parametrize(
    "configuration",
    CONFIGURATIONS_ALL,
)
def test_configurations_sanity_checks(configuration: PetscKspPcConfiguration):
    # 1. It should fetch our custom key.
    assert configuration.key == "custom_key"
    # 2. petsc_options should return something and it should be a dict.
    petsc_options = configuration.petsc_options(
        user_options={}, prefix="", dof_manager=MockDofManager()
    )
    assert isinstance(petsc_options, dict)
    # 3. petsc_assembly_config should return something and it should be a dict.
    config = configuration.petsc_assembly_config(
        user_options={}, prefix="", dof_manager=MockDofManager()
    )
    assert isinstance(config, dict)


def test_fieldsplit_bad_groups():
    # Should not let instantiate if overlapping groups are passed.
    with pytest.raises(ValueError):
        FieldSplitSchur(
            subsolver=Identity(groups=["g2", "g1"]),
            complement=Identity(groups=["g1"]),
            approximate_invertor=DiagonalInvertor(),
        )

    with pytest.raises(ValueError):
        FieldSplitAdditive(
            subsolvers=[
                Identity(groups=["g2"]),
                AMG(groups=["g2"]),
                Identity(groups=["g3"]),
            ],
        )


def test_composite_bad_groups():
    # Should not let instantiate different groups are passed.
    with pytest.raises(ValueError):
        CompositePreconditioner(
            subsolvers=[
                Identity(groups=["g2"]),
                Identity(groups=["g1"]),
            ]
        )
    # Order matters.
    with pytest.raises(ValueError):
        CompositePreconditioner(
            subsolvers=[
                Identity(groups=["g2", "g1"]),
                Identity(groups=["g1", "g2"]),
            ]
        )


@pytest.mark.parametrize(
    "configuration",
    CONFIGURATIONS_ALL,
)
@pytest.mark.parametrize("prefix", ["", "custom_prefix_"])
def test_user_options_and_prefix(configuration: PetscKspPcConfiguration, prefix: str):
    user_options = {
        "custom_key": {"ksp_type": "cg", "pc_type": "sor"},
        "this_key_should_be_ignored": {"ksp_type": "bcgs", "pc_type": "ilu"},
    }
    petsc_options = configuration.petsc_options(
        user_options=user_options, prefix=prefix, dof_manager=MockDofManager()
    )
    # User options should override defaults.
    if not isinstance(configuration, GMRES):
        assert petsc_options[f"{prefix}ksp_type"] == "cg"
        assert petsc_options[f"{prefix}pc_type"] == "sor"
    else:
        # This is the known edge case (preconditioner overrides gmres settings.)
        assert petsc_options[f"{prefix}ksp_type"] == "cg"
        assert petsc_options[f"{prefix}pc_type"] == "none"


def test_gmres_and_preconditioner_override_user_params():
    configuration = GMRES(
        preconditioner=Identity(groups=["g1"], key="preconditioner"),
        key="gmres",
    )
    user_options = {
        "gmres": {"ksp_type": "cg", "pc_type": "sor"},
        "preconditioner": {"ksp_type": "bcgs", "pc_type": "ilu"},
    }
    petsc_options = configuration.petsc_options(
        user_options=user_options, prefix="", dof_manager=MockDofManager()
    )
    # Preconditioner options are prioritized.
    assert petsc_options["ksp_type"] == "bcgs"
    assert petsc_options["pc_type"] == "ilu"


def test_nested_fieldsplits_schur():
    def make_fieldsplit(subsolver, complement, key):
        return FieldSplitSchur(
            subsolver=subsolver,
            complement=complement,
            approximate_invertor=DiagonalInvertor(),
            key=key,
            petsc_tag="elim",
            petsc_complement_tag="keep",
        )

    configuration = make_fieldsplit(
        key="fs1",
        subsolver=make_fieldsplit(
            key="fs2",
            subsolver=Identity(groups=["g1"], key="i1"),
            complement=Identity(groups=["g2"], key="i2"),
        ),
        complement=make_fieldsplit(
            key="fs3",
            subsolver=Identity(groups=["g3"], key="i3"),
            complement=Identity(groups=["g4", "g5"], key="i4"),
        ),
    )

    # Check that the root fielsplit fetched the groups in the right order.
    assert configuration.groups == ["g1", "g2", "g3", "g4", "g5"]

    # Passing options to each key, both leaves and fieldsplits.
    user_options = {
        "fs1": {"test_option": "fs1"},
        "fs2": {"test_option": "fs2"},
        "fs3": {"test_option": "fs3"},
        "i1": {"test_option": "i1"},
        "i2": {"test_option": "i2"},
        "i3": {"test_option": "i3"},
        "i4": {"test_option": "i4"},
    }
    petsc_options = configuration.petsc_options(
        user_options=user_options, prefix="", dof_manager=MockDofManager()
    )
    # Each option should be fetched with the corresponding petsc prefix.
    for expected_key, expected_value in {
        "test_option": "fs1",
        "fieldsplit_elim_test_option": "fs2",
        "fieldsplit_elim_fieldsplit_elim_test_option": "i1",
        "fieldsplit_elim_fieldsplit_keep_test_option": "i2",
        "fieldsplit_keep_test_option": "fs3",
        "fieldsplit_keep_fieldsplit_elim_test_option": "i3",
        "fieldsplit_keep_fieldsplit_keep_test_option": "i4",
    }.items():
        assert petsc_options[expected_key] == expected_value

    # Nested fieldsplits should return correct assembly configs.
    petsc_assembly_config = configuration.petsc_assembly_config(
        user_options=user_options, prefix="", dof_manager=MockDofManager()
    )
    assert petsc_assembly_config == {
        "": {
            "config_type": "fieldsplit_schur",
            "elim_tag": "elim",
            "keep_tag": "keep",
            "elim_groups": [0, 1],
            "keep_groups": [2, 3, 4],
        },
        "fieldsplit_elim_": {
            "config_type": "fieldsplit_schur",
            "elim_tag": "elim",
            "keep_tag": "keep",
            "elim_groups": [0],
            "keep_groups": [1],
        },
        "fieldsplit_keep_": {
            "config_type": "fieldsplit_schur",
            "elim_tag": "elim",
            "keep_tag": "keep",
            "elim_groups": [2],
            "keep_groups": [3, 4],
        },
    }


def test_nested_additive_fieldsplits():
    configuration = FieldSplitAdditive(
        key="fs1",
        subsolvers=[
            FieldSplitAdditive(
                key="fs2",
                subsolvers=[
                    Identity(groups=["g1"], key="i1"),
                    Identity(groups=["g2"], key="i2"),
                ],
            ),
            Identity(groups=["g3"], key="i3"),
            FieldSplitAdditive(
                key="fs3",
                subsolvers=[
                    Identity(groups=["g4"], key="i4"),
                    Identity(groups=["g5"], key="i5"),
                ],
            ),
        ],
    )

    # Check that the root fielsplit fetched the groups in the right order.
    assert configuration.groups == ["g1", "g2", "g3", "g4", "g5"]

    # Passing options to each key, both leaves and fieldsplits.
    user_options = {
        "fs1": {"test_option": "fs1"},
        "fs2": {"test_option": "fs2"},
        "fs3": {"test_option": "fs3"},
        "i1": {"test_option": "i1"},
        "i2": {"test_option": "i2"},
        "i3": {"test_option": "i3"},
        "i4": {"test_option": "i4"},
        "i5": {"test_option": "i5"},
    }
    petsc_options = configuration.petsc_options(
        user_options=user_options, prefix="", dof_manager=MockDofManager()
    )
    # Each option should be fetched with the corresponding petsc prefix.
    for expected_key, expected_value in {
        "test_option": "fs1",
        "fieldsplit_sub_0_test_option": "fs2",
        "fieldsplit_sub_0_fieldsplit_sub_0_test_option": "i1",
        "fieldsplit_sub_0_fieldsplit_sub_1_test_option": "i2",
        "fieldsplit_sub_1_test_option": "i3",
        "fieldsplit_sub_2_test_option": "fs3",
        "fieldsplit_sub_2_fieldsplit_sub_0_test_option": "i4",
        "fieldsplit_sub_2_fieldsplit_sub_1_test_option": "i5",
    }.items():
        assert petsc_options[expected_key] == expected_value

    # Nested fieldsplits should return correct assembly configs.
    petsc_assembly_config = configuration.petsc_assembly_config(
        user_options=user_options, prefix="", dof_manager=MockDofManager()
    )
    assert petsc_assembly_config == {
        "": {
            "config_type": "fieldsplit_additive",
            "subsolver_groups": [[0, 1], [2], [3, 4]],
        },
        "fieldsplit_sub_0_": {
            "config_type": "fieldsplit_additive",
            "subsolver_groups": [[0], [1]],
        },
        "fieldsplit_sub_2_": {
            "config_type": "fieldsplit_additive",
            "subsolver_groups": [[3], [4]],
        },
    }


def test_nested_composites():
    groups = ["g1", "g2", "g3"]
    configuration = CompositePreconditioner(
        key="c1",
        subsolvers=[
            CompositePreconditioner(
                key="c2",
                subsolvers=[
                    Identity(groups=groups, key="i1"),
                    Identity(groups=groups, key="i2"),
                ],
            ),
            Identity(groups=groups, key="i3"),
            Identity(groups=groups, key="i4"),
        ],
    )

    # Check that the root composite fetched the groups in the right order.
    assert configuration.groups == groups

    # Passing options to each key, both leaves and fieldsplits.
    user_options = {
        "c1": {"test_option": "c1"},
        "c2": {"test_option": "c2"},
        "i1": {"test_option": "i1"},
        "i2": {"test_option": "i2"},
        "i3": {"test_option": "i3"},
        "i4": {"test_option": "i4"},
    }
    petsc_options = configuration.petsc_options(
        user_options=user_options, prefix="", dof_manager=MockDofManager()
    )
    # Each option should be fetched with the corresponding petsc prefix.
    for expected_key, expected_value in {
        "test_option": "c1",
        "sub_0_test_option": "c2",
        "sub_0_sub_0_test_option": "i1",
        "sub_0_sub_1_test_option": "i2",
        "sub_1_test_option": "i3",
        "sub_2_test_option": "i4",
    }.items():
        assert petsc_options[expected_key] == expected_value

    # Nested composites should return correct assembly configs.
    petsc_assembly_config = configuration.petsc_assembly_config(
        user_options=user_options, prefix="", dof_manager=MockDofManager()
    )
    assert petsc_assembly_config == {
        "": {
            "config_type": "composite",
            "num_stages": 3,
        },
        "sub_0_": {
            "config_type": "composite",
            "num_stages": 2,
        },
    }


@pytest.mark.parametrize(
    "invertor", [DiagonalInvertor(), BlockDiagonalInvertor(), FixedStressInvertor()]
)
@pytest.mark.parametrize("prefix", ["", "custom_prefix"])
def test_approximate_invertors_petsc_options(invertor: PetscInvertor, prefix: str):
    petsc_options = invertor.petsc_options(
        prefix=prefix, tag="elim", complement_tag="keep"
    )
    assert isinstance(petsc_options, dict)
    for key in petsc_options.keys():
        assert key.startswith(prefix)


@pytest.mark.parametrize(
    "invertor",
    [
        DiagonalInvertor(),
        BlockDiagonalInvertor(),
        # FixedStressInvertor() is not tested here, because it requires the real PorePy
        # model. It is tested in test_fixed_stress.
    ],
)
def test_approximate_invertors_assembly_config(invertor: PetscInvertor):
    assert invertor.petsc_assembly_config(dof_manager=None) == {}


def test_python_permutation():
    groups = ["g1", "g2"]
    configuration = PythonPermutationWrapper(
        key="p1",
        permutation_groups=[["g1"], ["g2"]],
        inner_subsolver=Identity(groups=groups, key="i1"),
    )

    assert configuration.groups == groups

    user_options = {
        "i1": {"custom_option": "i1"},
        "p1": {"custom_option": "p1"},
    }
    petsc_options = configuration.petsc_options(
        user_options=user_options, prefix="", dof_manager=MockDofManager()
    )
    for expected_key, expected_value in {
        "custom_option": "p1",
        "python_custom_option": "i1",
    }.items():
        assert petsc_options[expected_key] == expected_value

    assembly_config = configuration.petsc_assembly_config(
        user_options={}, prefix="custom_prefix_", dof_manager=MockDofManager()
    )
    assert assembly_config["custom_prefix_"]["config_type"] == "python_permutation"
    assert assembly_config["custom_prefix_"]["permutation_groups"] == [[0], [1]]


@pytest.fixture
def block_linear_system() -> BlockLinearSystem:
    dofs_row, dofs_col = generate_reference_dofs_3_groups()
    return BlockLinearSystem(
        mat=generate_reference_matrix_3_groups(),
        rhs=generate_reference_rhs_3_groups(),
        indexer=LinearSystemIndexer(
            dofs_row=dofs_row,
            dofs_col=dofs_col,
        ),
    )


def test_petsc_ksp_scheme(block_linear_system: BlockLinearSystem):
    ksp_scheme = PetscKSPScheme(
        petsc_ksp_pc_configuration=GMRES(preconditioner=Identity(groups=["mock_g1"])),
        dof_manager=MockDofManager(),
    )
    krylov_solver = ksp_scheme.make_solver(
        mat_orig=block_linear_system, options={"gmres": {"ksp_type": "fgmres"}}
    )
    # Check that the custom option applied.
    assert krylov_solver.ksp.type == "fgmres"
    solution = krylov_solver.solve(block_linear_system.rhs)

    np.testing.assert_allclose(
        block_linear_system.mat @ solution,
        block_linear_system.rhs,
    )
