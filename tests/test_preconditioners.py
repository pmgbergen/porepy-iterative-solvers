"""This module tests that the petsc options are generated correctly by various
combinations of configuration classes, defined in `preconditioners.py`.

"""

from typing import Optional

import numpy as np
import pytest
from petsc4py import PETSc
from scipy.sparse import csr_matrix
from testing_utils import MockDofManager, generate_block_linear_system

from pp_solvers import dof_manager
from pp_solvers.block_linear_system import BlockLinearSystem
from pp_solvers.options_parsers import initialize_petsc_ksp
from pp_solvers.petsc_utils import (
    clear_petsc_options,
    csr_to_petsc,
    insert_petsc_options,
)
from pp_solvers.preconditioners import (
    AMG,
    GMRES,
    ILU,
    BlockDiagonalInverter,
    BlockDiagonalPreconditioner,
    CompositePreconditioner,
    DiagonalInverter,
    FieldSplit,
    FieldSplitSchur,
    FixedStressInverter,
    Identity,
    NoInverter,
    PetscInverter,
    PetscKspPcConfiguration,
    PythonPermutationWrapper,
    _validate_subsolvers_keys_are_unique,
    validate_all_keys_are_unique,
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
    GMRES(preconditioner=Identity(groups=["g1"]), key="custom_key"),
    CompositePreconditioner(
        subsolvers=[
            Identity(groups=["g1", "g2"], key="identity_0"),
            Identity(groups=["g1", "g2"], key="identity_1"),
        ],
        key="custom_key",
    ),
]
"""List of shallow configurations used for tests, which invoke `ksp.setUp()`."""

CONFIGURATIONS_ALL = CONFIGURATIONS_FOR_PETSC + [
    FieldSplitSchur(
        subsolver=Identity(groups=["g1"], key="identity_g1"),
        complement_solver=Identity(groups=["g2"], key="identity_g2"),
        approximate_inverter=DiagonalInverter(),
        key="custom_key",
    ),
    FieldSplit(
        subsolvers=[
            Identity(groups=["g1"], key="identity_g1"),
            Identity(groups=["g2"], key="identity_g2"),
            Identity(groups=["g3"], key="identity_g3"),
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
        user_options={}, dof_manager=MockDofManager(groups=configuration.groups)
    )
    assert isinstance(petsc_options, dict)
    insert_petsc_options(petsc_options)

    ksp.setFromOptions()
    # If the bad options were passed, it will raise here.
    ksp.setUp()

    # It should either set up a ksp or a pc.
    assert (
        "custom_key_ksp_type" in petsc_options or "custom_key_pc_type" in petsc_options
    )
    if "ksp_type" in petsc_options:
        assert ksp.type == petsc_options["custom_key_ksp_type"]
    if "pc_type" in petsc_options:
        assert ksp.getPC().type == petsc_options["custom_key_pc_type"]


@pytest.mark.parametrize(
    "configuration",
    CONFIGURATIONS_ALL,
)
def test_configurations_sanity_checks(configuration: PetscKspPcConfiguration):
    # 1. It should fetch our custom key.
    assert configuration.key == "custom_key"
    # 2. petsc_options should return something and it should be a dict.
    petsc_options = configuration.petsc_options(
        user_options={}, dof_manager=MockDofManager(groups=configuration.groups)
    )
    assert isinstance(petsc_options, dict)
    # 3. petsc_assembly_config should return something and it should be a dict.
    config = configuration.petsc_assembly_config(
        user_options={}, dof_manager=MockDofManager(groups=configuration.groups)
    )
    assert isinstance(config, dict)
    # 4. assert that get_children works.
    all_children = configuration.get_children()
    for child in all_children:
        assert isinstance(child, PetscKspPcConfiguration)


def test_fieldsplit_bad_groups():
    # Should not let instantiate if overlapping groups are passed.
    with pytest.raises(ValueError):
        FieldSplitSchur(
            subsolver=Identity(groups=["g2", "g1"]),
            complement_solver=Identity(groups=["g1"]),
            approximate_inverter=DiagonalInverter(),
        )

    with pytest.raises(ValueError):
        FieldSplit(
            subsolvers=[
                Identity(groups=["g2"], key="identity_g2"),
                AMG(groups=["g2"], key="amg_g2"),
                Identity(groups=["g3"], key="identity_g3"),
            ],
        )


def test_composite_bad_groups():
    # Should not let instantiate different groups are passed.
    with pytest.raises(ValueError):
        CompositePreconditioner(
            subsolvers=[
                Identity(groups=["g2"], key="identity_0"),
                Identity(groups=["g1"], key="identity_1"),
            ]
        )
    # Order matters.
    with pytest.raises(ValueError):
        CompositePreconditioner(
            subsolvers=[
                Identity(groups=["g2", "g1"], key="identity_0"),
                Identity(groups=["g1", "g2"], key="identity_1"),
            ]
        )


@pytest.mark.parametrize(
    "configuration",
    CONFIGURATIONS_ALL,
)
def test_user_options_and_prefix(configuration: PetscKspPcConfiguration):
    user_options = {
        "custom_key": {"ksp_type": "cg", "pc_type": "sor"},
        "this_key_should_be_ignored": {"ksp_type": "bcgs", "pc_type": "ilu"},
    }
    petsc_options = configuration.petsc_options(
        user_options=user_options,
        dof_manager=MockDofManager(groups=configuration.groups),
    )
    # User options should override defaults.
    assert petsc_options[f"custom_key_ksp_type"] == "cg"
    assert petsc_options[f"custom_key_pc_type"] == "sor"


def test_gmres_override_preconditioner_key():
    configuration = GMRES(
        key="gmres",
        preconditioner=Identity(groups=["g1"], key="preconditioner"),
    )
    user_options = {
        "gmres": {"ksp_type": "cg", "pc_type": "sor"},
        "preconditioner": {"ksp_type": "bcgs", "pc_type": "ilu"},
    }
    petsc_options = configuration.petsc_options(
        user_options=user_options,
        dof_manager=MockDofManager(groups=configuration.groups),
    )
    # The "preconditioner" key is ignored, read the GMRES class comment.
    assert petsc_options["gmres_ksp_type"] == "cg"
    assert petsc_options["gmres_pc_type"] == "sor"
    # All keys have "gmres" prefix.
    for key in petsc_options:
        assert key.startswith("gmres_")


def test_nested_fieldsplits_schur():
    def make_fieldsplit(subsolver, complement, key):
        return FieldSplitSchur(
            subsolver=subsolver,
            complement_solver=complement,
            approximate_inverter=DiagonalInverter(),
            key=key,
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
    dof_manager = MockDofManager(groups=configuration.groups)
    petsc_options = configuration.petsc_options(
        user_options=user_options,
        dof_manager=dof_manager,
    )
    # Each option should be fetched with the corresponding petsc prefix.
    for expected_key, expected_value in {
        "fs1_test_option": "fs1",
        "fs2_test_option": "fs2",
        "fs3_test_option": "fs3",
        "i1_test_option": "i1",
        "i2_test_option": "i2",
        "i3_test_option": "i3",
        "i4_test_option": "i4",
    }.items():
        assert petsc_options[expected_key] == expected_value

    # Nested fieldsplits should return correct assembly configs.
    petsc_assembly_config = configuration.petsc_assembly_config(
        user_options=user_options, dof_manager=dof_manager
    )
    assert petsc_assembly_config == {
        "fs1": {
            "config_type": "fieldsplit_schur",
            "elim_key": "fs2",
            "keep_key": "fs3",
            "elim_groups": [0, 1],
            "keep_groups": [2, 3, 4],
        },
        "fs2": {
            "config_type": "fieldsplit_schur",
            "elim_key": "i1",
            "keep_key": "i2",
            "elim_groups": [0],
            "keep_groups": [1],
        },
        "fs3": {
            "config_type": "fieldsplit_schur",
            "elim_key": "i3",
            "keep_key": "i4",
            "elim_groups": [2],
            "keep_groups": [3, 4],
        },
    }


def test_nested_additive_fieldsplits():
    configuration = FieldSplit(
        key="fs1",
        subsolvers=[
            FieldSplit(
                key="fs2",
                subsolvers=[
                    Identity(groups=["g1"], key="i1"),
                    Identity(groups=["g2"], key="i2"),
                ],
            ),
            Identity(groups=["g3"], key="i3"),
            FieldSplit(
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
    dof_manager = MockDofManager(groups=configuration.groups)
    petsc_options = configuration.petsc_options(
        user_options=user_options,
        dof_manager=dof_manager,
    )
    # Each option should be fetched with the corresponding petsc prefix.
    for expected_key, expected_value in {
        "fs1_test_option": "fs1",
        "fs2_test_option": "fs2",
        "fs3_test_option": "fs3",
        "i1_test_option": "i1",
        "i2_test_option": "i2",
        "i3_test_option": "i3",
        "i4_test_option": "i4",
        "i5_test_option": "i5",
    }.items():
        assert petsc_options[expected_key] == expected_value

    # Nested fieldsplits should return correct assembly configs.
    petsc_assembly_config = configuration.petsc_assembly_config(
        user_options=user_options, dof_manager=dof_manager
    )
    assert petsc_assembly_config == {
        "fs1": {
            "config_type": "fieldsplit_common",
            "subsolver_groups": [[0, 1], [2], [3, 4]],
            "subsolver_keys": ["fs2", "i3", "fs3"],
        },
        "fs2": {
            "config_type": "fieldsplit_common",
            "subsolver_groups": [[0], [1]],
            "subsolver_keys": ["i1", "i2"],
        },
        "fs3": {
            "config_type": "fieldsplit_common",
            "subsolver_groups": [[3], [4]],
            "subsolver_keys": ["i4", "i5"],
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
        user_options=user_options,
        dof_manager=MockDofManager(groups=configuration.groups),
    )
    # Each option should be fetched with the corresponding petsc prefix.
    for expected_key, expected_value in {
        "c1_test_option": "c1",
        "c2_test_option": "c2",
        "i1_test_option": "i1",
        "i2_test_option": "i2",
        "i3_test_option": "i3",
        "i4_test_option": "i4",
    }.items():
        assert petsc_options[expected_key] == expected_value

    # Nested composites should return correct assembly configs.
    petsc_assembly_config = configuration.petsc_assembly_config(
        user_options=user_options, dof_manager=MockDofManager(groups=groups)
    )
    assert petsc_assembly_config == {
        "c1": {
            "config_type": "composite",
            "subsolver_keys": ["c2", "i3", "i4"],
        },
        "c2": {
            "config_type": "composite",
            "subsolver_keys": ["i1", "i2"],
        },
    }


@pytest.mark.parametrize(
    "inverter",
    [NoInverter(), DiagonalInverter(), BlockDiagonalInverter(), FixedStressInverter()],
)
@pytest.mark.parametrize("key", ["key1", "key2"])
def test_approximate_inverters_petsc_options(inverter: PetscInverter, key: str):
    petsc_options = inverter.petsc_options(
        key=key,
        elim_key="elim",
        complement_key="keep",
        dof_manager=MockDofManager(groups=["g1", "g2"]),
    )
    assert isinstance(petsc_options, dict)
    for key in petsc_options.keys():
        assert key.startswith(key)


@pytest.mark.parametrize(
    "inverter",
    [
        DiagonalInverter(),
        BlockDiagonalInverter(),
        # FixedStressInverter() is not tested here, because it requires the real PorePy
        # model. It is tested in test_fixed_stress.
    ],
)
def test_approximate_inverters_assembly_config(inverter: PetscInverter):
    assert inverter.petsc_assembly_config(dof_manager=None) == {}


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
        user_options=user_options,
        dof_manager=MockDofManager(groups=configuration.groups),
    )
    for expected_key, expected_value in {
        "p1_custom_option": "p1",
        "i1_custom_option": "i1",
    }.items():
        assert petsc_options[expected_key] == expected_value

    assembly_config = configuration.petsc_assembly_config(
        user_options={}, dof_manager=MockDofManager(groups=configuration.groups)
    )
    assert assembly_config == {
        "p1": {
            "config_type": "python_permutation",
            "permutation_groups": [[0], [1]],
            "inner_key": "i1",
        }
    }


def test_petsc_ksp_scheme():
    block_linear_system = generate_block_linear_system()
    krylov_solver = initialize_petsc_ksp(
        block_linear_system=block_linear_system,
        dof_manager=MockDofManager(groups=["mock_g1"]),
        petsc_ksp_pc_configuration=GMRES(preconditioner=Identity(groups=["mock_g1"])),
        user_options={
            "gmres": {"ksp_type": "fgmres"},
            "delete_matrices": False,
        },
    )
    # Check that the custom option applied.
    assert krylov_solver.ksp.type == "fgmres"
    solution = krylov_solver.solve(block_linear_system.rhs)

    np.testing.assert_allclose(
        block_linear_system.mat @ solution,
        block_linear_system.rhs,
    )


@pytest.mark.parametrize(
    "params",
    [
        # Use all defaults: uses default keys "ilu" and "identity" for subsolvers.
        # Fieldsplit names itself based on the secondary subsolver ("fs_identity").
        {
            "elim_key": None,
            "keep_key": None,
            "expected_petsc_options": {
                "fs_identity_pc_type": "fieldsplit",
                "identity_pc_type": "none",
                "ilu_pc_type": "ilu",
            },
        },
        # Use default for the complement (ilu).
        {
            "elim_key": "custom_tag_1",
            "keep_key": None,
            "expected_petsc_options": {
                "fs_custom_tag_1_pc_type": "fieldsplit",
                "custom_tag_1_pc_type": "none",
                "ilu_pc_type": "ilu",
            },
        },
        # Use default for the secondary sub-solver (identity).
        {
            "elim_key": None,
            "keep_key": "custom_tag_2",
            "expected_petsc_options": {
                "fs_identity_pc_type": "fieldsplit",
                "identity_pc_type": "none",
                "custom_tag_2_pc_type": "ilu",
            },
        },
        # Custom values for both.
        {
            "elim_key": "custom_tag_1",
            "keep_key": "custom_tag_2",
            "expected_petsc_options": {
                "fs_custom_tag_1_pc_type": "fieldsplit",
                "custom_tag_1_pc_type": "none",
                "custom_tag_2_pc_type": "ilu",
            },
        },
    ],
)
def test_fieldsplit_schur_default_parameters(params: dict):
    """Tests non-trivial logic of creating FieldSplitSchur with default parameters
    elim_key and keep_key.

    """
    elim_key: Optional[str] = params["elim_key"]
    keep_key: Optional[str] = params["keep_key"]
    expected_petsc_options: dict = params["expected_petsc_options"]
    kwargs_subsolver = {"key": elim_key} if elim_key is not None else {}
    kwargs_complement = {"key": keep_key} if keep_key is not None else {}
    preconditioner = FieldSplitSchur(
        subsolver=Identity(groups=["mock_g1"], **kwargs_subsolver),
        complement_solver=ILU(groups=["mock_g2"], **kwargs_complement),
        approximate_inverter=DiagonalInverter(),
    )
    petsc_options = preconditioner.petsc_options(
        user_options={}, dof_manager=MockDofManager(groups=["mock_g1", "mock_g2"])
    )
    for key, value in expected_petsc_options.items():
        assert petsc_options[key] == value


@pytest.mark.parametrize(
    "conflicting_options",
    [
        {"fs": {"pc_fieldsplit_schur_precondition": "a11"}},
        {"elim": {"mat_block_size": 5}},
        {"keep": {"mat_schur_complement_ainv_type": "diag"}},
    ],
)
def test_fieldsplit_schur_raises_on_invertor_option_conflict(conflicting_options: dict):
    """FieldSplitSchur raises ValueError when user_options duplicate a key that the
    approximate_inverter already manages, preventing a silent option override.

    """
    preconditioner = FieldSplitSchur(
        subsolver=Identity(groups=["g1"], key="elim"),
        complement_solver=ILU(groups=["g2"], key="keep"),
        approximate_inverter=BlockDiagonalInverter(),
        key="fs",
    )

    with pytest.raises(ValueError, match="invertor options override solver options"):
        preconditioner.petsc_options(
            user_options=conflicting_options,
            dof_manager=MockDofManager(groups=preconditioner.groups),
        )


def test_validate_subsolvers_keys_are_unique():
    with pytest.raises(ValueError):
        _validate_subsolvers_keys_are_unique(
            [Identity(["g1"]), ILU(["g2"]), Identity(["g3"])], "root"
        )

    _validate_subsolvers_keys_are_unique(
        [Identity(["g1"]), ILU(["g2"]), AMG(["g3"])], "root"
    )


def test_validate_all_keys_are_unique():
    with pytest.raises(ValueError):
        validate_all_keys_are_unique(
            GMRES(
                FieldSplitSchur(
                    subsolver=Identity(groups=["g1"]),
                    approximate_inverter=NoInverter(),
                    complement_solver=FieldSplitSchur(
                        subsolver=ILU(groups=["g2"]),
                        approximate_inverter=NoInverter(),
                        complement_solver=FieldSplitSchur(
                            approximate_inverter=NoInverter(),
                            subsolver=Identity(groups=["g3"]),
                            complement_solver=AMG(groups=["g4"]),
                        ),
                    ),
                )
            )
        )

    with pytest.raises(ValueError):
        validate_all_keys_are_unique(
            CompositePreconditioner(
                subsolvers=[
                    Identity(groups=["g1", "g2"]),
                    FieldSplit(
                        subsolvers=[Identity(groups=["g1"]), ILU(groups=["g2"])]
                    ),
                ]
            )
        )

    validate_all_keys_are_unique(
        FieldSplitSchur(
            subsolver=Identity(groups=["g1"], key="i1"),
            approximate_inverter=NoInverter(),
            complement_solver=FieldSplitSchur(
                subsolver=ILU(groups=["g2"], key="ilu"),
                approximate_inverter=NoInverter(),
                complement_solver=FieldSplitSchur(
                    approximate_inverter=NoInverter(),
                    subsolver=Identity(groups=["g3"], key="i3"),
                    complement_solver=AMG(groups=["g4"], key="i4"),
                ),
            ),
        )
    )
