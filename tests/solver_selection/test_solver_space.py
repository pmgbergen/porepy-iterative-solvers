import pytest
from pp_solvers.solver_selection.solver_space import (
    SolverSpace,
    CategoricalChoices,
    NumericalChoices,
    explain_decisions,
)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {
                "scheme": {"a": "a1", "b": "b1"},
                "num_category_choices": 0,
                "num_numerical_choices": 0,
                "all_configs": [{"a": "a1", "b": "b1"}],
            },
            id="No choices",
        ),
        pytest.param(
            {
                "scheme": {"a": CategoricalChoices(["a1", "a2", "a3"]), "b": "b1"},
                "num_category_choices": 3,
                "num_numerical_choices": 0,
                "all_configs": [
                    {"a": "a1", "b": "b1"},
                    {"a": "a2", "b": "b1"},
                    {"a": "a3", "b": "b1"},
                ],
            },
            id="One category choice, 3 options",
        ),
        pytest.param(
            {
                "scheme": {
                    "a": CategoricalChoices(["a1", "a2", "a3"]),
                    "b": CategoricalChoices(["b1", "b2"]),
                },
                "num_category_choices": 5,
                "num_numerical_choices": 0,
                "all_configs": [
                    {"a": "a1", "b": "b1"},
                    {"a": "a1", "b": "b2"},
                    {"a": "a2", "b": "b1"},
                    {"a": "a2", "b": "b2"},
                    {"a": "a3", "b": "b1"},
                    {"a": "a3", "b": "b2"},
                ],
            },
            id="Two category choices, 5 options",
        ),
        pytest.param(
            {
                "scheme": {
                    "a": CategoricalChoices(
                        [
                            {
                                "b": CategoricalChoices(["b1", "b2"]),
                                "c": CategoricalChoices(["c1", "c2"]),
                            },
                            {
                                "d": CategoricalChoices(["d1", "d2"]),
                                "e": CategoricalChoices(["e1", "e2"]),
                            },
                            "a1",
                        ]
                    )
                },
                "num_category_choices": 11,
                "num_numerical_choices": 0,
                "all_configs": [
                    {"a": {"b": "b1", "c": "c1"}},
                    {"a": {"b": "b1", "c": "c2"}},
                    {"a": {"b": "b2", "c": "c1"}},
                    {"a": {"b": "b2", "c": "c2"}},
                    {"a": {"d": "d1", "e": "e1"}},
                    {"a": {"d": "d1", "e": "e2"}},
                    {"a": {"d": "d2", "e": "e1"}},
                    {"a": {"d": "d2", "e": "e2"}},
                    {"a": "a1"},
                ],
            },
            id="Nested categorical choices",
        ),
        pytest.param(
            {
                "scheme": CategoricalChoices(
                    [
                        {"a": "a1"},
                        {"b": "b1"},
                    ]
                ),
                "num_category_choices": 2,
                "num_numerical_choices": 0,
                "all_configs": [
                    {"a": "a1"},
                    {"b": "b1"},
                ],
            },
            id="Root-level categorical choice",
        ),
        pytest.param(
            {
                "scheme": CategoricalChoices(["a", "b", "c"]),
                "num_category_choices": 3,
                "num_numerical_choices": 0,
                "all_configs": ["a", "b", "c"],
            },
            id="Root-level string",
        ),
        pytest.param(
            {
                "scheme": NumericalChoices([1, 2, 3]),
                "num_category_choices": 0,
                "num_numerical_choices": 1,
                "all_configs": [1, 2, 3],
            },
            id="Root-level numerical choice",
        ),
        pytest.param(
            {
                "scheme": {"a": NumericalChoices([0.5, 1, 3])},
                "num_category_choices": 0,
                "num_numerical_choices": 1,
                "all_configs": [{"a": 0.5}, {"a": 1}, {"a": 3}],
            },
            id="Single numerical choice",
        ),
        pytest.param(
            {
                "scheme": CategoricalChoices(["a", CategoricalChoices(["b", "c"])]),
                "num_category_choices": 4,
                "num_numerical_choices": 0,
                "all_configs": ["a", "b", "c"],
            },
            id="Unnecessary nested categorical choices",
        ),
        pytest.param(
            {
                "scheme": CategoricalChoices([1, 2, NumericalChoices([3, 4, 5, 6])]),
                "num_category_choices": 3,
                "num_numerical_choices": 1,
                "all_configs": [1, 2, 3, 4, 5, 6],
            },
            id="Categorical with numerical inside",
        ),
        pytest.param(
            {
                "scheme": {
                    "a": CategoricalChoices(
                        [
                            {
                                "a1": NumericalChoices([0, 1]),
                                "a2": NumericalChoices([2, 3]),
                            },
                            {
                                "a3": NumericalChoices([0, 1]),
                                "a4": NumericalChoices([2, 3]),
                            },
                        ]
                    ),
                    "b": NumericalChoices([4, 5]),
                },
                "num_category_choices": 2,
                "num_numerical_choices": 5,
                "all_configs": [
                    {"a": {"a1": 0, "a2": 2}, "b": 4},
                    {"a": {"a1": 0, "a2": 2}, "b": 5},
                    {"a": {"a1": 1, "a2": 2}, "b": 5},
                    {"a": {"a1": 1, "a2": 2}, "b": 4},
                    {"a": {"a1": 0, "a2": 3}, "b": 5},
                    {"a": {"a1": 0, "a2": 3}, "b": 4},
                    {"a": {"a1": 1, "a2": 3}, "b": 5},
                    {"a": {"a1": 1, "a2": 3}, "b": 4},
                    #
                    {"a": {"a3": 0, "a4": 2}, "b": 4},
                    {"a": {"a3": 0, "a4": 2}, "b": 5},
                    {"a": {"a3": 1, "a4": 2}, "b": 5},
                    {"a": {"a3": 1, "a4": 2}, "b": 4},
                    {"a": {"a3": 0, "a4": 3}, "b": 5},
                    {"a": {"a3": 0, "a4": 3}, "b": 4},
                    {"a": {"a3": 1, "a4": 3}, "b": 5},
                    {"a": {"a3": 1, "a4": 3}, "b": 4},
                ],
            },
            id="Nested numerical choices",
        ),
        pytest.param(
            {
                "scheme": {
                    "mech": {"pc_type": CategoricalChoices(["hypre", "ilu"])},
                    "flow": {"pc_type": CategoricalChoices(["gamg", "ilu"])},
                },
                "num_category_choices": 4,
                "num_numerical_choices": 0,
                "all_configs": [
                    {
                        "mech": {"pc_type": "hypre"},
                        "flow": {"pc_type": "gamg"},
                    },
                    {
                        "mech": {"pc_type": "ilu"},
                        "flow": {"pc_type": "gamg"},
                    },
                    {
                        "mech": {"pc_type": "hypre"},
                        "flow": {"pc_type": "ilu"},
                    },
                    {
                        "mech": {"pc_type": "ilu"},
                        "flow": {"pc_type": "ilu"},
                    },
                ],
            },
            id="Same key for different categorical choices",
        ),
        pytest.param(
            {
                "scheme": {
                    "mech": {"strong_threshold": NumericalChoices([0.5, 0.7])},
                    "flow": {"strong_threshold": NumericalChoices([0.2, 0.5])},
                },
                "num_category_choices": 0,
                "num_numerical_choices": 2,
                "all_configs": [
                    {
                        "mech": {"strong_threshold": 0.5},
                        "flow": {"strong_threshold": 0.2},
                    },
                    {
                        "mech": {"strong_threshold": 0.7},
                        "flow": {"strong_threshold": 0.2},
                    },
                    {
                        "mech": {"strong_threshold": 0.5},
                        "flow": {"strong_threshold": 0.5},
                    },
                    {
                        "mech": {"strong_threshold": 0.7},
                        "flow": {"strong_threshold": 0.5},
                    },
                ],
            },
            id="Same key for different numerical choices",
        ),
        pytest.param(
            {
                "scheme": {
                    "a": CategoricalChoices(
                        [
                            {"b": "b1"},
                            {"c": "c1"},
                        ]
                    )
                },
                "num_category_choices": 2,
                "num_numerical_choices": 0,
                "all_configs": [
                    {"a": {"b": "b1"}},
                    {"a": {"c": "c1"}},
                ],
            },
            id="Dict inside categorical choice",
        ),
    ],
)
def test_solver_space(params: dict):
    """Tests various SolverSpaces. Ensures that the solver space builds successfully,
    and each decision (numpy array) can produce corresponding configuration.

    """
    scheme: dict = params["scheme"]
    num_category_choices: int = params["num_category_choices"]
    num_numerical_choices: int = params["num_numerical_choices"]
    all_configs: list[dict] = params["all_configs"]

    solver_space = SolverSpace(scheme)

    assert solver_space.num_category_choices == num_category_choices
    assert solver_space.num_numerical_choices == num_numerical_choices

    assert len(solver_space.all_decisions_encoding) == len(all_configs)

    for decision in solver_space.all_decisions_encoding:
        result_config = solver_space.config_from_decision(decision=decision)
        config_is_found = False
        for config in all_configs:
            if config == result_config:
                config_is_found = True
                break
        assert config_is_found, result_config

    # Check that this function does not raise.
    explain_decisions(solver_space, include_ids=False)
    explain_decisions(solver_space, include_ids=True)


def test_bad_nested_numerical_choices():
    """Tests creation of NumericalChoices with nested choices inside."""
    # This test is not parametrized because the cases here raise an error during the
    # NumericalChoices initialization. All the examples here make no practical sense.
    with pytest.raises(ValueError):
        _ = NumericalChoices([1, 2, CategoricalChoices([3, 4])])

    with pytest.raises(ValueError):
        _ = NumericalChoices([1, 2, NumericalChoices([3, 4])])


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {
                "x": CategoricalChoices(
                    [
                        {"a": "a1"},
                        {"b": "b1"},
                    ]
                ),
                "y": {"c": "c1"},
                "expected": CategoricalChoices(
                    [
                        {"a": "a1", "c": "c1"},
                        {"b": "b1", "c": "c1"},
                    ]
                ),
            },
            id="CategoricalChoices | dict",
        ),
        pytest.param(
            {
                "x": {"c": "c1"},
                "y": CategoricalChoices(
                    [
                        {"a": "a1"},
                        {"b": "b1"},
                    ]
                ),
                "expected": CategoricalChoices(
                    [
                        {"a": "a1", "c": "c1"},
                        {"b": "b1", "c": "c1"},
                    ]
                ),
            },
            id="dict | CategoricalChoices",
        ),
        pytest.param(
            {
                "x": CategoricalChoices(
                    [
                        {"a": "a1"},
                        {"b": "b1"},
                    ]
                ),
                "y": CategoricalChoices(
                    [
                        {"c": "c1"},
                        {"d": "d1"},
                    ]
                ),
                "should_raise": True,
            },
            id="CategoricalChoices | CategoricalChoices",
        ),
        pytest.param(
            {
                "x": CategoricalChoices([{"a": "a1"}, "b"]),
                "y": {"c": "c1"},
                "should_raise": True,
            },
            id="CategoricalChoices, but a choice is not a dict",
        ),
    ],
)
def test_union_categorical_choices(params):
    """Tests the "|" operator for CategoricalChoices."""
    x = params["x"]
    y = params["y"]
    should_raise = params.get("should_raise", False)

    if should_raise:
        with pytest.raises(ValueError):
            result = x | y
        return

    expected: CategoricalChoices = params["expected"]
    result: CategoricalChoices = x | y

    assert len(result.choices) == len(expected.choices)
    for choice in result.choices:
        assert choice in expected.choices
