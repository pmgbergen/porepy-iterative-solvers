"""Classes here express the solver space:

- SolverSpace describes the whole space.
- CategoricalChoices - wraps a list of categorical options.
- NumericalChoices - wraps a list of numerical options.

Also see utility functions:

- explain_decision
- explain_decisions

which help understand the solver configuration encoding and get insights about which
parameters affect solver performance.

"""

from __future__ import annotations

from itertools import count, product
from typing import Any, Iterable, Optional, Self

import numpy as np


class CategoricalChoices:
    """Wraps a list of categorical options. Example usage:
    ```
    CategoricalChoices(['ilu', 'gamg', 'pbjacobi'])
    ```
    describes a choice from 3 PETSc options.

    Options can be dicts:
    ```
    CategoricalChoices([
        {
            'pc_type': 'hypre',
            'hypre_boomeramg_strong_threshold': 0.7
        },
        {
            'pc_type': 'ilu',
            'pc_factor_levels': 1,
        },
    ])
    ```
    describes 2 configurations: either AMG with a custom threshold or ILU with a custom
    number of factorization levels.

    Options can be nested:
    ```
    CategoricalChoices([
        {
            'pc_type': 'hypre',
            'hypre_boomeramg_strong_threshold': 0.7
        },
        {
            'pc_type': 'ilu',
            'pc_factor_mat_ordering_type': CategoricalChoices(['natural', 'rcm']),
        },
    ])
    ```
    describes 3 configurations: either AMG or ilu with a natural ordering or ilu
    with an "rcm" ordering. Nested dictionaries are also allowed.

    CategoricalChoices with dictionaries of options can be merged with a `dict` of
    constant options using the "|" operator, same as the normal dictionaries with each
    other:
    ```
    CategoricalChoices([
        {
            'pc_type': 'hypre',
        },
        {
            'pc_type': 'ilu',
        },
    ]) | {
        'mat_type': 'mpiaij'
    }
    ```
    is equivalent to:
    ```
    CategoricalChoices([
        {
            'pc_type': 'hypre',
            'mat_type': 'mpiaij',
        },
        {
            'pc_type': 'ilu',
            'mat_type': 'mpiaij',
        },
    ])
    ```

    """

    def __init__(self, choices: list[dict | Any]) -> None:
        self.choices: list[dict | Any] = choices
        self.id: int = -1  # Will be assigned in _build_decision_tree function.

    def __repr__(self) -> str:
        return f"CategoricalChoices({self.choices})"

    def __or__(self: Self, value: Any) -> CategoricalChoices:
        if not isinstance(value, dict):
            raise ValueError(
                "Can only do `__or__` with CategoricalChoices and dict of constant "
                f"options. Passed: {type(value)}"
            )
        updated_choices = []
        for choice in self.choices:
            if not isinstance(choice, dict):
                raise ValueError(
                    "Can only do `__or__` with CategoricalChoices, where each choice is"
                    " a dict."
                )
            updated_choices.append(choice | value)
        return CategoricalChoices(choices=updated_choices)

    def __ror__(self: Self, value: Any) -> CategoricalChoices:
        return self.__or__(value)


class NumericalChoices:
    """Wraps a list of numerical options. Example usage:
    ```
    NumericalChoices([1, 3, 5])
    ```

    Can be used inside `CategoricalChoices`, but not the other way around:
    ```
    CategoricalChoices([
        {
            'pc_type': 'hypre',
            'hypre_boomeramg_strong_threshold': NumericalChoices([0.5, 0.7, 0.9]),
        },
        {
            'pc_type': 'ilu',
            'pc_factor_levels': NumericalChoices([0, 1]),
        },
    ])
    ```

    """

    def __init__(
        self,
        choices: Iterable,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        self.choices: np.ndarray = np.array(choices, dtype=dtype)
        if self.choices.dtype == object:
            raise ValueError("Nested choices not supported. Use sub-dictionaries.")
        self.dtype: np.dtype = dtype or self.choices.dtype

        self.tag: str | None = None  # Will be assigned in _build_decision_tree.
        self.id: int = -1  # Will be assigned in _initialize_decision_ids function.

    def __repr__(self) -> str:
        return f"NumericalChoices({self.tag}: {self.choices})"

    def __str__(self) -> str:
        return (
            f"{self.tag}: Choices from {min(self.choices)} to {max(self.choices)}, "
            f"len = {len(self.choices)}"
        )


class SolverSpace:
    """A finite space of solver configurations with the following read-only public
    properties:

    - `num_category_choices`: number of categorical choices in the solver space.

    - `num_numerical_choices`: number of categorical choices in the solver space.

    - `all_decisions_encoding`: array of
        `shape=(num_configurations, num_categorical_choices+num_numerical_choices)`
        where `all_decisions_encoding[i]` is a vector encoding the i-th solver
        configuration.

    Call `config_from_decision` method to construct a human-readable solver
    configuration from an encoded decision.

    """

    def __init__(
        self,
        solver_space_scheme: dict | CategoricalChoices | NumericalChoices | Any,
    ) -> None:
        self._solver_space_scheme: (
            dict | CategoricalChoices | NumericalChoices | Any
        ) = solver_space_scheme
        """The original scheme of the solver space, passed by a user."""
        self._decision_tree: _DecisionNode = _DecisionNode(solver_space_scheme)
        """An tree structure representing the solver space. Leaves are particular
        decisions, non-leaf nodes represents a choice to be made. A path from the tree
        root to a single or multiple leaves represents a solver configuration.
        
        """
        _build_decision_tree(
            solver_space_scheme,
            self._decision_tree,
            options_key="root",
            id_counter=count(0),
        )

        self._flat_complete_solver_confs: list[_FlatCompleteSolverConfig] = (
            self._decision_tree.list_possible_solvers()
        )
        """Each entry represents a particular solver configuration."""

        num_category_choices, num_numerical_choices = _initialize_decision_ids(
            self._flat_complete_solver_confs
        )
        self.num_category_choices: int = num_category_choices
        """Number of categorical choices in the solver space."""
        self.num_numerical_choices: int = num_numerical_choices
        """Number of numerical choices in the solver space."""
        self.all_decisions_encoding: np.ndarray = _make_all_decisions_encoding(
            flat_complete_solver_conf=self._flat_complete_solver_confs,
            num_category_choices=num_category_choices,
            num_numerical_choices=num_numerical_choices,
        )
        """Array of
        `shape=(num_configurations, num_categorical_choices+num_numerical_choices)`
        where `all_decisions_encoding[i]` is a vector encoding the i-th solver
        configuration."""

    def config_from_decision(self, decision: np.ndarray) -> dict | Any:
        """Constructs a human-readable solver configuration from an encoded decision.
        Config of this format should be passed to
            `pp_solvers.PetscKSPScheme.make_solver(mat_orig=..., options=this_config)`.

        Parameters:
            decision: i-th solver configuration from the solver space corresponds to
                `all_decisions_encoding[i]`.

        """
        # Recursion start.
        return self._config_from_decision(
            decision=decision,
            decision_tree=self._decision_tree,  # tree root
            solver_space_scheme=self._solver_space_scheme,  # full solver space scheme
        )

    def _config_from_decision(
        self,
        decision: np.ndarray,
        decision_tree: _DecisionNode,
        solver_space_scheme: dict | Any,
    ) -> dict | Any:
        """Recursively traverses `decision_tree` and `solver_space_scheme`. Replaces
        encountered `NumericalChoices` and `CategoricalChoices` by a particular decision
        encoded in `decision`.

        """
        # What is the current subset of solver_space_scheme we're working with?
        if isinstance(solver_space_scheme, dict):
            # Traverse each key-value pair.
            config = {}
            for key, value in solver_space_scheme.items():
                config[key] = self._config_from_decision(
                    decision=decision,
                    decision_tree=decision_tree,
                    solver_space_scheme=value,
                )

        elif isinstance(solver_space_scheme, NumericalChoices):
            assert solver_space_scheme.id != -1, (
                "The id must be initialized in _initialize_decision_ids."
            )
            # `decision` is arranged as [categorical_choices, numerical_choices].
            # We offset categorical decisions to get the right index...
            choice_idx = solver_space_scheme.id + self.num_category_choices
            decision_value = decision[choice_idx]
            # ... and extract the value.
            config = decision_value.astype(solver_space_scheme.dtype)

        elif isinstance(solver_space_scheme, CategoricalChoices):
            is_chosen = False
            assert solver_space_scheme.id != -1, (
                "The id must be initialized in _build_decision_tree."
            )

            # Our current subtree is a node which may hold multiple forks. Example when
            # it's more than one (think poromechanics):
            #       decision_tree   <- we are here
            #       /           \
            #     fork1         fork2
            #     mechanics     flow
            #     |    |        |   \
            #    gamg  hypre   gamg  hypre
            try:
                fork = next(
                    c for c in decision_tree.children if c.id == solver_space_scheme.id
                )
            except StopIteration:
                assert False, "Something really bad happened to the tree structure."

            # Now when we found the right fork, we need to find the chosen option.
            for choice in fork.children:
                choice_idx = choice.id
                is_chosen = decision[choice_idx] == 1  # Assuming it can be only 0 or 1.
                if is_chosen:
                    config = self._config_from_decision(
                        decision=decision,
                        decision_tree=choice,
                        solver_space_scheme=choice.solver_space_scheme,
                    )
                    break
            assert is_chosen, "Something really bad happened to the decision array."

        else:
            # Default case type(solver_space_scheme) not in [dict, NumericalChoices,
            # CategoricalChoices]. This can be anything the user put into the config,
            # and we have nothing to do with it.
            config = solver_space_scheme

        return config


def explain_decision(solver_space: SolverSpace, decision_idx: int, sep=" - ") -> str:
    """`solver_space.all_decisions_encoding[i]` correponds to the i-th solver
    configuration. But what does `solver_space.all_decisions_encoding[:, k]` refer to?

    This function answers this question.

    Parameters:
        solver_space: the solver space.
        decision_idx: explain what k-th entry in the decision vector means.
        sep: separator in a result string.

    Returns:
        Explained decision string.

    """
    # `decision` is arranged as [categorical, numerical]. First we figure out, which one
    # is ours.
    if decision_idx < solver_space.num_category_choices:
        is_categorical = True
    elif decision_idx < (
        solver_space.num_category_choices + solver_space.num_numerical_choices
    ):
        decision_idx -= solver_space.num_category_choices
        is_categorical = False
    else:
        raise IndexError(f"{decision_idx} is out of bounds.")

    prefix: list[str] = []

    def find_child(
        node: _DecisionNode | _ForkNode,
    ) -> _DecisionNode | _ForkNode | NumericalChoices | None:
        """Recursively traverse the decision tree to find the node referring to our
        decision.

        Appending the path we traversed to the decision to `prefix`.

        """
        if isinstance(node, _DecisionNode):
            # DecisionNode might have the `decision_idx` we are looking for.
            # Either in numerical choices...
            if not is_categorical:
                for numerical_choice in node.numerical_choices:
                    if numerical_choice.id == decision_idx:
                        assert numerical_choice.tag is not None
                        prefix.append(numerical_choice.tag)
                        return numerical_choice
            # Or in categorical choices...
            elif node.id == decision_idx:  # categorical
                return node

        # If not found in this node, looking at its children.
        for child in node.children:
            result = find_child(child)

            if result is None:
                continue  # Not found.

            # Found it! Now each parent node will append itself to the prefix.
            # This is how we trace the path from the root to the found node.
            if isinstance(result, _DecisionNode):
                if isinstance(result.solver_space_scheme, dict):
                    prefix.append(f"(id={result.id})")
                else:
                    prefix.append(str(result.solver_space_scheme))
            elif isinstance(result, _ForkNode):
                prefix.append(str(result.options_key))

            return node

        return None  # Nothing found.

    result = find_child(solver_space._decision_tree)
    assert result is not None

    return sep.join(reversed(prefix))


def explain_decisions(solver_space: SolverSpace) -> list[str]:
    """Calls `explain_decision` for each decision in the encoded decision space."""
    decision_names = []
    for i in range(solver_space.num_category_choices):
        decision_names.append(explain_decision(solver_space, i))

    for i in range(solver_space.num_numerical_choices):
        decision_names.append(
            explain_decision(solver_space, i + solver_space.num_category_choices)
        )
    return decision_names


class _FlatCompleteSolverConfig:
    """An internal representation of a complete solver configurations (all decisions are
    made).

    """

    def __init__(
        self,
        categorical: Optional[list[_DecisionNode]] = None,
        numerical: Optional[list[NumericalChoices]] = None,
    ) -> None:
        self.categorical: list[_DecisionNode] = categorical or []
        """Categorical decisions that lead to this solver configuration."""
        self.numerical: list[NumericalChoices] = numerical or []
        """Numerical decisions that lead to this solver configuration."""

    def __repr__(self) -> str:
        return f"FlatSolverConfig({self.categorical}, {self.numerical})"


class _DecisionNode:
    """A node of a tree describing the solver space. There are two types of nodes:
    `_DecisionNode` and `_ForkNode`. The difference is:
    - `_DecisionNode` describes a determined part of the solver configuration (with
        potentially undetermined childern).
    - `_ForkNode` describes a choice to be made during solver selection.

    Children of a `_DecisionNode` are always `_ForkNode`s, and vice versa.

    """

    def __init__(
        self, solver_space_scheme: dict | CategoricalChoices | NumericalChoices | Any
    ) -> None:
        self.solver_space_scheme: dict | CategoricalChoices | NumericalChoices | Any = (
            solver_space_scheme
        )
        """A part of the solver space configuration passed by a user, which produces
        this node and its children.

        """
        self.children: list[_ForkNode] = []
        """Child nodes."""
        self.numerical_choices: list[NumericalChoices] = []
        """Numerical choices can be attached to each `_DecisionNode`."""

        self.id: int = -1  # This will be set in _initialize_decision_ids.

    def string(self, prefix: str = "") -> str:
        """Human-readible representation of the node and its children for debug and
        analysis.

        """
        if not isinstance(self.solver_space_scheme, dict):
            return f"{prefix}{self.solver_space_scheme}"
        k, v = next(iter(self.solver_space_scheme.items()))
        if isinstance(v, CategoricalChoices):
            v = "CategoricalChoices"
        repr = f"{prefix}{k}: {v}"
        child_prefix = f"{prefix}| "
        if len(self.numerical_choices) > 0:
            numerical_repr = [f"{child_prefix}{v}" for v in self.numerical_choices]
            tmp = "\n".join(numerical_repr)
            repr = f"{repr}\n{tmp}"
        if len(self.children) > 0:
            child_repr = [child.string(prefix=child_prefix) for child in self.children]
            tmp = "\n".join(child_repr)
            repr = f"{repr}\n{tmp}"
        return repr

    def __repr__(self) -> str:
        if isinstance(self.solver_space_scheme, dict):
            k, v = next(iter(self.solver_space_scheme.items()))
            return f"DecisionNode({k}, {v})"

        return f"DecisionNode({self.solver_space_scheme})"

    def __str__(self) -> str:
        return self.string()

    def list_possible_solvers(self) -> list[_FlatCompleteSolverConfig]:
        """Generates a list of all complete configurations defined by this solver space.

        You should call this method on the tree root.

        """
        my_numerical_choices = list(self.numerical_choices)
        if len(self.children) == 0:
            flat_config = _FlatCompleteSolverConfig(numerical=my_numerical_choices)
            return [flat_config]

        children_solver_spaces = [c.list_possible_solvers() for c in self.children]

        merged_results = []
        for tuple_of_decisions in list(product(*children_solver_spaces)):
            cat = [x for conf in tuple_of_decisions for x in conf.categorical]
            num = [x for conf in tuple_of_decisions for x in conf.numerical]
            num.extend(my_numerical_choices)
            merged_results.append(
                _FlatCompleteSolverConfig(categorical=cat, numerical=num)
            )
        return merged_results


class _ForkNode:
    """A node of a tree describing the solver space. There are two types of nodes:
    `_DecisionNode` and `_ForkNode`. The difference is:
    - `_DecisionNode` describes a determined part of the solver configuration (with
        potentially undetermined childern).
    - `_ForkNode` describes a choice to be made during solver selection.

    Children of a `_ForkNode` are always `_DecisionNode`s, and vice versa.

    """

    def __init__(
        self, categorical_choices: CategoricalChoices, options_key: str, id: int
    ) -> None:
        self.options_key: str = options_key
        """The key this fork node refers to."""
        self.categorical_choices: CategoricalChoices = categorical_choices
        """The choices to choose from."""
        self.children: list[_DecisionNode] = []
        """Child nodes."""
        self.id: int = id
        """Initialized in `_build_decision_tree` function."""

    def __repr__(self) -> str:
        return f"ForkNode({self.options_key}, id={self.id})"

    def __str__(self) -> str:
        return self.string()

    def string(self, prefix: str = "") -> str:
        num = len(self.categorical_choices.choices)
        repr = f"{prefix}{self.options_key} (fork with {num} branches):"
        child_prefix = f"{prefix}| "
        if len(self.children) > 0:
            child_repr = [child.string(prefix=child_prefix) for child in self.children]
            tmp = "\n".join(child_repr)
            repr = f"{repr}\n{tmp}"
        return repr

    def list_possible_solvers(self) -> list[_FlatCompleteSolverConfig]:
        """Generates a list of all complete configurations defined by this solver space.

        You should call this method on the tree root.

        """
        assert len(self.children) != 0, (
            "Something went wrong in _build_decision_tree. Why _ForkNode with no "
            "children?"
        )

        solver_space = []
        for child in self.children:
            child_solver_space = child.list_possible_solvers()
            for solver in child_solver_space:
                solver.categorical.append(child)
            solver_space.extend(child_solver_space)
        return solver_space


def _build_decision_tree(
    solver_space_scheme: dict | CategoricalChoices | NumericalChoices | Any,
    current_decision_node: _DecisionNode,
    options_key: str,
    id_counter: count[int],
) -> None:
    """Recursively traverses the `solver_space_scheme` and builds the decision tree from
    the root node.

    Parameters:
        solver_space_scheme: The current sub-configuration to build a tree based on. You
            should pass a full configuration from the outside here.
        current_decision_node: The current sub-tree to grow leaves. You should pass a
            root node from the outside here.
        options_key: A key of the current subtree.
        id_counter: A generator of unique incrementing identifiers.

    """
    # What is our current sub-config?
    match solver_space_scheme:
        case solver_space_scheme if isinstance(solver_space_scheme, dict):
            # It is a dict - building leaves for each key-value pair.
            for key, value in solver_space_scheme.items():
                _build_decision_tree(
                    solver_space_scheme=value,
                    current_decision_node=current_decision_node,
                    options_key=key,
                    id_counter=id_counter,
                )

        case categorical_choice if isinstance(categorical_choice, CategoricalChoices):
            # It is a categorical decision. Making a fork node as a child.
            fork_node = _ForkNode(
                categorical_choice, options_key=options_key, id=next(id_counter)
            )
            assert categorical_choice.id == -1, "Id should not be initialized twice."

            # Categorical choice gets the id of the corresponding fork node.
            categorical_choice.id = fork_node.id
            current_decision_node.children.append(fork_node)
            for decision in categorical_choice.choices:
                # Appending decision nodes to represent chosen decisions within the fork
                # node.
                new_decision_node = _DecisionNode(decision)
                fork_node.children.append(new_decision_node)

                # Traversing sub-configs of each decision.
                _build_decision_tree(
                    solver_space_scheme=decision,
                    current_decision_node=new_decision_node,
                    options_key=options_key,
                    id_counter=id_counter,
                )

        case numerical_choices if isinstance(numerical_choices, NumericalChoices):
            # It is a numerical decision. Attaching numerical decision to the current
            # node.
            if numerical_choices.tag is None:
                numerical_choices.tag = options_key
            current_decision_node.numerical_choices.append(numerical_choices)

        # Default case: Do nothing (end of recursion).


def _initialize_decision_ids(
    flat_complete_solver_confs: list[_FlatCompleteSolverConfig],
) -> tuple[int, int]:
    """Initializes and counts ids for `_DecisionNode`s and `NumericalChoices`.

    Returns:
        Count of categorical and numerical choices in the solver space.
    """
    # At this stage, the ids must be uninitialized. We ensure it here.
    for solver in flat_complete_solver_confs:
        for choice in solver.categorical:
            assert choice.id == -1
        for numerical_choice in solver.numerical:
            assert numerical_choice.id == -1

    category_choices_counter = count()
    numerical_choices_counter = count()

    # We may encounter the same choice more than once. Only initializing those we've not
    # seen earlier.
    for solver in flat_complete_solver_confs:
        for choice in solver.categorical:
            if choice.id == -1:
                choice.id = next(category_choices_counter)

        for numerical_choice in solver.numerical:
            if numerical_choice.id == -1:
                numerical_choice.id = next(numerical_choices_counter)

    return next(category_choices_counter), next(numerical_choices_counter)


def _make_all_decisions_encoding(
    flat_complete_solver_conf: list[_FlatCompleteSolverConfig],
    num_category_choices: int,
    num_numerical_choices: int,
) -> np.ndarray:
    """Builds a 2D array of
    `shape=(num_configurations, num_categorical_choices+num_numerical_choices)`
    representing the whole solver space, encoded. Each element is a vector that encodes
    a complete solver configuration.

    The first `num_categorical_choices` elements of the vector are ones and zeros,
    describing chosen and not chosen options in the categorical forks.

    The last `num_numerical_choices` elements take real or integer values, each
    corresponding to a particular numerical choice.

    """
    all_possible_decisions: list[np.ndarray] = []
    for solver in flat_complete_solver_conf:
        categorical_decision = [choice.id for choice in solver.categorical]

        categorical_encoding = np.zeros((1, num_category_choices))
        categorical_encoding[:, categorical_decision] = 1

        numerical_encoding = [np.zeros(1) for _ in range(num_numerical_choices)]
        for choice in solver.numerical:
            numerical_encoding[choice.id] = choice.choices
        x = np.atleast_2d(np.meshgrid(*numerical_encoding, indexing="ij"))
        if x.size != 0:
            x = x.reshape(num_numerical_choices, -1).T
        categorical_encoding = np.broadcast_to(
            categorical_encoding, (x.shape[0], num_category_choices)
        )
        encoding = np.concatenate([categorical_encoding, x], axis=1)
        all_possible_decisions.append(encoding)
    return np.concatenate(all_possible_decisions, axis=0)
