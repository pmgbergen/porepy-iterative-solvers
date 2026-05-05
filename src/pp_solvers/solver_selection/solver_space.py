from itertools import count, product
from typing import Any, Iterable, Optional, Self

import numpy as np


class CategoricalChoices:
    def __init__(self, choices: list[dict | Any]):
        self.choices = choices
        self.id: int = -1  # Will be assigned in build_decision_tree.

    def __repr__(self):
        return f"CategoricalChoices({self.choices})"

    def __or__(self: Self, value: Any) -> "CategoricalChoices":
        # This needs to be test-covered hardly.
        if not isinstance(value, dict):
            raise ValueError(
                "Can only do `__or__` with CategoricalChoices, where each choice is"
                "a dict."
            )
        updated_choices = []
        for choice in self.choices:
            if not isinstance(choice, dict):
                raise ValueError(
                    "Can only do `__or__` with CategoricalChoices, where each choice is"
                    "a dict."
                )
            updated_choices.append(choice | value)
        return CategoricalChoices(choices=updated_choices)

    def __ror__(self: Self, value: Any) -> "CategoricalChoices":
        return self.__or__(value)


class NumericalChoices:
    def __init__(
        self,
        choices: Iterable,
        dtype: Optional[np.dtype] = None,
    ):
        self.choices: np.ndarray = np.array(choices, dtype=dtype)
        if self.choices.dtype == object:
            raise ValueError("Nested choices not supported. Use sub-dictionaries.")
        self.dtype: np.dtype = dtype or self.choices.dtype

        # Will be assigned during the SolverSpace initialization.
        self.tag: str | None = None
        self.id: int = -1

    def __repr__(self):
        return f"NumericalChoices({self.tag}: {self.choices})"

    def __str__(self) -> str:
        return (
            f"{self.tag}: Choices from {min(self.choices)} to {max(self.choices)}, "
            f"len = {len(self.choices)}"
        )


class SolverSpace:
    def __init__(
        self,
        solver_space_scheme: dict | CategoricalChoices | NumericalChoices | Any,
    ):
        self._solver_space_scheme: (
            dict | CategoricalChoices | NumericalChoices | Any
        ) = solver_space_scheme
        self._decision_tree = _DecisionNode(solver_space_scheme)
        _build_decision_tree(
            solver_space_scheme,
            self._decision_tree,
            options_key="root",
            id_counter=count(0),
        )

        self._flat_solver_decisions: list[_FlatSolverDecision] = (
            self._decision_tree.list_possible_solvers()
        )

        num_category_choices, num_numerical_choices = _make_choices_map(
            self._flat_solver_decisions
        )
        self.num_category_choices: int = num_category_choices
        self.num_numerical_choices: int = num_numerical_choices
        self.all_decisions_encoding: np.ndarray = _make_all_decisions_encoding(
            solver_space=self._flat_solver_decisions,
            num_category_choices=num_category_choices,
            num_numerical_choices=num_numerical_choices,
        )

    def config_from_decision(self, decision: np.ndarray):
        return self._config_from_decision(
            decision=decision,
            decision_tree=self._decision_tree,
            solver_space_scheme=self._solver_space_scheme,
        )

    def _config_from_decision(
        self,
        decision: np.ndarray,
        decision_tree: "_DecisionNode",
        solver_space_scheme: dict | Any,
    ):
        if isinstance(solver_space_scheme, dict):
            config = {}
            for key, value in solver_space_scheme.items():
                config[key] = self._config_from_decision(
                    decision=decision,
                    decision_tree=decision_tree,
                    solver_space_scheme=value,
                )

        elif isinstance(solver_space_scheme, NumericalChoices):
            assert solver_space_scheme.id != -1, "This should never happen"
            choice_idx = solver_space_scheme.id + self.num_category_choices
            decision_value = decision[choice_idx]
            config = decision_value.astype(solver_space_scheme.dtype)

        elif isinstance(solver_space_scheme, CategoricalChoices):
            is_chosen = False
            assert solver_space_scheme.id != -1, "This should never happen"
            try:
                fork = next(
                    c for c in decision_tree.children if c.id == solver_space_scheme.id
                )
            except StopIteration:
                assert False, "This should never happen"
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
            assert is_chosen

        else:
            config = solver_space_scheme

        return config


def explain_decision(
    solver_space: SolverSpace, decision_idx: int, sep=" - ", include_ids: bool = False
) -> tuple[str, list]:
    if decision_idx < solver_space.num_category_choices:
        is_categorical = True
    elif decision_idx < (
        solver_space.num_category_choices + solver_space.num_numerical_choices
    ):
        decision_idx -= solver_space.num_category_choices
        is_categorical = False
    else:
        raise IndexError(f"{decision_idx} is out of bounds.")

    def find_child(node: _DecisionNode | _ForkNode):
        if isinstance(node, _DecisionNode):
            if not is_categorical:
                for numerical_choice in node.numerical_choices:
                    if numerical_choice.id == decision_idx:
                        assert numerical_choice.tag is not None
                        prefix.append(numerical_choice.tag)
                        return numerical_choice

            elif node.id == decision_idx:  # categorical
                return node

        for child in node.children:
            result = find_child(child)

            if result is None:
                continue

            # Found it, now we roll the recursion back.

            if isinstance(result, _DecisionNode):
                if isinstance(result.solver_space_scheme, dict):
                    prefix.append(f"(id={result.id})")
                else:
                    prefix.append(str(result.solver_space_scheme))
            elif isinstance(result, _ForkNode):
                prefix.append(str(result.options_key))

            return node
        return None

    prefix: list[str] = []
    ranges: list[Any] = []

    result = find_child(solver_space._decision_tree)
    assert result is not None

    return sep.join(reversed(prefix)), ranges


def explain_decisions(solver_space: SolverSpace, include_ids: bool = False):
    decision_names = []
    decision_ranges = []
    for i in range(solver_space.num_category_choices):
        a, b = explain_decision(solver_space, i, include_ids=include_ids)
        decision_names.append(a)
        decision_ranges.append(b)

    for i in range(solver_space.num_numerical_choices):
        a, b = explain_decision(
            solver_space, i + solver_space.num_category_choices, include_ids=include_ids
        )
        decision_names.append(a)
        decision_ranges.append(b)
    return decision_names, decision_ranges


class _FlatSolverDecision:
    def __init__(
        self,
        categorical: Optional[list["_DecisionNode"]] = None,
        numerical: Optional[list[NumericalChoices]] = None,
    ):
        self.categorical: list[_DecisionNode] = categorical or []
        self.numerical: list[NumericalChoices] = numerical or []

    def __repr__(self):
        return f"FlatSolverConfig({self.categorical}, {self.numerical})"


class _DecisionNode:
    def __init__(
        self, solver_space_scheme: dict | CategoricalChoices | NumericalChoices | Any
    ):
        self.solver_space_scheme: dict | CategoricalChoices | NumericalChoices | Any = (
            solver_space_scheme
        )
        self.children: list[_ForkNode] = []
        self.numerical_choices: list[NumericalChoices] = []
        # This will be set during the initialization of SolverSpace
        self.id: int = -1
        self.tag: str | None = None

    def _str(self, prefix="") -> str:
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
            child_repr = [child._str(prefix=child_prefix) for child in self.children]
            tmp = "\n".join(child_repr)
            repr = f"{repr}\n{tmp}"
        return repr

    def __repr__(self) -> str:
        if isinstance(self.solver_space_scheme, dict):
            k, v = next(iter(self.solver_space_scheme.items()))
            return f"DecisionNode({k}, {v})"

        return f"DecisionNode({self.solver_space_scheme})"

    def __str__(self) -> str:
        return self._str()

    def list_possible_solvers(self) -> list[_FlatSolverDecision]:
        my_numerical_choices = list(self.numerical_choices)
        if len(self.children) == 0:
            flat_config = _FlatSolverDecision(numerical=my_numerical_choices)
            return [flat_config]

        children_solver_spaces = [c.list_possible_solvers() for c in self.children]

        merged_results = []
        for tuple_of_decisions in list(product(*children_solver_spaces)):
            cat = [x for conf in tuple_of_decisions for x in conf.categorical]
            num = [x for conf in tuple_of_decisions for x in conf.numerical]
            num.extend(my_numerical_choices)
            merged_results.append(_FlatSolverDecision(categorical=cat, numerical=num))
        return merged_results


class _ForkNode:
    def __init__(
        self, categorical_choices: CategoricalChoices, options_key: str, id: int
    ):
        self.options_key: str = options_key
        self.categorical_choices: CategoricalChoices = categorical_choices
        self.children: list[_DecisionNode] = []
        self.id: int = id

    def __repr__(self):
        return f"ForkNode({self.options_key}, id={self.id})"

    def __str__(self):
        return self._str()

    def _str(self, prefix="") -> str:
        num = len(self.categorical_choices.choices)
        repr = f"{prefix}{self.options_key} (fork with {num} branches):"
        child_prefix = f"{prefix}| "
        if len(self.children) > 0:
            child_repr = [child._str(prefix=child_prefix) for child in self.children]
            tmp = "\n".join(child_repr)
            repr = f"{repr}\n{tmp}"
        return repr

    def list_possible_solvers(self) -> list[_FlatSolverDecision]:
        if len(self.children) == 0:
            assert False, "Why Fork node with no options?"

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
    id_counter: count,
):
    match solver_space_scheme:
        case solver_space_scheme if isinstance(solver_space_scheme, dict):
            for key, value in solver_space_scheme.items():
                _build_decision_tree(
                    solver_space_scheme=value,
                    current_decision_node=current_decision_node,
                    options_key=key,
                    id_counter=id_counter,
                )
        case categorical_choice if isinstance(categorical_choice, CategoricalChoices):
            fork_node = _ForkNode(
                categorical_choice, options_key=options_key, id=next(id_counter)
            )
            assert categorical_choice.id == -1, "Id should not be initialized twice."
            categorical_choice.id = fork_node.id
            current_decision_node.children.append(fork_node)
            for decision in categorical_choice.choices:
                new_decision_node = _DecisionNode(decision)
                fork_node.children.append(new_decision_node)
                _build_decision_tree(
                    solver_space_scheme=decision,
                    current_decision_node=new_decision_node,
                    options_key=options_key,
                    id_counter=id_counter,
                )
        case numerical_choices if isinstance(numerical_choices, NumericalChoices):
            if numerical_choices.tag is None:
                numerical_choices.tag = options_key
            current_decision_node.numerical_choices.append(numerical_choices)
        # Default: Do nothing (end of recursion).


def _make_choices_map(
    solver_space: list[_FlatSolverDecision],
) -> tuple[int, int]:
    # At this stage, the ids must be uninitialized. We ensure it here.
    for solver in solver_space:
        for choice in solver.categorical:
            choice.id = -1
        for numerical_choice in solver.numerical:
            numerical_choice.id = -1

    category_choices_counter = count()
    numerical_choices_counter = count()

    # We may encounter the same choice more than once.
    for solver in solver_space:
        for choice in solver.categorical:
            if choice.id == -1:
                choice.id = next(category_choices_counter)

        for numerical_choice in solver.numerical:
            if numerical_choice.id == -1:
                numerical_choice.id = next(numerical_choices_counter)

    return next(category_choices_counter), next(numerical_choices_counter)


def _make_all_decisions_encoding(
    solver_space: list[_FlatSolverDecision],
    num_category_choices: int,
    num_numerical_choices: int,
) -> np.ndarray:
    all_possible_decisions: list[np.ndarray] = []
    for solver in solver_space:
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
