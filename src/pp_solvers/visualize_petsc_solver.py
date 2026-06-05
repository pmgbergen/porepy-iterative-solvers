"""Block-diagram visualization of PETSc solver configurations.

Each box in the diagram represents one PETSc KSP+PC object. The box shows
all PETSc command-line options that object owns, with ksp_type and pc_type
highlighted in bold.

Usage::

    from pp_solvers.preconditioners import mass_balance_factory
    from pp_solvers.visualize_petsc_solver import visualize_petsc_solver

    cfg = mass_balance_factory()
    visualize_petsc_solver(cfg, filename="my_solver", fmt="svg", view=True)

Requires the ``graphviz`` Python package (``pip install graphviz``) and the
Graphviz system binaries.
"""

from __future__ import annotations

import html
from typing import Union

from pp_solvers.dof_manager import DofManager
from pp_solvers.preconditioners import (
    GMRES,
    LinearSolverConfiguration,
    PetscKspPcConfiguration,
    PythonPermutationWrapper,
)

__all__ = ["visualize_petsc_solver"]

_BOLD_OPTIONS = frozenset({"ksp_type", "pc_type"})


def _collect_all_options(
    root: PetscKspPcConfiguration, user_options: dict, dof_manager: DofManager
) -> tuple[dict[str, str], dict[str, str]]:
    """Call petsc_options twice — with and without user_options.

    Returns:
        (default_options, custom_options) where keys in custom_options that
        differ from default_options were changed by user_options.
    """

    def _call(uo):
        try:
            return {
                k: str(v)
                for k, v in root.petsc_options(
                    user_options=uo, dof_manager=dof_manager
                ).items()
            }
        except Exception:
            return {}

    return _call({}), _call(user_options)


def _own_options(key: str, all_options: dict[str, str]) -> dict[str, str]:
    """Return options whose prefix matches *key*, stripping that prefix."""
    prefix = f"{key}_"
    return {k[len(prefix) :]: v for k, v in all_options.items() if k.startswith(prefix)}


def _get_children(node: PetscKspPcConfiguration) -> list[PetscKspPcConfiguration]:
    children = node.get_children()
    if not children and isinstance(node, PythonPermutationWrapper):
        children = [node.inner_subsolver]
    return children


def _node_label(
    node: PetscKspPcConfiguration,
    default_options: dict[str, str],
    custom_options: dict[str, str],
) -> str:
    class_name = type(node).__name__
    if isinstance(node, GMRES):
        class_name = f"GMRES + {type(node.preconditioner).__name__}"

    header = (
        f'<TR><TD COLSPAN="2" BGCOLOR="#2a4d8f">'
        f'<FONT COLOR="white"><B>{html.escape(node.key)}</B></FONT>'
        f'<BR/><FONT COLOR="#b8ccf0" POINT-SIZE="10">{html.escape(class_name)}</FONT>'
        f"</TD></TR>"
    )

    # Merge: show all options from custom_options, falling back to defaults for
    # any key not overridden. custom_options may add or change entries.
    merged = {
        **_own_options(node.key, default_options),
        **_own_options(node.key, custom_options),
    }
    defaults = _own_options(node.key, default_options)

    rows = []
    for name, value in merged.items():
        esc_name = html.escape(name)
        esc_value = html.escape(value)
        user_changed = defaults.get(name) != value
        bold = name in _BOLD_OPTIONS

        if user_changed:
            name_cell = f'<FONT COLOR="#cc0000" POINT-SIZE="9">{esc_name}</FONT>'
            value_cell = f'<FONT COLOR="#cc0000" POINT-SIZE="9">{esc_value}</FONT>'
            row_bg = ""
        elif bold:
            name_cell = f"<B>{esc_name}</B>"
            value_cell = f"<B>{esc_value}</B>"
            row_bg = ' BGCOLOR="#e8eefc"'
        else:
            name_cell = f'<FONT POINT-SIZE="9">{esc_name}</FONT>'
            value_cell = f'<FONT POINT-SIZE="9">{esc_value}</FONT>'
            row_bg = ""

        rows.append(
            f"<TR{row_bg}>"
            f'<TD ALIGN="LEFT">{name_cell}</TD>'
            f'<TD ALIGN="LEFT">{value_cell}</TD>'
            f"</TR>"
        )

    body = header + "".join(rows)
    return (
        f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
        f"{body}</TABLE>>"
    )


def _build_graph(
    dot,
    node: PetscKspPcConfiguration,
    default_options: dict[str, str],
    custom_options: dict[str, str],
    visited: set,
) -> None:
    node_id = str(id(node))
    if node_id in visited:
        return
    visited.add(node_id)

    label = _node_label(node, default_options, custom_options)
    dot.node(node_id, label=label, shape="none", margin="0")

    for child in _get_children(node):
        _build_graph(dot, child, default_options, custom_options, visited)
        dot.edge(node_id, str(id(child)))


def visualize_petsc_solver(
    configuration: Union[PetscKspPcConfiguration, LinearSolverConfiguration],
    user_options: dict,
    dof_manager: DofManager,
    filename: str = "petsc_solver",
    fmt: str = "svg",
    view: bool = False,
) -> str:
    """Render a PETSc solver configuration as a block diagram.

    Each box represents one PETSc KSP+PC object and lists the PETSc options it
    owns. ``ksp_type`` and ``pc_type`` are shown in bold. Options modified by
    ``user_options`` are highlighted in red.

    Parameters:
        configuration: Root solver or ``LinearSolverConfiguration`` wrapper.
        user_options: User-supplied option overrides (same format as passed to
            ``petsc_options``). Options that differ from the defaults are shown
            in red.
        dof_manager: The ``DofManager`` for the problem.
        filename: Output file base name (without extension).
        fmt: Graphviz output format — ``"svg"``, ``"png"``, ``"pdf"``, etc.
        view: Open the rendered file after creation.

    Returns:
        Path to the rendered output file.
    """
    import graphviz

    if isinstance(configuration, LinearSolverConfiguration):
        configuration = configuration.solver

    default_options, custom_options = _collect_all_options(
        configuration, user_options=user_options, dof_manager=dof_manager
    )

    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": "TB",
            "splines": "ortho",
            "nodesep": "0.6",
            "ranksep": "0.8",
        },
        node_attr={"fontname": "Helvetica"},
        edge_attr={"arrowhead": "open"},
    )

    _build_graph(dot, configuration, default_options, custom_options, set())

    return dot.render(filename, format=fmt, view=view, cleanup=True)
