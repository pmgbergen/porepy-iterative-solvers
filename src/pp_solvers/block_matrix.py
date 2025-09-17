from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import matplotlib
import numpy as np
import scipy.linalg
import scipy.sparse
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, spmatrix, coo_matrix


from .plot_utils import plot_mat, spy

__all__ = ["BlockMatrixStorage"]


def color_spy(
    mat: spmatrix,
    row_idx: list[list[int]],
    col_idx: list[list[int]],
    row_names: Optional[list[str]] = None,
    col_names: Optional[list[str]] = None,
    aspect: Literal["equal", "auto"] = "equal",
    show: bool = False,
    marker: Optional[str] = None,
    draw_marker: bool = True,
    color: bool = True,
    hatch: bool = True,
    alpha: float = 0.3,
) -> None:
    if draw_marker:
        spy(mat, show=False, aspect=aspect, marker=marker)
    else:
        spy(csr_matrix(mat.shape), show=False, aspect=aspect)

    row_sep = [0]
    for row in row_idx:
        row_sep.append(row[-1] + 1)
    row_sep = sorted(row_sep)

    col_sep = [0]
    for col in col_idx:
        col_sep.append(col[-1] + 1)
    col_sep = sorted(col_sep)

    if row_names is None:
        row_names = [str(i) for i in range(len(row_sep) - 1)]
    if col_names is None:
        col_names = [str(i) for i in range(len(col_sep) - 1)]

    hatch_types = itertools.cycle(["/", "\\"])

    ax = plt.gca()
    row_label_pos = []
    for i in range(len(row_names)):
        ystart, yend = row_sep[i : i + 2]
        row_label_pos.append(ystart + (yend - ystart) / 2)
        kwargs: dict[str, Any] = {}
        if color:
            kwargs["facecolor"] = f"C{i}"
        else:
            kwargs["fill"] = False
        if hatch:
            kwargs["hatch"] = next(hatch_types)
            # kwargs['color'] = 'none'
            kwargs["edgecolor"] = "red"
            # kwargs['facecolor'] = 'blue'

        plt.axhspan(ystart - 0.5, yend - 0.5, alpha=alpha, **kwargs)
    ax.yaxis.set_ticks(row_label_pos)
    ax.set_yticklabels(row_names, rotation=0)

    # hatch_types = itertools.cycle(["|", "-"])

    col_label_pos = []
    for i in range(len(col_names)):
        xstart, xend = col_sep[i : i + 2]
        col_label_pos.append(xstart + (xend - xstart) / 2)
        if color:
            kwargs["facecolor"] = f"C{i}"
        if hatch:
            kwargs["hatch"] = next(hatch_types)
        plt.axvspan(xstart - 0.5, xend - 0.5, alpha=alpha, **kwargs)
    ax.xaxis.set_ticks(col_label_pos)
    ax.set_xticklabels(col_names, rotation=0)

    if show:
        plt.show()


def get_nonzero_indices(
    A: csr_matrix, row_indices: list[np.ndarray], col_indices: list[np.ndarray]
) -> list[int]:
    """
    Get the indices of A.data that correspond to the specified subset of rows and
    columns.

    Parameters:
        A: The input sparse matrix.
        row_indices (list or array): The list of row indices to consider.
        col_indices (list or array): The list of column indices to consider.

    Returns:
    list: Indices in A.data corresponding to non-zero elements in the specified subset.
    """
    result_indices = []
    col_set = set(col_indices)  # For quick lookup

    for row in row_indices:
        start_ptr = A.indptr[row]
        end_ptr = A.indptr[row + 1]

        for data_idx in range(start_ptr, end_ptr):
            col_idx = A.indices[data_idx]
            if col_idx in col_set:
                result_indices.append(data_idx)

    return result_indices


class BlockMatrixStorage:
    """Storage class for block matrices with utility functions for indexing.

    The block matrix bridges three different levels of indexing:
    - The global indexes. These run over all rows and columns of the global matrix.
    - The block indices. Each of these represents an equation (row) or variable (column)
        defined on a geometric entity (e.g., a subdomain or an interface).
    - Groups of blocks. These are collections of blocks that are treated together in the
        solution process. For instance, a group can be all the blocks corresponding to
        fracture contact mechanics (on all fracture subdomains), or the mass
        conservation equation stated on all subdomains. Similar examples can be given
        for variables.

    """

    def __init__(
        self,
        mat: csr_matrix,
        global_dofs_row: list[np.ndarray],
        global_dofs_col: list[np.ndarray],
        groups_to_blocks_row: list[list[int]],
        groups_to_blocks_col: list[list[int]],
        local_dofs_row: Optional[list[np.ndarray]] = None,
        local_dofs_col: Optional[list[np.ndarray]] = None,
        active_groups_row: Optional[list[int]] = None,
        active_groups_col: Optional[list[int]] = None,
        group_names_row: Optional[list[str]] = None,
        group_names_col: Optional[list[str]] = None,
    ):
        self.mat: csr_matrix = mat
        """The matrix itself."""

        self.groups_to_blocks_row: list[list[int]] = groups_to_blocks_row
        """The outer list is the different equation groups specified for the matrix.
        The inner list is the blocks that belong to each equation group."""

        self.groups_to_blocks_col: list[list[int]] = groups_to_blocks_col
        """The outer list is the different variable groups specified for the matrix.
        The inner list is the blocks that belong to each variable group."""

        self.group_names_row: Optional[list[str]] = group_names_row
        """List of group names for the rows."""

        self.group_names_col: Optional[list[str]] = group_names_col
        """List of group names for the columns."""

        def init_global_dofs(global_dofs: list[np.ndarray]):
            # Cast dofs to numpy arrays.
            return [np.atleast_1d(x) for x in global_dofs]

        self.global_dofs_row: list[np.ndarray] = init_global_dofs(global_dofs_row)
        """List of global dofs for the rows. One list item per equation group (EK
        believes)."""

        self.global_dofs_col: list[np.ndarray] = init_global_dofs(global_dofs_col)
        """List of global dofs for the columns. One list item per variable group (EK
        believes)."""

        def init_local_dofs(
            local_dofs: list[np.ndarray] | None, global_dofs: list[np.ndarray]
        ):
            # Cast the local dofs to 1d numpy arrays, unless the list item is None,
            # in which case it is left as None.
            if local_dofs is None:
                local_dofs = global_dofs
            return [np.atleast_1d(x) if x is not None else x for x in local_dofs]

        self.local_dofs_row: list[np.ndarray] = init_local_dofs(
            local_dofs_row, self.global_dofs_row
        )
        """List of local dofs for the rows. One list item per equation group. A list
        item None corresponds to a group of the global matrix that is not active in this
        local matrix."""

        self.local_dofs_col: list[np.ndarray] = init_local_dofs(
            local_dofs_col, self.global_dofs_col
        )

        """List of local dofs for the columns. One list item per variable group. A list
        item None corresponds to a group of the global matrix that is not active in this
        local matrix."""

        def init_active_groups(
            groups_to_blocks: list[list[int]], active_groups: list[int] | None
        ) -> list[int]:
            if active_groups is not None:
                tmp = active_groups
            else:
                tmp = list(
                    np.argsort([x[0] if len(x) else -1 for x in groups_to_blocks])
                )
            # Filter empty groups, e.g., when no fractures are present.
            return [group_idx for group_idx in tmp if len(groups_to_blocks[group_idx])]

        # TODO: What is an active group?
        self.active_groups: tuple[list[int], list[int]] = (
            init_active_groups(groups_to_blocks_row, active_groups_row),
            init_active_groups(groups_to_blocks_col, active_groups_col),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the matrix."""
        return self.mat.shape

    def __repr__(self) -> str:
        return (
            f"BlockMatrixStorage of shape {self.shape} with {self.mat.nnz} elements "
            f"with {len(self.active_groups[0])}x{len(self.active_groups[1])} "
            "active groups"
        )

    def _correct_getitem_key(
        self, key: int | list | slice | tuple
    ) -> tuple[list[int], list[int]]:
        """Helper function to process the key for __getitem__ and __setitem__. See the
        former method for permissible formats.

        """
        # Since the key is defined as a single argument (see __getitem__), passing
        # multiple arguments (e.g., both row and column indices) will be interpreted as
        # a tuple. If the key is a list or a slice, we will assign the same key to the
        # row and column indices.
        if isinstance(key, list):
            key = key, key
        if isinstance(key, slice):
            key = key, key

        # By now, we should have a tuple with two elements, corresponding to the row and
        # column block indices to be extracted.
        assert isinstance(key, tuple)
        assert len(key) == 2

        def correct_key(key_: slice | int, total: int):
            # Convert slice or int to list of indices. Total is the maximum upper bound
            # of a slice, in case it is given on the form `1:` or similar.
            if isinstance(key_, slice):
                start = key_.start or 0
                stop = key_.stop or total
                step = key_.step or 1
                key_ = list(range(start, stop, step))
            try:
                # Try to iterate over the key. If not successful (which means this is an
                # int?), convert to a list.
                iter(key_)
            except TypeError:
                key_ = [key_]
            return key_

        groups_row, groups_col = key
        # Convert the key to a list of indices.
        groups_row = correct_key(groups_row, total=len(self.groups_to_blocks_row))
        groups_col = correct_key(groups_col, total=len(self.groups_to_blocks_col))
        return groups_row, groups_col

    def __getitem__(self, key: list | slice | tuple) -> BlockMatrixStorage:
        """Get a subset of blocks from the matrix. The block indexing is defined
        according to the groups


        The following indexing is supported:

        - `1, 2`: Get the block corresponding to row block index 1 and column block
           index 2. Results in submatrix [J_12].
        - `1, 2]`: Get the blocks corresponding row block indices 1 and 2 and column
           block indices 1 and 2. Results in the submatrix [[J_11, J_12], [J_21, J_22]].
        - `([1, 2], [3, 4])`: Get the blocks corresponding to row block indices 1 and 2
           and column block indices 3 and 4. Results in the submatrix
           [[J_13, J_14], [J_23, J_24]].
        - `:, [1, 2]: Get all row blocks and column blocks 1 and 2. Results in the
           submatrix [[J_11, J_12], [J_21, J_22], ..., [J_m1, J_m2]], where m is the
           maximum row block index.
        - `[1, 2], :`: Get row blocks 1 and 2 and all column blocks. Results in the
           submatrix [[J_11, J_12, ..., J_1n], [J_21, J_22, ..., J_2n]], where n is the
           maximum column block index.
        - `[1, 2], 1:4`: Get row blocks 1 and 2 and column blocks 1 to 3. Results in
           the submatrix [[J_11, J_12, J_13], [J_21, J_22, J_23]].

        Indices can be given by tuples as well as lists. The indexing is 0-based.

        Only active groups can be taken. That is, under nested indexing, a group that is
        not selected in the outer index cannot be selected in the inner index.

        Parameters:
            key: The key to index the matrix. See above for permissible formats.

        Raises:
            ValueError: If an inactive group is selected.

        Returns:
            A block matrix storage object containing the selected blocks.

        """
        # Process input arguments to get lists of row and column indices.
        groups_row, groups_col = self._correct_getitem_key(key)

        def inner(
            input_dofs_idx: list[np.ndarray],
            take_groups: list[int],
            all_groups: list[list[int]],
        ):
            """Expand indices from groups to matrix indices.

            Parameters:
                input_dofs_idx: The local indices for the row or column to be expanded.
                take_groups: The groups to be taken.
                all_groups: All groups available.

            """
            dofs_global_idx = []
            # Initialize the local indices to None. Groups that remain active after this
            # take operation will have their local indices set to the corresponding
            # matrix indices.
            dofs_local_idx = [None] * len(input_dofs_idx)
            offset = 0
            # Loop over the groups that are to be taken.
            for group in take_groups:
                # Loop over all available groups.
                for dof_idx in all_groups[group]:
                    # An inactive group will have a None entry in the local dofs,
                    # instead of matrix indices. This is checked here, and an error is
                    # raised if an inactive group is selected.
                    if input_dofs_idx[dof_idx] is None:
                        raise ValueError(f"Taking inactive row {group}")

                    # Append the global indices for the selected group.
                    dofs_global_idx.append(input_dofs_idx[dof_idx])
                    # Append the local indices for the selected group.
                    dofs_local_idx[dof_idx] = (
                        np.arange(len(input_dofs_idx[dof_idx])) + offset
                    )
                    offset += len(input_dofs_idx[dof_idx])
            if len(dofs_global_idx):
                return np.concatenate(dofs_global_idx), dofs_local_idx
            else:
                return np.array([], dtype=int), dofs_local_idx

        row_idx, local_row_idx = inner(
            self.local_dofs_row, groups_row, self.groups_to_blocks_row
        )
        col_idx, local_col_idx = inner(
            self.local_dofs_col, groups_col, self.groups_to_blocks_col
        )

        rows_expanded, cols_expanded = np.meshgrid(
            row_idx, col_idx, sparse=True, indexing="ij", copy=False
        )
        submat = self.mat[rows_expanded, cols_expanded]

        # Return a new block matrix storage object with the selected blocks. Compared to
        # the current object, the new object potentially has a subset of active groups,
        # with a corresponding subset of local indices.
        return BlockMatrixStorage(
            mat=submat,
            local_dofs_row=local_row_idx,
            local_dofs_col=local_col_idx,
            global_dofs_row=self.global_dofs_row,
            global_dofs_col=self.global_dofs_col,
            groups_to_blocks_col=self.groups_to_blocks_col,
            groups_to_blocks_row=self.groups_to_blocks_row,
            active_groups_row=groups_row,
            active_groups_col=groups_col,
            group_names_col=self.group_names_col,
            group_names_row=self.group_names_row,
        )

    def __setitem__(
        self, key: list | slice | tuple, value: BlockMatrixStorage | spmatrix
    ) -> None:
        """Set method for a BlockMatrixStorage object.

        See __getitem__ for permissible formats of the key.

        Parameters:
            key: The key to index the matrix. See above for permissible formats.
            value: The value to set. This can be a BlockMatrixStorage object, or a
                sparse matrix.

        Raises:
            ValueError: If an inactive group is selected.
        """
        groups_i, groups_j = self._correct_getitem_key(key)

        if isinstance(value, BlockMatrixStorage):
            value = value.mat

        def inner(input_dofs_idx, take_groups, all_groups):
            dofs_idx = []
            for group in take_groups:
                for dof_idx in all_groups[group]:
                    if input_dofs_idx[dof_idx] is None:
                        raise ValueError(f"Taking inactive row {group}")
                    dofs_idx.append(input_dofs_idx[dof_idx])
            return np.concatenate(dofs_idx)

        row_idx = inner(self.local_dofs_row, groups_i, self.groups_to_blocks_row)
        col_idx = inner(self.local_dofs_col, groups_j, self.groups_to_blocks_col)
        row_expanded, col_expanded = np.meshgrid(
            row_idx, col_idx, sparse=True, indexing="ij", copy=False
        )
        self.mat[row_expanded, col_expanded] = value

    def copy(self) -> BlockMatrixStorage:
        res = self.empty_container()
        res.mat = self.mat.copy()
        return res

    def empty_container(self) -> BlockMatrixStorage:
        """Create container with the same structure as the current one, but with an
        empty matrix.
        """

        return BlockMatrixStorage(
            mat=scipy.sparse.csr_matrix(self.mat.shape),
            local_dofs_row=self.local_dofs_row,
            local_dofs_col=self.local_dofs_col,
            global_dofs_row=self.global_dofs_row,
            global_dofs_col=self.global_dofs_col,
            groups_to_blocks_row=self.groups_to_blocks_row,
            groups_to_blocks_col=self.groups_to_blocks_col,
            active_groups_row=self.active_groups[0],
            active_groups_col=self.active_groups[1],
            group_names_col=self.group_names_col,
            group_names_row=self.group_names_row,
        )

    def project_rhs_to_local(self, global_rhs: np.ndarray) -> np.ndarray:
        """Global rhs is the rhs arranged in the porepy model manner. This method
        permutes and restricts the global rhs to make it match the current matrix
        arrangement.

        Parameters:
            global_rhs: The global right hand side.

        Returns:
            np.ndarray: The part of the rhs corresponding to the local dofs, as
                specified by the active groups.

        """
        row_idx = [
            self.global_dofs_row[j]
            for i in self.active_groups[0]
            for j in self.groups_to_blocks_row[i]
        ]
        row_idx = np.concatenate(row_idx)
        return global_rhs[row_idx]

    def project_rhs_to_global(self, local_rhs: np.ndarray) -> np.ndarray:
        """Local rhs is the rhs arranged to match the current matrix. This method
        permutes and prolongates with zeros the local rhs to restore the global
        arrangement.

        Parameters:
            local_rhs: The local right hand side.

        Returns:
            np.ndarray: The global right hand side, with zeros in the items that are
                not part of the active groups.

        """
        row_idx = np.concatenate(
            [
                self.global_dofs_row[j]
                for i in self.active_groups[0]
                for j in self.groups_to_blocks_row[i]
            ]
        )
        total_size = sum(x.size for x in self.global_dofs_col)
        result = np.zeros(total_size, dtype=local_rhs.dtype)
        result[row_idx] = local_rhs
        return result

    def project_solution_to_global(self, x: np.ndarray) -> np.ndarray:
        """The same as `project_rhs_to_global, but in the solution space."""
        col_idx = np.concatenate(
            [
                self.global_dofs_col[j]
                for i in self.active_groups[1]
                for j in self.groups_to_blocks_col[i]
            ]
        )
        total_size = sum(x.size for x in self.global_dofs_col)
        result = np.zeros(total_size)
        result[col_idx] = x
        return result

    def _project_to_global(self, vec: np.ndarray, row: bool) -> np.ndarray:
        # TODO: Replace the two above methods with calls to this one.
        if row:
            idx = self.global_dofs_row
            blocks = self.groups_to_blocks_row
        else:
            idx = self.global_dofs_col
            blocks = self.groups_to_blocks_col
        all_idx = np.concatenate(
            [idx[j] for i in self.active_groups[0] for j in blocks[i]]
        )
        total_size = sum(x.size for x in self.global_dofs_col)
        result = np.zeros(total_size, dtype=vec.dtype)
        result[all_idx] = vec
        return result

    def set_zeros(
        self, group_row_idx: list[int] | int, group_col_idx: list[int] | int
    ) -> None:
        """Set the values in the given block rows and columns to zeros. Does not change
        the sparsity pattern, so this is much cheaper than doing it in the naive way."""
        group_row_idx, group_col_idx = self._correct_getitem_key(
            (group_row_idx, group_col_idx)
        )
        all_rows, all_cols = self.get_active_local_dofs(grouped=True)

        groups_row, groups_col = self.active_groups

        nonzero_idx = get_nonzero_indices(
            A=self.mat,
            row_indices=np.concatenate(
                [all_rows[groups_row.index(i)] for i in group_row_idx]
            ),
            col_indices=np.concatenate(
                [all_cols[groups_col.index(i)] for i in group_col_idx]
            ),
        )
        self.mat.data[nonzero_idx] = 0

    def set_diagonal(
        self, groups: list[int] | int, values: np.ndarray, additive: bool = False
    ) -> None:
        """Adds the values to the main diagonal of the given groups. This method avoids
        allocating a dense matrix.

        """
        groups, _ = self._correct_getitem_key(groups)
        row_idx = np.concatenate(
            [
                self.local_dofs_row[block]
                for group in groups
                for block in self.groups_to_blocks_row[group]
            ]
        )
        col_idx = np.concatenate(
            [
                self.local_dofs_col[block]
                for group in groups
                for block in self.groups_to_blocks_col[group]
            ]
        )
        assert row_idx.size == col_idx.size == values.size

        if not additive:
            values = values - np.array(self.mat[row_idx, col_idx]).ravel()
        tmp = coo_matrix((values, (row_idx, col_idx))).tocsr()
        self.mat += tmp

    # Visualization

    def get_active_local_dofs(self, grouped=False):
        def inner(idx, groups, active_groups):
            data = []
            for active_group in active_groups:
                group_i = groups[active_group]
                group_data = []
                for i in group_i:
                    dofs = idx[i]
                    if dofs is not None:
                        group_data.append(dofs)
                if len(group_data) > 0:
                    data.append(group_data)
            return data

        row_idx = inner(
            self.local_dofs_row, self.groups_to_blocks_row, self.active_groups[0]
        )
        col_idx = inner(
            self.local_dofs_col, self.groups_to_blocks_col, self.active_groups[1]
        )
        if not grouped:
            row_idx = [y for x in row_idx for y in x]
            col_idx = [y for x in col_idx for y in x]
        else:
            row_idx = [np.concatenate(x) for x in row_idx]
            col_idx = [np.concatenate(x) for x in col_idx]
        return row_idx, col_idx

    def get_active_group_names(self):
        def inner(group_names, active_groups):
            if group_names is not None:
                names = [
                    f"{i}: {group_names[i]}" if group_names[i] != "" else str(i)
                    for i in active_groups
                ]
            else:
                names = active_groups
            return names

        row_names = inner(self.group_names_row, self.active_groups[0])
        col_names = inner(self.group_names_col, self.active_groups[1])
        return row_names, col_names

    def color_spy(
        self,
        groups=True,
        show=True,
        aspect: Literal["equal", "auto"] = "equal",
        marker=None,
        color=True,
        hatch=False,
        draw_marker=True,
        alpha=0.3,
    ):
        row_idx, col_idx = self.get_active_local_dofs(grouped=groups)
        if not groups:
            row_names = col_names = None
        else:
            row_names, col_names = self.get_active_group_names()
        color_spy(
            self.mat,
            row_idx,
            col_idx,
            row_names=row_names,
            col_names=col_names,
            show=show,
            aspect=aspect,
            marker=marker,
            alpha=alpha,
            color=color,
            hatch=hatch,
            draw_marker=draw_marker,
        )

    def matshow(
        self,
        log=True,
        show=True,
        threshold: float = 1e-30,
        aspect: Literal["equal", "auto"] = "equal",
    ):
        plot_mat(self.mat, log=log, show=show, threshold=threshold, aspect=aspect)

    def matshow_blocks(self, log=True, show=True, groups=True):
        self.matshow(log=log, show=False)
        self.color_spy(
            show=show, groups=groups, color=False, hatch=True, draw_marker=False
        )

    def plot_max(
        self,
        groups=True,
        annot=True,
        mean=False,
    ):
        row_idx, col_idx = self.get_active_local_dofs(grouped=groups)
        data = []

        for row in row_idx:
            row_data = []
            for col in col_idx:
                ind_i, ind_j = np.meshgrid(
                    row, col, sparse=True, indexing="ij", copy=False
                )
                submat = self.mat[ind_i, ind_j]
                if submat.data.size == 0:
                    row_data.append(np.nan)
                else:
                    if not mean:
                        row_data.append(abs(submat).max())
                    else:
                        row_data.append(abs(submat).mean())
            data.append(row_data)

        if groups:
            y_tick_labels, x_tick_labels = self.get_active_group_names()
        else:
            y_tick_labels = x_tick_labels = "auto"

        ax = plt.gca()
        sns.heatmap(
            data=np.array(data),
            square=False,
            annot=annot,
            norm=matplotlib.colors.LogNorm(),
            fmt=".1e",
            xticklabels=x_tick_labels,
            yticklabels=y_tick_labels,
            ax=ax,
            linewidths=0.01,
            linecolor="grey",
            cbar=False,
            cmap=sns.color_palette("coolwarm", as_cmap=True),
        )

    def color_left_vector(
        self, local_rhs: np.ndarray, groups: bool = True, log: bool = True, label=None
    ):
        y_tick_labels, x_tick_labels = self.get_active_group_names()
        row_idx, col_idx = self.get_active_local_dofs(grouped=groups)
        row_names = y_tick_labels
        alpha = 0.3

        # this repeats the code of color_spy()
        row_sep = [0]
        for row in row_idx:
            row_sep.append(row[-1] + 1)
        row_sep = sorted(row_sep)

        if row_names is None:
            row_names = list(range(len(row_sep) - 1))

        ax = plt.gca()
        row_label_pos = []
        for i in range(len(row_names)):
            ystart, yend = row_sep[i : i + 2]
            row_label_pos.append(ystart + (yend - ystart) / 2)
            kwargs = {}
            kwargs["facecolor"] = f"C{i}"
            plt.axvspan(ystart - 0.5, yend - 0.5, alpha=alpha, kwargs=kwargs)
        ax.xaxis.set_ticks(row_label_pos)
        ax.set_xticklabels(row_names, rotation=45)
        if log:
            local_rhs = abs(local_rhs)
            plt.yscale("log")

        plt.plot(local_rhs, label=label)
