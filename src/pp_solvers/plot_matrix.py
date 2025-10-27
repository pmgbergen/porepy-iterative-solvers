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


def spy(
    mat: spmatrix,
    show: bool = True,
    aspect: Literal["equal", "auto"] = "equal",
    marker: Optional[str] = None,
) -> None:
    if marker is None:
        marker = "+"
        if max(*mat.shape) > 300:
            marker = ","
    plt.spy(mat, marker=marker, markersize=4, color="black", aspect=aspect)
    if show:
        plt.show()


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


def plot_mat(
    mat: spmatrix,
    log: bool = True,
    show: bool = True,
    threshold: float = 1e-30,
    aspect: Literal["equal", "auto"] = "equal",
) -> None:
    mat = mat.copy()
    try:
        mat = mat.toarray()
    except AttributeError:
        pass

    mat[abs(mat) < threshold] = np.nan
    if log:
        mat = np.log10(abs(mat))

    plt.matshow(mat, fignum=0, aspect=aspect)
    plt.colorbar()
    if show:
        plt.show()
