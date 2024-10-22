from warnings import warn

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from .utils import *
from .utils import _make_iterable, _is_iter


def _plot_grid(plot_func, data, x, y, color=None, **kwargs):
    MSG_ONLY_2D = "Only 2 of 3 dimensions can be iterables to create grids: x axis, y axis, or color."

    # Determine split_axis
    if _is_iter(x):
        if _is_iter(y):
            assert not _is_iter(color), MSG_ONLY_2D
        elif _is_iter(color):
            assert not _is_iter(y), MSG_ONLY_2D
        else:
            return _plot_grid_from_1d(plot_func, data, x, y, color, "x", **kwargs)
        return _plot_2d_grid(plot_func, data, x, y, color, **kwargs)
    elif _is_iter(y):
        if _is_iter(x):
            assert not _is_iter(color), MSG_ONLY_2D
        elif _is_iter(color):
            assert not _is_iter(x), MSG_ONLY_2D
        else:
            return _plot_grid_from_1d(plot_func, data, x, y, color, "y", **kwargs)
        return _plot_2d_grid(plot_func, data, x, y, color, **kwargs)
    elif _is_iter(color):
        if _is_iter(x):
            assert not _is_iter(y), MSG_ONLY_2D
        elif _is_iter(y):
            assert not _is_iter(x), MSG_ONLY_2D
        else:
            return _plot_grid_from_1d(plot_func, data, x, y, color, "color", **kwargs)
        return _plot_2d_grid(plot_func, data, x, y, color, **kwargs)
    else:
        return _plot_grid_from_1d(plot_func, data, x, y, [color], "color", **kwargs)


def _plot_grid_from_1d(
    plot_func,
    data,
    x,
    y,
    color=None,
    split_axis="color",  # x, y, or color
    ncols=4,
    zero_line_x=False,
    zero_line_y=False,
    linewidth=0,
    zero_linewidth=1,
    legend=True,
    legend_prop=None,
    palette=None,
    sharex=False,
    sharey=False,
    modifier=None,
    rotate_x_labels=None,
    **kwargs,
):
    x = maybe_factor_indices_to_factors(x)
    y = maybe_factor_indices_to_factors(y)
    color = maybe_factor_indices_to_factors(color)

    # Define the variable for the split
    # and plot axes labels
    if split_axis == "color":
        split_vars = maybe_factor_indices_to_factors(_make_iterable(color))
        x_label = x
        y_label = y

    elif split_axis == "x":
        split_vars = maybe_factor_indices_to_factors(_make_iterable(x))
        y_label = y

    elif split_axis == "y":
        split_vars = maybe_factor_indices_to_factors(_make_iterable(y))
        x_label = x

    # Set default colour to black if none set
    if "c" not in kwargs and (color is None or color == [None]):
        kwargs["color"] = "black"

    legend_str = "brief" if (legend and color is not None) else False

    # Figure out rows & columns for the grid with plots
    ncols = min(ncols, len(split_vars))
    nrows = int(np.ceil(len(split_vars) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex=sharex,
        sharey=sharey,
        figsize=(
            ncols * rcParams["figure.figsize"][0],
            nrows * rcParams["figure.figsize"][1],
        ),
    )
    if ncols == 1:
        axes = np.array(axes).reshape(-1, 1)
    if nrows == 1:
        axes = np.array(axes).reshape(1, -1)

    for i, split_var in enumerate(split_vars):
        color_var = color if split_axis != "color" else split_var

        ri = i // ncols
        ci = i % ncols

        # data_ = data.sort_values(color) if color is not None and color != [None] else data

        with sns.axes_style("ticks"), sns.color_palette(palette or "Set2"):

            g = plot_func(
                x=x if split_axis != "x" else split_var,
                y=y if split_axis != "y" else split_var,
                data=data,
                hue=color_var,
                linewidth=linewidth,
                legend=legend_str,
                palette=palette,
                ax=axes[ri, ci],
                **kwargs,
            )
            # Otherwise sns.violinplot still plots the legend
            if legend_str is False:
                try:
                    g.get_legend().remove()
                except AttributeError:
                    pass

            if modifier:
                modifier(split_var=split_var, color_var=color_var, ax=g)

            sns.despine(offset=10, trim=True, ax=g)

            if split_axis == "x":
                x_label = f"Factor{split_var+1}" if isinstance(x, int) else split_var
            elif split_axis == "y":
                y_label = f"Factor{split_var+1}" if isinstance(y, int) else split_var
            g.set(
                xlabel=f"{x_label}",
                ylabel=f"{y_label}",
                title=split_var,
            )

            if legend and color_var:
                if is_numeric_dtype(data[color_var]):
                    means = data.groupby(color_var)[color_var].mean()
                    norm = plt.Normalize(means.min(), means.max())
                    cmap = (
                        palette
                        if palette is not None
                        else sns.cubehelix_palette(as_cmap=True)
                    )
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    try:
                        g.figure.colorbar(sm, ax=axes[ri, ci])
                        g.get_legend().remove()
                    except Exception:
                        warn("Cannot make a proper colorbar")
                else:
                    g.legend(
                        bbox_to_anchor=(1.05, 1),
                        loc=2,
                        borderaxespad=0.0,
                        prop=legend_prop,
                    )

            if rotate_x_labels:
                plt.setp(g.get_xticklabels(), rotation=rotate_x_labels)

            if zero_line_y:
                axes[ri, ci].axhline(
                    0, ls="--", color="lightgrey", linewidth=zero_linewidth, zorder=0
                )
            if zero_line_x:
                axes[ri, ci].axvline(
                    0, ls="--", color="lightgrey", linewidth=zero_linewidth, zorder=0
                )

        # Remove unused axes
        for i in range(len(split_vars), ncols * nrows):
            ri = i // ncols
            ci = i % ncols
            try:
                fig.delaxes(axes[ri, ci])
            except KeyError:
                pass

        plt.tight_layout()

    return g


def _plot_2d_grid(*args, **kwargs):
    raise NotImplementedError
