from .core import mofa_model
from .utils import *

import sys
from warnings import warn
from typing import Union, Optional, List, Iterable, Sequence
from functools import partial

import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from .utils import maybe_factor_indices_to_factors, _make_iterable, _is_iter
from .plot_utils import _plot_grid


### VARIANCE EXPLAINED ###


def plot_r2(
    model: mofa_model,
    x="Group",
    y="Factor",
    factors: Union[int, List[int], str, List[str]] = None,
    group_label: str = None,
    views=None,
    groups=None,
    cmap="Blues",
    vmin=None,
    vmax=None,
    **kwargs,
):
    """
    Plot R2 values for the model

    Parameters
    ----------
    model : mofa_model
        Factor model
    x : str
        Dimension along X axis: Group (default), View, or Factor
    y : str
        Dimension along Y axis: Group, View, or Factor (default)
    factors : optional
        Index of a factor (or indices of factors) to use (all factors by default)
    views : optional
        Make a plot for certain views (None by default to plot all views)
    groups : optional
        Make a plot for certain groups (None by default to plot all groups)
    group_label : optional
        Sample (cell) metadata column to be used as group assignment
    cmap : optional
        The colourmap for the heatmap (default is 'Blues' with darker colour for higher R2)
    vmin : optional
        Display all R2 values smaller than vmin as vmin (0 by default)
    vmax : optional
        Display all R2 values larger than vmax as vmax (derived from the data by default)
    """
    if group_label is None:
        r2 = model.get_variance_explained(
            factors=factors,
            groups=groups,
            views=views,
        )
    else:
        r2 = model.calculate_variance_explained(
            factors=factors,
            groups=groups,
            views=views,
            group_label=group_label,
            per_factor=True,
        )

    vmax = r2.R2.max() if vmax is None else vmax
    vmin = 0 if vmin is None else vmin

    split_by = [dim for dim in ["Group", "View", "Factor"] if dim not in [x, y]]
    assert (
        len(split_by) == 1
    ), "x and y values should be different and be one of Group, View, or Factor"
    split_by = split_by[0]

    split_by_items = r2[split_by].unique()
    fig, axes = plt.subplots(ncols=len(split_by_items), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    if len(split_by_items) == 1:
        axes = [axes]

    for i, item in enumerate(split_by_items):
        r2_sub = r2[r2[split_by] == item]
        r2_df = r2_sub.sort_values("R2").pivot(index=y, columns=x, values="R2")

        if y == "Factor":
            # Sort by factor index
            r2_df.index = r2_df.index.astype("category")
            r2_df.index = r2_df.index.reorder_categories(
                sorted(r2_df.index.categories, key=lambda x: int(x.split("Factor")[1]))
            )
            r2_df = r2_df.sort_values("Factor")

        if x == "Factor":
            # Re-order columns by factor index
            r2_df.columns = r2_df.columns.astype("category")
            r2_df.columns = r2_df.columns.reorder_categories(
                sorted(
                    r2_df.columns.categories, key=lambda x: int(x.split("Factor")[1])
                )
            )
            r2_df = r2_df[r2_df.columns.sort_values()]

        g = sns.heatmap(
            r2_df.sort_index(level=0, ascending=False),
            ax=axes[i],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=i == 0,
            cbar_ax=None if i else cbar_ax,
            **kwargs,
        )

        axes[i].set_title(item)
        axes[i].tick_params(axis="both", which="both", length=0)
        if i == 0:
            g.set_yticklabels(g.yaxis.get_ticklabels(), rotation=0)
        else:
            axes[i].set_ylabel("")

    plt.close()
    return fig


def plot_r2_pvalues(
    model: mofa_model,
    factors: Union[int, List[int], str, List[str]] = None,
    n_iter: int = 100,
    groups_df: pd.DataFrame = None,
    group_label: str = None,
    view=0,
    fdr: bool = True,
    cmap="binary_r",
    **kwargs,
):
    """
    Plot R2 values for the model

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Index of a factor (or indices of factors) to use (all factors by default)
    view : optional
        Make a plot for a cetrain view (first view by default)
    groups_df : optional pd.DataFrame
        Data frame with samples (cells) as index and first column as group assignment
    group_label : optional
        Sample (cell) metadata column to be used as group assignment
    fdr : optional bool
        If plot corrected PValues (FDR)
    cmap : optional
        The colourmap for the heatmap (default is 'binary_r' with darker colour for smaller PValues)
    """
    r2 = model._get_r2_null(
        factors=factors,
        groups_df=groups_df,
        group_label=group_label,
        n_iter=n_iter,
        return_pvalues=True,
        fdr=fdr,
    )
    pvalue_column = "FDR" if fdr else "PValue"
    # Select a certain view if necessary
    if view is not None:
        view = model.views[view] if isinstance(view, int) else view
        r2 = r2[r2["View"] == view]
    r2_df = r2.sort_values("PValue").pivot(
        index="Factor", columns="Group", values=pvalue_column
    )

    # Sort by factor index
    r2_df.index = r2_df.index.astype("category")
    r2_df.index = r2_df.index.reorder_categories(
        sorted(r2_df.index.categories, key=lambda x: int(x.split("Factor")[1]))
    )
    r2_df = r2_df.sort_values("Factor")

    g = sns.heatmap(r2_df.sort_index(level=0, ascending=False), cmap=cmap, **kwargs)

    g.set_yticklabels(g.yaxis.get_ticklabels(), rotation=0)

    return g


def plot_r2_barplot(
    model: mofa_model,
    factors: Union[int, List[int], str, List[str]] = None,
    view=0,
    groups_df: pd.DataFrame = None,
    group_label: str = None,
    x="Factor",
    y="R2",
    groupby="Group",
    xticklabels_size=10,
    linewidth=0,
    stacked=False,
    **kwargs,
):
    """
    Plot R2 values for the model

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Index of a factor (or indices of factors) to use (all factors by default)
    view : optional
        Make a plot for a cetrain view (first view by default)
    groups_df : optional pd.DataFrame
        Data frame with samples (cells) as index and first column as group assignment
    group_label : optional
        Sample (cell) metadata column to be used as group assignment
    x : optional
        Value to plot along the x axis (default is Factor)
    y : optional
        Value to plot along the y axis (default is R2)
    groupby : optional
        Column to group bars for R2 values by (default is Group)
    xticklabels_size : optional
        Font size for group labels (default is 10)
    linewidth : optional
        Linewidth (0 by default)
    stacked : optional
        Plot a stacked barplot instead of a grouped barplot
    """
    r2 = model.get_r2(
        factors=factors, groups_df=groups_df, group_label=group_label, per_factor=True
    )
    # Select a certain view if necessary
    if view is not None:
        view = model.views[view] if isinstance(view, int) else view
        r2 = r2[r2["View"] == view]
    r2_df = r2.sort_values("R2")

    # Sort by factor index
    r2_df.Factor = r2_df.Factor.astype("category")
    r2_df.Factor = r2_df.Factor.cat.reorder_categories(
        sorted(r2_df.Factor.cat.categories, key=lambda x: int(x.split("Factor")[1]))
    )
    r2_df = r2_df.sort_values("Factor")

    if stacked:
        g = r2_df.pivot(index="Factor", columns="Group", values="R2").plot(
            kind="bar", stacked=True, linewidth=linewidth, **kwargs
        )
        plt.ylabel("R2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    else:
        # Grouped barplot
        g = sns.barplot(
            data=r2_df.sort_index(level=0, ascending=False),
            x=x,
            y=y,
            hue=groupby,
            linewidth=linewidth,
            **kwargs,
        )

        g.set_xticklabels(g.xaxis.get_ticklabels(), rotation=90, size=xticklabels_size)

    return g
