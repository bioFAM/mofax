from .core import mofa_model
from .utils import *

from typing import Union, Optional, List, Iterable, Sequence
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from .utils import _make_iterable
from .plot_utils import _plot_grid


def plot_interpolated_factors(
    model: mofa_model,
    factors: Union[int, List[int]] = None,
    groups=None,
    only_mean: bool = False,
    show_observed: bool = True,
    color=None,
    alpha=0.3,
    zero_line=False,
    linewidth=2,
    dot_linewidth=0,
    zero_linewidth=1,
    size=20,
    legend=True,
    legend_prop=None,
    palette=None,
    ncols=4,
    sharex=False,
    sharey=False,
    **kwargs,
):
    """
    Plot samples features such as factor values,
    samples metadata or covariates

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Index of a factor (or indices of factors) to use (all factors by default)
    groups : optional
        Subset of groups to consider
    only_mean : optional
        If not to plot confidence for the interpolated values (False by default)
    show_observed : optional
        If not to plot oberved factor values along the transformed covariate (True by default)
    group_label : optional
        Sample (cell) metadata column to be used as group assignment ('group' by default)
    color : optional
        Grouping variable by default, alternatively a feature name can be provided (when no kde).
        If a list of features is provided, they will be plot on one figure.
        Use palette argument to provide a colour map.
    alpha : optional
        Opacity when plotting confidence for the interpolated values (when only_mean=False)
    zero_line : optional
        Boolean values if to add Z=0 line
    linewidth : optional
        Linewidth argument for lines (default is 2)
    dot_linewidth : optional
        Linewidth argument for dots (default is 0)
    zero_linewidth : optional
        Linewidth argument for the zero line (default is 1)
    size : optional
        Size argument for dots (ms for plot, s for jointplot and scatterplot; default is 5)
    legend : optional bool
        If to show the legend (e.g. colours matching groups)
    legend_prop : optional
        The font properties of the legend
    palette : optional
        cmap describing colours, default is None (cubehelix)
        Example palette: seaborn.cubehelix_palette(8, start=.5, rot=-.75. as_cmap=True)
    ncols : optional
        Number of columns if multiple colours are defined (4 by default)
    sharex: optional
        Common X axis across plots on the grid
    sharey: optional
        Common Y axis across plots on the grid
    """

    # Process input arguments
    if color is None:
        color = "group"
    if color != "group":
        raise ValueError(
            "Only colouring by group is supported when plotting interpolated factors"
        )

    factors, factor_indices = model._check_factors(factors)

    # Get factors
    zi = model.get_interpolated_factors(
        factors=factors, df_long=True
    )  # this includes the group

    new_values_dim = model.interpolated_factors["new_values"].shape[1]
    if new_values_dim == 1 and len(model.covariates_names) == 1:
        new_value = f"{model.covariates_names[0]}_transformed"
    else:
        # If the new values are multi-dimensional, address them by index
        # since multi-dimensional plots are not supported
        new_value = "new_value"

    # Subset groups
    if groups is not None:
        zi = zi[zi["group"].isin(_make_iterable(groups))]

    zi_mean = (
        zi.pivot(
            index=["new_sample", new_value, "group"], columns="factor", values="mean"
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values([new_value, "group"])
    )
    zi_var = (
        zi.pivot(
            index=["new_sample", new_value, "group"],
            columns="factor",
            values="variance",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values([new_value, "group"])
    )

    z_observed = None
    if show_observed:
        covs = [f"{v}_transformed" for v in model.covariates_names]
        if len(covs) > 1:
            raise NotImplemented(
                "Only data with a single covariate is currently supported"
            )
        z_observed = model.fetch_values([*factors, covs[0], "group"]).sort_values(
            ["group"]
        )
        # Subset groups
        if groups is not None:
            z_observed = z_observed[z_observed["group"].isin(_make_iterable(groups))]

    plot = partial(
        sns.lineplot,
    )

    modifier = None
    if (not only_mean) or show_observed:

        def modifier(
            data,
            ax,
            split_var,
            color_var,
            alpha=alpha,
            show_observed=show_observed,
            only_mean=only_mean,
        ):
            m, v = data["mean"], data["var"]

            # Add confidence intervals
            if not only_mean:
                get_conf = lambda f, mean, var: f(mean, 1.96 * np.sqrt(var))
                for group in m[color_var].unique():
                    m_g = m[m[color_var] == group]
                    v_g = v[v[color_var] == group]
                    g_mean, g_var = m_g[split_var].values, v_g[split_var].values
                    ax.fill_between(
                        m_g[new_value],
                        get_conf(np.subtract, g_mean, g_var),
                        get_conf(np.add, g_mean, g_var),
                        alpha=alpha,
                    )

            # Add dots
            if show_observed:
                sns.scatterplot(
                    data=data["observed"],
                    x=new_value,
                    y=split_var,
                    hue=color_var,
                    linewidth=dot_linewidth,
                    ax=ax,
                )

        modifier = partial(
            modifier, data={"mean": zi_mean, "var": zi_var, "observed": z_observed}
        )

    g = _plot_grid(
        plot,
        zi_mean,
        x=new_value,
        y=factors,
        color=color,
        zero_line_x=False,
        zero_line_y=zero_line,
        linewidth=linewidth,
        zero_linewidth=zero_linewidth,
        legend=legend,
        legend_prop=legend_prop,
        palette=palette,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        modifier=modifier,
        **kwargs,
    )

    return g


### MEFISTO ###


def plot_group_kernel(
    model, groups=None, factors=None, palette=None, vmin=-1, vmax=1, ncols=4, **kwargs
):

    z = model.get_factors(factors=factors, groups=groups)
    factor_indices, factors = model._check_factors(factors, unique=True)

    groups = model._check_groups(groups)
    all_groups = np.array(model.groups)
    group_indices = [np.where(all_groups == gr)[0][0] for gr in groups]

    # Get group kernels
    Kgs = model.get_group_kernel()[factor_indices, :, :][:, group_indices, :][
        :, :, group_indices
    ]

    df_list = [
        pd.DataFrame(Kgs[i], index=groups, columns=groups) for i in range(Kgs.shape[0])
    ]

    if palette is None:
        palette = "RdBu_r"

    # Only use clustermap when there's one factors to plot
    # since it creates its own figure
    if len(factors) == 1:
        g = sns.clustermap(df_list[0], cmap=palette, vmin=vmin, vmax=vmax, **kwargs)
        g.ax_heatmap.set(ylabel="Group", xlabel="Group", title=factors[0])
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.yaxis.get_ticklabels(), rotation=0)

        sns.despine(offset=10, trim=True, ax=g.ax_heatmap)

    else:
        # Figure out rows & columns for the grid with plots
        ncols = min(ncols, len(factors))
        nrows = int(np.ceil(len(factors) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(
                ncols * rcParams["figure.figsize"][0],
                nrows * rcParams["figure.figsize"][1],
            ),
        )

        if ncols == 1:
            axes = np.array(axes).reshape(-1, 1)
        if nrows == 1:
            axes = np.array(axes).reshape(1, -1)

        for i, factor in enumerate(factors):

            ri = i // ncols
            ci = i % ncols

            g = sns.heatmap(
                df_list[i],
                square=True,
                cmap=palette,
                vmin=vmin,
                vmax=vmax,
                ax=axes[ri, ci],
                **kwargs,
            )
            g.set(ylabel="Group", xlabel="Group", title=factor)
            g.set_yticklabels(g.yaxis.get_ticklabels(), rotation=0)

            sns.despine(offset=10, trim=True, ax=g)

        # Remove unused axes
        for i in range(len(factors), ncols * nrows):
            ri = i // ncols
            ci = i % ncols
            fig.delaxes(axes[ri, ci])

    plt.tight_layout()

    return g


def plot_sharedness(
    model, groups=None, factors=None, color="#B8CF87", return_data=False, **kwargs
):
    GROUPS_MSG = "Multiple groups are required to determine sharedness"
    assert model.ngroups > 1, GROUPS_MSG
    if groups is not None:
        assert not isinstance(groups, str) and len(groups) > 1, GROUPS_MSG

    z = model.get_factors(factors=factors, groups=groups)
    factor_indices, factors = model._check_factors(factors, unique=True)

    groups = model._check_groups(groups)
    all_groups = np.array(model.groups)
    group_indices = [np.where(all_groups == gr)[0][0] for gr in groups]

    # Get group kernels
    Kgs = model.get_group_kernel()[factor_indices, :, :][:, group_indices, :][
        :, :, group_indices
    ]

    # Calculate distance
    gr = np.array(
        [
            np.abs(Kgs[i, :, :])[np.tril(Kgs[i, :, :], -1).astype(bool)].mean()
            for i in range(Kgs.shape[0])
        ]
    )

    df = pd.DataFrame({"factor": factors, "shared": gr, "non_shared": 1 - gr})

    if return_data:
        return df

    sns.barplot(data=df, y="factor", x="shared", color="lightgrey")
    g = sns.barplot(data=df, color=color, y="factor", x="shared", **kwargs)

    g.set(xlabel="Sharedness", ylabel="Factor")

    sns.despine(offset=10, trim=True, ax=g)

    plt.tight_layout()

    return g


def plot_smoothness(model, factors=None, color="#5F9EA0", return_data=False, **kwargs):
    z = model.get_factors(factors=factors)
    factor_indices, factors = model._check_factors(factors, unique=True)

    # Get scales
    scales = np.array(model.model["training_stats"]["scales"])

    df = pd.DataFrame({"factor": factors, "smooth": scales, "non_smooth": 1 - scales})

    if return_data:
        return df

    sns.barplot(data=df, y="factor", x="smooth", color="lightgrey")
    g = sns.barplot(data=df, color=color, y="factor", x="smooth", **kwargs)

    g.set(xlabel="Smoothness", ylabel="Factor")

    sns.despine(offset=10, trim=True, ax=g)

    plt.tight_layout()

    return g
