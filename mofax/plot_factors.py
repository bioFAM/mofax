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


### FACTORS ###


def plot_factors_scatter(
    model: mofa_model,
    x="Factor1",
    y="Factor2",
    dist=False,
    groups=None,
    group_label="group",
    color=None,
    zero_line_x=False,
    zero_line_y=False,
    linewidth=0,
    zero_linewidth=1,
    size=20,
    legend=True,
    legend_str=True,
    legend_prop=None,
    palette=None,
    ncols=4,
    sharex=False,
    sharey=False,
    rotate_x_labels=None,
    **kwargs,
):
    """
    Plot samples features such as factor values,
    samples metadata or covariates

    Parameters
    ----------
    model : mofa_model
        Factor model
    x : optional
        Factor or variable (metadata column, feature, or covariate) to plot along X axis (Factor1 by default).
        A list can be provided to create a grid of plots.
    y : optional
        Factor or variable (metadata column, feature, or covariate) to plot along Y axis (Factor2 by default).
        A list can be provided to create a grid of plots.
    dist : optional
        Boolean value if to add marginal distributions or histograms to the scatterplot (jointplot)
    groups : optional
        Subset of groups to consider
    group_label : optional
        Sample (cell) metadata column to be used as group assignment ('group' by default)
    color : optional
        Grouping variable by default, alternatively a feature name can be provided (when no kde).
        If a list of features is provided, they will be plot on one figure.
        Use palette argument to provide a colour map.
    zero_line : optional
        Boolean values if to add Z=0 line
    linewidth : optional
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
    rotate_x_labels: int, optional
        Rotate x-axis labels (default is None)
    """

    # Process input arguments
    if group_label == "group" and color is None:
        color = "group"
    color_vars = maybe_factor_indices_to_factors(_make_iterable(color))

    assert not (
        len(color_vars) > 1 and dist
    ), "When plotting distributions, only one color can be provided"

    assert not (
        (_is_iter(x) or _is_iter(y)) and dist
    ), "When plotting distributions, only scalar x and y axes can be defined"

    if dist:
        # Get values
        z = model.fetch_values([x, y], unique=True)

        # Add group and colour information
        vars = [group_label, *color_vars]
        vars = [v for v in vars if v not in z.columns]
        if any([not (not (i)) for i in vars]):
            meta = model.fetch_values(variables=vars)
            z = z.rename_axis("sample").reset_index()
            z = z.set_index("sample").join(meta).reset_index()

        # Subset groups (incl. custom groups of samples)
        if group_label and groups is not None:
            z = z[z[group_label].isin(groups)]

        g = sns.jointplot(
            x=x,
            y=y,
            hue=color_vars[0],
            data=z.sort_values(color_vars[0]) if color_vars[0] else z,
            linewidth=linewidth,
            s=size,
            legend=legend_str,
            palette=palette,
            **kwargs,
        )
        sns.despine(offset=10, trim=True, ax=g.ax_joint)
        x_factor_label = maybe_factor_indices_to_factors(x)
        y_factor_label = maybe_factor_indices_to_factors(y)
        g.ax_joint.set(
            xlabel=f"{x_factor_label} value", ylabel=f"{y_factor_label} value"
        )
        if legend:
            g.ax_joint.legend(
                bbox_to_anchor=(1.4, 1), loc=2, borderaxespad=0.0, prop=legend_prop
            )
        if zero_line_y:
            g.axhline(0, ls="--", color="lightgrey", linewidth=linewidth, zorder=0)
        if zero_line_x:
            g.axvline(0, ls="--", color="lightgrey", linewidth=linewidth, zorder=0)
        if rotate_x_labels:
            plt.xticks(rotation=rotate_x_labels)
        plt.tight_layout()
    else:
        plot = partial(
            sns.scatterplot,
            s=size,
        )
        g = _plot_factors(
            plot,
            model,
            x,
            y,
            color,
            groups=groups,
            group_label=group_label,
            zero_line_x=zero_line_x,
            zero_line_y=zero_line_y,
            linewidth=linewidth,
            zero_linewidth=zero_linewidth,
            legend=legend,
            legend_prop=legend_prop,
            palette=palette,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            rotate_x_labels=rotate_x_labels,
            **kwargs,
        )

    return g


plot_factors = plot_factors_scatter


def _plot_factors(
    plot_func,
    model: mofa_model,
    x="Factor1",
    y="Factor2",
    color=None,
    groups=None,
    group_label="group",
    zero_line_x=False,
    zero_line_y=False,
    linewidth=0,
    zero_linewidth=1,
    legend=True,
    legend_prop=None,
    palette=None,
    ncols=4,
    sharex=False,
    sharey=False,
    rotate_x_labels=None,
    **kwargs,
):
    """
    Plot samples features such as factor values,
    samples metadata or covariates

    Parameters
    ----------
    model : mofa_model
        Factor model
    x : optional
        Factor or variable (metadata column, feature, or covariate) to plot along X axis (Factor1 by default).
        A list can be provided to create a grid of plots.
    y : optional
        Factor or variable (metadata column, feature, or covariate) to plot along Y axis (Factor2 by default).
        A list can be provided to create a grid of plots.
    dist : optional
        Boolean value if to add marginal distributions or histograms to the scatterplot (jointplot)
    groups : optional
        Subset of groups to consider
    group_label : optional
        Sample (cell) metadata column to be used as group assignment ('group' by default)
    color : optional
        Grouping variable by default, alternatively a feature name can be provided (when no kde).
        If a list of features is provided, they will be plot on one figure.
        Use palette argument to provide a colour map.
    zero_line : optional
        Boolean values if to add Z=0 line
    linewidth : optional
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
    rotate_x_labels: int, optional
        Rotate x-axis labels (default is None)
    """

    # Process input arguments
    if group_label == "group" and color is None:
        color = "group"
    color_vars = maybe_factor_indices_to_factors(_make_iterable(color))

    # Accept x and y as lists to create a grid
    x_vars = maybe_factor_indices_to_factors(_make_iterable(x))
    y_vars = maybe_factor_indices_to_factors(_make_iterable(y))

    # Get values
    z = model.fetch_values([*x_vars, *y_vars], unique=True)

    # Add group and colour information
    vars = [group_label, *color_vars]
    vars = [v for v in vars if v not in z.columns]
    if any([not (not (i)) for i in vars]):
        meta = model.fetch_values(variables=vars)
        z = z.rename_axis("sample").reset_index()
        z = z.set_index("sample").join(meta).reset_index()

    # Subset groups (incl. custom groups of samples)
    if group_label and groups is not None:
        z = z[z[group_label].isin(groups)]

    g = _plot_grid(
        plot_func,
        z,
        x,
        y,
        color,
        zero_line_x=zero_line_x,
        zero_line_y=zero_line_y,
        linewidth=linewidth,
        zero_linewidth=zero_linewidth,
        legend=legend,
        legend_prop=legend_prop,
        palette=palette,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        rotate_x_labels=rotate_x_labels,
        **kwargs,
    )

    return g


def plot_factors_violin(
    model: mofa_model,
    factors: Union[int, List[int]] = None,
    color="group",
    violins=True,
    dots=False,
    zero_line=True,
    group_label="group",
    groups=None,
    linewidth=0,
    zero_linewidth=1,
    size=20,
    legend=True,
    legend_prop=None,
    palette=None,
    alpha=None,
    violins_alpha=None,
    ncols=4,
    sharex=False,
    sharey=False,
    **kwargs,
):
    """
    Plot factor values as violinplots or stripplots (jitter plots)

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Index of a factor (or indices of factors) to use (all factors by default)
    x : optional
        Variable to plot along X axis (factor identity by default)
    y : optional
        Variable to plot along Y axis (factor value by default)
    color : optional
        Variable to split & colour dots by (cell group by default)
    groups : optional
        Subset of groups to consider
    group_label : optional
        Sample (cell) metadata column to be used as group assignment ('group' by default)
    violins : optional
        Boolean value if to add violin plots
    dots : optional
        Boolean value if to add dots to the plots
    zero_line : optional
        Boolean values if to add Z=0 line
    linewidth : optional
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
    alpha : optional
        Dots opacity
    violins_alpha : optional
        Violins opacity
    ncols : optional
        Number of columns if multiple colours are defined (4 by default)
    sharex: optional
        Common X axis across plots on the grid
    sharey: optional
        Common Y axis across plots on the grid
    """

    # Process input arguments
    if group_label == "group" and color is None:
        color = "group"
    color_vars = maybe_factor_indices_to_factors(_make_iterable(color))

    assert violins or dots, "Either violins=True or dots=True"

    # Get factors
    z = model.get_factors(factors=factors, df=True)
    z = z.rename_axis("sample").reset_index()
    # Make the table long for plotting
    z = z.melt(id_vars="sample", var_name="factor", value_name="value")

    # Add group and colour information
    vars = [group_label, *color_vars]
    if any([not (not (i)) for i in vars]):
        meta = model.fetch_values(variables=vars)
        z = z.set_index("sample").join(meta).reset_index()

    # Subset groups (incl. custom groups of samples)
    if group_label and groups is not None:
        z = z[z[group_label].isin(groups)]

    z["factor_idx"] = z.factor.str.lstrip("Factor").astype(int)
    z = z.sort_values(by="factor_idx")

    modifier = None
    if dots:

        def modifier(
            data,
            ax,
            split_var,
            color_var,
            alpha=alpha,
        ):
            # Add dots
            sns.stripplot(
                data=z,
                dodge=True,
                size=size,
                x="factor",
                y="value",
                hue=color_var,
                alpha=alpha,
                linewidth=linewidth,
                ax=ax,
            )

        modifier = partial(modifier, data=z)

    plot = partial(
        sns.violinplot,
        inner=None,
        s=size,
    )

    g = _plot_grid(
        plot,
        z,
        x="factor",
        y="value",
        color=color,
        zero_line_x=False,
        zero_line_y=zero_line,
        linewidth=linewidth,
        zero_linewidth=zero_linewidth,
        size=size,
        legend=legend,
        legend_prop=legend_prop,
        palette=palette,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        modifier=modifier,
        **kwargs,
    )

    # Adjust violins alpha
    if violins:
        if violins_alpha:
            for path in g.collections:
                if path.__class__.__name__ == "PolyCollection":
                    path.set_alpha(violins_alpha)
    if violins is None or not violins or violins_alpha == 0:
        n_plots = len(g.collections)
        i = 0
        while i < len(g.collections):
            path = g.collections[i]
            if path.__class__.__name__ == "PolyCollection":
                path.remove()
            else:
                i += 1

    return g


def plot_factors_umap(
    model: mofa_model,
    factors: Optional[Union[int, List[int]]] = None,
    groups=None,
    group_label: Optional[str] = None,
    color=None,
    linewidth=0,
    size=20,
    n_neighbors=10,
    spread=1,
    min_dist=0.5,
    random_state=None,
    umap_kwargs={},
    legend=True,
    legend_prop=None,
    palette=None,
    ncols=4,
    sharex=False,
    sharey=False,
    **kwargs,
):
    """
    Plot UMAP on factor values.

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Index of a factor (or indices of factors) to use (all factors by default)
    groups : optional
        Subset of groups to consider
    group_label : optional
        Sample (cell) metadata column to be used as group assignment
    color : optional
        Grouping variable by default, alternatively a feature name can be provided
    linewidth : optional
        Linewidth argument for dots (default is 0)
    size : optional
        Size argument for dots (ms for plot, s for jointplot and scatterplot; default is 5)
    n_neighbors : optional
        n_neighbors parameter for UMAP
    spread : optional
        spread parameter for UMAP
    random_state : optional
        random_state parameter for UMAP
    umap_kwargs : optional
        Additional arguments to umap.UMAP()
    legend : optional bool
        If to show the legend (e.g. colours matching groups)
    legend_prop : optional
        The font properties of the legend
    palette : optional
        cmap describing colours, default is None (cubehelix)
        Example palette: seaborn.cubehelix_palette(8, start=.5, rot=-.75. as_cmap=True)
    ncols : optional
        Number of columns if multiple colours are defined (4 by default)
    """

    # Process input arguments
    if group_label == "group" and not color:
        color = "group"
    color_vars = maybe_factor_indices_to_factors(_make_iterable(color))

    # Check if UMAP has be pre-computed
    get_umap_cols = lambda m: np.where(
        [s.startswith("UMAP") for s in m.samples_metadata.columns.values]
    )[0]
    umap_cols = get_umap_cols(model)
    if len(umap_cols) == 0:
        print("Computing UMAP coordinates...")
        model.run_umap(
            factors=factors,
            n_neighbors=n_neighbors,
            spread=spread,
            min_dist=min_dist,
            random_state=random_state,
        )
        umap_cols = get_umap_cols(model)

    embedding = model.samples_metadata.iloc[:, umap_cols]

    x, y, *_ = embedding.columns

    # Add group and colour information
    vars = [group_label, *color_vars]
    vars = [v for v in vars if v not in embedding.columns.values]
    if any([not (not (i)) for i in vars]):
        meta = model.fetch_values(variables=vars)
        embedding = embedding.rename_axis("sample").reset_index()
        embedding = embedding.set_index("sample").join(meta).reset_index()

    # Subset groups (incl. custom groups of samples)
    if group_label and groups is not None:
        embedding = embedding[embedding[group_label].isin(groups)]

    plot = partial(
        sns.scatterplot,
        s=size,
    )

    g = _plot_grid(
        plot,
        embedding,
        x=x,
        y=y,
        color=color,
        zero_line_x=False,
        zero_line_y=False,
        linewidth=linewidth,
        size=size,
        legend=legend,
        legend_prop=legend_prop,
        palette=palette,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        **kwargs,
    )

    return g


# The following functions do not use the grid interface


def plot_factors_matrix(
    model: mofa_model,
    factors: Optional[Union[int, List[int], str, List[str]]] = None,
    group_label: Optional[str] = None,
    groups: Optional[Union[int, List[int], str, List[str]]] = None,
    agg="mean",
    cmap="viridis",
    vmin=None,
    vmax=None,
    **kwargs,
):
    """
    Average factor value per group and plot it as a heatmap

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Factor idices or names (all factors by default)
    group_label : optional
        Sample (cell) metadata column to be used as group assignment
    groups : optional
        Group indices or names (all groups by default)
    avg : optional
        Aggregation function to average factor values per group (mean by default)
    cmap : optional
        Heatmap cmap argument ("viridis" by default)
    """

    z = model.get_factors(factors=factors, groups=groups, df=True)
    z = z.rename_axis("sample").reset_index()
    # Make the table long for plotting
    z = z.melt(id_vars="sample", var_name="factor", value_name="value")

    # Assign a group to every sample (cell) if it is provided
    if group_label is None:
        group_label = "group"
    groups_df = model.samples_metadata.loc[:, [group_label]]

    groups_df.rename(columns={groups_df.columns[0]: "group"}, inplace=True)

    # Add group information for samples (cells)
    z = z.set_index("sample").join(groups_df).reset_index()

    z = (
        z.groupby(["group", "factor"])
        .agg({"value": agg})
        .reset_index()
        .pivot(index="factor", columns="group", values="value")
    )

    z.index = z.index.astype("category")
    z.index = z.index.reorder_categories(
        sorted(z.index.categories, key=lambda x: int(x.split("Factor")[1]))
    )
    z = z.sort_values("factor", ascending=False)

    ax = sns.heatmap(z, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set(ylabel="Factor", xlabel="Group")
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), rotation=0)

    return ax


def plot_factors_dotplot(
    model: mofa_model,
    factors: Optional[Union[int, List[int], str, List[str]]] = None,
    group_label: Optional[str] = None,
    groups: Optional[Union[int, List[int], str, List[str]]] = None,
    palette=None,
    vmin=None,
    vmax=None,
    size=100,
    xticklabels_size=8,
    yticklabels_size=8,
    **kwargs,
):
    """
    Average factor value per group and plot it as a heatmap.
    Colour indicates mean factor value
    and size relates to its variance.

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Factor idices or names (all factors by default)
    group_label : optional
        Sample (cell) metadata column to be used as group assignment
    groups : optional
        Group indices or names (all groups by default)
    palette : optional
        Colour palette (blue-white-red by default)
    vmin : optional
        Min color value
    vmax : optional
        Max colour value
    size : optional
        Dot plot size scaling coefficient
    xticklabels_size : optional
        Font size for features labels (default is 10)
    yticklabels_size : optional
        Font size for factors labels (default is None)
    """

    if palette is None:
        palette = sns.diverging_palette(240, 10, sep=3, center="light", as_cmap=True)

    z = model.get_factors(factors=factors, groups=groups, df=True)
    z = z.rename_axis("sample").reset_index()
    # Make the table long for plotting
    z = z.melt(id_vars="sample", var_name="factor", value_name="value")

    # Assign a group to every sample (cell) if it is provided
    if group_label is None:
        group_label = "group"
    groups_df = model.samples_metadata.loc[:, [group_label]]

    groups_df.rename(columns={groups_df.columns[0]: "group"}, inplace=True)

    # Add group information for samples (cells)
    z = z.set_index("sample").join(groups_df).reset_index()

    z = z.groupby(["group", "factor"]).agg({"value": ["mean", "var"]}).reset_index()
    z.columns = ["group", "factor", "value_mean", "value_var"]
    z["abs_mean"] = np.abs(z.value_mean)

    z.factor = z.factor.astype("category")
    z.factor = z.factor.cat.reorder_categories(
        sorted(z.factor.cat.categories, key=lambda x: int(x.split("Factor")[1]))
    )
    z = z.sort_values("factor")

    # Fix group order
    groups = (
        z.pivot(columns="group", index="factor", values="abs_mean")
        .idxmax(axis=0)
        .sort_values(key=lambda x: x.str.split("Factor").str[1].astype(int))
        .index.values
    )
    z.group = z.group.astype("category")
    z.group = z.group.cat.reorder_categories(np.flip(groups))

    z = z.sort_values(["factor", "group"])

    # Normalise colour map
    abs_mean_max = max(abs(z.value_mean.min()), z.value_mean.max())
    norm = plt.Normalize(-abs_mean_max, abs_mean_max)
    # norm = plt.Normalize(z.value_mean.min(), z.value_mean.max())
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    # from matplotlib.colors import DivergingNorm
    # divnorm = DivergingNorm(vmin=-abs_mean, vcenter=0, vmax=abs_mean)
    sm.set_array([])

    # Scale dot size
    z.value_var = z.value_var / z.value_var.max() * size

    # Create the plot
    fig, ax = plt.subplots()
    g = plt.scatter(
        x=z.factor,
        y=z.group,
        c=z.value_mean,
        s=z.value_var,
        cmap=palette,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    try:
        g.figure.colorbar(sm)
        g.get_legend().remove()
    except Exception:
        warn("Cannot make a proper colorbar")

    plt.xticks(rotation=90, size=xticklabels_size)
    plt.yticks(size=yticklabels_size)

    plt.ylabel(group_label)

    return g


def plot_factors_correlation(
    model: mofa_model,
    factors: Optional[Union[int, List[int]]] = None,
    groups: Optional[Union[int, List[int], str, List[str]]] = None,
    covariates=None,
    pvalues=False,
    linewidths=0,
    diag=False,
    cmap=None,
    square=True,
    **kwargs,
):
    """
    Plot correlation of factors and, if provided, covariates

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Index of a factor (or indices of factors) to use (all factors by default)
    groups : optional
        Subset of groups to consider
    covarites : optional
        A vector, a matrix, or a data frame with covariates (one per column)
    pvalues
        Plot BH-adjusted p-values instead of correlation coefficient
    linewidths : optional
        Heatmap linewidths argument (default is 0)
    diag : optional
        If to only plot lower triangle of the correlation matrix (False by default)
    cmap : optional
        Heatmap cmap argument
    square : optional
        Heatmap square argument (True by default)
    """

    z = model.get_factors(factors=factors, groups=groups)

    # pearsonr returns (r, pvalue)
    value_index = 1 if pvalues else 0

    if covariates is not None:
        n_cov = covariates.shape[1]
        # Transform a vector to a data frane
        # Also ransform matrices and ndarrays to a data frame
        if len(covariates.shape) == 1 or not isinstance(covariates, pd.DataFrame):
            covariates = pd.DataFrame(covariates)
        corr = np.ndarray(shape=(z.shape[1], n_cov))
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                corr[i, j] = pearsonr(z[:, i], covariates.iloc[:, j])[value_index]
    else:
        # Inter-factor correlations
        corr = np.ndarray(shape=(z.shape[1], z.shape[1]))
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                corr[i, j] = pearsonr(z[:, i], z[:, j])[value_index]

    if pvalues:
        corr = padjust_fdr_2d(corr)

        corr = -np.log10(corr)

        if np.sum(np.isinf(corr)) > 0:
            warn("Some p-values are 0, these values will be capped.")
            corr[np.isinf(corr)] = np.ceil(corr[~np.isinf(corr)].max() * 10)

    mask = None
    if diag:
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots()

    if cmap is None:
        # Generate a custom diverging colormap
        cmap = "Reds" if pvalues else sns.diverging_palette(220, 10, as_cmap=True)

    # Generate labels for the heatmap
    if factors is None:
        factors = range(z.shape[1])
    fnames = [f"Factor{fi+1}" if isinstance(fi, int) else fi for fi in factors]
    if covariates is not None:
        if isinstance(covariates, pd.DataFrame):
            cnames = covariates.columns.values
        else:
            cnames = [f"Covar{ci+1}" for ci in covariates.shape[1]]
        xticklabels = cnames
        yticklabels = fnames
    else:
        xticklabels = fnames
        yticklabels = fnames

    center = 0 if not pvalues else None
    cbar_kws = {"shrink": 0.5}
    cbar_kws["label"] = (
        "Correlation coefficient" if not pvalues else "-log10(adjusted p-value)"
    )

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(
        corr,
        cmap=cmap,
        mask=mask,
        center=center,
        square=True,
        linewidths=0.5,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar_kws=cbar_kws,
        **kwargs,
    )

    g.set_yticklabels(g.yaxis.get_ticklabels(), rotation=0)

    return g


def plot_factors_covariates_correlation(
    model: mofa_model,
    covariates: Union[np.ndarray, np.matrix, pd.DataFrame],
    pvalues: bool = False,
    **kwargs,
):
    """
    Plot correlation of factors and covariates

    Parameters
    ----------
    model : mofa_model
        Factor model
    covarites
        A vector, a matrix, or a data frame with covariates (one per column)
    pvalues
        Plot BH-adjusted p-values instead of correlation coefficient
    **kwargs
        Other arguments to plot_factors_correlation
    """
    return plot_factors_correlation(
        model, covariates=covariates, pvalues=pvalues, square=False, **kwargs
    )


### Projection ###


def plot_projection(
    model: mofa_model,
    data,
    data_name: str = "projected_data",
    view: Union[str, int] = None,
    with_orig: bool = False,
    x="Factor1",
    y="Factor2",
    groups=None,
    groups_df=None,
    color=None,
    linewidth=0,
    size=20,
    legend=False,
    legend_loc="best",
    legend_prop=None,
    feature_intersection=False,
    **kwargs,
):
    """
    Project new data onto the factor space of the model.

    For the projection, a pseudo-inverse of the weights matrix is calculated
    and its product with the provided data matrix is calculated.

    Parameters
    ----------
    model : mofa_model
        Factor model
    data
        Numpy array or Pandas DataFrame with the data matching the number of features
    data_name : optional
        A name for the projected dataset ("projected_data" by default)
    view : optional
        A view of the model to consider (first view by default)
    with_orig : optional
        Boolean value if to plot data from the model (False by default)
    x : optional
        Factor to plot along X axis (Factor1 by default)
    y : optional
        Factor to plot along Y axis (Factor2 by default)
    groups : optional
        Subset of groups to consider
    groups_df : optional pd.DataFrame
        Data frame with samples (cells) as index and first column as group assignment
    color : optional
        A feature name to colour the dots by its expression
    linewidth : optional
        Linewidth argument for dots (default is 0)
    size : optional
        Size argument for dots (ms for plot, s for jointplot and scatterplot; default is 5)
    legend : optional bool
        If to show the legend (e.g. colours matching groups)
    legend_loc : optional
        Legend location (e.g. 'upper left', 'center', or 'best')
    legend_prop : optional
        The font properties of the legend
    feature_intersection : optional
        Feature intersection flag for project_data
    """
    zpred = model.project_data(
        data=data,
        view=view,
        factors=[x, y],
        df=True,
        feature_intersection=feature_intersection,
    )
    zpred.columns = ["x", "y"]

    # Get and prepare Z matrix from the model if required
    if with_orig:
        z = model.get_factors(factors=[x, y], groups=groups, df=True)
        z.columns = ["x", "y"]

        # Assign a group to every sample (cell) if it is provided
        if groups_df is None:
            groups_df = model.get_samples().set_index("sample")

        z = z.rename_axis("sample").reset_index()
        z = z.set_index("sample").join(groups_df).reset_index()
        grouping_var = groups_df.columns[0]

        # Assign colour to every sample (cell) if colouring by feature expression
        if color is None:
            color_var = grouping_var
        else:
            color_var = color
            color_df = model.get_data(features=color, df=True)
            z = z.set_index("sample").join(color_df).reset_index()
            z = z.sort_values(color_var)
    else:
        grouping_var = "group"
    zpred[grouping_var] = data_name

    # Assign colour to every sample (cell) in the new data if colouring by feature expression
    if color is None:
        color_var = grouping_var
    else:
        if isinstance(data, pd.DataFrame):
            color_var = color
            color_df = data.loc[:, [color]]
            zpred = zpred.join(color_df).reset_index()
            zpred = zpred.sort_values(color_var)

    if with_orig:
        z = z.append(zpred)
    else:
        z = zpred

    # Define plot axes labels
    x_factor_label = f"Factor{x+1}" if isinstance(x, int) else x
    y_factor_label = f"Factor{y+1}" if isinstance(y, int) else y

    # Set default colour to black if none set
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["color"] = "black"

    g = sns.scatterplot(
        x="x",
        y="y",
        data=z,
        linewidth=linewidth,
        s=size,
        hue=color_var,
        legend=legend,
        **kwargs,
    )
    sns.despine(offset=10, trim=True, ax=g)
    g.set(xlabel=f"{x_factor_label} value", ylabel=f"{y_factor_label} value")

    return g
