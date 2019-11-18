from .core import mofa_model

from typing import Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")
sns.set_palette("Set2")


def plot_weights(
    model: mofa_model,
    factor="Factor1",
    view=0,
    n_features: int = 10,
    label_size=5,
    x_rank_offset=10,
    y_repel_coef=0.03,
    attract_to_points=True,
    **kwargs,
):
    """
    Plot loadings for a specific factor

    Parameters
    ----------
    model : mofa_model
        Factor model
    factor : optional
        Factor to use (default is Factor1)
    view : options
        The view to get the loadings of the factor for (first view by default)
    n_features : optional
        Number of features to label with most positive and most negative loadings
    label_size : optional
        Font size of feature labels (default is 5)
    x_rank_offset : optional
        Offset the feature labels from the left/right side (by 10 points by default)
    y_repel_coef : optional
        Parameter to repel feature labels along the y axis (0.03 by default)
    attract_to_points : optional
        If place labels according to the Y coordinate of the point (False by default)
    """
    w = model.get_weights(views=view, factors=factor, df=True)
    w = pd.melt(
        w.reset_index().rename(columns={"index": "feature"}),
        id_vars="feature",
        var_name="factor",
        value_name="value",
    )
    w["abs_value"] = abs(w.value)

    # Assign ranks to features, per factor
    w["rank"] = w.groupby("factor")["value"].rank(ascending=False)
    w["abs_rank"] = w.groupby("factor")["abs_value"].rank(ascending=False)
    w = w.sort_values(["factor", "abs_rank"], ascending=True)

    # Construct the plot
    plot = sns.lineplot(
        x="rank",
        y="value",
        data=w,
        c="black",
        markers=True,
        dashes=False,
        linewidth=0.5,
    )
    sns.despine(offset=10, trim=True)

    # Plot top features as dots
    sns.scatterplot(
        x="rank",
        y="value",
        data=w[w["abs_rank"] < n_features],
        linewidth=0.2,
        s=25,
        alpha=0.75,
    )

    # Label top loadings
    y_start_pos = w[w.value > 0].sort_values("abs_rank").iloc[0].value
    y_start_neg = w[w.value < 0].sort_values("abs_rank").iloc[0].value

    y_prev = y_start_pos
    for i, point in (
        w[(w["abs_rank"] < n_features) & (w["value"] >= 0)].reset_index().iterrows()
    ):
        y_loc = y_prev - y_repel_coef if i != 0 else y_start_pos
        y_loc = min(point["value"], y_loc) if attract_to_points else y_loc
        plot.text(
            x_rank_offset,
            y_loc,
            point["feature"],
            horizontalalignment="left",
            size=label_size,
            color="black",
            weight="regular",
        )
        y_prev = y_loc

    y_prev = y_start_neg
    for i, point in (
        w[(w["abs_rank"] < n_features) & (w["value"] < 0)].reset_index().iterrows()
    ):
        y_loc = y_prev + y_repel_coef if i != 0 else y_start_neg
        y_loc = max(point["value"], y_loc) if attract_to_points else y_loc
        plot.text(
            w.shape[0] - x_rank_offset,
            y_loc,
            point["feature"],
            horizontalalignment="left",
            size=label_size,
            color="black",
            weight="regular",
        )
        y_prev = y_loc

    # Set plot axes labels
    factor_label = f"Factor{factor+1}" if isinstance(factor, int) else factor
    plot.set(ylabel=f"{factor_label} value", xlabel="Feature rank")

    return plot


def plot_weights_heatmap(
    model: mofa_model,
    factors: Union[int, List[int]] = None,
    n_features: int = None,
    w_threshold: float = None,
    w_abs: bool = False,
    features_col: pd.DataFrame = None,
    cmap=None,
    xticklabels_size=10,
    cluster_factors=True,
    cluster_features=True,
    **kwargs,
):
    """
    Plot loadings for top features in a heatmap

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Factors to use (all factors in the model by default)
    n_features : optional
        Number of features for each factor by their absolute value (10 by default)
    w_threshold : optional
        Absolute loading threshold for a feature to plot (no threshold by default)
    w_abs : optional
        If plot absolute loadings values
    features_col : optional
        Pandas data frame with index by feature name with the first column 
        containing the colour for every feature
    cmap : optional
        Color map (blue-to-red divergent palette with by default)
    xticklabels_size : optional
        Font size for features labels (default is 10)
    cluster_factors : optional
        If cluster factors (in rows; default is True)
    cluster_features : optional
        If cluster features (in columns; default in True)
    """

    # Set defaults
    n_features_default = 10
    if factors is None:
        factors = list(range(model.nfactors))
    if cmap is None:
        cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)

    # Fetch weights for the relevant factors
    w = (
        model.get_weights(factors=factors, df=True, absolute_values=w_abs)
        .rename_axis("feature")
        .reset_index()
    )
    wm = w.melt(id_vars="feature", var_name="factor", value_name="value")
    wm = wm.assign(value_abs=lambda x: x.value.abs())
    wm["factor"] = wm["factor"].astype("category")

    if n_features is None and w_threshold is not None:
        features = wm[wm.value_abs >= w_threshold].feature.unique()
    else:
        if n_features is None:
            n_features = n_features_default
        # Get a subset of features
        wm = wm.sort_values(["factor", "value_abs"], ascending=False).groupby("factor")
        if w_threshold is None:
            features = wm.head(n_features).feature.unique()
        else:
            features = wm[wm.value_abs >= w_threshold].head(n_features).feature.unique()

    w = w[w.feature.isin(features)].set_index("feature").T

    col_colors = (
        list(features_col.loc[features, :].iloc[:, 0])
        if features_col is not None
        else None
    )

    cg = sns.clustermap(
        w,
        cmap=cmap,
        col_colors=col_colors,
        xticklabels=True,
        row_cluster=cluster_factors,
        col_cluster=cluster_features,
        **kwargs,
    )
    sns.despine(offset=10, trim=True)

    plt.setp(cg.ax_heatmap.xaxis.get_ticklabels(), rotation=90, size=xticklabels_size)

    return cg


def plot_weights_scatter(
    model: mofa_model,
    x="Factor1",
    y="Factor2",
    hist=False,
    n_features: int = 10,
    label_size: int = 5,
    **kwargs,
):
    """
    Plot factor loadings for two factors

    Parameters
    ----------
    model : mofa_model
        Factor model
    x : optional
        Factor which loadings to plot along X axis (Factor1 by default)
    y : optional
        Factor which loadings to plot along Y axis (Factor2 by default)
    hist : optional
        Boolean value if to add marginal histograms to the scatterplot (jointplot)
    n_features : optional
        Number of features to label (default is 10)
    label_size : optional
        Font size of feature labels (default is 5)
    """
    w = model.get_weights(factors=[x, y], df=True).rename_axis("feature").reset_index()

    # Get features to label
    wm = w.melt(id_vars="feature", var_name="factor", value_name="value")
    wm = wm.assign(value_abs=lambda x: x.value.abs())
    wm["factor"] = wm["factor"].astype("category")

    sns_plot = sns.jointplot if hist else sns.scatterplot
    plot = sns_plot(x=x, y=y, data=w, **kwargs)
    sns.despine(offset=10, trim=True)

    # Label some features
    add_text = plot.ax_joint.text if hist else plot.text
    if n_features is not None and n_features > 0:
        # Get a subset of features
        wm = wm.sort_values(["factor", "value_abs"], ascending=False).groupby("factor")
        features = wm.head(n_features).feature.unique()
        w_label = w[w.feature.isin(features)].set_index("feature")
        del wm

        # Add labels to the plot
        for i, point in w_label.iterrows():
            add_text(
                point[x],
                point[y],
                point.name,
                horizontalalignment="left",
                size=label_size,
                color="black",
                weight="regular",
            )

    return plot


def plot_factors(
    model: mofa_model,
    x="Factor1",
    y="Factor2",
    hist=True,
    kde=False,
    groups_df=None,
    linewidth=0,
    size=10,
    **kwargs,
):
    """
    Plot factor values for two factors

    Parameters
    ----------
    model : mofa_model
        Factor model
    x : optional
        Factor to plot along X axis (Factor1 by default)
    y : optional
        Factor to plot along Y axis (Factor2 by default)
    hist : optional
        Boolean value if to add marginal histograms to the scatterplot (jointplot)
    kde : optional
        Boolean value if to add marginal distributions to the scatterplot (jointplot)
    groups_df : optional pd.DataFrame
        Data frame with cells as index and first column as group assignment
    linewidth : optional
        Linewidth argument for dots (default is 0)
    size : optional
        Size argument for dots (ms for plot, s for jointplot and scatterplot; default is 10)
    """
    z = model.get_factors(factors=[x, y], df=True)

    # Assign a group to every cell if it is provided
    if groups_df is not None:
        z = z.rename_axis("cell").reset_index()
        z = z.set_index("cell").join(groups_df).reset_index()
        grouping_var = groups_df.columns[0]

    if hist or kde:
        if group_df is not None:
            # Construct a custom joint plot
            # in order to colour cells
            g = sns.JointGrid(x, y, z)
            for group, group_cells in z.groupby(grouping_var):
                sns.distplot(group_cells[x], ax=g.ax_marg_x, kde=kde, hist=hist)
                sns.distplot(
                    group_cells[y], ax=g.ax_marg_y, vertical=True, kde=kde, hist=hist
                )
                g.ax_joint.plot(group_cells[x], group_cells[y], "o", ms=size, **kwargs)
        else:
            g = sns.jointplot(x=x, y=y, data=z, linewidth=linewidth, s=size, **kwargs)
    else:
        if group_df is not None:
            g = sns.scatterplot(
                x=x,
                y=y,
                data=z,
                linewidth=linewidth,
                s=size,
                hue=grouping_var,
                **kwargs,
            )
        else:
            g = sns.scatterplot(x=x, y=y, data=z, linewidth=linewidth, s=size, **kwargs)
    sns.despine(offset=10, trim=True)

    return g


def plot_factor(
    model: mofa_model,
    factors: Union[int, List[int]] = None,
    x="factor",
    y="value",
    hue="group",
    violin=False,
    **kwargs,
):
    """
    Plot factor values as stripplots (jitter plots)

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
    hue : optional
        Variable to split & colour dots by (cell group by default)
    violin : optional
        Boolean value if to add violin plots
    """
    z = model.get_factors(factors=factors, df=True)
    z = z.rename_axis("cell").reset_index()
    # Make the table long for plotting
    z = z.melt(id_vars="cell", var_name="factor", value_name="value")
    # Add group information for cells
    z = z.set_index("cell").join(model.get_cells().set_index("cell")).reset_index()

    if violin:
        ax = sns.violinplot(x=x, y=y, hue=hue, data=z, inner=None, color=".9")
    ax = sns.stripplot(x=x, y=y, hue=hue, data=z, dodge=True, **kwargs)
    sns.despine(offset=10, trim=True)

    return ax


def plot_r2(
    model: mofa_model,
    factors: Union[int, List[int], str, List[str]] = None,
    view=0,
    **kwargs,
):
    """
    Plot R2 values for the model (draft)

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Index of a factor (or indices of factors) to use (all factors by default)
    view : optional
        Make a plot for a cetrain view (first view by default)
    """
    r2 = model.get_r2(factors=factors)
    # Select a certain view if necessary
    if view is not None:
        view = model.views[view] if isinstance(view, int) else view
        r2 = r2[r2["View"] == view]
    r2_df = r2.sort_values("R2").pivot(index="Factor", columns="Group", values="R2")

    # Sort by factor index
    r2_df.index = r2_df.index.astype("category")
    r2_df.index = r2_df.index.reorder_categories(
        sorted(r2_df.index.categories, key=lambda x: int(x.split("Factor")[1]))
    )
    r2_df = r2_df.sort_values("Factor")

    g = sns.heatmap(r2_df.sort_index(level=0, ascending=False), **kwargs)

    plt.setp(g.yaxis.get_ticklabels(), rotation=0)

    return g


def plot_r2_custom_groups(
    model: mofa_model,
    groups_df: pd.DataFrame,
    factors: Union[int, List[int], str, List[str]] = None,
    view=0,
    **kwargs,
):
    """
    Plot R2 values for the model (draft)

    Parameters
    ----------
    model : mofa_model
        Factor model
    groups_df : pd.DataFrame
        Data frame with cells as index and first column as group assignment
    factors : optional
        Index of a factor (or indices of factors) to use (all factors by default)
    view : optional
        Make a plot for a cetrain view (first view by default)
    """
    r2 = model.get_r2_custom_groups(factors=factors, groups_df=groups_df)
    # Select a certain view if necessary
    if view is not None:
        view = model.views[view] if isinstance(view, int) else view
        r2 = r2[r2["View"] == view]
    r2_df = r2.sort_values("R2").pivot(index="Factor", columns="Group", values="R2")

    # Sort by factor index
    r2_df.index = r2_df.index.astype("category")
    r2_df.index = r2_df.index.reorder_categories(
        sorted(r2_df.index.categories, key=lambda x: int(x.split("Factor")[1]))
    )
    r2_df = r2_df.sort_values("Factor")

    g = sns.heatmap(r2_df.sort_index(level=0, ascending=False), **kwargs)

    plt.setp(g.yaxis.get_ticklabels(), rotation=0)

    return g
