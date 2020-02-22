from .core import mofa_model

from typing import Union, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")
sns.set_palette("Set2")


### WEIGHTS (LOADINGS) ###


def plot_weights(
    model: mofa_model,
    factor="Factor1",
    view=0,
    n_features: int = 10,
    label_size=5,
    x_rank_offset=10,
    x_rank_offset_neg=0,
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
    x_rank_offset_neg : optional
        Offset but for the negative loadings only (i.e. from the right side)
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

    # Set default colour to black if none set
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["color"] = "black"

    # Construct the plot
    ax = sns.lineplot(
        x="rank", y="value", data=w, markers=True, dashes=False, linewidth=0.5, **kwargs
    )
    sns.despine(offset=10, trim=True, ax=ax)

    # Plot top features as dots
    sns.scatterplot(
        x="rank",
        y="value",
        data=w[w["abs_rank"] < n_features],
        linewidth=0.2,
        s=25,
        alpha=0.75,
        **kwargs,
    )

    # Label top loadings

    # Positive weights
    y_start_pos = w[w.value > 0].sort_values("abs_rank").iloc[0].value

    y_prev = y_start_pos
    for i, point in (
        w[(w["abs_rank"] < n_features) & (w["value"] >= 0)].reset_index().iterrows()
    ):
        y_loc = y_prev - y_repel_coef if i != 0 else y_start_pos
        y_loc = min(point["value"], y_loc) if attract_to_points else y_loc
        ax.text(
            x_rank_offset,
            y_loc,
            point["feature"],
            horizontalalignment="left",
            size=label_size,
            color="black",
            weight="regular",
        )
        y_prev = y_loc

    # Negative weights
    y_start_neg = w[w.value < 0].sort_values("abs_rank").iloc[0].value

    y_prev = y_start_neg
    for i, point in (
        w[(w["abs_rank"] < n_features) & (w["value"] < 0)].reset_index().iterrows()
    ):
        y_loc = y_prev + y_repel_coef if i != 0 else y_start_neg
        y_loc = max(point["value"], y_loc) if attract_to_points else y_loc
        ax.text(
            w.shape[0] - x_rank_offset_neg,
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
    ax.set(ylabel=f"{factor_label} loading", xlabel="Feature rank")

    return ax


def plot_weights_scaled(
    model: mofa_model,
    x="Factor1",
    y="Factor2",
    view=0,
    n_features: int = 10,
    w_scaled: bool = True,
    label_size=5,
    y_repel_coef=0.05,
    attract_to_points=True,
    **kwargs,
):
    """
    Plot scaled loadings for 2 factors

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
    y_repel_coef : optional
        Parameter to repel feature labels along the y axis (0.03 by default)
    attract_to_points : optional
        If place labels according to the Y coordinate of the point (False by default)
    """
    w = model.get_weights(views=view, factors=[x, y], df=True)
    w.columns = ["x", "y"]

    if w_scaled:
        w.x = w.x / abs(w.loc[abs(w.x).idxmax()].x)
        w.y = w.y / abs(w.loc[abs(w.y).idxmax()].y)

    wm = (
        w.rename_axis("feature")
        .reset_index()
        .melt(var_name="factor", id_vars=["feature"])
        .assign(
            value_abs=lambda x: np.abs(x.value), value_sign=lambda x: np.sign(x.value)
        )
        .sort_values("value_abs", ascending=False)
        .head(n_features)
        .sort_values(["factor", "value_sign"], ascending=True)
        .drop_duplicates("feature")
    )

    top_features = wm.sort_values("factor", ascending=True).feature.values

    # Construct the plot
    ax = sns.scatterplot("x", "y", data=w, linewidth=0, color="#CCCCCC", **kwargs)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect(1)
    for factor in wm.factor.unique():
        for sign in wm[wm.factor == factor].value_sign.unique():
            feature_set = wm[
                (wm.factor == factor) & (wm.value_sign == sign)
            ].feature.values
            w_set = w.loc[feature_set].sort_values("y", ascending=False)
            y_start_pos = w_set.y.max()
            y_prev = y_start_pos
            for i, row in enumerate(w_set.iterrows()):
                name, point = row
                y_loc = y_prev - y_repel_coef if i != 0 else y_start_pos
                y_loc = min(point.y, y_loc) if attract_to_points else y_loc
                y_prev = y_loc
                ax.text(point.x, y_loc, str(name), size=label_size)
                ax.plot([0, point.x], [0, point.y], linewidth=0.5, color="#333333")

    sns.despine(offset=10, trim=True, ax=ax)
    ax.set_xticks(np.arange(-1, 2.0, step=1.0))
    ax.set_yticks(np.arange(-1, 2.0, step=1.0))

    # Set plot axes labels
    x_factor_label = f"Factor{x+1}" if isinstance(x, int) else x
    y_factor_label = f"Factor{y+1}" if isinstance(y, int) else y
    ax.set(xlabel=f"{x_factor_label} loading", ylabel=f"{y_factor_label} loading")

    return ax


def plot_weights_heatmap(
    model: mofa_model,
    factors: Union[int, List[int]] = None,
    view=0,
    n_features: int = None,
    w_threshold: float = None,
    w_abs: bool = False,
    features_col: pd.DataFrame = None,
    cmap=None,
    xticklabels_size=10,
    yticklabels_size=None,
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
    view : options
        The view to get the loadings of the factor for (first view by default)
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
    yticklabels_size : optional
        Font size for factors labels (default is None)
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
        model.get_weights(views=view, factors=factors, df=True, absolute_values=w_abs)
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

    col_colors = features_col.loc[features, :] if features_col is not None else None

    if not isinstance(factors, Iterable) or len(factors) < 2:
        cluster_factors = False

    cg = sns.clustermap(
        w,
        cmap=cmap,
        col_colors=col_colors,
        xticklabels=True,
        row_cluster=cluster_factors,
        col_cluster=cluster_features,
        **kwargs,
    )

    cg.ax_heatmap.set_xticklabels(
        cg.ax_heatmap.xaxis.get_ticklabels(), rotation=90, size=xticklabels_size
    )
    cg.ax_heatmap.set_yticklabels(
        cg.ax_heatmap.yaxis.get_ticklabels(), rotation=0, size=yticklabels_size
    )

    return cg


def plot_weights_dotplot(
    model: mofa_model,
    factors: Union[int, List[int]] = None,
    view=0,
    n_features: int = None,
    w_threshold: float = None,
    w_abs: bool = False,
    col_wrap: Optional[int] = 4,
    yticklabels_size: int = 10,
    **kwargs,
):
    """
    Plot loadings for top features as a dotplot

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Factors to use (all factors in the model by default)
    view : options
        The view to get the loadings of the factor for (first view by default)
    n_features : optional
        Number of features for each factor by their absolute value (10 by default)
    w_threshold : optional
        Absolute loading threshold for a feature to plot (no threshold by default)
    w_abs : optional
        If plot absolute loadings values
    col_wrap : optional
        Number of columns per row when plotting multiple factors
    yticklabels_size : optional
        Font size for features labels (default is 10)
    """

    # Set defaults
    n_features_default = 10
    if factors is None:
        factors = list(range(model.nfactors))

    # Fetch weights for the relevant factors
    w = (
        model.get_weights(views=view, factors=factors, df=True, absolute_values=w_abs)
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
        wmf = wm.sort_values(["factor", "value_abs"], ascending=False).groupby("factor")
        if w_threshold is None:
            features = wmf.head(n_features).feature.unique()
        else:
            features = (
                wmf[wmf.value_abs >= w_threshold].head(n_features).feature.unique()
            )

    wm = wm.apply(lambda x: x.reset_index(drop=True))
    wm = wm[wm.feature.isin(features)]

    sns.set(style="whitegrid")

    # Make the PairGrid
    g = sns.FacetGrid(wm, col="factor", col_wrap=col_wrap)

    # Draw a dot plot using the stripplot function
    g.map(
        sns.stripplot,
        "value",
        "feature",
        "value_abs",
        size=10,
        orient="h",
        order=features,
        palette="ch:s=1,r=-.1",
        linewidth=1,
        edgecolor="w",
    )

    # Use the same x axis limits on all columns and add better labels
    g.set(xlabel="Loadings", ylabel="")

    # Use semantically meaningful titles for the columns
    titles = [f"Factor{i+1}" for i in factors]

    for ax, title in zip(g.axes.flat, titles):

        # Set a different title for each axes
        ax.set(title=title)

        # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.set_yticklabels(ax.yaxis.get_ticklabels(), size=yticklabels_size)

    sns.despine(left=True, bottom=True, offset=10, trim=True)

    return g


def plot_weights_scatter(
    model: mofa_model,
    x="Factor1",
    y="Factor2",
    view=0,
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
    view : options
        The view to get the loadings of the factor for (first view by default)
    hist : optional
        Boolean value if to add marginal histograms to the scatterplot (jointplot)
    n_features : optional
        Number of features to label (default is 10)
    label_size : optional
        Font size of feature labels (default is 5)
    """
    w = (
        model.get_weights(views=view, factors=[x, y], df=True)
        .rename_axis("feature")
        .reset_index()
    )

    # Get features to label
    wm = w.melt(id_vars="feature", var_name="factor", value_name="value")
    wm = wm.assign(value_abs=lambda x: x.value.abs())
    wm["factor"] = wm["factor"].astype("category")

    # Set default colour to darkgrey if none set
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["color"] = "darkgrey"

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


### FACTOR VALUES ###


def plot_factors_scatter(
    model: mofa_model,
    x="Factor1",
    y="Factor2",
    hist=False,
    kde=False,
    groups=None,
    groups_df=None,
    group_label=None,
    color=None,
    linewidth=0,
    size=5,
    legend=False,
    legend_loc="best",
    legend_prop=None,
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
    groups : optional
        Subset of groups to consider
    groups_df : optional pd.DataFrame
        Data frame with samples (cells) as index and first column as group assignment
    group_label : optional
        Sample (cell) metadata column to be used as group assignment
    color : optional
        Grouping variable by default, alternatively a feature name can be provided (when no kde/hist)
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
    """
    z = model.get_factors(factors=[x, y], groups=groups, df=True)
    z.columns = ["x", "y"]

    # Assign a group to every cell if it is provided
    if groups_df is None and group_label is None:
        group_label = "group"

    if groups_df is None:
        groups_df = model.samples_metadata.loc[:,[group_label]]

    z = z.rename_axis("sample").reset_index()
    z = z.set_index("sample").join(groups_df).reset_index()
    grouping_var = groups_df.columns[0]

    # Assign colour to every cell if colouring by feature expression
    if color is None:
        color_var = grouping_var
    else:
        color_var = color
        color_df = model.get_data(features=color, df=True)
        z = z.set_index("sample").join(color_df).reset_index()
        z = z.sort_values(color_var)

    # Define plot axes labels
    x_factor_label = f"Factor{x+1}" if isinstance(x, int) else x
    y_factor_label = f"Factor{y+1}" if isinstance(y, int) else y

    # Set default colour to black if none set
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["color"] = "black"

    if hist or kde:
        if groups_df is not None:
            # Construct a custom joint plot
            # in order to colour samples (cells)
            g = sns.JointGrid(x="x", y="y", data=z)
            group_labels = []
            for group, group_samples in z.groupby(grouping_var):
                sns.distplot(group_samples["x"], ax=g.ax_marg_x, kde=kde, hist=hist)
                sns.distplot(
                    group_samples["y"], ax=g.ax_marg_y, vertical=True, kde=kde, hist=hist
                )
                g.ax_joint.plot(
                    group_samples["x"], group_samples["y"], "o", ms=size, **kwargs
                )
                group_labels.append(group)
            if legend:
                legend = g.ax_joint.legend(
                    labels=group_labels, loc=legend_loc, prop=legend_prop
                )
        else:
            # DEPRECATED
            g = sns.jointplot(
                x="x", y="y", data=z, linewidth=linewidth, s=size, **kwargs
            )
        sns.despine(offset=10, trim=True, ax=g.ax_joint)
        g.ax_joint.set(
            xlabel=f"{x_factor_label} value", ylabel=f"{y_factor_label} value"
        )
    else:
        if groups_df is not None:
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
        else:
            g = sns.scatterplot(
                x="x",
                y="y",
                data=z,
                linewidth=linewidth,
                s=size,
                legend=legend,
                **kwargs,
            )
        sns.despine(offset=10, trim=True, ax=g)
        g.set(xlabel=f"{x_factor_label} value", ylabel=f"{y_factor_label} value")

    return g


def plot_factors(
    model: mofa_model,
    factors: Union[int, List[int]] = None,
    x="factor",
    y="value",
    hue="group",
    violin=False,
    groups_df=None,
    group_label: Optional[str] = None,
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
    groups_df : optional pd.DataFrame
        Data frame with samples (cells) as index and first column as group assignment
    group_label : optional
        Sample (cell) metadata column to be used as group assignment
    """
    z = model.get_factors(factors=factors, df=True)
    z = z.rename_axis("sample").reset_index()
    # Make the table long for plotting
    z = z.melt(id_vars="sample", var_name="factor", value_name="value")

    # Assign a group to every cell if it is provided
    if groups_df is None and group_label is None:
        group_label = "group"

    if groups_df is None:
        groups_df = model.samples_metadata.loc[:,[group_label]]

    # Add group information for samples (cells)
    z = z.set_index("sample").join(groups_df).reset_index()

    if violin:
        ax = sns.violinplot(x=x, y=y, hue=hue, data=z, inner=None, color=".9")
    ax = sns.stripplot(x=x, y=y, hue=hue, data=z, dodge=True, **kwargs)
    sns.despine(offset=10, trim=True)

    ax.set(xlabel="", ylabel="Factor value")

    return ax


def plot_factors_matrixplot(
    model: mofa_model,
    factors: Optional[Union[int, List[int], str, List[str]]] = None,
    groups_df=None,
    group_label: Optional[str] = None,
    groups: Optional[Union[int, List[int], str, List[str]]] = None,
    agg="mean",
    cmap="viridis",
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
    groups_df : optional pd.DataFrame
        Data frame with samples (cells) as index and first column as group assignment
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
    if groups_df is None and group_label is None:
        group_label = "group"

    if groups_df is None:
        groups_df = model.samples_metadata.loc[:,[group_label]]

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

    ax = sns.heatmap(z, cmap=cmap, **kwargs)
    ax.set(ylabel="Factor", xlabel="Group")
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), rotation=0)

    return ax


def plot_factors_umap(
    model: mofa_model,
    embedding: pd.DataFrame = None,
    factors: Optional[Union[int, List[int]]] = None,
    groups=None,
    groups_df=None,
    group_label: Optional[str] = None,
    color=None,
    linewidth=0,
    size=5,
    legend=False,
    legend_loc="best",
    legend_prop=None,
    n_neighbors=10,
    spread=1,
    **kwargs,
):
    """
    Plot factor values for two factors

    Parameters
    ----------
    model : mofa_model
        Factor model
    embedding : optional pd.DataFrame
        Output of UMAP embedding from mofax.umap (or any other embedding with samples (cells) as index)
    factors : optional
        Index of a factor (or indices of factors) to use (all factors by default)
    groups : optional
        Subset of groups to consider
    groups_df : optional pd.DataFrame
        Data frame with samples (cells) as index and first column as group assignment
    group_label : optional
        Sample (cell) metadata column to be used as group assignment
    color : optional
        Grouping variable by default, alternatively a feature name can be provided
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
    n_neighbors : optional
        n_neighbors parameter for UMAP
    spread : optional
        spread parameter for UMAP
    """

    if embedding is None:
        z = model.get_factors(factors=factors, groups=groups)

        import umap

        embedding = pd.DataFrame(
            umap.UMAP(n_neighbors=n_neighbors, spread=spread).fit_transform(z)
        )

        embedding.columns = ["UMAP1", "UMAP2"]
        embedding.index = model.get_samples().sample

    x, y, *_ = embedding.columns

    # Assign a group to every sample (cell) if it is provided
    if groups_df is None and group_label is None:
        group_label = "group"

    if groups_df is None:
        groups_df = model.samples_metadata.loc[:,[group_label]]

    embedding = embedding.rename_axis("sample").reset_index()
    embedding = embedding.set_index("sample").join(groups_df).reset_index()
    grouping_var = groups_df.columns[0]

    # Assign colour to every sample (cell) if colouring by feature expression
    if color is None:
        color_var = grouping_var
    else:
        color_var = color
        color_df = model.get_data(features=color, df=True)
        embedding = embedding.set_index("sample").join(color_df).reset_index()
        embedding = embedding.sort_values(color_var)

    # Define plot axes labels
    x_factor_label = f"Factor{x+1}" if isinstance(x, int) else x
    y_factor_label = f"Factor{y+1}" if isinstance(y, int) else y

    # Set default colour to black if none set
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["color"] = "black"

    if groups_df is not None:
        g = sns.scatterplot(
            x=x,
            y=y,
            data=embedding,
            linewidth=linewidth,
            s=size,
            hue=color_var,
            legend=legend,
            **kwargs,
        )
    return g


### VARIANCE EXPLAINED ###


def plot_r2(
    model: mofa_model,
    factors: Union[int, List[int], str, List[str]] = None,
    groups_df: pd.DataFrame = None,
    group_label: str = None,
    view=0,
    group=None,
    cmap="Blues",
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
        Make a plot for a certain view (first view by default)
    group : optional
        Make a plot for a certain group (None by default to plot all groups)
    group_label : optional
        Sample (cell) metadata column to be used as group assignment
    groups_df : optional pd.DataFrame
        Data frame with samples (cells) as index and first column as group assignment
    cmap : optional
        The colourmap for the heatmap (default is 'Blues' with darker colour for higher R2)
    """
    r2 = model.get_r2(factors=factors, group_label=group_label, groups_df=groups_df)
    # Select a certain view if necessary
    if view is not None:
        view = model.views[view] if isinstance(view, int) else view
        r2 = r2[r2["View"] == view]
    # Select a certain group if necessary
    if group is not None:
        group = model.groups[group] if isinstance(group, int) else group
        r2 = r2[r2["Group"] == group]
    r2_df = r2.sort_values("R2").pivot(index="Factor", columns="Group", values="R2")

    # Sort by factor index
    r2_df.index = r2_df.index.astype("category")
    r2_df.index = r2_df.index.reorder_categories(
        sorted(r2_df.index.categories, key=lambda x: int(x.split("Factor")[1]))
    )
    r2_df = r2_df.sort_values("Factor")

    g = sns.heatmap(r2_df.sort_index(level=0, ascending=False), cmap=cmap, **kwargs)

    g.set_yticklabels(g.yaxis.get_ticklabels(), rotation=0)

    return g


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
    r2 = model.get_r2_null(
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
    """
    r2 = model.get_r2(factors=factors, groups_df=groups_df, group_label=group_label)
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

    g = sns.barplot(
        data=r2_df.sort_index(level=0, ascending=False), x=x, y=y, hue=groupby, **kwargs
    )

    g.set_xticklabels(g.xaxis.get_ticklabels(), rotation=90, size=xticklabels_size)

    return g


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
    size=5,
    legend=False,
    legend_loc="best",
    legend_prop=None,
    feature_intersection=False,
    **kwargs,
):
    """
    Project new data onto the factor space of the model.
    
    For the projection, a pseudo-inverse of the loadings matrix is calculated 
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
