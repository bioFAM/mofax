from .core import mofa_model, umap, padjust_fdr_2d

import sys
from warnings import warn
from typing import Union, Optional, List, Iterable, Sequence

import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

sns.set_style("ticks")
sns.set_palette("Set2")


### WEIGHTS (LOADINGS) ###


def plot_weights(
    model: mofa_model,
    factors=None,
    views=None,
    n_features: int = 5,
    w_scaled: bool = False,
    w_abs: bool = False,
    size=2,
    color="black",
    label_size=5,
    x_offset=.01,
    y_offset=.15,
    jitter=.01,
    line_width=.5,
    line_color="black",
    line_alpha=.2,
    zero_line=False,
    zero_line_width=1,
    ncols=4,
    **kwargs,
):
    """
    Plot weights (loadings) for a specific factor

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : str or int or list of str or None
        Factors to use (default is all)
    views : str or int or list of str or None
        The views to get the factors weights for (first view by default)
    n_features : int
        Number of features to label with most positive and most negative weights
    w_scaled : bool
        If scale weights to unite variance (False by default)
    w_abs : bool
        If plot absolute weight values (False by default)
    size : float
        Dot size (2 by default)
    color : str
        Colour for the labelled dots (black by default)
    label_size : int or float
        Font size of feature labels (default is 5)
    x_offset : int or float
        Offset the feature labels from the left/right side (by 0.03 points by default)
    y_offset : int or float
        Parameter to repel feature labels along the y axis (0.1 by default)
    jitter : bool
        Jitter dots per factors (True by default)
    line_width : int or float
        Width of the lines connecting labels with dots (0.5 by default)
    line_color : str
        Color of the lines connecting labels with dots (black by default)
    line_alpha : float
        Alpha level for the lines connecting labels with dots (0.2 by default)
    zero_line : bool
        If to plot a dotted line at zero (False by default)
    zero_line_width : int or float
        Width of the line at 0 (1 by default)
    ncols : int
        Number of columns in the grid of multiple plots, one plot per view (4 by default)
    """
    w = model.get_weights(views=views, factors=factors, df=True, 
                          scaled_values=w_scaled, absolute_values=w_abs) \
             .join(model.features_metadata.loc[:,["view"]])
    w = w.rename_axis("feature").reset_index()
    # Make the table long for plotting
    wm = w.melt(id_vars=["feature", "view"], var_name="factor", value_name="value")
    wm["abs_value"] = abs(wm.value)

    # Assign ranks to features, per factor
    wm["rank"] = wm.groupby("factor")["value"].rank(ascending=False)
    wm["abs_rank"] = wm.groupby("factor")["abs_value"].rank(ascending=False)
    wm = wm.sort_values(["factor", "abs_rank"], ascending=True)

    # Sort factors
    wm["factor"] = wm["factor"].astype("category")
    wm["factor"] = wm["factor"].cat.reorder_categories(
        sorted(wm["factor"].cat.categories, key=lambda x: int(x.split("Factor")[1]))
    )

    # Set default colour to black if none set
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["color"] = "black"

    # Fetch top features to label
    features_to_label = model.get_top_features(factors=factors, views=views, n_features=n_features, df=True)
    features_to_label["to_label"] = True
    wm = features_to_label \
        .loc[:,["feature", "view", "factor", "to_label"]] \
        .set_index(["feature", "view", "factor"]) \
        .join(wm.set_index(["feature", "factor", "view"]), how="right") \
        .reset_index() \
        .fillna({"to_label": False}) \
        .sort_values(["factor", "to_label"])

    # Figure out rows & columns for the grid with plots (one plot per view)
    view_vars = wm.view.unique()
    ncols = min(ncols, len(view_vars))
    nrows = int(np.ceil(len(view_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, 
                             sharex=True, sharey=True,
                             figsize=(ncols * rcParams['figure.figsize'][0],
                                      nrows * rcParams['figure.figsize'][1]))
    if ncols == 1:
        axes = np.array(axes).reshape(-1, 1)
    if nrows == 1:
        axes = np.array(axes).reshape(1, -1)

    for m, view in enumerate(view_vars):
        ri = m // ncols
        ci = m % ncols

        wm_view = wm.query('view == @view')

        # Construct the plot
        g = sns.stripplot(data=wm_view, x='value', y='factor', jitter=jitter, size=size,
                          hue="to_label", palette=["lightgrey", color], ax=axes[ri,ci])
        sns.despine(offset=10, trim=True, ax=g)
        g.legend().remove()

        # Label some points
        for fi, factor in enumerate(wm_view.factor.cat.categories):
            for sign_i in [1, -1]:
                to_label = features_to_label.query('factor == @factor & view == @view').feature.tolist()
                w_set = wm_view.query('factor == @factor & value * @sign_i > 0 & feature == @to_label & view == @view') \
                               .sort_values("abs_value", ascending=False)

                x_start_pos = sign_i * (w_set.abs_value.max() + x_offset)
                y_start_pos = fi - ((w_set.shape[0] - 1) // 2) * y_offset
                y_prev = y_start_pos

                for i, row in enumerate(w_set.iterrows()):
                    name, point = row
                    y_loc = y_prev + y_offset if i != 0 else y_start_pos

                    g.annotate(point["feature"],
                        xy=(point.value, fi),
                        xytext=(x_start_pos, y_loc),
                        arrowprops=dict(arrowstyle="-",
                                        connectionstyle="arc3",
                                        color=line_color,
                                        alpha=line_alpha,
                                        linewidth=line_width),
                        horizontalalignment="left" if sign_i > 0 else "right",
                        size=label_size,
                        color="black",
                        weight="regular",
                        alpha=.9
                    )
                    y_prev = y_loc

        # Set plot axes labels
        g.set(ylabel="", xlabel="Feature weight", title=view)

        if zero_line:
            axes[ri,ci].axvline(0, ls='--', color="lightgrey", linewidth=zero_line_width, zorder=0)

    # Remove unused axes
    for i in range(len(view_vars), ncols * nrows):
        ri = i // ncols
        ci = i % ncols
        fig.delaxes(axes[ri,ci])


    return g


def plot_weights_ranked(
    model: mofa_model,
    factor="Factor1",
    view=0,
    n_features: int = 10,
    size : int = 25,
    label_size=5,
    x_rank_offset=10,
    x_rank_offset_neg=0,
    y_repel_coef=0.03,
    attract_to_points=True,
    **kwargs,
):
    """
    Plot weights (loadings) for a specific factor

    Parameters
    ----------
    model : mofa_model
        Factor model
    factor : optional
        Factor to use (default is Factor1)
    view : options
        The view to get the factors weights for (first view by default)
    n_features : optional
        Number of features to label with most positive and most negative weights
    size : int
        Dit size for labelled features (default is 25)
    label_size : optional
        Font size of feature labels (default is 5)
    x_rank_offset : optional
        Offset the feature labels from the left/right side (by 10 points by default)
    x_rank_offset_neg : optional
        Offset but for the negative weights only (i.e. from the right side)
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
        s=size,
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
    ax.set(ylabel=f"{factor_label} weight", xlabel="Feature rank")

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
    Plot scaled weights (loadings) for 2 factors

    Parameters
    ----------
    model : mofa_model
        Factor model
    factor : optional
        Factor to use (default is Factor1)
    view : options
        The view to get the factors weights for (first view by default)
    n_features : optional
        Number of features to label with most positive and most negative weights
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
    ax.set(xlabel=f"{x_factor_label} weight", ylabel=f"{y_factor_label} weight")

    return ax


def plot_weights_heatmap(
    model: mofa_model,
    factors: Union[int, List[int]] = None,
    view=0,
    n_features: int = None,
    w_threshold: float = None,
    w_abs: bool = False,
    only_positive: bool = False,
    only_negative: bool = False,
    features_col: pd.DataFrame = None,
    cmap=None,
    xticklabels_size=10,
    yticklabels_size=None,
    cluster_factors=True,
    cluster_features=True,
    **kwargs,
):
    """
    Plot weights (loadings) for top features in a heatmap

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Factors to use (all factors in the model by default)
    view : options
        The view to get the factors weights for (first view by default)
    n_features : optional
        Number of features for each factor by their absolute value (10 by default)
    w_threshold : optional
        Absolute weight threshold for a feature to plot (no threshold by default)
    w_abs : optional
        If to plot absolute weight values
    only_positive : optional
        If to plot only positive weights
    only_negative : optional
        If to plot only negative weights
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

    if only_positive and only_negative:
        print("Please specify either only_positive or only_negative")
        sys.exit(1)
    elif only_positive:
        wm = wm[wm.value > 0]
    elif only_negative:
        wm = wm[wm.value < 0]

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
    only_positive: bool = False,
    only_negative: bool = False,
    palette = None,
    size: int = 30,
    linewidth: int = 1,
    xticklabels_size=8,
    yticklabels_size=5,
    **kwargs,
):
    """
    Plot weights (loadings) for top features in a heatmap

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Factors to use (all factors in the model by default)
    view : options
        The view to get the factors weights for (first view by default)
    n_features : optional
        Number of features for each factor by their absolute value (5 by default)
    w_threshold : optional
        Absolute weight threshold for a feature to plot (no threshold by default)
    w_abs : optional
        If to plot absolute weight values
    only_positive : optional
        If to plot only positive weights
    only_negative : optional
        If to plot only negative weights
    palette : optional
        Color map (blue-to-red divergent palette with by default)
    size : optional
        Dot size (default in 30)
    lienwidth : optional
        Dot outline width (default is 1)
    xticklabels_size : optional
        Font size for features labels (default is 10)
    yticklabels_size : optional
        Font size for factors labels (default is None)
    """

    # Set defaults
    n_features_default = 5
    if factors is None:
        factors = list(range(model.nfactors))
    if palette is None:
        palette = sns.diverging_palette(240, 10, n=9, as_cmap=True)

    # Fetch weights for the relevant factors
    w = (
        model.get_weights(views=view, factors=factors, df=True, absolute_values=w_abs)
        .rename_axis("feature")
        .reset_index()
    )
    wm = w.melt(id_vars="feature", var_name="factor", value_name="value")
    wm = wm.assign(value_abs=lambda x: x.value.abs())
    wm["factor"] = wm["factor"].astype("category")

    if only_positive and only_negative:
        print("Please specify either only_positive or only_negative")
        sys.exit(1)
    elif only_positive:
        wm = wm[wm.value > 0]
    elif only_negative:
        wm = wm[wm.value < 0]

    
    # Fix factors order
    wm.factor = wm.factor.astype("category")
    wm.factor = wm.factor.cat.reorder_categories(
        sorted(wm.factor.cat.categories, key=lambda x: int(x.split("Factor")[1]))
    )
    wm.sort_values("factor")

    if n_features is None and w_threshold is not None:
        features = wm[wm.value_abs >= w_threshold].feature.unique()
    else:
        if n_features is None:
            n_features = n_features_default
        # Get a subset of features
        wm_g = wm.sort_values(["factor", "value_abs"], ascending=False).groupby("factor")
        if w_threshold is None:
            features = wm_g.head(n_features).feature.unique()
        else:
            features = wm_g[wm_g.value_abs >= w_threshold].head(n_features).feature.unique()

    
    wm = wm[wm.feature.isin(features)]
    
    # Fix features order
    wm.feature = wm.feature.astype("category")
    wm.feature = wm.feature.cat.reorder_categories(features)
    
    wm = wm.sort_values(["factor", "feature"])

    g = sns.scatterplot(
        data=wm,
        x="factor",
        y="feature",
        hue="value",
        linewidth=linewidth,
        s=size,
        palette=palette,
        **kwargs,
    )

    norm = plt.Normalize(wm.value.min(), wm.value.max())
    cmap = palette if palette is not None else sns.diverging_palette(220, 20, as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    try:
        g.figure.colorbar(sm)
        g.get_legend().remove()
    except Exception:
        warn("Cannot make a proper colorbar")
    
    plt.draw()
    
    g.set_xticklabels(g.get_xticklabels(), rotation=90, size=xticklabels_size)
    g.set_yticklabels(g.get_yticklabels(), size=yticklabels_size)


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
    Plot weights (loadings) for two factors

    Parameters
    ----------
    model : mofa_model
        Factor model
    x : optional
        Factor which weights to plot along X axis (Factor1 by default)
    y : optional
        Factor which weights to plot along Y axis (Factor2 by default)
    view : options
        The view to get the factors weights for (first view by default)
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


def plot_weights_correlation(
    model: mofa_model,
    factors: Optional[Union[int, List[int]]] = None,
    views=None,
    covariates=None,
    linewidths=0,
    diag=False,
    full=True,
    cmap=None,
    square=True,
    **kwargs,
):
    """
    Plot correlation of weights and, if provided, covariates

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
    linewidths : optional
        Heatmap linewidths argument (default is 0)
    diag : optional
        If to only plot lower triangle of the correlation matrix (False by default)
    full : optional
        If covariates are provided, also plot inter-factor and inter-covariates correlation coefficients (True by default)
    square : optional
        Heatmap square argument (True by default)
    cmap : optional
        Heatmap cmap argument
    """

    w = model.get_weights(factors=factors, views=views)
    if covariates is not None:
        # Transform a vector to a matrix
        if len(covariates.shape) == 1:
            covariates = pd.DataFrame(covariates)
        corr = np.corrcoef(w.T, covariates.T)
    else:
        corr = np.corrcoef(w.T)

    if covariates is not None:
        if not full:
            n_cov = covariates.shape[1]
            corr = corr[0:-n_cov,-n_cov:]

    mask = None
    if diag:
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    if cmap is None:
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Generate labels for the heatmap
    if factors is None:
        factors = range(w.shape[1])
    fnames = [f"Factor{fi+1}" if isinstance(fi, int) else fi for fi in factors]
    if covariates is not None:
        if isinstance(covariates, pd.DataFrame):
            cnames = covariates.columns.values
        else:
            cnames = [f"Covar{ci+1}" for ci in covariates.shape[1]]
        xticklabels = cnames if not full else np.concatenate((fnames, cnames))
        yticklabels = fnames if not full else np.concatenate((fnames, cnames))
    else:
        xticklabels = fnames
        yticklabels = fnames

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(corr, cmap=cmap, mask=mask, center=0,
                    square=True, linewidths=.5,
                    xticklabels=xticklabels, yticklabels=yticklabels,
                    cbar_kws={"shrink": .5}, **kwargs)

    g.set_yticklabels(g.yaxis.get_ticklabels(), rotation=0)

    return g


### FACTOR VALUES ###

def plot_factors_scatter(
    model: mofa_model,
    x="Factor1",
    y="Factor2",
    dist=False,
    groups=None,
    group_label='group',
    color=None,
    zero_line=False,
    linewidth=0,
    zero_linewidth=1,
    size=5,
    legend=True,
    legend_prop=None,
    palette=None,
    ncols=4,
    sharex=False,
    sharey=False,
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
    """

    # Process input arguments
    if group_label == 'group' and not color:
        color = 'group'
    color_vars = [color] if not color or isinstance(color, str) else color

    assert not (len(color_vars) > 1 and dist), "When plotting distributions, only one color can be provided at the moment"

    # Get factors
    z = model.get_factors(factors=[x, y], df=True)
    z.columns = ["x", "y"]

    # Add group and colour information
    vars = [group_label, *color_vars]
    if any([not(not(i)) for i in vars]):
        meta = model.fetch_values(variables=vars)
        z = z.rename_axis("sample").reset_index()
        z = z.set_index("sample").join(meta).reset_index()

    # Subset groups (incl. custom groups of samples)
    if group_label and groups is not None:
        z = z[z[group_label].isin(groups)]

    # Define plot axes labels
    x_factor_label = f"Factor{x+1}" if isinstance(x, int) else x
    y_factor_label = f"Factor{y+1}" if isinstance(y, int) else y

    # Set default colour to black if none set
    if "c" not in kwargs and not color:
        kwargs["color"] = "black"


    legend_str = 'brief' if (legend and color) else False

    if dist:
        g = sns.jointplot(x="x", y="y",
                      hue=color_vars[0],
                      data=z.sort_values(color_vars[0]) if color_vars[0] else z,
                      linewidth=linewidth,
                      s=size,
                      legend=legend_str,
                      palette=palette,
                      **kwargs
                      )
        sns.despine(offset=10, trim=True, ax=g.ax_joint)
        g.ax_joint.set(
            xlabel=f"{x_factor_label} value", ylabel=f"{y_factor_label} value"
        )
        if legend:
            g.ax_joint.legend(bbox_to_anchor=(1.4, 1), loc=2, borderaxespad=0., prop=legend_prop)
        if zero_line:
            g.axhline(0, ls='--', color="lightgrey", linewidth=linewidth, zorder=0)
            g.axvline(0, ls='--', color="lightgrey", linewidth=linewidth, zorder=0)
    else:
        # Figure out rows & columns for the grid with plots
        ncols = min(ncols, len(color_vars))
        nrows = int(np.ceil(len(color_vars) / ncols))
        fig, axes = plt.subplots(nrows, ncols, 
                                 sharex=sharex, sharey=sharey,
                                 figsize=(ncols * rcParams['figure.figsize'][0],
                                          nrows * rcParams['figure.figsize'][1]))
        if ncols == 1:
            axes = np.array(axes).reshape(-1, 1)
        if nrows == 1:
            axes = np.array(axes).reshape(1, -1)

        for i, color_var in enumerate(color_vars):
            ri = i // ncols
            ci = i % ncols
            g = sns.scatterplot(
                x="x",
                y="y",
                data=z.sort_values(color_var) if color_var else z,
                linewidth=linewidth,
                s=size,
                hue=color_var,
                legend=legend_str,
                palette=palette,
                ax=axes[ri, ci],
                **kwargs,
            )
            sns.despine(offset=10, trim=True, ax=g)
            g.set(xlabel=f"{x_factor_label} value", ylabel=f"{y_factor_label} value", title=color_var)

            if legend and color_var:
                if is_numeric_dtype(z[color_var]):
                    means = z.groupby(group_label)[color_var].mean() if group_label else z[color_var].mean()
                    norm = plt.Normalize(means.min(), means.max())
                    cmap = palette if palette is not None else sns.cubehelix_palette(as_cmap=True)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    try:
                        g.figure.colorbar(sm, ax=axes[ri, ci])
                        g.get_legend().remove()
                    except Exception:
                        warn("Cannot make a proper colorbar")
                else:
                    g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop=legend_prop)

            if zero_line:
                axes[ri,ci].axhline(0, ls='--', color="lightgrey", linewidth=zero_linewidth, zorder=0)
                axes[ri,ci].axvline(0, ls='--', color="lightgrey", linewidth=zero_linewidth, zorder=0)

        # Remove unused axes
        for i in range(len(color_vars), ncols * nrows):
            ri = i // ncols
            ci = i % ncols
            fig.delaxes(axes[ri,ci])

    plt.tight_layout()

    return g

plot_factors = plot_factors_scatter

def plot_factors_violin(
    model: mofa_model,
    factors: Union[int, List[int]] = None,
    x="factor",
    y="value",
    color="group",
    violins=True,
    dots=False,
    zero_line=True,
    group_label='group',
    groups=None,
    linewidth=0,
    zero_linewidth=1,
    size=4,
    legend=True,
    legend_prop=None,
    palette=None,
    ncols=4,
    sharex=False,
    sharey=False,
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
    ncols : optional
        Number of columns if multiple colours are defined (4 by default)
    sharex: optional
        Common X axis across plots on the grid
    sharey: optional
        Common Y axis across plots on the grid
    """

    # Process input arguments
    if group_label == 'group' and not color:
        color = 'group'
    color_vars = [color] if not color or isinstance(color, str) else color

    assert violins or dots, "Either violins=True or dots=True"

    # Get factors
    z = model.get_factors(factors=factors, df=True)
    z = z.rename_axis("sample").reset_index()
    # Make the table long for plotting
    z = z.melt(id_vars="sample", var_name="factor", value_name="value")

    # Add group and colour information
    vars = [group_label, *color_vars]
    if any([not(not(i)) for i in vars]):
        meta = model.fetch_values(variables=vars)
        z = z.set_index("sample").join(meta).reset_index()

    # Subset groups (incl. custom groups of samples)
    if group_label and groups is not None:
        z = z[z[group_label].isin(groups)]

    z["factor_idx"] = z.factor.str.lstrip("Factor").astype(int)
    z = z.sort_values(by="factor_idx")

    # Set default colour to black if none set
    if "c" not in kwargs and not color:
        kwargs["color"] = "black"

    legend_str = 'brief' if (legend and color) else False

    ncols = min(ncols, len(color_vars))
    nrows = int(np.ceil(len(color_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, 
                             sharex=True, sharey=True,
                             figsize=(ncols * rcParams['figure.figsize'][0],
                                      nrows * rcParams['figure.figsize'][1]))
    if nrows == 1:
        axes = np.array(axes).reshape(1, -1)
        
    for i, color_var in enumerate(color_vars):
        ri = i // ncols
        ci = i % ncols
        
        legend_str = 'brief' if (legend and color) else False

        if zero_line:
            axes[ri,ci].axhline(0, ls='--', color="lightgrey", linewidth=zero_linewidth, zorder=0)
        if violins:
            g = sns.violinplot(x=x, y=y, data=z.sort_values(color_var).sort_values("factor_idx"),
                               hue=color_var, linewidth=linewidth, s=size, 
                               legend=legend_str, palette=palette, inner=None, ax=axes[ri,ci])
        if dots:
            g = sns.stripplot(x=x, y=y, data=z.sort_values(color_var).sort_values("factor_idx"),
                              hue=color_var, linewidth=linewidth, s=size, 
                              palette=palette, dodge=True, ax=axes[ri,ci], **kwargs)
        sns.despine(offset=10, trim=True, ax=g)
        g.set(xlabel="", ylabel="Factor value")

        if legend and color_var:
            g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop=legend_prop)
                    
        else:
            try:
                g.get_legend().remove()
            except Exception:
                pass

    # Remove unused axes
    for i in range(len(color_vars), ncols * nrows):
        ri = i // ncols
        ci = i % ncols
        fig.delaxes(axes[ri,ci])

    plt.tight_layout()

    return g


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

    ax = sns.heatmap(z, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set(ylabel="Factor", xlabel="Group")
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), rotation=0)

    return ax


def plot_factors_umap(
    model: mofa_model,
    embedding: pd.DataFrame = None,
    factors: Optional[Union[int, List[int]]] = None,
    groups=None,
    group_label: Optional[str] = None,
    color=None,
    linewidth=0,
    size=5,
    n_neighbors=10,
    spread=1,
    random_state=None,
    umap_kwargs={},
    legend=True,
    legend_prop=None,
    palette=None,
    ncols=4,
    **kwargs,
):
    """
    Plot UMAP on factor values

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
    if group_label == 'group' and not color:
        color = 'group'
    color_vars = [color] if not color or isinstance(color, str) else color

    if embedding is None:
        embedding = umap(model.get_factors(factors=factors, groups=groups, df=True),
                                           n_neighbors=n_neighbors, spread=spread, random_state=random_state, 
                                           **umap_kwargs)
        embedding.columns = ["UMAP1", "UMAP2"]
        embedding.index = model.get_samples()['sample'].values

    x, y, *_ = embedding.columns

    # Add group and colour information
    vars = [group_label, *color_vars]
    if any([not(not(i)) for i in vars]):
        meta = model.fetch_values(variables=vars)
        embedding = embedding.rename_axis("sample").reset_index()
        embedding = embedding.set_index("sample").join(meta).reset_index()

    # Subset groups (incl. custom groups of samples)
    if group_label and groups is not None:
        embedding = embedding[embedding[group_label].isin(groups)]

    # Define plot axes labels
    x_factor_label = f"Factor{x+1}" if isinstance(x, int) else x
    y_factor_label = f"Factor{y+1}" if isinstance(y, int) else y

    # Set default colour to black if none set
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["color"] = "black"

    legend_str = 'brief' if (legend and color) else False

    # Figure out rows & columns for the grid with plots
    ncols = min(ncols, len(color_vars))
    nrows = int(np.ceil(len(color_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, 
                             sharex=True, sharey=True,
                             figsize=(ncols * rcParams['figure.figsize'][0],
                                      nrows * rcParams['figure.figsize'][1]))
    if ncols == 1:
        axes = np.array(axes).reshape(-1, 1)
    if nrows == 1:
        axes = np.array(axes).reshape(1, -1)

    for i, color_var in enumerate(color_vars):
        ri = i // ncols
        ci = i % ncols

        g = sns.scatterplot(
            x=x,
            y=y,
            data=embedding,
            linewidth=linewidth,
            s=size,
            hue=color_var,
            legend=legend_str,
            palette=palette,
            ax=axes[ri,ci],
            **kwargs,
        )
        sns.despine(offset=10, trim=True, ax=g)
        g.set(title=color_var)

        if legend and color_var:
            if is_numeric_dtype(embedding[color_var]):
                means = embedding.groupby(group_label)[color_var].mean() if group_label else embedding[color_var].mean()
                norm = plt.Normalize(means.min(), means.max())
                cmap = palette if palette is not None else sns.cubehelix_palette(as_cmap=True)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                try:
                    g.figure.colorbar(sm, ax=axes[ri, ci])
                    g.get_legend().remove()
                except Exception:
                    warn("Cannot make a proper colorbar")
            else:
                g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop=legend_prop)


    # Remove unused axes
    for i in range(len(color_vars), ncols * nrows):
        ri = i // ncols
        ci = i % ncols
        fig.delaxes(axes[ri,ci])

    plt.tight_layout()

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
                corr[i,j] = pearsonr(z[:,i], covariates.iloc[:,j])[value_index]
    else:
        # Inter-factor correlations
        corr = np.ndarray(shape=(z.shape[1], z.shape[1]))
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                corr[i,j] = pearsonr(z[:,i], z[:,j])[value_index]

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
    f, ax = plt.subplots(figsize=(11, 9))

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
    cbar_kws = {"shrink": .5}
    cbar_kws["label"] = "Correlation coefficient" if not pvalues else "-log10(adjusted p-value)"    

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(corr, cmap=cmap, mask=mask, center=center,
                    square=True, linewidths=.5,
                    xticklabels=xticklabels, yticklabels=yticklabels,
                    cbar_kws=cbar_kws, **kwargs)

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
    return plot_factors_correlation(model, covariates=covariates, pvalues=pvalues, square=False, **kwargs)



### VARIANCE EXPLAINED ###


def plot_r2(
    model: mofa_model,
    x="Group",
    y="Factor",
    factors: Union[int, List[int], str, List[str]] = None,
    groups_df: pd.DataFrame = None,
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
    groups_df : optional pd.DataFrame
        Data frame with samples (cells) as index and first column as group assignment
    cmap : optional
        The colourmap for the heatmap (default is 'Blues' with darker colour for higher R2)
    vmin : optional
        Display all R2 values smaller than vmin as vmin (0 by default)
    vmax : optional
        Display all R2 values larger than vmax as vmax (derived from the data by default)
    """
    r2 = model.get_r2(factors=factors, groups=groups, views=views, group_label=group_label, groups_df=groups_df)
    
    vmax = r2.R2.max() if vmax is None else vmax
    vmin = 0 if vmin is None else vmin

    split_by = [dim for dim in ["Group", "View", "Factor"] if dim not in [x, y]]
    assert len(split_by) == 1, "x and y values should be different and be one of Group, View, or Factor"
    split_by = split_by[0]

    split_by_items = r2[split_by].unique()
    fig, axes = plt.subplots(ncols=len(split_by_items), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
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
                sorted(r2_df.columns.categories, key=lambda x: int(x.split("Factor")[1]))
            )
            r2_df = r2_df[r2_df.columns.sort_values()]
        
        g = sns.heatmap(r2_df.sort_index(level=0, ascending=False), ax=axes[i], cmap=cmap, 
                        vmin=vmin, vmax=vmax,
                        cbar=i==0, cbar_ax=None if i else cbar_ax, **kwargs)
        
        axes[i].set_title(item)
        axes[i].tick_params(axis='both', which='both', length=0)
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

    if stacked:
        g = r2_df.pivot(index='Factor', columns='Group', values='R2').plot(kind='bar', stacked=True, linewidth=linewidth, **kwargs)
        plt.ylabel("R2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    else:
        # Grouped barplot
        g = sns.barplot(
            data=r2_df.sort_index(level=0, ascending=False), x=x, y=y, hue=groupby, linewidth=linewidth, **kwargs
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
