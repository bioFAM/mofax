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


### WEIGHTS ###


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
    x_offset=0.01,
    y_offset=0.15,
    jitter=0.01,
    line_width=0.5,
    line_color="black",
    line_alpha=0.2,
    zero_line=True,
    zero_line_width=1,
    ncols=4,
    sharex=True,
    sharey=False,
    **kwargs,
):
    """
    Plot weights for a specific factor

    Parameters
    ----------
    model : mofa_model
        An instance of the mofa_model class
    factors : str or int or list of str or None
        Factors to use (default is all)
    views : str or int or list of str or None
        The views to get the factors weights for (first view by default)
    n_features : int
        Number of features with the largest weights to label (in absolute values)
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
    sharex : bool
        If to use the same X axis across panels (True by default)
    sharey : bool
        If to use the same Y axis across panels (False by default)
    """
    w = model.get_weights(
        views=views,
        factors=factors,
        df=True,
        scale=w_scaled,
        absolute_values=w_abs,
    )
    wm = (
        w.join(model.features_metadata.loc[:, ["view"]])
        .rename_axis("feature")
        .reset_index()
        .melt(id_vars=["feature", "view"], var_name="factor", value_name="value")
    )

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
    features_to_label = model.get_top_features(
        factors=factors, views=views, n_features=n_features, df=True
    )
    features_to_label["to_label"] = True
    wm = (
        features_to_label.loc[:, ["feature", "view", "factor", "to_label"]]
        .set_index(["feature", "view", "factor"])
        .join(wm.set_index(["feature", "factor", "view"]), how="right")
        .reset_index()
        .fillna({"to_label": False})
        .sort_values(["factor", "to_label"])
    )

    # Figure out rows & columns for the grid with plots (one plot per view)
    view_vars = wm.view.unique()
    ncols = min(ncols, len(view_vars))
    nrows = int(np.ceil(len(view_vars) / ncols))
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

    for m, view in enumerate(view_vars):
        ri = m // ncols
        ci = m % ncols

        wm_view = wm.query("view == @view")

        # Construct the plot
        g = sns.stripplot(
            data=wm_view,
            x="value",
            y="factor",
            jitter=jitter,
            size=size,
            hue="to_label",
            palette=["lightgrey", color],
            ax=axes[ri, ci],
        )
        sns.despine(offset=10, trim=True, ax=g)
        g.legend().remove()

        # Label some points
        for fi, factor in enumerate(wm_view.factor.cat.categories):
            for sign_i in [1, -1]:
                to_label = features_to_label.query(
                    "factor == @factor & view == @view"
                ).feature.tolist()
                w_set = wm_view.query(
                    "factor == @factor & value * @sign_i > 0 & feature == @to_label & view == @view"
                ).sort_values("abs_value", ascending=False)

                x_start_pos = sign_i * (w_set.abs_value.max() + x_offset)
                y_start_pos = fi - ((w_set.shape[0] - 1) // 2) * y_offset
                y_prev = y_start_pos

                for i, row in enumerate(w_set.iterrows()):
                    name, point = row
                    y_loc = y_prev + y_offset if i != 0 else y_start_pos

                    g.annotate(
                        point["feature"],
                        xy=(point.value, fi),
                        xytext=(x_start_pos, y_loc),
                        arrowprops=dict(
                            arrowstyle="-",
                            connectionstyle="arc3",
                            color=line_color,
                            alpha=line_alpha,
                            linewidth=line_width,
                        ),
                        horizontalalignment="left" if sign_i > 0 else "right",
                        size=label_size,
                        color="black",
                        weight="regular",
                        alpha=0.9,
                    )
                    y_prev = y_loc

        # Set plot axes labels
        g.set(ylabel="", xlabel="Feature weight", title=view)

        if zero_line:
            axes[ri, ci].axvline(
                0, ls="--", color="lightgrey", linewidth=zero_line_width, zorder=0
            )

    # Remove unused axes
    for i in range(len(view_vars), ncols * nrows):
        ri = i // ncols
        ci = i % ncols
        fig.delaxes(axes[ri, ci])

    return g


def plot_weights_ranked(
    model: mofa_model,
    factor="Factor1",
    view=0,
    n_features: int = 10,
    size: int = 25,
    label_size=5,
    x_rank_offset=10,
    x_rank_offset_neg=0,
    y_repel_coef=0.03,
    attract_to_points=True,
    **kwargs,
):
    """
    Plot weights for a specific factor

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

    # Label top features

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
    Scatterplot of feature weights for two factors

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

    # top_features = wm.sort_values("factor", ascending=True).feature.values

    # Construct the plot
    ax = sns.scatterplot(data=w, x="x", y="y", linewidth=0, color="#CCCCCC", **kwargs)
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
    Plot weights for top features in a heatmap

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
        wm = wm.sort_values(["factor", "value_abs"], ascending=False).groupby(
            "factor", observed=False
        )
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
    palette=None,
    size: int = 30,
    linewidth: int = 1,
    xticklabels_size=8,
    yticklabels_size=5,
    ncols=1,
    sharex=True,
    sharey=False,
    **kwargs,
):
    """
    Plot weights for top features in a heatmap

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
    ncols : optional
        Number of columns when plotting multiple views (default is 1)
    sharex : bool
        If to use the same X axis across panels (True by default)
    sharey : bool
        If to use the same Y axis across panels (False by default)
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
        .join(model.features_metadata.loc[:, ["view"]])
        .reset_index()
    )
    wm = w.melt(id_vars=["feature", "view"], var_name="factor", value_name="value")
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
        wm_g = wm.sort_values(["factor", "value_abs"], ascending=False).groupby(
            ["factor", "view"],
            observed=False,
        )
        if w_threshold is None:
            features = wm_g.head(n_features).feature.unique()
        else:
            features = (
                wm_g[wm_g.value_abs >= w_threshold].head(n_features).feature.unique()
            )

    wm = wm[wm.feature.isin(features)]

    # Fix features order
    wm.feature = wm.feature.astype("category")
    wm.feature = wm.feature.cat.reorder_categories(features)

    wm = wm.sort_values(["factor", "feature"])

    # Figure out rows & columns for the grid with plots (one plot per view)
    view_vars = wm.view.unique()
    ncols = min(ncols, len(view_vars))
    nrows = int(np.ceil(len(view_vars) / ncols))
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

    for m, view in enumerate(view_vars):
        ri = m // ncols
        ci = m % ncols

        wm_view = wm.query("view == @view")

        # Construct the plot
        g = sns.scatterplot(
            data=wm_view,
            x="factor",
            y="feature",
            hue="value",
            linewidth=linewidth,
            s=size,
            palette=palette,
            ax=axes[ri, ci],
            **kwargs,
        )
        sns.despine(offset=10, trim=True, ax=g)
        g.legend().remove()

        norm = plt.Normalize(wm_view.value.min(), wm_view.value.max())
        cmap = (
            palette
            if palette is not None
            else sns.diverging_palette(220, 20, as_cmap=True)
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        try:
            g.figure.colorbar(sm, ax=axes[ri, ci])
            g.get_legend().remove()
        except Exception:
            warn("Cannot make a proper colorbar")

        plt.draw()

        g.set_title(view)

        g.set_xticklabels(g.get_xticklabels(), rotation=90, size=xticklabels_size)
        g.set_yticklabels(g.get_yticklabels(), size=yticklabels_size)

    # Remove unused axes
    for i in range(len(view_vars), ncols * nrows):
        ri = i // ncols
        ci = i % ncols
        fig.delaxes(axes[ri, ci])

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
    Plot weights for two factors

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
        wm = wm.sort_values(["factor", "value_abs"], ascending=False).groupby(
            "factor", observed=False
        )
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
            corr = corr[0:-n_cov, -n_cov:]

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
    g = sns.heatmap(
        corr,
        cmap=cmap,
        mask=mask,
        center=0,
        square=True,
        linewidths=0.5,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar_kws={"shrink": 0.5},
        **kwargs,
    )

    g.set_yticklabels(g.yaxis.get_ticklabels(), rotation=0)

    return g
